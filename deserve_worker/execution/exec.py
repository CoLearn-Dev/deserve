from dataclasses import dataclass

import torch
from flashinfer import (  # type: ignore
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from deserve_worker.execution.response import (
    BatchResult,
    BatchUpdate,
    LLMResponse,
    TraceResult,
)
from deserve_worker.kvcache.kvcache import KVCacheManager, main_device
from deserve_worker.kvcache.packed_kvcache import PackedKVCacheManager
from deserve_worker.kvcache.page_pool import PagePool
from deserve_worker.kvcache.paged_kvcache import PagedKVCacheManager
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.model.context.paged import PagedDecodeCtx, PagedPrefillCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.task import TaskData
from deserve_worker.trace import OpId

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=main_device)
decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, enable_flash_attn: bool = False
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    if enable_flash_attn:
        freqs_cis = torch.stack([freqs.cos(), freqs.sin()])  # flash_attn
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0, False).to(main_device)
flash_global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0, True).to(main_device)


@dataclass
class LLMExec:
    xs: torch.Tensor  # stored in ragged format or packed format
    layer_storage: LayerStorage
    page_pool: PagePool
    task_datas: list[TaskData]
    bsz: int

    def post_process(self, result: torch.Tensor) -> list[LLMResponse]:
        responses: list[LLMResponse] = []
        if self.layer_storage.need_sample:
            ongoing_tokens, ongoing_ids, all_tokens, all_ids, done_ids = (
                self.layer_storage.sample(result, self.task_datas)
            )
            if len(ongoing_tokens) > 0:
                responses.append(BatchResult(torch.cat(ongoing_tokens), ongoing_ids))
            responses.append(BatchUpdate(all_tokens, all_ids, done_ids))
        else:
            for task in self.task_datas:
                task.start_pos += task.seqlen  # this has also be done in sampling
            responses.append(
                BatchResult(result, [task.task_id for task in self.task_datas])
            )
        return responses


@dataclass
class BatchDecode(LLMExec):
    """
    The tensor is stored in ragged format.
    """

    @staticmethod
    def merge(decodes: list["BatchDecode"]) -> "BatchDecode":
        xs = torch.cat([decode.xs for decode in decodes])
        layer_storage = decodes[0].layer_storage
        task_datas = [
            task_data for decode in decodes for task_data in decode.task_datas
        ]
        page_pool = decodes[0].page_pool
        return BatchDecode(xs, layer_storage, page_pool, task_datas, len(task_datas))

    def step(self) -> list[LLMResponse]:
        with torch.inference_mode():
            decode_ctx = PagedDecodeCtx.init_paged_decode_ctx(
                self.page_pool, self.task_datas, decode_wrapper
            )
            decode_wrapper.begin_forward(
                indptr=decode_ctx.kv_page_indptr,
                indices=decode_ctx.kv_page_indices,
                last_page_len=decode_ctx.kv_last_page_lens,
                num_qo_heads=64,
                num_kv_heads=8,
                head_dim=128,
                page_size=256,
            )
            result = self.layer_storage.forward(self.xs, decode_ctx)
            decode_wrapper.end_forward()
            return self.post_process(result)


@dataclass
class BatchPrefill(LLMExec):
    """
    The tensor is stored in ragged format.
    """

    seqlens: list[int]
    total_seqlen: int

    @staticmethod
    def merge(prefills: list["BatchPrefill"]) -> "BatchPrefill":
        xs = torch.cat([prefill.xs for prefill in prefills])
        task_datas = [
            task_data for prefill in prefills for task_data in prefill.task_datas
        ]
        seqlens = [task_data.seqlen for task_data in task_datas]
        layer_storage = prefills[0].layer_storage
        page_pool = prefills[0].page_pool
        return BatchPrefill(
            xs,
            layer_storage,
            page_pool,
            task_datas,
            len(task_datas),
            seqlens,
            total_seqlen=sum(seqlens),
        )

    def step(self) -> list[LLMResponse]:
        with torch.inference_mode():
            prefill_ctx = PagedPrefillCtx.init_page_prefill_ctx(
                self.page_pool,
                self.task_datas,
                self.seqlens,
                prefill_wrapper,
            )
            prefill_wrapper.begin_forward(
                qo_indptr=prefill_ctx.indptr,
                paged_kv_indptr=prefill_ctx.kv_page_indptr,
                paged_kv_indices=prefill_ctx.kv_page_indices,
                paged_kv_last_page_len=prefill_ctx.kv_last_page_lens,
                num_qo_heads=64,
                num_kv_heads=8,
                head_dim=128,
                page_size=256,
            )
            result = self.layer_storage.forward(self.xs, prefill_ctx)
            prefill_wrapper.end_forward()
            return self.post_process(result)


@dataclass
class SingleTrace(LLMExec):
    """
    The tensor is stored in packed format. Besides, these kinds of reequests are not allowed to be batched.
    """

    traces: dict[OpId, torch.Tensor]

    def step(self) -> list[LLMResponse]:
        with torch.inference_mode():
            forward_ctx = TraceForwardCtx.init_trace_forward_ctx(
                self.page_pool,
                self.task_datas,
                [task_data.seqlen for task_data in self.task_datas],
                global_freqs_cis,
                self.traces,
            )
            result = self.layer_storage.forward(self.xs, forward_ctx).view(
                self.task_datas[0].seqlen, -1
            )
            prefill_wrapper.end_forward()
            return self.post_process(result)

    def post_process(self, result: torch.Tensor) -> list[LLMResponse]:
        responses: list[LLMResponse] = []
        if self.layer_storage.need_sample:
            _, _, all_tokens, all_ids, _ = self.layer_storage.sample(
                result, self.task_datas
            )
            assert len(all_tokens) == 1
            responses.append(TraceResult(all_tokens[0], all_ids[0], self.traces))
        else:
            for task_data in self.task_datas:
                task_data.start_pos += task_data.seqlen
            assert len(self.task_datas) == 1
            responses.append(
                TraceResult(result, self.task_datas[0].task_id, self.traces)
            )
        return responses
