from dataclasses import dataclass
from typing import Sequence

import torch
from flashinfer import (  # type: ignore
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from deserve_worker.execution.result import (
    BatchAct,
    BatchUpdate,
    ExecResult,
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
class BatchExec:
    xs: torch.Tensor  # stored in ragged format or packed format
    layer_storage: LayerStorage
    page_pool: PagePool
    task_datas: list[TaskData]

    def bsz(self) -> int:
        return len(self.task_datas)

    def seqlens(self) -> list[int]:
        return [task_data.seqlen for task_data in self.task_datas]

    def total_seqlen(self) -> int:
        return sum(self.seqlens())

    @staticmethod
    def merge(execs: Sequence["BatchExec"]) -> "BatchExec":
        xs = torch.cat([exec.xs for exec in execs])
        layer_storage = execs[0].layer_storage
        page_pool = execs[0].page_pool
        task_datas = [task_data for exec in execs for task_data in exec.task_datas]
        return BatchExec(xs, layer_storage, page_pool, task_datas)

    def post_process(self, result: torch.Tensor) -> list[ExecResult]:
        responses: list[ExecResult] = []
        if self.layer_storage.need_sample:
            ongoing_tokens, ongoing_datas, all_tokens, all_datas, done_datas = (
                self.layer_storage.sample(result, self.task_datas)
            )
            if len(ongoing_tokens) > 0:
                responses.append(BatchAct(ongoing_datas, torch.cat(ongoing_tokens)))
            responses.append(BatchUpdate(all_datas, all_tokens, done_datas))
        else:
            for task in self.task_datas:
                task.start_pos += task.seqlen  # this has also be done in sampling
            responses.append(BatchAct(self.task_datas, result))
        return responses


@dataclass
class BatchDecode(BatchExec):
    """
    The tensor is stored in ragged format.
    """

    @staticmethod
    def merge(decodes: Sequence["BatchExec"]) -> "BatchDecode":
        exec = BatchExec.merge(decodes)
        return BatchDecode(exec.xs, exec.layer_storage, exec.page_pool, exec.task_datas)

    def step(self) -> list[ExecResult]:
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
class BatchPrefill(BatchExec):
    """
    The tensor is stored in ragged format.
    """

    @staticmethod
    def merge(prefills: Sequence["BatchExec"]) -> "BatchPrefill":
        exec = BatchExec.merge(prefills)
        return BatchPrefill(
            exec.xs, exec.layer_storage, exec.page_pool, exec.task_datas
        )

    def step(self) -> list[ExecResult]:
        with torch.inference_mode():
            prefill_ctx = PagedPrefillCtx.init_page_prefill_ctx(
                self.page_pool,
                self.task_datas,
                self.seqlens(),
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
class SingleTrace(BatchExec):
    """
    The tensor is stored in packed format. Besides, these kinds of reequests are not allowed to be batched.
    """

    traces: dict[OpId, torch.Tensor]

    def step(self) -> list[ExecResult]:
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

    def post_process(self, result: torch.Tensor) -> list[ExecResult]:
        responses: list[ExecResult] = []
        if self.layer_storage.need_sample:
            _, _, all_tokens, all_datas, _ = self.layer_storage.sample(
                result, self.task_datas
            )
            assert len(all_tokens) == 1
            responses.append(TraceResult(all_datas, all_tokens[0], self.traces))
        else:
            for task_data in self.task_datas:
                task_data.start_pos += task_data.seqlen
            assert len(self.task_datas) == 1
            responses.append(TraceResult(self.task_datas, result, self.traces))
        return responses
