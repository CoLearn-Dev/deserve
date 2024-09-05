from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from flashinfer import (  # type: ignore
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from deserve_worker.execution.result import BatchResult
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import GpuPagePool
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.model.context.flash import FlashDecodeCtx, FlashPrefillCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.task import TaskData, main_device
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
    task_datas: list[TaskData]
    kvcaches: list[PagedKVCache[GpuPagePool]]

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
        task_datas = [task_data for exec in execs for task_data in exec.task_datas]
        kvcaches = [kvcaches for exec in execs for kvcaches in exec.kvcaches]
        return BatchExec(xs, layer_storage, task_datas, kvcaches)

    def split(self) -> Sequence["BatchExec"]:
        start_pos = 0
        execs = []
        for task_data, kvcaches in zip(self.task_datas, self.kvcaches):
            execs.append(
                BatchExec(
                    self.xs[start_pos : start_pos + task_data.seqlen],
                    self.layer_storage,
                    [task_data],
                    [kvcaches],
                )
            )
            start_pos += task_data.seqlen
        return execs

    def pop(self) -> "BatchExec":
        task_data = self.task_datas.pop()
        kvcaches = self.kvcaches.pop()
        seqlen = task_data.seqlen
        result = BatchExec(
            self.xs[-seqlen:],
            self.layer_storage,
            [task_data],
            [kvcaches],
        )
        self.xs = self.xs[:-seqlen]
        return result

    def post_process(self, result: torch.Tensor) -> BatchResult:
        if self.layer_storage.need_sample:
            (
                ongoing_tokens,
                ongoing_datas,
                all_tokens,
                all_datas,
                done_datas,
                needed_probs,
            ) = self.layer_storage.sample(result, self.task_datas)
            if len(ongoing_tokens) == 0:
                xs = torch.empty(0, dtype=torch.int, device=main_device)
            else:
                xs = torch.cat(ongoing_tokens)
            return BatchResult(
                xs,
                [task_data.task_id for task_data in ongoing_datas],
                torch.cat(all_tokens),
                [task_data.task_id for task_data in all_datas],
                [task_data.task_id for task_data in done_datas],
                {
                    task_data.task_id: needed_prob
                    for task_data, needed_prob in needed_probs
                },
            )
        else:
            return BatchResult(
                result,
                [task_data.task_id for task_data in self.task_datas],
                result,
                [task_data.task_id for task_data in self.task_datas],
                [],
                {},
            )


@dataclass
class BatchDecode(BatchExec):
    """
    The tensor is stored in ragged format.
    """

    @staticmethod
    def merge(decodes: Sequence["BatchExec"]) -> "BatchDecode":
        exec = BatchExec.merge(decodes)
        return BatchDecode(exec.xs, exec.layer_storage, exec.task_datas, exec.kvcaches)

    def split(self) -> Sequence["BatchDecode"]:
        results = super().split()
        return [
            BatchDecode(
                result.xs, result.layer_storage, result.task_datas, result.kvcaches
            )
            for result in results
        ]

    def pop(self) -> "BatchDecode":
        result = super().pop()
        return BatchDecode(
            result.xs, result.layer_storage, result.task_datas, result.kvcaches
        )

    def step(self) -> BatchResult:
        with torch.inference_mode():
            decode_ctx = FlashDecodeCtx.init_paged_decode_ctx(
                self.task_datas, self.kvcaches, decode_wrapper
            )
            model_args = self.layer_storage.model_args
            decode_wrapper.begin_forward(
                indptr=decode_ctx.kv_page_indptr,
                indices=decode_ctx.kv_page_indices,
                last_page_len=decode_ctx.kv_last_page_lens,
                num_qo_heads=model_args.n_heads,
                num_kv_heads=model_args.n_kv_heads,
                head_dim=model_args.dim // model_args.n_heads,
                page_size=model_args.page_size,
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
        return BatchPrefill(exec.xs, exec.layer_storage, exec.task_datas, exec.kvcaches)

    def pop(self) -> "BatchPrefill":
        result = super().pop()
        return BatchPrefill(
            result.xs, result.layer_storage, result.task_datas, result.kvcaches
        )

    def step(self) -> BatchResult:
        with torch.inference_mode():
            prefill_ctx = FlashPrefillCtx.init_paged_prefill_ctx(
                self.task_datas,
                self.kvcaches,
                prefill_wrapper,
            )
            model_args = self.layer_storage.model_args
            prefill_wrapper.begin_forward(
                qo_indptr=prefill_ctx.indptr,
                paged_kv_indptr=prefill_ctx.kv_page_indptr,
                paged_kv_indices=prefill_ctx.kv_page_indices,
                paged_kv_last_page_len=prefill_ctx.kv_last_page_lens,
                num_qo_heads=model_args.n_heads,
                num_kv_heads=model_args.n_kv_heads,
                head_dim=model_args.dim // model_args.n_heads,
                page_size=model_args.page_size,
            )
            result = self.layer_storage.forward(self.xs, prefill_ctx)
            prefill_wrapper.end_forward()
            return self.post_process(result)


@dataclass
class SingleTrace:
    xs: torch.Tensor  # stored in ragged format or packed format
    layer_storage: LayerStorage
    task_datas: list[TaskData]
    traces: dict[OpId, torch.Tensor]
    output2input: dict[OpId, list[OpId]]

    def step(self) -> BatchResult:
        with torch.inference_mode():
            forward_ctx = TraceForwardCtx.init_trace_forward_ctx(
                self.task_datas,
                global_freqs_cis,
                self.traces,
                self.output2input,
            )
            result = self.layer_storage.forward(self.xs, forward_ctx).squeeze(dim=0)
            return self.post_process(result)

    def post_process(self, result: torch.Tensor) -> BatchResult:
        if self.layer_storage.need_sample:
            _, _, all_tokens, all_datas, _, needed_probs = self.layer_storage.sample(
                result, self.task_datas
            )
            xs = torch.empty(0, dtype=torch.int, device=main_device)
            return BatchResult(
                xs,
                [],
                torch.cat(all_tokens),
                [task_data.task_id for task_data in all_datas],
                [task_data.task_id for task_data in all_datas],
                {
                    task_data.task_id: needed_prob
                    for task_data, needed_prob in needed_probs
                },
            )  # because we only serve one request in batch
        else:
            return BatchResult(
                result,
                [task_data.task_id for task_data in self.task_datas],
                result,
                [task_data.task_id for task_data in self.task_datas],
                [],
                {},
            )
