from dataclasses import dataclass
from typing import cast

import torch
from flashinfer import (  # type: ignore
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import GpuPagePool
from deserve_worker.model.args import ModelArgs
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.task import TaskData, main_device


@dataclass
class PagedForwardCtx(ForwardCtx):
    indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_page_indptr: torch.Tensor
    kv_last_page_lens: torch.Tensor

    @staticmethod
    def init_paged_forward_ctx(
        task_datas: list[TaskData],
        kvcaches: list[PagedKVCache[GpuPagePool]],
    ) -> "PagedForwardCtx":
        kv_last_page_lens = []
        for task_data, kvcache in zip(task_datas, kvcaches):
            total_len = task_data.start_pos + task_data.seqlen
            kvcache.extend(total_len)
            kv_last_page_lens.append((total_len - 1) % kvcache.pool.page_size + 1)
        kvcache_list = [kvcache.page_table for kvcache in kvcaches]

        len_list = [0] + [kvcache.shape[0] for kvcache in kvcache_list]
        seqlens = [task_data.seqlen for task_data in task_datas]
        kv_page_indices = torch.cat(kvcache_list).view(-1)
        kv_page_indptr = torch.tensor(
            len_list, dtype=torch.int32, device=main_device
        ).cumsum(dim=0, dtype=torch.int32)
        indptr = torch.tensor(
            [0] + seqlens, dtype=torch.int32, device=main_device
        ).cumsum(dim=0, dtype=torch.int32)
        last_page_lens_tch = torch.tensor(
            kv_last_page_lens, dtype=torch.int32, device=main_device
        )
        offsets = torch.tensor(
            [task.start_pos for task in task_datas],
            device=main_device,
            dtype=torch.int32,
        )
        return PagedForwardCtx(
            page_pool=kvcaches[0].pool,
            offsets=offsets,
            bsz=len(task_datas),
            seqlens=torch.tensor(seqlens, dtype=torch.int32, device=main_device),
            layer_id=0,
            indptr=indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=last_page_lens_tch,
        )


@dataclass
class PagedDecodeCtx(PagedForwardCtx):
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper

    @staticmethod
    def init_paged_decode_ctx(
        task_datas: list[TaskData],
        kvcaches: list[PagedKVCache[GpuPagePool]],
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
    ) -> "PagedDecodeCtx":
        forward_ctx = PagedForwardCtx.init_paged_forward_ctx(task_datas, kvcaches)
        return PagedDecodeCtx(
            page_pool=forward_ctx.page_pool,
            offsets=forward_ctx.offsets,
            bsz=forward_ctx.bsz,
            seqlens=forward_ctx.seqlens,
            layer_id=forward_ctx.layer_id,
            indptr=forward_ctx.indptr,
            kv_page_indices=forward_ctx.kv_page_indices,
            kv_page_indptr=forward_ctx.kv_page_indptr,
            kv_last_page_lens=forward_ctx.kv_last_page_lens,
            decode_wrapper=wrapper,
        )


@dataclass
class PagedPrefillCtx(PagedForwardCtx):
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper

    @staticmethod
    def init_paged_prefill_ctx(
        task_datas: list[TaskData],
        kvcaches: list[PagedKVCache[GpuPagePool]],
        wrapper: BatchPrefillWithPagedKVCacheWrapper,
    ) -> "PagedPrefillCtx":
        forward_ctx = PagedForwardCtx.init_paged_forward_ctx(task_datas, kvcaches)
        return PagedPrefillCtx(
            page_pool=forward_ctx.page_pool,
            offsets=forward_ctx.offsets,
            bsz=forward_ctx.bsz,
            seqlens=forward_ctx.seqlens,
            layer_id=forward_ctx.layer_id,
            indptr=forward_ctx.indptr,
            kv_page_indices=forward_ctx.kv_page_indices,
            kv_page_indptr=forward_ctx.kv_page_indptr,
            kv_last_page_lens=forward_ctx.kv_last_page_lens,
            prefill_wrapper=wrapper,
        )
