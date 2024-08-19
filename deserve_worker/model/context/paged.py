from dataclasses import dataclass
from typing import cast

import torch
from flashinfer import (  # type: ignore
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from deserve_worker.kvcache.kvcache import PersistentKVCache, main_device
from deserve_worker.kvcache.page_pool import PagePool
from deserve_worker.kvcache.paged_kvcache import PagedKVCache, PagedKVCacheManager
from deserve_worker.model.args import ModelArgs
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.task import TaskData


@dataclass
class PagedForwardCtx(ForwardCtx):
    indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_page_indptr: torch.Tensor
    kv_last_page_lens: torch.Tensor

    @staticmethod
    def init_paged_forward_ctx(
        page_pool: PagePool, task_datas: list[TaskData], seqlens: list[int]
    ) -> "PagedForwardCtx":
        kv_last_page_lens = []
        for task_data, seqlen in zip(task_datas, seqlens):
            if isinstance(task_data.kvcache, PersistentKVCache):
                print(f"Reload {task_data.task_id} to GPU")
                kvcache = task_data.kvcache.into_memory()
                assert isinstance(kvcache, PagedKVCache)
                task_data.kvcache = kvcache
            else:
                kvcache = cast(PagedKVCache, task_data.kvcache)

            total_len = task_data.start_pos + seqlen
            if not kvcache.renew(total_len):
                raise RuntimeError("KV cache renew failed")
            kv_last_page_lens.append((total_len - 1) % kvcache.manager.block_size + 1)
        kvcache_list = [
            cast(PagedKVCache, task_data.kvcache).block_table.view(-1)
            for task_data in task_datas
        ]

        len_list = [0] + [kvcache.shape[0] for kvcache in kvcache_list]
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
            page_pool=page_pool,
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
        page_pool: PagePool,
        task_datas: list[TaskData],
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
    ) -> "PagedDecodeCtx":
        seqlens = [1 for _ in task_datas]
        forward_ctx = PagedForwardCtx.init_paged_forward_ctx(
            page_pool, task_datas, seqlens
        )
        return PagedDecodeCtx(
            page_pool=page_pool,
            offsets=forward_ctx.offsets,
            bsz=forward_ctx.bsz,
            seqlens=forward_ctx.seqlens,
            layer_id=0,
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
    def init_page_prefill_ctx(
        page_pool: PagePool,
        task_datas: list[TaskData],
        seqlens: list[int],
        wrapper: BatchPrefillWithPagedKVCacheWrapper,
    ) -> "PagedPrefillCtx":
        forward_ctx = PagedForwardCtx.init_paged_forward_ctx(
            page_pool, task_datas, seqlens
        )
        return PagedPrefillCtx(
            page_pool=page_pool,
            offsets=forward_ctx.offsets,
            bsz=forward_ctx.bsz,
            seqlens=forward_ctx.seqlens,
            layer_id=0,
            indptr=forward_ctx.indptr,
            kv_page_indices=forward_ctx.kv_page_indices,
            kv_page_indptr=forward_ctx.kv_page_indptr,
            kv_last_page_lens=forward_ctx.kv_last_page_lens,
            prefill_wrapper=wrapper,
        )
