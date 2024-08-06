from dataclasses import dataclass
from typing import Optional, cast

import torch

from deserve_worker.kvcache.context import ForwardCtx
from deserve_worker.kvcache.page_pool import PagePool

from .kvcache import KVCache, KVCacheManager, main_device, main_dtype


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        page_pool: PagePool,
    ):
        self.page_pool = page_pool

    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += self.page_pool.block_size
        return cur

    def alloc(self, bsz: int, seqlen: int) -> Optional["PagedKVCache"]:
        len_token = self.get_kv_cache_length(0, seqlen)
        len_block = len_token // self.page_pool.block_size
        total_block = len_block * bsz
        blocks = self.page_pool.alloc(total_block)
        if blocks is None:
            return None
        else:
            return PagedKVCache(blocks.reshape(bsz, -1), self)

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PagedKVCache, kvcache)
        self.page_pool.recycle(kvcache.block_table.flatten())
        kvcache.block_table = torch.empty((0, 0), device=main_device, dtype=torch.int32)

    def renew(self, kvcache: KVCache, bsz: int, seqlen: int, start_pos: int) -> bool:
        kvcache = cast(PagedKVCache, kvcache)
        if (
            start_pos + seqlen
            > kvcache.block_table.shape[1] * self.page_pool.block_size
        ):
            len_block = (
                self.get_kv_cache_length(
                    kvcache.block_table.shape[1] * self.page_pool.block_size,
                    start_pos + seqlen,
                )
                // self.page_pool.block_size
            )
            total_block = (len_block - kvcache.block_table.shape[1]) * bsz
            blocks = self.page_pool.alloc(total_block)
            if blocks is None:
                return False
            else:
                new_block_table = torch.zeros(
                    (
                        bsz,
                        len_block,
                    ),
                    device=main_device,
                    dtype=torch.int32,
                )
                new_block_table[:, : kvcache.block_table.shape[1]] = (
                    kvcache.block_table[:, :]
                )
                new_block_table[:, kvcache.block_table.shape[1] :] = blocks.reshape(
                    bsz, -1
                )
                kvcache.block_table = new_block_table
        return True


class PagedKVCache(KVCache):
    def __init__(
        self,
        block_table: torch.Tensor,
        manager: PagedKVCacheManager,
    ):
        self.block_table = block_table
        self.manager = manager

    def renew(
        self,
        bsz: int,
        seqlen: int,
        start_pos: int,
    ) -> bool:
        return self.manager.renew(self, bsz, seqlen, start_pos)

    def clear(self) -> None:
        self.manager.recycle(self)

    def shape(self) -> torch.Size:
        return self.block_table.shape


@dataclass
class PagedForwardCtx(ForwardCtx):
    block_table: torch.Tensor
    kvcache_manager: KVCacheManager
