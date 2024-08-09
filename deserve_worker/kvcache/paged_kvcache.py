from dataclasses import dataclass
from typing import Optional, cast

import torch
from flashinfer import (  # type: ignore
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from deserve_worker.kvcache.page_pool import PagePool

from .kvcache import KVCache, KVCacheManager, main_device


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        page_pool: PagePool,
    ):
        self.page_pool = page_pool
        self.block_size = page_pool.block_size

    def alloc(self, total_len: int) -> Optional["PagedKVCache"]:
        len_block = (total_len + self.block_size - 1) // self.page_pool.block_size
        blocks = self.page_pool.alloc(len_block)
        if blocks is None:
            return None
        else:
            return PagedKVCache(blocks.view(1, -1), self)

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PagedKVCache, kvcache)
        self.page_pool.recycle(kvcache.block_table.flatten())
        kvcache.block_table = torch.empty((0, 0), device=main_device, dtype=torch.int32)

    def renew(self, kvcache: KVCache, total_len: int) -> bool:
        kvcache = cast(PagedKVCache, kvcache)
        if total_len > kvcache.block_table.shape[1] * self.block_size:
            len_block = (total_len + self.block_size - 1) // self.page_pool.block_size
            delta_block = len_block - kvcache.block_table.shape[1]
            blocks = self.page_pool.alloc(delta_block)
            if blocks is None:
                return False
            else:
                new_block_table = torch.empty(
                    (
                        1,
                        len_block,
                    ),
                    device=main_device,
                    dtype=torch.int32,
                )
                new_block_table[:, : kvcache.block_table.shape[1]] = (
                    kvcache.block_table[:, :]
                )
                new_block_table[:, kvcache.block_table.shape[1] :] = blocks.view(1, -1)
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

    def renew(self, total_len: int) -> bool:
        return self.manager.renew(self, total_len)

    def clear(self) -> None:
        self.manager.recycle(self)

    def shape(self) -> torch.Size:
        return self.block_table.shape
