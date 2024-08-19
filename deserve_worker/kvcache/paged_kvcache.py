from dataclasses import dataclass
from typing import Optional, cast

import torch

from deserve_worker.kvcache.page_pool import PagePool, calc_pages_needed_num

from .kvcache import KVCache, KVCacheManager, PersistentKVCache, main_device


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        page_pool: PagePool,
    ):
        self.page_pool = page_pool
        self.block_size = page_pool.page_size

    def new(self) -> "PagedKVCache":
        return PagedKVCache(
            torch.empty((0,), device=main_device, dtype=torch.int32), self
        )

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PagedKVCache, kvcache)
        self.page_pool.recycle(kvcache.block_table)
        kvcache.block_table = torch.empty((0,), device=main_device, dtype=torch.int32)

    def renew(self, kvcache: KVCache, total_len: int) -> bool:
        kvcache = cast(PagedKVCache, kvcache)
        if total_len > kvcache.block_table.shape[0] * self.block_size:
            len_block = calc_pages_needed_num(total_len, self.block_size)
            delta_block = len_block - kvcache.block_table.shape[0]
            blocks = self.page_pool.alloc(delta_block)
            if blocks is None:
                return False
            else:
                new_block_table = torch.empty(
                    (len_block,),
                    device=main_device,
                    dtype=torch.int32,
                )
                new_block_table[: kvcache.block_table.shape[0]] = kvcache.block_table[:]
                new_block_table[kvcache.block_table.shape[0] :] = blocks
                kvcache.block_table = new_block_table
        return True

    def reinsert(self, kvcache: PersistentKVCache) -> Optional["PagedKVCache"]:
        num_pages = kvcache.get_num_pages()
        pages = self.page_pool.alloc(num_pages)
        if pages is None:
            return None
        else:
            for i in range(len(kvcache.storage_k)):
                self.page_pool.pages_k[i][pages] = kvcache.storage_k[i].to(main_device)
                self.page_pool.pages_v[i][pages] = kvcache.storage_v[i].to(main_device)
            return PagedKVCache(pages, self)


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

    def into_persistent(self) -> PersistentKVCache:
        storage_k, storage_v = self.manager.page_pool.retrieve(self.block_table)
        storage_k = [storage.cpu() for storage in storage_k]
        storage_v = [storage.cpu() for storage in storage_v]
        persistent = PersistentKVCache(storage_k, storage_v, self.manager)
        self.clear()
        return persistent
