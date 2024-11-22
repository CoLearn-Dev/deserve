from typing import Generic

import torch
from typing_extensions import TypeVar

from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool, PagePool
from deserve_worker.kvcache.paged.page_table import PageTableAllocator, PageTableHandle

P = TypeVar("P", bound="PagePool")


class PagedKVCache(Generic[P]):
    def __init__(self, page_table: PageTableHandle, pool: P) -> None:
        self.page_table = page_table
        self.pool = pool

    @staticmethod
    def empty(allocator: PageTableAllocator, pool: P) -> "PagedKVCache[P]":
        return PagedKVCache(allocator.alloc(), pool)

    def extend(self, num_tokens: int) -> bool:
        if num_tokens > self.page_table.occupied * self.pool.page_size:
            new_num_pages = self.pool.calc_num_pages(num_tokens)
            delta = new_num_pages - self.page_table.occupied
            page_indices = self.pool.alloc(delta)
            if page_indices is None:
                return False
            self.page_table.extend(page_indices)
        return True

    def free(self) -> None:
        self.pool.free(self.page_table.retrieve())
        self.page_table.free()


GpuPagedKVCache = PagedKVCache[GpuPagePool]
CpuPagedKVCache = PagedKVCache[CpuPagePool]
