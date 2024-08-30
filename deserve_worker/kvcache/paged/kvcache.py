from typing import Generic

import torch
from typing_extensions import TypeVar

from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool, PagePool

P = TypeVar("P", bound="PagePool")


class PagedKVCache(Generic[P]):
    def __init__(self, page_table: torch.Tensor, pool: P) -> None:
        self.page_table = page_table
        self.pool = pool

    @staticmethod
    def empty(pool: P) -> "PagedKVCache[P]":
        return PagedKVCache(
            torch.empty((0,), device=pool.main_device, dtype=torch.int32), pool
        )

    def extend(self, size: int) -> bool:
        if size > self.page_table.shape[0] * self.pool.page_size:
            len = self.pool.calc_num_pages(size)
            delta = len - self.page_table.shape[0]
            page_indices = self.pool.alloc(delta)
            if page_indices is None:
                return False
            else:
                new_page_table = torch.empty(
                    (len,),
                    device=self.pool.main_device,
                    dtype=torch.int32,
                )
                new_page_table[: self.page_table.shape[0]] = self.page_table[:]
                new_page_table[self.page_table.shape[0] :] = page_indices
                self.page_table = new_page_table
        return True

    def free(self) -> None:
        self.pool.free(self.page_table)
        self.page_table = torch.empty(
            (0,), device=self.pool.main_device, dtype=torch.int32
        )


GpuPagedKVCache = PagedKVCache[GpuPagePool]
CpuPagedKVCache = PagedKVCache[CpuPagePool]
