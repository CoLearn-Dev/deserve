from typing import TypeVar

import torch

from deserve_worker.kvcache.manager.portable import PortableKVCacheManager
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool, PagePool
from deserve_worker.kvcache.pinned.kvcache import PinnedKVCache
from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory

P = TypeVar("P", bound="PagePool")


class BulkKVCacheManager(PortableKVCacheManager):
    def __init__(
        self,
        gpu_page_pool: GpuPagePool,
        cpu_page_pool: CpuPagePool,
        pinned_memory: PinnedMemory,
    ):
        super().__init__(gpu_page_pool, cpu_page_pool)
        self.pinned_memory = pinned_memory

    def copy_paged_to_pinned(
        self,
        paged_kvcaches: list[PagedKVCache[P]],
    ) -> list[PinnedKVCache]:
        """
        Non-blocking is always enabled for copying.
        In this function, we do not clean up paged kvcaches, because the copy is async.
        The user should clean up paged kvcaches after stream synchronization.
        """
        if len(paged_kvcaches) == 0:
            return []

        # get page tables and begin to copy
        page_pool = paged_kvcaches[0].pool
        page_tables = torch.cat(
            [paged_kvcache.page_table for paged_kvcache in paged_kvcaches]
        )
        pages = page_pool.pages[:, page_tables, :, :, :, :].view(-1)
        self.pinned_memory.memory[: pages.shape[0]].copy_(pages, non_blocking=True)
        self.pinned_memory.size = pages.shape[0]

        # generate pinned kvcaches for recovery
        ptr = 0
        pinned_kvcaches = []
        for paged_kvcache in paged_kvcaches:
            page_table = paged_kvcache.page_table
            page_table_len = page_table.shape[0]
            pinned_kvcaches.append(
                PinnedKVCache(
                    memory=self.pinned_memory,
                    ptr=ptr,
                    size=page_table_len,
                )
            )
            ptr += page_table_len

        return pinned_kvcaches

    def copy_pinned_to_paged(
        self,
        pinned_kvcaches: list[PinnedKVCache],
    ) -> list[PagedKVCache[GpuPagePool]]:
        """
        Non-blocking is always enabled for copying.
        """
        if len(pinned_kvcaches) == 0:
            return []

        page_tables = self.gpu_page_pool.alloc(self.pinned_memory.size)
        if page_tables is None:
            raise RuntimeError("Failed to allocate page tables")
        self.gpu_page_pool.pages[:, page_tables, :, :, :, :].copy_(
            self.pinned_memory.memory[: self.pinned_memory.size], non_blocking=True
        )
        self.pinned_memory.size = 0
        paged_kvcaches = []
        for pinned_kvcache in pinned_kvcaches:
            page_table = page_tables[
                pinned_kvcache.ptr : pinned_kvcache.ptr + pinned_kvcache.size
            ]
            paged_kvcaches.append(
                PagedKVCache(
                    page_table=page_table,
                    pool=self.gpu_page_pool,
                )
            )
        return paged_kvcaches
