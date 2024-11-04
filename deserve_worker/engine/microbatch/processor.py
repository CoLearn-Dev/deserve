from typing import cast

import torch

from deserve_worker.execution.exec import BatchDecode, BatchPrefill
from deserve_worker.execution.result import BatchResult
from deserve_worker.kvcache.manager import KVCacheManager
from deserve_worker.kvcache.paged.chunk_pool import ChunkHandle
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool
from deserve_worker.kvcache.pinned.kvcache import PinnedKVCache
from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory
from deserve_worker.kvcache.virtual import (
    CopiedPinnedMemory,
    VirtualPagedKVCache,
    VirtualPagePool,
)
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.task import TaskManager


class MicroBatchProcessor:
    def __init__(
        self,
        kvcache_manager: KVCacheManager,
        task_data_manager: TaskManager,
        layer_storage: LayerStorage,
    ) -> None:
        self.kvcache_manager = kvcache_manager
        self.pinned_memory = CopiedPinnedMemory(
            size=kvcache_manager.virtual_page_pool.calc_swap_size(),
            num_pages_swap=kvcache_manager.virtual_page_pool.num_pages_swap,
            device=kvcache_manager.virtual_page_pool.main_device,
        )
        self.task_data_manager = task_data_manager
        self.layer_storage = layer_storage
        self.ongoing_paged_kvcaches: dict[str, VirtualPagedKVCache] = {}
        self.cuda_stream = torch.cuda.Stream()  # type: ignore

    def join(
        self,
        task_ids: list[str],
    ) -> None:
        for task_id in task_ids:
            self.ongoing_paged_kvcaches[task_id] = VirtualPagedKVCache.empty(
                self.kvcache_manager.virtual_page_pool
            )

    def adjust(self) -> None:
        with torch.inference_mode():
            vpp = self.kvcache_manager.virtual_page_pool
            offset = vpp.num_pages_swap if vpp.on == 1 else -vpp.num_pages_swap
            for paged_kvcache in self.ongoing_paged_kvcaches.values():
                paged_kvcache.adjust(offset)

    def offload(self, task_ids: list[str]) -> list[ChunkHandle]:
        """
        Suspend the execution of the task in a blocking manner.
        """
        paged_kvcaches = [
            self.ongoing_paged_kvcaches.pop(task_id) for task_id in task_ids
        ]
        cpu_chunk_handles = [
            self.kvcache_manager.copy_gpu_to_cpu(paged_kvcache)
            for paged_kvcache in paged_kvcaches
        ]
        for paged_kvcache in paged_kvcaches:
            paged_kvcache.free()
        return cpu_chunk_handles

    def reload(self, task_ids: list[str], cpu_chunk_handles: list[ChunkHandle]) -> None:
        """
        Resume the execution of the task in a blocking manner.
        """
        gpu_paged_kvcaches = [
            self.kvcache_manager.copy_cpu_to_gpu(cpu_chunk_handle)
            for cpu_chunk_handle in cpu_chunk_handles
        ]
        for task_id, gpu_paged_kvcache in zip(task_ids, gpu_paged_kvcaches):
            self.ongoing_paged_kvcaches[task_id] = gpu_paged_kvcache
        for cpu_chunk_handle in cpu_chunk_handles:
            cpu_chunk_handle.free()

    def cancel(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            if task_id in self.ongoing_paged_kvcaches:
                gpu_paged_kvcache = self.ongoing_paged_kvcaches.pop(task_id)
                gpu_paged_kvcache.free()

    def exec(self, xs: torch.Tensor, task_ids: list[str]) -> BatchResult:
        task_datas = [self.task_data_manager.get(task_id) for task_id in task_ids]
        kvcaches = [self.ongoing_paged_kvcaches[task_id] for task_id in task_ids]
        decode = BatchDecode(
            xs,
            self.layer_storage,
            task_datas,
            cast(list[PagedKVCache[GpuPagePool]], kvcaches),
        )
        return decode.step()

    def step(
        self,
        task_ids: list[str],
        xs: torch.Tensor,
        prev_group: "MicroBatchProcessor",
        next_group: "MicroBatchProcessor",
    ) -> BatchResult:
        task_datas = [self.task_data_manager.get(task_id) for task_id in task_ids]
        kvcaches = [self.ongoing_paged_kvcaches[task_id] for task_id in task_ids]
        all_decode = all(task_data.seqlen == 1 for task_data in task_datas)
        if all_decode:
            decode = BatchDecode(
                xs,
                self.layer_storage,
                task_datas,
                cast(list[PagedKVCache[GpuPagePool]], kvcaches),
            )
            decode.prepare()
            self.kvcache_manager.synchronize()  # for previous offloading and reloading
            self.kvcache_manager.virtual_page_pool.swap2(
                next_group.pinned_memory, prev_group.pinned_memory
            )
            return decode.step()
        else:
            prefill = BatchPrefill(
                xs,
                self.layer_storage,
                task_datas,
                cast(list[PagedKVCache[GpuPagePool]], kvcaches),
            )
            prefill.prepare()
            self.kvcache_manager.synchronize()  # for previous offloading and reloading
            self.kvcache_manager.virtual_page_pool.swap2(
                next_group.pinned_memory, prev_group.pinned_memory
            )
            return prefill.step()

    def synchronize(self) -> None:
        self.cuda_stream.synchronize()  # type: ignore
