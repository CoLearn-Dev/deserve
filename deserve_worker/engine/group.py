import torch

from deserve_worker.execution.exec import BatchDecode
from deserve_worker.execution.result import BatchResult
from deserve_worker.kvcache.manager.bulk import BulkKVCacheManager
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool
from deserve_worker.kvcache.pinned.kvcache import PinnedKVCache
from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.task import TaskManager


class Group:
    def __init__(
        self,
        cpu_page_pool: CpuPagePool,
        gpu_page_pool: GpuPagePool,
        pinned_memory: PinnedMemory,
        task_data_manager: TaskManager,
        layer_storage: LayerStorage,
    ) -> None:
        self.cpu_page_pool = cpu_page_pool
        self.gpu_page_pool = gpu_page_pool
        self.pinned_memory = pinned_memory
        self.kvcache_manager = BulkKVCacheManager(
            gpu_page_pool, cpu_page_pool, pinned_memory
        )
        self.task_data_manager = task_data_manager
        self.layer_storage = layer_storage
        self.ongoing_paged_kvcaches: dict[str, PagedKVCache[GpuPagePool]] = {}
        self.offloaded_pinned_kvcaches: dict[str, PinnedKVCache] = {}
        self.cuda_stream = torch.cuda.Stream()  # type: ignore

    def suspend(self, task_ids: list[str]) -> list[PagedKVCache[CpuPagePool]]:
        """
        Suspend the execution of the task in a blocking manner.
        """
        paged_kvcaches = [
            self.ongoing_paged_kvcaches.pop(task_id) for task_id in task_ids
        ]
        cpu_paged_kvcaches = [
            self.kvcache_manager.copy_gpu_to_cpu(paged_kvcache)
            for paged_kvcache in paged_kvcaches
        ]
        for paged_kvcache in paged_kvcaches:
            paged_kvcache.free()
        return cpu_paged_kvcaches

    def join(
        self, task_ids: list[str], gpu_paged_kvcaches: list[PagedKVCache[GpuPagePool]]
    ) -> None:
        for task_id, gpu_paged_kvcache in zip(task_ids, gpu_paged_kvcaches):
            self.ongoing_paged_kvcaches[task_id] = gpu_paged_kvcache

    def resume(
        self, task_ids: list[str], cpu_paged_kvcaches: list[PagedKVCache[CpuPagePool]]
    ) -> None:
        """
        Resume the execution of the task in a blocking manner.
        """
        gpu_paged_kvcaches = [
            self.kvcache_manager.copy_cpu_to_gpu(cpu_paged_kvcache)
            for cpu_paged_kvcache in cpu_paged_kvcaches
        ]
        for task_id, gpu_paged_kvcache in zip(task_ids, gpu_paged_kvcaches):
            self.ongoing_paged_kvcaches[task_id] = gpu_paged_kvcache
        for cpu_paged_kvcache in cpu_paged_kvcaches:
            cpu_paged_kvcache.free()

    def offload(self, task_ids: list[str]) -> None:
        with torch.cuda.stream(self.cuda_stream):
            paged_kvcaches = [
                self.ongoing_paged_kvcaches.pop(task_id) for task_id in task_ids
            ]
            pinned_kvcaches = self.kvcache_manager.copy_paged_to_pinned(paged_kvcaches)
            for task_id, pinned_kvcache in zip(task_ids, pinned_kvcaches):
                self.offloaded_pinned_kvcaches[task_id] = pinned_kvcache

    def reload(self, task_ids: list[str]) -> None:
        with torch.cuda.stream(self.cuda_stream):
            pinned_kvcaches = [
                self.offloaded_pinned_kvcaches.pop(task_id) for task_id in task_ids
            ]
            paged_kvcaches = self.kvcache_manager.copy_pinned_to_paged(pinned_kvcaches)
            for task_id, paged_kvcache in zip(task_ids, paged_kvcaches):
                self.ongoing_paged_kvcaches[task_id] = paged_kvcache

    def cancel(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            if task_id in self.ongoing_paged_kvcaches:
                gpu_paged_kvcache = self.ongoing_paged_kvcaches.pop(task_id)
                gpu_paged_kvcache.free()
            if task_id in self.offloaded_pinned_kvcaches:
                self.offloaded_pinned_kvcaches.pop(task_id)

    def exec(self, xs: torch.Tensor, task_ids: list[str]) -> BatchResult:
        task_datas = [self.task_data_manager.get(task_id) for task_id in task_ids]
        kvcaches = [self.ongoing_paged_kvcaches[task_id] for task_id in task_ids]
        decode = BatchDecode(
            xs,
            self.layer_storage,
            task_datas,
            kvcaches,
        )
        return decode.step()

    def synchronize(self) -> None:
        self.cuda_stream.synchronize()  # type: ignore
