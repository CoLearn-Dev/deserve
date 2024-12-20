import torch

from deserve_worker.engine.microbatch.processor import MicroBatchProcessor
from deserve_worker.kvcache.manager import KVCacheManager
from deserve_worker.kvcache.paged.page_table import PageTableAllocator
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.task import TaskManager, main_device, main_dtype


class MicroBatchScheduler(MicroBatchProcessor):
    def __init__(
        self,
        kvcache_manager: KVCacheManager,
        task_data_manager: TaskManager,
        layer_storage: LayerStorage,
        page_table_allocator: PageTableAllocator,
        ignore_eos: bool,
    ) -> None:
        super().__init__(
            kvcache_manager,
            task_data_manager,
            layer_storage,
            page_table_allocator,
            ignore_eos,
        )
        self.suspended_decodes: dict[str, torch.Tensor] = {}
        self.suspended_prefills: dict[str, torch.Tensor] = {}

    def suspend(self, task_ids: list[str], xs: torch.Tensor) -> None:
        # here, we assume seqlen = 1
        for i, task_id in enumerate(task_ids):
            self.suspended_decodes[task_id] = xs[i : i + 1]

    def resume(self, task_ids: list[str]) -> torch.Tensor:
        xs = torch.empty((0,), dtype=main_dtype, device=main_device)
        for task_id in task_ids:
            xs = torch.cat([xs, self.suspended_decodes.pop(task_id)], dim=0)
        return xs
