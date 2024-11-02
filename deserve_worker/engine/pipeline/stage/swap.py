import torch

from deserve_worker.request import StepRequest
from deserve_worker.task import TaskManager

from . import Stage


class SwapStage(Stage):
    def __init__(
        self,
        manager: TaskManager,
        local_suspended_prefills: dict[str, torch.Tensor],
        local_suspended_decodes: dict[str, torch.Tensor],
        local_offloaded_prefills: dict[str, torch.Tensor],
        local_offloaded_decodes: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(manager)
        self.local_suspended_prefills = local_suspended_prefills
        self.local_suspended_decodes = local_suspended_decodes
        self.local_offloaded_prefills = local_offloaded_prefills
        self.local_offloaded_decodes = local_offloaded_decodes

    def process(self, rest_pages: int, request: StepRequest) -> tuple[int, StepRequest]:
        if rest_pages < 0:
            suspended_prefills = sorted(
                [(k, x) for k, x in self.local_suspended_prefills.items()],
                key=lambda x: x[1].shape[0],
            )
            while rest_pages < 0 and len(suspended_prefills) > 0:
                task_id, x = suspended_prefills.pop()
                delta_pages = self.manager.calc_initial_space([task_id])
                rest_pages += delta_pages
                self.local_suspended_prefills.pop(task_id)
                self.local_offloaded_prefills[task_id] = x
                request.offload_task_ids.append(task_id)
            suspended_decodes = sorted(
                [(k, x) for k, x in self.local_suspended_decodes.items()],
                key=lambda x: x[1].shape[0],
            )
            while rest_pages < 0 and len(suspended_decodes) > 0:
                task_id, x = suspended_decodes.pop()
                delta_pages = self.manager.calc_occupied_space([task_id])
                rest_pages += delta_pages
                self.local_suspended_decodes.pop(task_id)
                self.local_offloaded_decodes[task_id] = x
                request.offload_task_ids.append(task_id)
            if rest_pages < 0:
                assert all(seqlen == 1 for seqlen in request.exec_seqlens)
                for i in reversed(range(len(request.exec_task_ids))):
                    todo_task_ids = request.exec_task_ids[: i + 1]
                    to_offload_decode_task_ids = request.exec_task_ids[i + 1 :]
                    appended_pages = self.manager.calc_occupied_space(
                        to_offload_decode_task_ids
                    ) + self.manager.calc_extended_space(to_offload_decode_task_ids)
                    if rest_pages + appended_pages >= 0:
                        rest_pages += appended_pages
                        request.offload_task_ids.extend(to_offload_decode_task_ids)
                        request.exec_task_ids = todo_task_ids
                        sep = len(todo_task_ids)
                        removed_xs = request.xs[sep:]
                        request.xs = request.xs[:sep]
                        for i, task_id in enumerate(to_offload_decode_task_ids):
                            self.local_offloaded_decodes[task_id] = removed_xs[
                                i : i + 1
                            ]
                            assert self.local_offloaded_decodes[task_id].numel() > 0
                        return rest_pages, request
                raise RuntimeError("Unable to fetch enough pages")
        else:
            offloaded_prefills = sorted(
                [(k, x) for k, x in self.local_offloaded_prefills.items()],
                key=lambda x: x[1].shape[0],
            )
            while rest_pages > 0 and len(offloaded_prefills) > 0:
                task_id, x = offloaded_prefills.pop()
                delta_pages = self.manager.calc_initial_space([task_id])
                if rest_pages >= delta_pages:
                    rest_pages -= delta_pages
                    self.local_offloaded_prefills.pop(task_id)
                    self.local_suspended_prefills[task_id] = (
                        x  # is not added into execution tasks
                    )
                    request.reload_task_ids.append(task_id)
                else:
                    break
            offloaded_decodes = sorted(
                [(k, x) for k, x in self.local_offloaded_decodes.items()],
                key=lambda x: x[1].shape[0],
            )
            while rest_pages > 0 and len(offloaded_decodes) > 0:
                task_id, x = offloaded_decodes.pop()
                delta_pages = self.manager.calc_occupied_space([task_id])
                if rest_pages >= delta_pages:
                    rest_pages -= delta_pages
                    self.local_offloaded_decodes.pop(task_id)
                    self.local_suspended_decodes[task_id] = x
                    request.reload_task_ids.append(task_id)
                else:
                    break
        return rest_pages, request
