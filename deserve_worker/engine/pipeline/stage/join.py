from dataclasses import dataclass

import torch

from deserve_worker.request import StepRequest
from deserve_worker.task import SamplingParams, TaskManager

from . import Stage


class DecodeFirstAgggregatedJoinStage(Stage):
    def __init__(
        self,
        manager: TaskManager,
        global_pending_prefills: dict[str, tuple[torch.Tensor, SamplingParams]],
        local_suspended_prefills: dict[str, torch.Tensor],
        local_suspended_decodes: dict[str, torch.Tensor],
        max_batch_size: int,
    ) -> None:
        super().__init__(manager)
        self.global_pending_prefills = global_pending_prefills
        self.local_suspended_prefills = local_suspended_prefills
        self.local_suspended_decodes = local_suspended_decodes
        self.max_batch_size = max_batch_size
        self.max_extra_batch_size = max_batch_size

    def process(self, rest_pages: int, request: StepRequest) -> tuple[int, StepRequest]:
        rest_pages -= self.manager.calc_extended_space(request.exec_task_ids)
        current_bs = request.get_bsz()
        if (
            len(self.global_pending_prefills) > 0
            or len(self.local_suspended_prefills) > 0
        ) and rest_pages > 0:
            local_prefills = sorted(
                [(k, x) for k, x in self.local_suspended_prefills.items()],
                key=lambda x: x[1].shape[0],
            )
            for task_id, x in local_prefills:
                if current_bs + x.shape[0] <= self.max_batch_size:
                    current_bs += x.shape[0]
                    self.local_suspended_prefills.pop(task_id)
                    request.append_exec(task_id, x, None)
                    if current_bs == self.max_batch_size:
                        break
                else:
                    delta = self.max_batch_size - current_bs
                    current_bs += delta
                    request.append_exec(task_id, x[:delta], None)
                    self.local_suspended_prefills[task_id] = x[delta:]
                    break

            if current_bs < self.max_batch_size:
                global_prefills = sorted(
                    [(k, x, sp) for k, (x, sp) in self.global_pending_prefills.items()],
                    key=lambda x: x[1].shape[0],
                )
                for task_id, x, sp in global_prefills:
                    appended_pages = self.manager.calc_initial_space([task_id])
                    if rest_pages >= appended_pages:
                        rest_pages -= appended_pages
                        self.global_pending_prefills.pop(task_id)
                        if current_bs + x.shape[0] <= self.max_batch_size:
                            current_bs += x.shape[0]
                            request.append_exec(task_id, x, (x.shape[0], sp))
                            if current_bs == self.max_batch_size:
                                break
                        else:
                            delta = self.max_batch_size - current_bs
                            current_bs += delta
                            request.append_exec(task_id, x[:delta], (x.shape[0], sp))
                            self.local_suspended_prefills[task_id] = x[delta:]
                            break

        if current_bs < self.max_batch_size:
            local_decodes = list(self.local_suspended_decodes.items())
            for task_id, x in local_decodes:
                appended_pages = self.manager.calc_extended_space([task_id])
                if rest_pages >= appended_pages:
                    rest_pages -= appended_pages
                    current_bs += x.shape[0]
                    assert x.shape[0] == 1
                    self.local_suspended_decodes.pop(task_id)
                    request.append_exec(task_id, x, None)
                    if current_bs == self.max_batch_size:
                        break

        return rest_pages, request


class PrefillFirstAgggregatedJoinStage(Stage):
    def __init__(
        self,
        manager: TaskManager,
        global_pending_prefills: dict[str, tuple[torch.Tensor, SamplingParams]],
        local_suspended_prefills: dict[str, torch.Tensor],
        local_suspended_decodes: dict[str, torch.Tensor],
        max_batch_size: int,
    ) -> None:
        super().__init__(manager)
        self.global_pending_prefills = global_pending_prefills
        self.local_suspended_prefills = local_suspended_prefills
        self.local_suspended_decodes = local_suspended_decodes
        self.max_batch_size = max_batch_size

    def process(self, rest_pages: int, request: StepRequest) -> tuple[int, StepRequest]:
        current_bs = 0
        if (
            len(self.global_pending_prefills) > 0
            or len(self.local_suspended_prefills) > 0
        ):
            new_request = StepRequest.empty(request.microbatch_id, request.xs.dtype)
            local_prefills = sorted(
                [(k, x) for k, x in self.local_suspended_prefills.items()],
                key=lambda x: x[1].shape[0],
            )
            for task_id, x in local_prefills:
                if current_bs + x.shape[0] <= self.max_batch_size:
                    current_bs += x.shape[0]
                    self.local_suspended_prefills.pop(task_id)
                    new_request.append_exec(task_id, x, None)
                    if current_bs == self.max_batch_size:
                        break
                else:
                    delta = self.max_batch_size - current_bs
                    current_bs += delta
                    new_request.append_exec(task_id, x[:delta], None)
                    self.local_suspended_prefills[task_id] = x[delta:]
                    break

            if current_bs < self.max_batch_size:
                global_prefills = sorted(
                    [(k, x, sp) for k, (x, sp) in self.global_pending_prefills.items()],
                    key=lambda x: x[1].shape[0],
                )
                for task_id, x, sp in global_prefills:
                    appended_pages = self.manager.calc_initial_space([task_id])
                    if rest_pages >= appended_pages:
                        rest_pages -= appended_pages
                        self.global_pending_prefills.pop(task_id)
                        if current_bs + x.shape[0] <= self.max_batch_size:
                            current_bs += x.shape[0]
                            new_request.append_exec(task_id, x, (x.shape[0], sp))
                            if current_bs == self.max_batch_size:
                                break
                        else:
                            delta = self.max_batch_size - current_bs
                            current_bs += delta
                            new_request.append_exec(
                                task_id, x[:delta], (x.shape[0], sp)
                            )
                            self.local_suspended_prefills[task_id] = x[delta:]
                            break

            if len(new_request.exec_task_ids) > 0:
                for i, task_id in enumerate(request.exec_task_ids):
                    self.local_suspended_decodes[task_id] = request.xs[i : i + 1]
                new_request.cancel_task_ids.extend(
                    request.cancel_task_ids
                )  # inherit cancel task ids
                request = new_request
            else:
                rest_pages -= self.manager.calc_extended_space(request.exec_task_ids)
        else:
            rest_pages -= self.manager.calc_extended_space(request.exec_task_ids)

        if current_bs < self.max_batch_size:
            local_decodes = list(self.local_suspended_decodes.items())
            for task_id, x in local_decodes:
                appended_pages = self.manager.calc_extended_space([task_id])
                if rest_pages >= appended_pages:
                    rest_pages -= appended_pages
                    current_bs += x.shape[0]
                    assert x.shape[0] == 1
                    self.local_suspended_decodes.pop(task_id)
                    request.append_exec(task_id, x, None)
                    if current_bs == self.max_batch_size:
                        break

        return rest_pages, request


class VanillaJoinStage(Stage):
    def __init__(
        self,
        manager: TaskManager,
        global_pending_prefills: dict[str, tuple[torch.Tensor, SamplingParams]],
        local_suspended_decodes: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(manager)
        self.global_pending_prefills = global_pending_prefills
        self.local_suspended_decodes = local_suspended_decodes

    def process(self, rest_pages: int, request: StepRequest) -> tuple[int, StepRequest]:
        if len(self.global_pending_prefills) > 0:
            new_request = StepRequest.empty(request.microbatch_id, request.xs.dtype)
            global_prefills = sorted(
                [(k, x, sp) for k, (x, sp) in self.global_pending_prefills.items()],
                key=lambda x: x[1].shape[0],
            )
            for task_id, x, sp in global_prefills:
                appended_pages = self.manager.calc_initial_space([task_id])
                if rest_pages >= appended_pages:
                    rest_pages -= appended_pages
                    self.global_pending_prefills.pop(task_id)
                    new_request.append_exec(task_id, x, (x.shape[0], sp))
                    break

            if len(new_request.exec_task_ids) > 0:
                for i, task_id in enumerate(request.exec_task_ids):
                    self.local_suspended_decodes[task_id] = request.xs[i : i + 1]
                new_request.cancel_task_ids.extend(
                    request.cancel_task_ids
                )  # inherit cancel task ids
                request = new_request
                return rest_pages, request

        rest_pages -= self.manager.calc_extended_space(request.exec_task_ids)
        local_decodes = list(self.local_suspended_decodes.items())
        for task_id, x in local_decodes:
            appended_pages = self.manager.calc_extended_space([task_id])
            if rest_pages >= appended_pages:
                rest_pages -= appended_pages
                assert x.shape[0] == 1
                self.local_suspended_decodes.pop(task_id)
                request.append_exec(task_id, x, None)

        return rest_pages, request
