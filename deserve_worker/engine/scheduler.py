import time
import traceback
from queue import Queue
from typing import Optional

import torch

from deserve_worker.engine.group import Group
from deserve_worker.engine.processor import Processor
from deserve_worker.request import (
    DecodeRequest,
    JoinRequest,
    LLMRequest,
    PrefillRequest,
    TraceRequest,
)
from deserve_worker.task import TaskData, main_dtype


class Scheduler(Processor):
    def __init__(
        self,
        num_rounds: int,
        num_pages: int,
        page_size: int,
        batch_size: int,
        layers: list[str],
        worker_url: str,
        next_worker_url: str,
        controller_url: str,
    ) -> None:
        super().__init__(
            num_rounds,
            num_pages,
            page_size,
            batch_size,
            layers,
            worker_url,
            next_worker_url,
            controller_url,
        )
        self.pending_decode_xs: dict[str, torch.Tensor] = {}
        self.suspended_prefill_xs: dict[str, torch.Tensor] = {}
        self.buffer: Queue[DecodeRequest] = Queue()

        if layers[0].endswith("tok_embeddings"):
            dtype = torch.int
        else:
            dtype = main_dtype
        for i in range(self.num_rounds):
            self.buffer.put(DecodeRequest.empty(i, dtype))

    def run(self) -> None:
        try:
            last_time = time.time()
            while True:
                request = self.queue.get()
                print((time.time() - last_time) * 1000, "ms")
                last_time = time.time()
                if request is None:
                    break

                next_request: Optional[LLMRequest] = None
                if isinstance(request, PrefillRequest):
                    self.task_manager.add(
                        TaskData.empty(
                            request.task_id,
                            request.x.shape[0],
                            request.sampling_params,
                        )
                    )
                    next_request = self.process_prefill(request)
                elif isinstance(request, DecodeRequest):
                    if request.is_empty():
                        self.buffer.put(request)
                    else:
                        self.clear_buffer()
                        next_request = self.process_decode(request)
                elif isinstance(request, JoinRequest):
                    self.process_join(request)
                elif isinstance(request, TraceRequest):
                    next_request = self.process_trace(request)
                else:
                    raise ValueError(f"Unknown request type: {request}")
                if next_request is not None:
                    self.send_request(next_request)
        except Exception as e:
            traceback.print_exc()

    def clear_buffer(self) -> None:
        while not self.buffer.empty():
            request = self.process_decode(self.buffer.get())
            self.send_request(request)

    def process_prefill(self, request: PrefillRequest) -> Optional[LLMRequest]:
        """
        If there is enough space in the GPU cache, resume the prefill.
        Otherwise, suspend the prefill.
        """
        task_id = request.task_id
        if (
            self.task_manager.calc_extended_space([task_id])
            <= self.gpu_page_pool.num_avails
        ):
            return super().process_prefill(request)
        else:
            self.suspended_prefill_xs[task_id] = request.x
            return None

    def process_decode(self, request: DecodeRequest) -> LLMRequest:
        group = self.groups[request.group_id]
        if self.num_rounds > 2:
            prev_group = self.groups[
                (request.group_id - 1 + self.num_rounds) % self.num_rounds
            ]
            next_group = self.groups[(request.group_id + 1) % self.num_rounds]
            request.offload_task_ids = self.decide_offload(prev_group)
            request.reload_task_ids = self.decide_reload(
                next_group,
                self.task_manager.calc_occupied_space(request.offload_task_ids),
            )
        original_xs = request.xs
        self.decide_exec(request)  # modify request inside
        result = super().process_decode(request)

        for task_id in request.r2e_task_ids:
            self.pending_decode_xs.pop(task_id)
        for task_id in request.s2e_task_ids:
            self.pending_decode_xs.pop(task_id)
        suspended_decode_xs = original_xs[len(request.exec_task_ids) :]
        for i, task_id in enumerate(request.e2s_task_ids):
            self.pending_decode_xs[task_id] = suspended_decode_xs[i : i + 1]
        return result

    def process_join(self, request: JoinRequest) -> None:
        self.pending_decode_xs[request.task_id] = request.x
        self.clear_buffer()

    def decide_offload(self, group: Group) -> list[str]:
        ongoing_task_ids = list(group.ongoing_paged_kvcaches.keys())
        for i in range(len(ongoing_task_ids)):
            todo_task_ids = ongoing_task_ids[i:]
            if (
                self.task_manager.calc_occupied_space(todo_task_ids)
                <= self.limit_offload_len
            ):
                return todo_task_ids
        return []

    def decide_reload(self, group: Group, last_offload_len: int) -> list[str]:
        offload_task_ids = list(group.offloaded_pinned_kvcaches.keys())
        ongoing_task_ids = list(group.ongoing_paged_kvcaches.keys())
        for i in reversed(range(len(offload_task_ids))):
            todo_task_ids = offload_task_ids[: i + 1]
            if self.task_manager.calc_occupied_space(
                todo_task_ids
            ) <= last_offload_len + max(
                0,
                self.gpu_page_pool.num_avails
                - self.task_manager.calc_extended_space(ongoing_task_ids),
            ):
                return todo_task_ids
        return []

    def decide_exec(self, request: DecodeRequest) -> None:
        """
        Return execution tensor, exec task ids, suspended prefill task ids, and suspended decode task ids.
        """
        request.refresh()
        resumed_decode_task_ids = list(self.resumed_decode_kvcaches.keys())
        suspended_prefill_task_ids = list(self.suspended_prefill_xs.keys())
        suspended_decode_task_ids = list(self.suspended_decode_kvcaches.keys())
        rest_space = self.gpu_page_pool.num_avails
        extended_space = self.task_manager.calc_extended_space(request.exec_task_ids)
        extra_space = 0

        # try to execute decode tasks that is just prefilled
        while len(resumed_decode_task_ids) > 0:
            task_id = resumed_decode_task_ids[0]
            x = self.pending_decode_xs.get(task_id)
            appended_space = self.task_manager.calc_extended_space([task_id])
            if (
                extended_space + appended_space <= rest_space
                and x is not None
                and request.get_bsz() <= self.max_batch_size
            ):
                extended_space += appended_space
                resumed_decode_task_ids.pop(0)
                request.r2e_task_ids.append(task_id)
                request.append_exec(task_id, x)
            else:
                break

        # try to execute prefill tasks
        # while len(suspended_prefill_task_ids) > 0:
        #     task_id = suspended_prefill_task_ids[0]
        #     x = self.suspended_prefills[task_id]
        #     if (
        #         original_space + self.task_manager.calc_occupied([task_id])
        #         <= self.task_manager.space
        #     ) and x is not None:
        #         entity.s2e_prefill_task_ids.append(task_id)
        #         xs = torch.cat([xs, x])
        #     else:
        #         break

        # try to execute decode tasks that is suspended due to no space before
        while len(suspended_decode_task_ids) > 0:
            task_id = suspended_decode_task_ids[0]
            x = self.pending_decode_xs.get(task_id)
            appended_space = self.task_manager.calc_occupied_space(
                [task_id]
            ) + self.task_manager.calc_extended_space([task_id])
            if (
                extended_space + appended_space <= rest_space
                and x is not None
                and request.get_bsz() <= self.max_batch_size
            ):
                extended_space += appended_space
                suspended_decode_task_ids.pop(0)
                request.s2e_task_ids.append(task_id)
                request.append_exec(task_id, x)
            else:
                break

        # try to suspend decode tasks that is just prefilled
        while (
            extended_space > rest_space + extra_space
            and len(resumed_decode_task_ids) > 0
        ):
            task_id = resumed_decode_task_ids.pop()
            extra_space += self.task_manager.calc_occupied_space([task_id])
            request.r2s_task_ids.append(task_id)

        # try to suspend decode tasks that is executed
        if len(request.exec_task_ids) > 0:
            for i in reversed(range(len(request.exec_task_ids))):
                todo_task_ids = request.exec_task_ids[: i + 1]
                suspended_decode_task_ids = request.exec_task_ids[i + 1 :]
                if extended_space - self.task_manager.calc_extended_space(
                    suspended_decode_task_ids
                ) <= rest_space + extra_space + self.task_manager.calc_occupied_space(
                    suspended_decode_task_ids
                ):
                    request.e2s_task_ids = suspended_decode_task_ids
                    request.exec_task_ids = todo_task_ids
                    exec_xs = request.xs[
                        : self.task_manager.calc_seqlens(todo_task_ids)
                    ]
                    request.xs = exec_xs
                    # print(
                    #     "rest space:",
                    #     rest_space
                    #     + extra_space
                    #     + self.task_manager.calc_occupied_space(
                    #         suspended_decode_task_ids
                    #     )
                    #     + self.task_manager.calc_extended_space(
                    #         suspended_decode_task_ids
                    #     )
                    #     - extended_space,
                    # )
                    return
            assert False
