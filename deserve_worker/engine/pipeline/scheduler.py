import time
import traceback
from queue import Queue
from typing import Optional

import torch

from deserve_worker.engine.microbatch.scheduler import MicroBatchScheduler
from deserve_worker.engine.pipeline.processor import PipelineProcessor
from deserve_worker.request import InitRequest, LLMRequest, StepRequest, TraceRequest
from deserve_worker.task import SamplingParams, TaskData, main_dtype


class PipelineScheduler(PipelineProcessor):
    def __init__(
        self,
        num_rounds: int,
        num_pages_main: int,
        num_pages_swap: int,
        page_size: int,
        batch_size: int,
        layers: list[str],
        worker_url: str,
        next_worker_url: str,
        controller_url: str,
        simulated_latency: float,
    ) -> None:
        super().__init__(
            num_rounds,
            num_pages_main,
            num_pages_swap,
            page_size,
            batch_size,
            layers,
            worker_url,
            next_worker_url,
            controller_url,
            simulated_latency,
        )
        self.pending_prefill_datas: dict[str, tuple[torch.Tensor, SamplingParams]] = {}
        self.offloaded_decode_xs: dict[str, torch.Tensor] = {}
        self.offloaded_prefill_xs: dict[str, torch.Tensor] = {}
        self.buffer: Queue[StepRequest] = Queue()
        self.max_chunk_prefill_len = 256

        if layers[0].endswith("tok_embeddings"):
            dtype = torch.int
        else:
            dtype = main_dtype
        for i in range(self.num_rounds):
            self.buffer.put(StepRequest.empty(i, dtype))

    def run(self) -> None:
        try:
            last_time = time.time()
            while True:
                request = self.queue.get()
                if request is None:
                    break

                self.staged_time_log.append(time.time() - last_time)
                last_time = time.time()
                next_request: Optional[LLMRequest] = None
                if isinstance(request, InitRequest):
                    self.process_init(request)
                elif isinstance(request, StepRequest):
                    if request.is_empty():
                        self.buffer.put(request)
                    else:
                        self.clear_buffer()
                        self.synchronize()
                        next_request = self.process_step(request)
                elif isinstance(request, TraceRequest):
                    next_request = self.process_trace(request)
                if next_request is not None:
                    self.send_request(next_request)
        except Exception as e:
            traceback.print_exc()

    def clear_buffer(self) -> None:
        while not self.buffer.empty():
            self.synchronize()
            request = self.process_step(self.buffer.get())
            self.send_request(request)

    def process_init(self, request: InitRequest) -> None:
        self.task_manager.add(
            TaskData.empty(request.task_id, request.x.shape[0], request.sampling_params)
        )
        self.pending_prefill_datas[request.task_id] = (
            request.x,
            request.sampling_params,
        )
        self.clear_buffer()

    def process_step(self, request: StepRequest) -> StepRequest:
        request = self.convert_step_request(request)
        result = super().process_step(request)
        return result

    def convert_step_request(self, request: StepRequest) -> StepRequest:
        microbatch = self.microbatches[request.microbatch_id]
        rest_space = (
            self.virtual_page_pool.num_avails
            + self.task_manager.calc_occupied_space(request.cancel_task_ids)
            # - microbatch.reserved_space
        )
        request.refresh()

        total_prefill_len = 0
        global_pending_prefill_task_ids = set(self.pending_prefill_datas.keys())
        local_pending_prefill_task_ids = set(microbatch.pending_prefill_xs.keys())

        offloaded_task_ids = list(self.offloaded_kvcaches.keys())
        offloaded_prefill_task_ids = [
            task_id
            for task_id in offloaded_task_ids
            if task_id in self.offloaded_prefill_xs
        ]
        offloaded_decode_task_ids = [
            task_id
            for task_id in offloaded_task_ids
            if task_id in self.offloaded_decode_xs
        ]
        assert len(offloaded_decode_task_ids) + len(offloaded_prefill_task_ids) == len(
            offloaded_task_ids
        )

        if (
            len(global_pending_prefill_task_ids) > 0
            or len(local_pending_prefill_task_ids) > 0
            or len(self.offloaded_prefill_xs) > 0
        ):
            new_request = StepRequest.empty(request.microbatch_id, request.xs.dtype)

            if len(offloaded_prefill_task_ids) > 0:
                task_id = offloaded_prefill_task_ids[0]
                appended_space = self.task_manager.calc_initial_space([task_id])
                if rest_space >= appended_space:
                    rest_space -= appended_space
                    microbatch.reserved_space += self.task_manager.calc_rest_space(
                        task_id
                    )
                    request.reload_task_ids.append(task_id)
                    xs = self.offloaded_prefill_xs.pop(task_id)
                    if xs.shape[0] <= self.max_chunk_prefill_len:
                        total_prefill_len += xs.shape[0]
                        new_request.append_exec(task_id, xs, None)
                    else:
                        total_prefill_len += self.max_chunk_prefill_len
                        new_request.append_exec(
                            task_id, xs[: self.max_chunk_prefill_len], None
                        )
                        microbatch.pending_prefill_xs[task_id] = xs[
                            self.max_chunk_prefill_len :
                        ]

            # find all requests that can be prefilled and cannot be prefilled in once
            for task_id in list(microbatch.pending_prefill_xs.keys()):
                x = microbatch.pending_prefill_xs[task_id]
                if x.shape[0] > self.max_chunk_prefill_len:
                    microbatch.reserved_space -= self.task_manager.calc_delta_space(
                        task_id, self.max_chunk_prefill_len
                    )
                    new_request.append_exec(
                        task_id, x[: self.max_chunk_prefill_len], None
                    )
                    total_prefill_len += self.max_chunk_prefill_len
                    microbatch.pending_prefill_xs[task_id] = x[
                        self.max_chunk_prefill_len :
                    ]
                    break

            if total_prefill_len < self.max_chunk_prefill_len:
                for task_id in list(self.pending_prefill_datas.keys()):
                    x, sp = self.pending_prefill_datas[task_id]
                    if x.shape[0] > self.max_chunk_prefill_len:
                        appended_space = self.task_manager.calc_initial_space([task_id])
                        if rest_space >= appended_space:
                            microbatch.reserved_space += appended_space
                            rest_space -= appended_space
                            microbatch.reserved_space -= (
                                self.task_manager.calc_delta_space(
                                    task_id, self.max_chunk_prefill_len
                                )
                            )
                            new_request.append_exec(
                                task_id,
                                x[: self.max_chunk_prefill_len],
                                (x.shape[0], sp),
                            )
                            total_prefill_len += self.max_chunk_prefill_len
                            microbatch.pending_prefill_xs[task_id] = x[
                                self.max_chunk_prefill_len :
                            ]
                            self.pending_prefill_datas.pop(task_id)
                            break

            # find all requests that can be prefilled in once
            for task_id in list(microbatch.pending_prefill_xs.keys()):
                x = microbatch.pending_prefill_xs[task_id]
                if (
                    x.shape[0] <= self.max_chunk_prefill_len
                    and total_prefill_len < self.max_chunk_prefill_len
                ):
                    if total_prefill_len + x.shape[0] <= self.max_chunk_prefill_len:
                        microbatch.reserved_space -= self.task_manager.calc_delta_space(
                            task_id, x.shape[0]
                        )
                        new_request.append_exec(task_id, x, None)
                        total_prefill_len += x.shape[0]
                        microbatch.pending_prefill_xs.pop(task_id)
                    else:
                        current_prefill_len = min(
                            x.shape[0],
                            self.max_chunk_prefill_len - total_prefill_len,
                        )
                        microbatch.reserved_space -= self.task_manager.calc_delta_space(
                            task_id, current_prefill_len
                        )
                        new_request.append_exec(task_id, x[:current_prefill_len], None)
                        total_prefill_len += current_prefill_len
                        microbatch.pending_prefill_xs[task_id] = x[current_prefill_len:]

            for task_id in list(self.pending_prefill_datas.keys()):
                x, sp = self.pending_prefill_datas[task_id]
                if (
                    x.shape[0] <= self.max_chunk_prefill_len
                    and total_prefill_len < self.max_chunk_prefill_len
                ):
                    appended_space = self.task_manager.calc_initial_space([task_id])
                    if rest_space >= appended_space:
                        rest_space -= appended_space
                        if total_prefill_len + x.shape[0] <= self.max_chunk_prefill_len:
                            new_request.append_exec(task_id, x, (x.shape[0], sp))
                            self.pending_prefill_datas.pop(task_id)
                        else:
                            microbatch.reserved_space += appended_space
                            current_prefill_len = min(
                                x.shape[0],
                                self.max_chunk_prefill_len - total_prefill_len,
                            )
                            microbatch.reserved_space -= self.task_manager.calc_space(
                                current_prefill_len
                            )
                            new_request.append_exec(
                                task_id, x[:current_prefill_len], (x.shape[0], sp)
                            )
                            total_prefill_len += current_prefill_len
                            microbatch.pending_prefill_xs[task_id] = x[
                                current_prefill_len:
                            ]
                            self.pending_prefill_datas.pop(task_id)

            if len(new_request.exec_task_ids) > 0:
                for i, task_id in enumerate(request.exec_task_ids):
                    if (
                        task_id not in global_pending_prefill_task_ids
                        and task_id not in local_pending_prefill_task_ids
                    ):
                        # the prefill task ids are the old version
                        microbatch.suspended_decode_xs[task_id] = request.xs[i : i + 1]
                new_request.cancel_task_ids.extend(
                    request.cancel_task_ids
                )  # inherit cancel task ids
                request = new_request
            else:
                rest_space -= self.task_manager.calc_extended_space(
                    request.exec_task_ids
                )
        else:
            rest_space -= self.task_manager.calc_extended_space(request.exec_task_ids)

        # assigned here because pending_prefill_xs may be changed in previous code
        pending_prefill_task_ids = list(microbatch.pending_prefill_xs.keys())
        suspended_decode_task_ids = list(microbatch.suspended_decode_xs.keys())

        # try to add more decode tasks to this microbatch
        while len(suspended_decode_task_ids) > 0:
            task_id = suspended_decode_task_ids[0]
            appended_space = self.task_manager.calc_extended_space([task_id])
            if (
                rest_space >= appended_space
                and request.get_bsz() < self.max_batch_size
                and total_prefill_len < self.max_chunk_prefill_len
            ):
                rest_space -= appended_space
                suspended_decode_task_ids.pop(0)
                request.append_exec(
                    task_id, microbatch.suspended_decode_xs.pop(task_id), None
                )
                total_prefill_len += 1
            else:
                break

        # try to add more decode tasks to this microbatch
        while len(offloaded_decode_task_ids) > 0:
            task_id = offloaded_decode_task_ids[0]
            appended_space = self.task_manager.calc_occupied_space(
                [task_id]
            ) + self.task_manager.calc_extended_space([task_id])
            if (
                rest_space >= appended_space
                and request.get_bsz() <= self.max_batch_size
                and total_prefill_len < self.max_chunk_prefill_len
            ):
                rest_space -= appended_space
                offloaded_decode_task_ids.pop(0)
                request.append_exec(
                    task_id, self.offloaded_decode_xs.pop(task_id), None
                )
                request.reload_task_ids.append(task_id)
                total_prefill_len += 1
            else:
                break

        while rest_space < 0 and len(pending_prefill_task_ids) > 0:
            task_id = pending_prefill_task_ids.pop()
            delta_space = self.task_manager.calc_initial_space([task_id])
            rest_space += delta_space
            microbatch.reserved_space -= delta_space
            request.offload_task_ids.append(task_id)
            self.offloaded_prefill_xs[task_id] = microbatch.pending_prefill_xs.pop(
                task_id
            )

        # try to suspend decode tasks that is just prefilled
        while rest_space < 0 and len(suspended_decode_task_ids) > 0:
            task_id = suspended_decode_task_ids.pop()
            rest_space += self.task_manager.calc_occupied_space([task_id])
            request.offload_task_ids.append(task_id)
            self.offloaded_decode_xs[task_id] = microbatch.suspended_decode_xs.pop(
                task_id
            )
            assert self.offloaded_decode_xs[task_id].numel() > 0

        # try to suspend decode tasks that is running
        if rest_space < 0 and len(request.exec_task_ids) > 0:
            assert all(seqlen == 1 for seqlen in request.exec_seqlens)
            for i in reversed(range(len(request.exec_task_ids))):
                todo_task_ids = request.exec_task_ids[: i + 1]
                to_offload_decode_task_ids = request.exec_task_ids[i + 1 :]
                if (
                    rest_space
                    + self.task_manager.calc_occupied_space(to_offload_decode_task_ids)
                    + self.task_manager.calc_extended_space(to_offload_decode_task_ids)
                    >= 0
                ):
                    request.offload_task_ids.extend(to_offload_decode_task_ids)
                    request.exec_task_ids = todo_task_ids
                    sep = self.task_manager.calc_seqlens(todo_task_ids)
                    removed_xs = request.xs[sep:]
                    request.xs = request.xs[:sep]
                    for i, task_id in enumerate(to_offload_decode_task_ids):
                        self.offloaded_decode_xs[task_id] = removed_xs[i : i + 1]
                        assert self.offloaded_decode_xs[task_id].numel() > 0
                    return request
            assert False

        return request
