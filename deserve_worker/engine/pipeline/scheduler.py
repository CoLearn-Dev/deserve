import time
import traceback
from queue import Queue
from typing import Optional

import torch

from deserve_worker.engine.microbatch.scheduler import MicroBatchScheduler
from deserve_worker.engine.pipeline.processor import PipelineProcessor
from deserve_worker.request import (
    DecodeRequest,
    InitRequest,
    LLMRequest,
    PrefillRequest,
    TraceRequest,
)
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
        )
        self.pending_prefill_datas: dict[str, tuple[torch.Tensor, SamplingParams]] = {}
        self.offloaded_decode_xs: dict[str, torch.Tensor] = {}
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
            last_sync = time.time()
            while True:
                request = self.queue.get()
                if request is None:
                    break

                print(f"stage: {(time.time() - last_time) * 1000:.2f}ms")
                last_time = time.time()
                next_request: Optional[LLMRequest] = None
                if isinstance(request, InitRequest):
                    self.process_init(request)
                elif isinstance(request, DecodeRequest) or isinstance(
                    request, PrefillRequest
                ):
                    if request.is_empty():
                        self.buffer.put(request)
                    else:
                        self.clear_buffer()
                        self.synchronize()
                        last_sync = time.time()
                        next_request = self.process_step(request)
                elif isinstance(request, TraceRequest):
                    next_request = self.process_trace(request)
                if next_request is not None:
                    self.send_request(next_request)
                print(f"all: {(time.time() - last_sync) * 1000:.2f}ms")
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

    def process_step(self, request: DecodeRequest) -> DecodeRequest:
        begin = time.time()
        request = self.convert_step_request(request)
        print(f"convert: {(time.time() - begin) * 1000:.2f}ms")
        return super().process_step(request)

    def convert_step_request(self, request: DecodeRequest) -> DecodeRequest:
        rest_space = (
            self.virtual_page_pool.num_avails
            + self.task_manager.calc_occupied_space(request.cancel_task_ids)
        )
        microbatch = self.microbatches[request.microbatch_id]
        request.refresh()

        pending_prefill_task_ids = list(self.pending_prefill_datas.keys())
        if len(pending_prefill_task_ids) > 0:
            prefill_task_id = pending_prefill_task_ids[0]
            if rest_space >= self.task_manager.calc_extended_space([prefill_task_id]):
                for i, task_id in enumerate(request.exec_task_ids):
                    microbatch.suspended_decode_xs[task_id] = request.xs[i : i + 1]

                x, sp = self.pending_prefill_datas.pop(prefill_task_id)
                request.exec_task_ids = [prefill_task_id]
                request.xs = x

                return PrefillRequest.from_decode_request(
                    request, [prefill_task_id], [sp]
                )

        suspended_decode_task_ids = list(microbatch.suspended_decode_xs.keys())
        offloaded_decode_task_ids = list(self.offloaded_decode_kvcaches.keys())
        rest_space -= self.task_manager.calc_extended_space(request.exec_task_ids)

        # try to add more decode tasks to this microbatch
        while len(suspended_decode_task_ids) > 0:
            task_id = suspended_decode_task_ids[0]
            appended_space = self.task_manager.calc_extended_space([task_id])
            if (
                rest_space >= appended_space
                and request.get_bsz() <= self.max_batch_size
            ):
                rest_space -= appended_space
                suspended_decode_task_ids.pop(0)
                request.append_exec(
                    task_id, microbatch.suspended_decode_xs.pop(task_id)
                )
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
            ):
                rest_space -= appended_space
                offloaded_decode_task_ids.pop(0)
                request.append_exec(task_id, self.offloaded_decode_xs.pop(task_id))
                request.reload_task_ids.append(task_id)
            else:
                break

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
        if len(request.exec_task_ids) > 0:
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
                    return request.into_decode_request()
            assert False

        return request.into_decode_request()
