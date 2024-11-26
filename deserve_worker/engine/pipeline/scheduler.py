import time
import traceback
from queue import Queue
from typing import Optional

import torch

from deserve_network import Server
from deserve_worker.engine.microbatch.scheduler import MicroBatchScheduler
from deserve_worker.engine.pipeline.processor import PipelineProcessor
from deserve_worker.engine.pipeline.stage import Stage
from deserve_worker.engine.pipeline.stage.join import (
    DecodeFirstAgggregatedJoinStage,
    PrefillFirstAgggregatedJoinStage,
    VanillaJoinStage,
)
from deserve_worker.engine.pipeline.stage.swap import SwapStage
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
        prefill_first_aggregate: bool,
        decode_first_aggregate: bool,
        buddy_height: int,
        ignore_eos: bool,
        server: Server,
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
            buddy_height,
            ignore_eos,
            server,
        )
        self.pending_prefills: dict[str, tuple[torch.Tensor, SamplingParams]] = {}
        self.offloaded_decodes: dict[str, torch.Tensor] = {}
        self.offloaded_prefills: dict[str, torch.Tensor] = {}
        self.join_stages: list[Stage] = (
            [
                VanillaJoinStage(
                    self.task_manager,
                    self.pending_prefills,
                    microbatch.suspended_decodes,
                )
                for microbatch in self.microbatches
            ]
            if not prefill_first_aggregate and not decode_first_aggregate
            else (
                [
                    DecodeFirstAgggregatedJoinStage(
                        self.task_manager,
                        self.pending_prefills,
                        microbatch.suspended_prefills,
                        microbatch.suspended_decodes,
                        512,
                    )
                    for microbatch in self.microbatches
                ]
                if decode_first_aggregate
                else [
                    PrefillFirstAgggregatedJoinStage(
                        self.task_manager,
                        self.pending_prefills,
                        microbatch.suspended_prefills,
                        microbatch.suspended_decodes,
                        512,
                    )
                    for microbatch in self.microbatches
                ]
            )
        )
        self.swap_stages: list[SwapStage] = [
            SwapStage(
                self.task_manager,
                microbatch.suspended_prefills,
                microbatch.suspended_decodes,
                self.offloaded_prefills,
                self.offloaded_decodes,
            )
            for microbatch in self.microbatches
        ]
        self.buffer: Queue[StepRequest] = Queue()

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
                    self.network_executor.submit(self.send_request, next_request)
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
        self.pending_prefills[request.task_id] = (
            request.x,
            request.sampling_params,
        )
        self.clear_buffer()

    def process_step(self, request: StepRequest) -> StepRequest:
        request = self.convert_step_request(request)
        result = super().process_step(request)
        return result

    def convert_step_request(self, request: StepRequest) -> StepRequest:
        join_stage = self.join_stages[request.microbatch_id]
        swap_stage = self.swap_stages[request.microbatch_id]
        rest_pages = (
            self.virtual_page_pool.num_avails
            + self.task_manager.calc_occupied_space(request.cancel_task_ids)
        )
        request.refresh()

        rest_pages, request = join_stage.process(rest_pages, request)
        rest_pages, request = swap_stage.process(rest_pages, request)
        return request
