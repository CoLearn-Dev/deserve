import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from typing import Any, Optional, cast

import requests
import torch

from deserve_utils.trace import OpId
from deserve_worker.engine.microbatch.scheduler import MicroBatchScheduler
from deserve_worker.execution.exec import BatchPrefill, SingleTrace
from deserve_worker.kvcache.manager import KVCacheManager
from deserve_worker.kvcache.paged.chunk_pool import ChunkHandle, CpuChunkPool
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool
from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory
from deserve_worker.kvcache.virtual import VirtualPagePool
from deserve_worker.layer_storage import LayerManager
from deserve_worker.model.args import get_model_args, llama_3_70b_args
from deserve_worker.model.utils import dumps
from deserve_worker.request import (
    DecodeRequest,
    InitRequest,
    LLMRequest,
    PrefillRequest,
    StepRequest,
    TraceRequest,
)
from deserve_worker.resource import ResourceCollector
from deserve_worker.task import TaskData, TaskManager, main_device, main_dtype

LOG_INTERVAL = 5


class PipelineProcessor:
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
        buddy_height: int,
    ) -> None:
        self.num_rounds = num_rounds
        self.num_layers = sum(layer.count(".") for layer in layers)
        self.max_batch_size = batch_size
        self.cpu_chunk_pool = CpuChunkPool(
            self.num_layers,
            buddy_height,
            page_size,
            main_dtype,
        )
        self.virtual_page_pool = VirtualPagePool(
            self.num_layers,
            num_pages_main,
            num_pages_swap,
            page_size,
            main_device,
            main_dtype,
        )
        self.kvcache_manager = KVCacheManager(
            self.virtual_page_pool, self.cpu_chunk_pool
        )
        self.task_manager = TaskManager(num_pages_main + num_pages_swap, page_size)
        self.offloaded_kvcaches: dict[str, ChunkHandle] = {}

        self.layer_manager = LayerManager(main_device)
        self.layer_storage = self.layer_manager.get_layer_storage(layers)
        self.model_args = get_model_args(layers[0].split("/")[0])
        self.model_args.num_pages = num_pages_main + num_pages_swap * 2
        self.resource_collector = ResourceCollector(self.model_args)
        self.resource_collector.print_resources()
        self.microbatches = [
            MicroBatchScheduler(
                self.kvcache_manager,
                self.task_manager,
                self.layer_storage,
            )
            for _ in range(num_rounds)
        ]
        self.queue: Queue[LLMRequest] = Queue()
        self.worker_url = worker_url
        self.next_worker_url = next_worker_url
        self.controller_url = controller_url
        self.network_executor = ThreadPoolExecutor(max_workers=64)
        self.stream = torch.cuda.Stream(device=main_device)  # type: ignore
        self.staged_time_log: list[float] = []
        self.recv_bytes_log: int = 0
        self.send_bytes_log: int = 0
        self.highest_upload_speed: float = 0
        self.highest_download_speed: float = 0
        self.current_microbatch_id = 0

    def staged_print(self) -> None:
        total_batch_size = sum(
            len(microbatch.ongoing_paged_kvcaches) for microbatch in self.microbatches
        )
        if len(self.staged_time_log) > 0:
            avg_stage_time = (
                f"{sum(self.staged_time_log) / len(self.staged_time_log) * 1000:.2f}ms"
            )
        else:
            avg_stage_time = "N/A"
        avg_available_pages = (
            self.kvcache_manager.virtual_page_pool.num_avails
            - self.microbatches[self.current_microbatch_id].pinned_memory.count
        ) + sum(
            microbatch.pinned_memory.count for microbatch in self.microbatches
        ) / self.num_rounds
        upload_speed = self.send_bytes_log / LOG_INTERVAL / 1024 / 1024
        download_speed = self.recv_bytes_log / LOG_INTERVAL / 1024 / 1024
        self.highest_upload_speed = max(self.highest_upload_speed, upload_speed)
        self.highest_download_speed = max(self.highest_download_speed, download_speed)

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Total batch size: {total_batch_size}; Avg stage time: {avg_stage_time}; Avg available pages: {avg_available_pages}; Current upload speed: {upload_speed:.2f} MB/s; Current download speed: {download_speed:.2f} MB/s; Highest upload speed: {self.highest_upload_speed:.2f} MB/s; Highest download speed: {self.highest_download_speed:.2f} MB/s"
        )

        self.recv_bytes_log = 0
        self.send_bytes_log = 0
        self.staged_time_log.clear()

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
                if isinstance(request, StepRequest):
                    self.synchronize()
                    for task_id, (initial_seqlen, sp) in request.init_tasks.items():
                        self.task_manager.add(
                            TaskData.empty(task_id, initial_seqlen, sp)
                        )
                    next_request = self.process_step(request)
                elif isinstance(request, TraceRequest):
                    next_request = self.process_trace(request)
                if next_request is not None:
                    self.send_request(next_request)
        except Exception as e:
            traceback.print_exc()

    def add_request(self, request: LLMRequest) -> None:
        self.queue.put(request)
        self.recv_bytes_log += request.get_tensors_size()

    def _post_request(
        self, url: str, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
    ) -> None:
        data = dumps(tensors, metadata)
        requests.post(url, data=data)

    def send_request(self, request: LLMRequest) -> None:
        if isinstance(request, InitRequest):
            url = f"{self.next_worker_url}/init"
        elif isinstance(request, StepRequest):
            url = f"{self.next_worker_url}/step"
        elif isinstance(request, TraceRequest):
            url = f"{self.next_worker_url}/trace"
        else:
            raise ValueError(f"Unknown request type: {request}")
        tensors, metadata = request.into_safetensors()
        self.network_executor.submit(self._post_request, url, tensors, metadata)
        self.send_bytes_log += request.get_tensors_size()

    def send_result(
        self,
        tokens: torch.Tensor,
        task_ids: list[str],
        needed_probs: dict[str, list[tuple[int, float]]],
    ) -> None:
        url = f"{self.controller_url}/update_tasks"
        tosend = [
            {
                "task_id": task_id,
                "output_token": token,
                "needed_probs": needed_probs.get(task_id),
            }
            for task_id, token in zip(task_ids, tokens.tolist())
        ]
        self.network_executor.submit(requests.post, url, json=tosend)

    def send_traces(
        self,
        token: Optional[torch.Tensor],
        probs: Optional[list[tuple[int, float]]],
        traces: dict[OpId, torch.Tensor],
        output2input: dict[OpId, list[OpId]],
        task_id: str,
    ) -> None:
        url = f"{self.controller_url}/update_traces"
        str_output2input = {
            str(op_id): [str(input_op_id) for input_op_id in input_op_ids]
            for op_id, input_op_ids in output2input.items()
        }
        metadata: dict[str, Any] = {
            "task_id": task_id,
            "output2input": str_output2input,
        }
        if token is not None:
            metadata["token"] = token.tolist()[0]
        if probs is not None:
            metadata["probs"] = probs
        tensors = {str(op_id): tensor for op_id, tensor in traces.items()}
        self.network_executor.submit(requests.post, url, data=dumps(tensors, metadata))

    def synchronize(self) -> None:
        self.kvcache_manager.virtual_page_pool.stream.synchronize()  # type: ignore
        self.kvcache_manager.virtual_page_pool.stream2.synchronize()  # type: ignore
        self.kvcache_manager.virtual_page_pool.switch()

    def process_step(self, request: StepRequest) -> StepRequest:
        microbatch = self.microbatches[request.microbatch_id]
        assert request.microbatch_id == self.current_microbatch_id
        self.current_microbatch_id = (self.current_microbatch_id + 1) % self.num_rounds
        if self.num_rounds % 2 == 1:
            microbatch.adjust()
        prev_microbatch = self.microbatches[
            (request.microbatch_id - 1 + self.num_rounds) % self.num_rounds
        ]
        next_microbatch = self.microbatches[
            (request.microbatch_id + 1) % self.num_rounds
        ]

        # init all sequence lengths
        for task_id, seqlen in zip(request.exec_task_ids, request.exec_seqlens):
            task_data = self.task_manager.get(task_id)
            task_data.init(seqlen)

        # cancel
        microbatch.cancel(request.cancel_task_ids)

        # exec
        cpu_chunk_handles = microbatch.offload(request.offload_task_ids)
        for task_id, cpu_chunk_handle in zip(
            request.offload_task_ids, cpu_chunk_handles
        ):
            self.offloaded_kvcaches[task_id] = cpu_chunk_handle

        microbatch.reload(
            request.reload_task_ids,
            [
                self.offloaded_kvcaches.pop(task_id)
                for task_id in request.reload_task_ids
            ],
        )
        microbatch.kvcache_manager.synchronize()

        if len(request.exec_task_ids) > 0:
            microbatch.join(list(request.init_tasks.keys()))

            result = microbatch.step(
                request.exec_task_ids,
                request.xs,
                prev_microbatch,
                next_microbatch,
            )  # synchronize for previous offloading and reloading inside
            if self.layer_storage.need_sample:
                self.send_result(
                    result.all_xs, result.all_task_ids, result.needed_probs
                )

            for task_id in request.exec_task_ids:
                task_data = self.task_manager.get(task_id)
                task_data.step()

            if self.layer_storage.need_sample:
                request.cancel_task_ids = result.done_task_ids
            request.xs = result.ongoing_xs
            request.exec_task_ids = result.ongoing_task_ids
        else:
            # self.kvcache_manager.synchronize()  # for previous offloading and reloading
            self.kvcache_manager.virtual_page_pool.swap2(
                next_microbatch.pinned_memory, prev_microbatch.pinned_memory
            )
            if self.layer_storage.need_sample:
                request.cancel_task_ids = []
        return request

    def process_trace(self, request: TraceRequest) -> Optional[LLMRequest]:
        traces: dict[OpId, torch.Tensor] = {}
        output2input: dict[OpId, list[OpId]] = {}
        trace = SingleTrace(
            xs=request.x,
            layer_storage=self.layer_storage,
            task_datas=[
                TaskData.empty(
                    request.task_id,
                    request.x.shape[0],
                    request.sampling_params,
                )
            ],
            traces=traces,
            output2input=output2input,
        )
        result = trace.step()
        if self.layer_storage.need_sample:
            self.send_traces(
                result.all_xs,
                result.needed_probs[request.task_id],
                traces,
                output2input,
                request.task_id,
            )
            return None
        else:
            request.x = result.all_xs
            return request

    def heartbeat(self) -> None:
        is_start = self.layer_storage.need_tokenize
        while True:
            url = f"{self.controller_url}/heartbeat"
            self.network_executor.submit(
                requests.post,
                url,
                json={
                    "worker_url": self.worker_url,
                    "is_start": is_start,
                    "next_worker_url": self.next_worker_url,
                },
            )
            time.sleep(1)

    def log(self) -> None:
        while True:
            time.sleep(LOG_INTERVAL)
            self.staged_print()
