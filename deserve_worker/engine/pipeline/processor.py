import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Optional, cast

import requests
import torch

from deserve_utils.trace import OpId
from deserve_worker.engine.microbatch.scheduler import MicroBatchScheduler
from deserve_worker.execution.exec import BatchPrefill, SingleTrace
from deserve_worker.kvcache.manager import KVCacheManager
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
    TraceRequest,
)
from deserve_worker.resource import ResourceCollector
from deserve_worker.task import TaskData, TaskManager, main_device, main_dtype


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
        simulated_latency: float,
    ) -> None:
        self.num_rounds = num_rounds
        self.num_layers = sum(layer.count(".") for layer in layers)
        self.max_batch_size = batch_size
        self.cpu_page_pool = CpuPagePool(
            self.num_layers,
            (num_pages_main + num_pages_swap * 2) * 8,
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
            self.virtual_page_pool, self.cpu_page_pool
        )
        self.task_manager = TaskManager(num_pages_main + num_pages_swap, page_size)
        self.offloaded_decode_kvcaches: dict[str, PagedKVCache[CpuPagePool]] = {}

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
        self.simulated_latency = simulated_latency

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
                if isinstance(request, DecodeRequest) or isinstance(
                    request, PrefillRequest
                ):
                    self.synchronize()
                    last_sync = time.time()
                    if isinstance(request, PrefillRequest):
                        # here, we assume that we do not do batch prefill
                        self.task_manager.add(
                            TaskData.empty(
                                request.task_ids[0],
                                request.xs.shape[0],
                                request.sampling_params[0],
                            )
                        )
                    next_request = self.process_step(request)
                elif isinstance(request, TraceRequest):
                    next_request = self.process_trace(request)
                if next_request is not None:
                    self.send_request(next_request)
                print(f"all: {(time.time() - last_sync) * 1000:.2f}ms")
        except Exception as e:
            traceback.print_exc()

    def add_request(self, request: LLMRequest) -> None:
        self.queue.put(request)

    def _post_request(
        self, url: str, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]
    ) -> None:
        begin = time.time()
        if self.simulated_latency > 0:
            default_stream = torch.cuda.Stream(device=main_device)  # type: ignore
            default_stream.synchronize()  # type: ignore
            time.sleep(self.simulated_latency)
        requests.post(url, data=dumps(tensors, metadata))
        print(f"post: {(time.time() - begin) * 1000:.2f}ms")

    def send_request(self, request: LLMRequest) -> None:
        if isinstance(request, InitRequest):
            url = f"{self.next_worker_url}/init"
        elif isinstance(request, DecodeRequest) or isinstance(request, PrefillRequest):
            url = f"{self.next_worker_url}/step"
        elif isinstance(request, TraceRequest):
            url = f"{self.next_worker_url}/trace"
        else:
            raise ValueError(f"Unknown request type: {request}")
        tensors, metadata = request.into_safetensors()
        self.network_executor.submit(self._post_request, url, tensors, metadata)

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
        begin = time.time()
        self.kvcache_manager.virtual_page_pool.stream.synchronize()  # type: ignore
        self.kvcache_manager.virtual_page_pool.stream2.synchronize()  # type: ignore
        self.kvcache_manager.virtual_page_pool.switch()
        end = time.time()
        print(f"synchronize: {(end - begin) * 1000:.2f}ms")

    def process_step(self, request: DecodeRequest) -> DecodeRequest:
        begin = time.time()
        microbatch = self.microbatches[request.microbatch_id]
        if self.num_rounds % 2 == 1:
            microbatch.adjust()
        print(f"adjust: {(time.time() - begin) * 1000:.2f}ms")
        prev_microbatch = self.microbatches[
            (request.microbatch_id - 1 + self.num_rounds) % self.num_rounds
        ]
        next_microbatch = self.microbatches[
            (request.microbatch_id + 1) % self.num_rounds
        ]

        # cancel
        microbatch.cancel(request.cancel_task_ids)

        # exec
        cpu_kvcaches = microbatch.offload(request.offload_task_ids)
        for task_id, cpu_kvcache in zip(request.offload_task_ids, cpu_kvcaches):
            self.offloaded_decode_kvcaches[task_id] = cpu_kvcache

        microbatch.reload(
            request.reload_task_ids,
            [
                self.offloaded_decode_kvcaches.pop(task_id)
                for task_id in request.reload_task_ids
            ],
        )

        if len(request.exec_task_ids) > 0:
            if isinstance(request, PrefillRequest):  # init
                microbatch.join(request.task_ids)

            result = microbatch.step(
                request.exec_task_ids,
                request.xs,
                isinstance(request, PrefillRequest),
                prev_microbatch,
                next_microbatch,
            )
            if self.layer_storage.need_sample:
                self.send_result(
                    result.all_xs, result.all_task_ids, result.needed_probs
                )

            for task_id in result.ongoing_task_ids:
                task_data = self.task_manager.get(task_id)
                task_data.step()

            if self.layer_storage.need_sample:
                request.cancel_task_ids = result.done_task_ids
            request.xs = result.ongoing_xs
            request.exec_task_ids = result.ongoing_task_ids
        else:
            self.kvcache_manager.virtual_page_pool.swap2(
                next_microbatch.pinned_memory, prev_microbatch.pinned_memory
            )
            if self.layer_storage.need_sample:
                request.cancel_task_ids = []
        print(f"prepare: {(time.time() - begin) * 1000:.2f}ms")
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
