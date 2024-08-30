import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

import requests
import torch

from deserve_worker.engine.group import Group
from deserve_worker.execution.exec import BatchPrefill
from deserve_worker.kvcache.manager.portable import PortableKVCacheManager
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool, GpuPagePool
from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory
from deserve_worker.layer_storage import LayerManager
from deserve_worker.model.utils import dumps
from deserve_worker.request import (
    DecodeRequest,
    JoinRequest,
    LLMRequest,
    PrefillRequest,
)
from deserve_worker.task import TaskData, TaskManager, main_device, main_dtype

PINNED_MEMORY_SIZE = 512 * 1024 * 1024


class Processor:
    def __init__(
        self,
        num_rounds: int,
        num_pages: int,
        page_size: int,
        batch_size: int,
        layers: list[str],
        next_worker_url: str,
        controller_url: str,
    ) -> None:
        self.num_rounds = num_rounds
        self.num_layers = sum(layer.count(".") for layer in layers)
        self.max_batch_size = batch_size
        self.cpu_page_pool = CpuPagePool(
            self.num_layers, num_pages, page_size, main_dtype
        )
        self.gpu_page_pool = GpuPagePool(
            self.num_layers, num_pages, page_size, main_device, main_dtype
        )
        self.kvcache_manager = PortableKVCacheManager(
            self.gpu_page_pool, self.cpu_page_pool
        )
        self.task_manager = TaskManager(num_pages, page_size)
        self.suspended_decode_kvcaches: dict[str, PagedKVCache[CpuPagePool]] = {}
        self.resumed_decode_kvcaches: dict[str, PagedKVCache[GpuPagePool]] = {}
        self.layer_manager = LayerManager(main_device)
        self.layer_storage = self.layer_manager.get_layer_storage(layers)
        self.groups = [
            Group(
                self.cpu_page_pool,
                self.gpu_page_pool,
                PinnedMemory(PINNED_MEMORY_SIZE),
                self.task_manager,
                self.layer_storage,
            )
            for _ in range(num_rounds)
        ]
        self.queue: Queue[LLMRequest] = Queue()
        self.limit_offload_len = (
            PINNED_MEMORY_SIZE // page_size // 2048 // self.num_layers
        )
        print("Offload max len:", self.limit_offload_len)
        self.next_worker_url = next_worker_url
        self.controller_url = controller_url
        self.network_executor = ThreadPoolExecutor(max_workers=64)

    def run(self) -> None:
        try:
            while True:
                request = self.queue.get()
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
                    next_request = self.process_decode(request)
                elif isinstance(request, JoinRequest):
                    self.process_join(request)
                else:
                    raise ValueError(f"Unknown request type: {request}")
                if next_request is not None:
                    self.send_request(next_request)
        except Exception as e:
            traceback.print_exc()

    def add_request(self, request: LLMRequest) -> None:
        self.queue.put(request)

    def send_request(self, request: LLMRequest) -> None:
        if isinstance(request, PrefillRequest):
            url = f"{self.next_worker_url}/prefill"
        elif isinstance(request, DecodeRequest):
            url = f"{self.next_worker_url}/decode"
        elif isinstance(request, JoinRequest):
            url = f"{self.next_worker_url}/join"
        else:
            raise ValueError(f"Unknown request type: {request}")
        tensors, metadata = request.into_safetensors()
        self.network_executor.submit(requests.post, url, data=dumps(tensors, metadata))

    def send_tokens(self, tokens: torch.Tensor, task_ids: list[str]) -> None:
        url = f"{self.controller_url}/update_tasks"
        tosend = [
            {"task_id": task_id, "output_tokens": [token]}
            for task_id, token in zip(task_ids, tokens.tolist())
        ]
        self.network_executor.submit(requests.post, url, json=tosend)

    def process_prefill(self, request: PrefillRequest) -> Optional[LLMRequest]:
        task_id = request.task_id
        kvcache = PagedKVCache.empty(self.gpu_page_pool)
        prefill = BatchPrefill(
            xs=request.x,
            layer_storage=self.layer_storage,
            task_datas=[self.task_manager.get(task_id)],
            kvcaches=[kvcache],
        )
        self.resumed_decode_kvcaches[task_id] = kvcache
        result = prefill.step()
        for task_id in result.ongoing_task_ids:
            task_data = self.task_manager.get(task_id)
            task_data.step()
        if self.layer_storage.need_sample:
            self.send_tokens(result.all_xs, result.all_task_ids)
        if len(result.ongoing_task_ids) > 0:
            if self.layer_storage.need_sample:
                return JoinRequest(x=result.all_xs, task_id=task_id)
            else:
                return PrefillRequest(
                    x=result.all_xs,
                    task_id=task_id,
                    sampling_params=request.sampling_params,
                )
        else:
            return None

    def process_decode(self, request: DecodeRequest) -> LLMRequest:
        # print(request)
        group = self.groups[request.group_id]
        prev_group = self.groups[
            (request.group_id - 1 + self.num_rounds) % self.num_rounds
        ]
        next_group = self.groups[(request.group_id + 1) % self.num_rounds]

        # cancel
        group.cancel(request.cancel_task_ids)

        # offload
        prev_group.offload(request.offload_task_ids)

        # reload
        next_group.reload(request.reload_task_ids)

        # exec
        for task_id in request.r2s_task_ids:
            gpu_kvcache = self.resumed_decode_kvcaches.pop(task_id)
            cpu_kvcahce = self.kvcache_manager.copy_gpu_to_cpu(gpu_kvcache)
            self.suspended_decode_kvcaches[task_id] = cpu_kvcahce
            gpu_kvcache.free()

        cpu_kvcaches = group.suspend(request.e2s_task_ids)
        for task_id, cpu_kvcache in zip(request.e2s_task_ids, cpu_kvcaches):
            self.suspended_decode_kvcaches[task_id] = cpu_kvcache

        group.resume(
            request.s2e_task_ids,
            [
                self.suspended_decode_kvcaches.pop(task_id)
                for task_id in request.s2e_task_ids
            ],
        )

        gpu_kvcaches = [
            self.resumed_decode_kvcaches.pop(task_id)
            for task_id in request.r2e_task_ids
        ]
        group.join(request.r2e_task_ids, gpu_kvcaches)

        if len(request.exec_task_ids) > 0:
            result = group.exec(request.xs, request.exec_task_ids)
            # print(
            #     "page pool:",
            #     self.gpu_page_pool.num_avails,
            # )
            if self.layer_storage.need_sample:
                self.send_tokens(result.all_xs, result.all_task_ids)

            for task_id in result.ongoing_task_ids:
                task_data = self.task_manager.get(task_id)
                task_data.step()

            if self.layer_storage.need_sample:
                request.cancel_task_ids = result.done_task_ids
            request.xs = result.ongoing_xs
            request.exec_task_ids = result.ongoing_task_ids
            return request
        else:
            if self.layer_storage.need_sample:
                request.cancel_task_ids = []
            return request

    def process_join(self, request: JoinRequest) -> None:
        raise NotImplementedError
