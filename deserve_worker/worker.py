import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import requests
import torch

from deserve_worker.engine.event.base import EngineEvent
from deserve_worker.engine.event.exec import NewExecEvent

from .engine.llm_engine import LLMEngine
from .execution.exec import BatchDecode, BatchPrefill, SingleTrace
from .execution.result import (
    BatchAct,
    BatchPersist,
    BatchUpdate,
    ExecResult,
    TraceResult,
)
from .kvcache.kvcache import KVCache, main_device, main_dtype
from .kvcache.packed_kvcache import PackedKVCacheManager
from .kvcache.page_pool import PagePool
from .kvcache.paged_kvcache import PagedKVCacheManager
from .layer_storage import LayerManager
from .model.args import llama_3_70b_args
from .model.utils import dumps
from .resource import ResourceCollector
from .task import PlanStep, SamplingParams, TaskData, TaskDataPlaceholder, TaskInfo


class Worker:
    def __init__(
        self,
        worker_id: str,
        worker_url: str,
        max_total_bsz: int,
        controller_url: str,
    ):
        self.worker_id = worker_id
        self.worker_url = worker_url
        self.controller_url = controller_url
        self.task_datas: dict[str, TaskData] = {}
        self.relay_queue = queue.Queue[ExecResult]()
        self.event_queue = queue.Queue[
            EngineEvent
        ]()  # TODO: after we use async engine, we do not need this anymore
        self.llm_engine = LLMEngine(
            max_total_bsz,
            256,
            self.event_queue,
            self.relay_queue,
            is_scheduler=worker_url.endswith("8080"),
        )
        self.layer_manager = LayerManager(main_device)
        self.page_pool = PagePool(
            40,
            llama_3_70b_args.num_pages,
            llama_3_70b_args.page_size,
            self.event_queue,
            main_device,
            main_dtype,
        )
        self.paged_kvcache_manager = PagedKVCacheManager(self.page_pool)
        self.packed_kvcache_manager = PackedKVCacheManager(self.page_pool)
        self.resource_collector = ResourceCollector(llama_3_70b_args)
        self.resource_collector.print_resources()
        self.network_executor = ThreadPoolExecutor(max_workers=max_total_bsz)

        threading.Thread(target=self.llm_engine.run, daemon=True).start()
        threading.Thread(target=self.relay, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def locate_in_plan(self, plan: list[PlanStep]) -> Optional[int]:
        return next(
            (i for i, worker in enumerate(plan) if worker.worker_id == self.worker_id),
            None,
        )

    def init_task_data(
        self, task_info: TaskInfo, is_trace: bool
    ) -> TaskData | TaskDataPlaceholder:
        """
        When placeholder is placed, we should
        """
        if task_info.round == 0 and task_info.task_id not in self.task_datas:
            kvcache: KVCache
            if is_trace:
                kvcache = self.packed_kvcache_manager.new()
            else:
                kvcache = self.paged_kvcache_manager.new()
            task_data = TaskData(
                task_id=task_info.task_id,
                start_pos=0,
                plan=task_info.plan,
                round=0,
                seqlen=task_info.seqlen,
                sampling_params=task_info.sampling_params,
                kvcache=kvcache,
            )
            self.task_datas[task_info.task_id] = task_data
        elif task_info.task_id in self.task_datas:
            task_data = self.task_datas[task_info.task_id]
            task_data.round = task_info.round
            task_data.seqlen = task_info.seqlen
        else:
            return task_info.into_task_data_placeholder()
        return task_data

    def batch_forward(
        self,
        xs: torch.Tensor,  # in shape [bsz * seqlen, vocab_size]
        task_infos: list[TaskInfo],
    ) -> None:
        round = task_infos[0].round
        plan = task_infos[0].plan
        index = self.locate_in_plan(plan)
        assert index is not None
        layer_storage = self.layer_manager.get_layer_storage(
            task_infos[0].plan[index].layers
        )

        task_datas = []
        ptr = 0
        for task_info in task_infos:
            task_data = self.init_task_data(task_info, is_trace=False)
            if isinstance(task_data, TaskData):
                task_datas.append(task_data)
                ptr += task_data.seqlen
            else:
                xs = torch.cat([xs[:ptr], xs[ptr + task_info.seqlen :]])
        if len(task_datas) == 0:
            return  # nothing happens

        # currently, we do not allow prefills and decodes to be mixed in the same batch
        if round == 0:
            prefill = BatchPrefill(
                xs=xs.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=task_datas,
            )
            self.llm_engine.add_event(NewExecEvent(prefill))
        else:
            decode = BatchDecode(
                xs=xs.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=task_datas,
            )
            self.llm_engine.add_event(NewExecEvent(decode))

    def forward(
        self,
        x: torch.Tensor,  # in shape [seqlen, vocab_size]
        task_id: str,
        round: int,
        plan: list[PlanStep],
        sampling_params: SamplingParams,
    ) -> None:
        index = self.locate_in_plan(plan)
        if index is None:
            return None

        layer_storage = self.layer_manager.get_layer_storage(plan[index].layers)
        seqlen = x.shape[0]
        task_data = self.init_task_data(
            TaskInfo(
                task_id=task_id,
                plan=plan,
                round=round,
                seqlen=seqlen,
                sampling_params=sampling_params,
            ),
            is_trace=False,
        )
        if isinstance(task_data, TaskDataPlaceholder):
            return
        if round == 0:
            prefill = BatchPrefill(
                xs=x.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=[task_data],
            )
            self.llm_engine.add_event(NewExecEvent(prefill))
        else:
            decode = BatchDecode(
                xs=x.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=[task_data],
            )
            self.llm_engine.add_event(NewExecEvent(decode))

    def trace(
        self,
        x: torch.Tensor,
        task_id: str,
        round: int,
        plan: list[PlanStep],
        sampling_params: SamplingParams,
    ) -> None:
        index = self.locate_in_plan(plan)
        if index is None:
            return None

        seqlen = x.shape[0]
        layer_storage = self.layer_manager.get_layer_storage(plan[index].layers)
        task_data = self.init_task_data(
            TaskInfo(
                task_id=task_id,
                plan=plan,
                round=round,
                seqlen=seqlen,
                sampling_params=sampling_params,
            ),
            is_trace=True,
        )
        if isinstance(task_data, TaskDataPlaceholder):
            return

        trace = SingleTrace(
            xs=x.to(main_device),
            layer_storage=layer_storage,
            task_datas=[task_data],
            page_pool=self.page_pool,
            traces={},
        )
        self.llm_engine.add_event(NewExecEvent(trace))

    def relay(self) -> None:
        q = self.relay_queue
        while True:
            result = q.get()
            if isinstance(result, BatchAct):
                task_data = result.task_datas[0]
                plan = task_data.plan
                index = self.locate_in_plan(plan)
                assert index is not None
                next_index = (index + 1) % len(plan)
                next_worker_url = plan[next_index].worker_url
                data = dumps(
                    {"x": result.xs},
                    {
                        "task_infos": [
                            task_data.into_task_info().model_dump()
                            for task_data in result.task_datas
                        ],
                    },
                )
                self.network_executor.submit(
                    requests.post,
                    f"{next_worker_url}/batch_forward",
                    data=data,
                )
            elif isinstance(result, BatchUpdate):
                updated_tasks = []
                for tokens, task_data in zip(result.tokens, result.task_datas):
                    updated_tasks.append(
                        {
                            "task_id": task_data.task_id,
                            "output_tokens": tokens.tolist(),
                        }
                    )
                self.network_executor.submit(
                    requests.post,
                    f"{self.controller_url}/update_tasks",
                    json=updated_tasks,
                )
                for task_data in result.cancel_datas:
                    self.cancel(task_data.task_id, None, task_data.plan)
            elif isinstance(result, TraceResult):
                task_data = result.task_datas[0]
                plan = task_data.plan
                index = self.locate_in_plan(plan)
                assert index is not None
                next_index = (index + 1) % len(plan)
                if next_index != 0:
                    next_worker_url = plan[next_index].worker_url
                    data = dumps(
                        {"x": result.x},
                        task_data.into_task_info().model_dump(),
                    )
                    self.network_executor.submit(
                        requests.post,
                        f"{next_worker_url}/trace",
                        data=data,
                    )
                metadata: dict[str, Any] = {"task_id": task_data.task_id}
                if next_index == 0:  # last worker
                    metadata["token"] = result.x.tolist()
                data = dumps(
                    {str(key): value for key, value in result.trace.items()},
                    metadata,
                )
                self.network_executor.submit(
                    requests.post,
                    f"{self.controller_url}/update_traces",
                    data=data,
                )
            elif isinstance(result, BatchPersist):
                for task_data in result.task_datas:
                    self.persist(task_data.task_id, None, task_data.plan)

    def cancel(
        self, task_id: str, start_index: Optional[int], plan: list[PlanStep]
    ) -> None:
        index = next(
            (i for i, x in enumerate(plan) if x.worker_id == self.worker_id), None
        )
        if index is None:
            return
        if start_index is None:
            start_index = index

        task_data = self.task_datas.pop(task_id, None)
        if task_data is not None:
            task_data.kvcache.clear()
        next_index = (index + 1) % len(plan)
        if next_index != start_index:
            requests.post(
                f"{plan[next_index].worker_url}/cancel",
                json={
                    "task_id": task_id,
                    "start_index": start_index,
                    "plan": [step.model_dump() for step in plan],
                },
            )

    def persist(
        self, task_id: str, start_index: Optional[int], plan: list[PlanStep]
    ) -> None:
        index = next(
            (i for i, x in enumerate(plan) if x.worker_id == self.worker_id), None
        )
        if index is None:
            return
        if start_index is None:
            start_index = index
        else:
            # the first worker has already persist the data inside LLM engine
            task_data = self.task_datas.get(task_id)
            if task_data is not None and isinstance(task_data.kvcache, KVCache):
                task_data.kvcache = task_data.kvcache.into_persistent()
        next_index = (index + 1) % len(plan)
        if next_index != start_index:
            requests.post(
                f"{plan[next_index].worker_url}/persist",
                json={
                    "task_id": task_id,
                    "start_index": start_index,
                    "plan": [step.model_dump() for step in plan],
                },
            )

    def heartbeat(self) -> None:
        while True:
            self.network_executor.submit(
                requests.post,
                f"{self.controller_url}/heartbeat",
                json={
                    "worker_id": self.worker_id,
                    "worker_url": self.worker_url,
                },
            )
            time.sleep(1)
