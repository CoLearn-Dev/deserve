import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import requests
import torch
from transformers import AutoTokenizer  # type: ignore

from deserve_worker.execution.response import (
    BatchResult,
    BatchUpdate,
    LLMResponse,
    TraceResult,
)
from deserve_worker.kvcache.packed_kvcache import PackedKVCacheManager
from deserve_worker.kvcache.page_pool import PagePool

from .execution.exec import BatchDecode, BatchPrefill, SingleTrace
from .kvcache.kvcache import KVCache, main_device, main_dtype
from .kvcache.paged_kvcache import PagedKVCacheManager
from .layer_storage import LayerManager
from .llm_engine import LLMEngine
from .model.utils import dumps
from .task import PlanStep, SamplingParams, TaskData, TaskInfo

EOS_TOKEN_ID = 128001  # for llama 3 only

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


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
        self.relay_queue = queue.Queue[LLMResponse]()
        self.llm_engine = LLMEngine(max_total_bsz, 256, self.relay_queue)
        self.layer_manager = LayerManager(main_device)
        self.page_pool = PagePool(40, 290, 256, main_device, main_dtype)
        # TODO: in future, different cache manager could allocate on same memory
        self.paged_kvcache_manager = PagedKVCacheManager(self.page_pool)
        self.packed_kvcache_manager = PackedKVCacheManager(self.page_pool)
        self.network_executor = ThreadPoolExecutor(max_workers=max_total_bsz)

        threading.Thread(target=self.llm_engine.run, daemon=True).start()
        threading.Thread(target=self.relay, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def locate_in_plan(self, plan: list[PlanStep]) -> Optional[int]:
        return next(
            (i for i, worker in enumerate(plan) if worker.worker_id == self.worker_id),
            None,
        )

    def init_forward_task_data(self, x: torch.Tensor, task_info: TaskInfo) -> TaskData:
        if task_info.round == 0:
            # TODO: need double check whether request is repeated
            task_data = TaskData(
                task_id=task_info.task_id,
                start_pos=0,
                plan=task_info.plan,
                round=0,
                seqlen=task_info.seqlen,
                sampling_params=task_info.sampling_params,
                kvcache=cast(
                    KVCache, self.paged_kvcache_manager.alloc(task_info.seqlen)
                ),
            )
            self.task_datas[task_info.task_id] = task_data
        else:
            task_data = self.task_datas[task_info.task_id]
            task_data.round = task_info.round
            task_data.seqlen = task_info.seqlen

        return task_data

    def init_trace_task_data(self, x: torch.Tensor, task_info: TaskInfo) -> TaskData:
        if task_info.round == 0:
            task_data = TaskData(
                task_id=task_info.task_id,
                start_pos=0,
                plan=task_info.plan,
                round=0,
                seqlen=task_info.seqlen,
                sampling_params=task_info.sampling_params,
                kvcache=cast(
                    KVCache, self.packed_kvcache_manager.alloc(task_info.seqlen)
                ),
            )
            self.task_datas[task_info.task_id] = task_data
        else:
            task_data = self.task_datas[task_info.task_id]
            task_data.round = task_info.round
            task_data.seqlen = task_info.seqlen

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
        task_datas = [
            self.init_forward_task_data(xs, task_info) for task_info in task_infos
        ]
        seqlens = [task_info.seqlen for task_info in task_infos]
        assert sum(seqlens) == xs.shape[0]
        if round == 0:
            prefill = BatchPrefill(
                xs=xs.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=task_datas,
                bsz=len(task_infos),
                seqlens=seqlens,
                total_seqlen=sum(seqlens),
            )
            self.llm_engine.add_request(prefill)
        else:
            decode = BatchDecode(
                xs=xs.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=task_datas,
                bsz=len(task_infos),
            )
            self.llm_engine.add_request(decode)

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
        task_datas = [
            self.init_forward_task_data(
                x,
                TaskInfo(
                    task_id=task_id,
                    plan=plan,
                    round=round,
                    seqlen=seqlen,
                    sampling_params=sampling_params,
                ),
            )
        ]
        if round == 0:
            prefill = BatchPrefill(
                xs=x.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=task_datas,
                bsz=1,
                seqlens=[seqlen],
                total_seqlen=seqlen,
            )
            self.llm_engine.add_request(prefill)
        else:
            decode = BatchDecode(
                xs=x.to(main_device),
                layer_storage=layer_storage,
                page_pool=self.page_pool,
                task_datas=task_datas,
                bsz=1,
            )
            self.llm_engine.add_request(decode)

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
        trace = SingleTrace(
            xs=x.to(main_device),
            layer_storage=layer_storage,
            task_datas=[
                self.init_trace_task_data(
                    x,
                    TaskInfo(
                        task_id=task_id,
                        plan=plan,
                        round=round,
                        seqlen=seqlen,
                        sampling_params=sampling_params,
                    ),
                )
            ],
            bsz=1,
            page_pool=self.page_pool,
            traces={},
        )
        self.llm_engine.add_request(trace)

    def relay(self) -> None:
        q = self.relay_queue
        while True:
            result = q.get()
            if isinstance(result, BatchResult):
                task_id = result.task_ids[0]
                task_info = self.task_datas[task_id]
                plan = task_info.plan
                index = self.locate_in_plan(plan)
                assert index is not None
                next_index = (index + 1) % len(plan)
                next_worker_url = plan[next_index].worker_url
                data = dumps(
                    {"x": result.xs},
                    {
                        "task_infos": [
                            {
                                "task_id": task_id,
                                "round": self.task_datas[task_id].round,
                                "plan": plan,
                                "seqlen": (
                                    1
                                    if self.task_datas[task_id].round != 0
                                    else self.task_datas[task_id].seqlen
                                ),
                                "sampling_params": self.task_datas[
                                    task_id
                                ].sampling_params,
                            }
                            for task_id in result.task_ids
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
                for tokens, task_id in zip(result.tokens, result.task_ids):
                    updated_tasks.append(
                        {
                            "task_id": task_id,
                            "output_tokens": tokens.tolist(),
                        }
                    )
                self.network_executor.submit(
                    requests.post,
                    f"{self.controller_url}/update_tasks",
                    json=updated_tasks,
                )
                for task_id in result.cancel_ids:
                    self.cancel(task_id, None, self.task_datas[task_id].plan)
            elif isinstance(result, TraceResult):
                task_id = result.task_id
                task_info = self.task_datas[task_id]
                plan = task_info.plan
                index = self.locate_in_plan(plan)
                assert index is not None
                next_index = (index + 1) % len(plan)
                if next_index != 0:
                    next_worker_url = plan[next_index].worker_url
                    data = dumps(
                        {"x": result.x},
                        {
                            "task_id": task_id,
                            "round": self.task_datas[task_id].round,
                            "plan": plan,
                            "sampling_params": self.task_datas[task_id].sampling_params,
                        },
                    )
                    self.network_executor.submit(
                        requests.post,
                        f"{next_worker_url}/trace",
                        data=data,
                    )
                metadata: dict[str, Any] = {"task_id": task_id}
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

        task_info = self.task_datas.pop(task_id, None)
        if task_info is not None:
            task_info.kvcache.clear()
        next_index = (index + 1) % len(plan)
        if next_index != start_index:
            requests.post(
                f"{plan[next_index].worker_url}/cancel",
                json={
                    "task_id": task_id,
                    "start_index": index,
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
