import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, cast

import requests
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

from .forward_engine import ForwardEngine
from .kvcache import KVCacheBase
from .layer_storage import global_layer_manager
from .model import dumps
from .paged_kvcache import PagedKVCache, global_paged_memory
from .task import LayerForward, PlanStep, ResultBack, SamplingParams, TaskData, TaskInfo

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]

stop_tokens = torch.tensor(STOP_TOKEN_IDS)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class Worker:
    def __init__(self, worker_id: str, max_total_bsz: int, controller_url: str):
        self.worker_id = worker_id
        self.controller_url = controller_url
        self.task_datas: dict[str, TaskData] = {}
        self.relay_queue = queue.Queue[list[ResultBack]]()
        self.forward_engine = ForwardEngine(max_total_bsz, self.relay_queue)
        threading.Thread(target=self.forward_engine.run, daemon=True).start()
        threading.Thread(target=self.relay, daemon=True).start()
        self.network_executor = ThreadPoolExecutor(max_workers=max_total_bsz)

    def locate_in_plan(self, plan: list[PlanStep]) -> Optional[int]:
        return next(
            (i for i, worker in enumerate(plan) if worker.worker_id == self.worker_id),
            None,
        )

    def init_task_data(
        self, x: torch.Tensor, index: int, task_info: TaskInfo
    ) -> TaskData:
        if task_info.round == 0:
            kvcaches = {}
            for full_layer_name in task_info.plan[index].layers:
                _, layer_name = full_layer_name.split("/")
                if layer_name.startswith("layers."):
                    layer_id = int(layer_name.split(".")[1])
                    kvcaches[layer_id] = PagedKVCache(x, 0, torch.device("cuda"))

            # TODO: need double check whether request is repeated
            task_data = TaskData(
                task_id=task_info.task_id,
                start_pos=0,
                plan=task_info.plan,
                round=0,
                sampling_params=task_info.sampling_params,
                kvcaches=cast(dict[int, KVCacheBase], kvcaches),
            )
            self.task_datas[task_info.task_id] = task_data
        else:
            task_data = self.task_datas[task_info.task_id]
            task_data.round = task_info.round

        return task_data

    def batch_forward(
        self,
        xs: torch.Tensor,
        task_infos: list[TaskInfo],
    ) -> None:
        ptr = 0
        forwards = []
        begin = time.time()
        plan = task_infos[0].plan
        index = self.locate_in_plan(plan)
        assert index is not None
        layer_storage = global_layer_manager.get_layer_storage(
            task_infos[0].plan[index].layers
        )
        xs_cuda = xs.to("cuda")
        for ptr, task_info in enumerate(task_infos):
            x_cuda = xs_cuda[ptr : ptr + 1]
            forwards.append(
                LayerForward(
                    layer_storage=layer_storage,
                    h=x_cuda,
                    task_data=self.init_task_data(
                        x_cuda,
                        index,
                        task_info,
                    ),
                    need_sample=(index == len(plan) - 1),
                )
            )
        print("process time:", (time.time() - begin) * 1000)
        self.forward_engine.add_layer_forward(forwards)

    def forward(
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

        layer_storage = global_layer_manager.get_layer_storage(plan[index].layers)
        layer_forward = LayerForward(
            layer_storage=layer_storage,
            h=x.to("cuda"),
            task_data=self.init_task_data(
                x,
                index,
                TaskInfo(
                    task_id=task_id,
                    plan=plan,
                    round=round,
                    sampling_params=sampling_params,
                ),
            ),
            need_sample=(index == len(plan) - 1),
        )
        self.forward_engine.add_layer_forward([layer_forward])

    def relay(self) -> None:
        q = self.relay_queue
        while True:
            results: list[ResultBack] = q.get()
            while True:
                try:
                    tasks = q.get(block=False)
                    results.extend(tasks)
                except queue.Empty:
                    break

            updated_tasks = []
            forward_tasks = []
            forward_tensors = []
            for result in results:
                task_id = result.task_id
                task_info = self.task_datas[task_id]
                plan = task_info.plan
                index = self.locate_in_plan(plan)
                assert index is not None

                cancel = False
                if index == len(plan) - 1:
                    tokens = result.x.tolist()

                    updated_tasks.append(
                        {
                            "task_id": task_id,
                            "output_tokens": tokens,
                        }
                    )

                    if tokens[0][0] in STOP_TOKEN_IDS:
                        cancel = True

                next_index = (index + 1) % len(plan)
                next_worker_url = task_info.plan[next_index].worker_url
                if cancel:
                    task_info = self.task_datas.pop(task_id)
                    for kvcache in task_info.kvcaches.values():
                        kvcache.clear()
                    if next_index != len(plan) - 1:
                        self.network_executor.submit(
                            requests.post,
                            f"{next_worker_url}/cancel",
                            json={
                                "task_id": task_id,
                                "plan": [step.model_dump() for step in plan],
                            },
                        )
                else:
                    forward_tasks.append(
                        {
                            "task_id": task_id,
                            "round": task_info.round,
                            "plan": plan,
                            "sampling_params": task_info.sampling_params,
                        }
                    )
                    forward_tensors.append(result.x)

            self.network_executor.submit(
                requests.post,
                f"{self.controller_url}/update_tasks",
                json=updated_tasks,
            )

            if len(forward_tasks) > 0:
                x = torch.cat(forward_tensors).to("cpu")
                data = dumps(
                    {"x": x},
                    {
                        "task_infos": forward_tasks,
                    },
                )
                self.network_executor.submit(
                    requests.post,
                    f"{next_worker_url}/batch_forward",
                    data=data,
                )

    def cancel(self, task_id: str, plan: list[PlanStep]) -> None:
        index = next(
            (i for i, x in enumerate(plan) if x.worker_id == self.worker_id), None
        )
        if index is None:
            return

        task_info = self.task_datas.pop(task_id, None)
        if task_info is not None:
            for kvcache in task_info.kvcaches.values():
                kvcache.clear()
        next_index = (index + 1) % len(plan)
        if next_index != len(plan) - 1:
            requests.post(
                f"{plan[next_index].worker_url}/cancel",
                json={
                    "task_id": task_id,
                    "plan": plan,
                },
            )
