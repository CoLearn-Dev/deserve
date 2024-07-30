import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, cast

import requests
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

from deserve_worker.paged_kvcache import PagedKVCache

from .forward_engine import ForwardEngine, LayerForward
from .kvcache import KVCacheBase
from .layer_storage import global_layer_manager
from .model import dumps

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]

stop_tokens = torch.tensor(STOP_TOKEN_IDS)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class PlanStep(BaseModel):
    worker_id: str
    worker_url: str
    layers: list[str]


@dataclass
class TaskInfo:
    start_pos: int

    kvcaches: dict[int, KVCacheBase]
    """
    When flash attention is enabled, we use paged attention, otherwise the standard attention is adopted.
    """


class SamplingParams(BaseModel):
    temperature: float
    top_p: float
    max_total_len: int


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class Worker:
    def __init__(self, worker_id: str, max_total_bsz: int, controller_url: str):
        self.worker_id = worker_id
        self.controller_url = controller_url
        self.task_infos: dict[str, TaskInfo] = {}
        self.forward_engine = ForwardEngine(max_total_bsz)
        threading.Thread(target=self.forward_engine.run, daemon=True).start()
        self.network_executor = ThreadPoolExecutor(max_workers=max_total_bsz)

    def forward(
        self,
        x: torch.Tensor,
        task_id: str,
        round: int,
        plan: list[PlanStep],
        sampling_params: SamplingParams,
    ) -> None:
        try:
            index = next(
                (
                    i
                    for i, worker in enumerate(plan)
                    if worker.worker_id == self.worker_id
                ),
                None,
            )
            if index is None:
                return None

            if round == 0:
                kvcaches = {}
                for full_layer_name in plan[index].layers:
                    _, layer_name = full_layer_name.split("/")
                    if layer_name.startswith("layers."):
                        layer_id = int(layer_name.split(".")[1])
                        kvcaches[layer_id] = PagedKVCache(x, 0, torch.device("cuda"))

                # TODO: need double check whether request is repeated
                self.task_infos[task_id] = TaskInfo(
                    start_pos=0,
                    kvcaches=cast(dict[int, KVCacheBase], kvcaches),
                )

            bsz, seqlen = x.shape[:2]  # currently bsz is not used
            task_info = self.task_infos[task_id]
            layer_storage = global_layer_manager.get_layer_storage(plan[index].layers)
            back = queue.Queue[
                torch.Tensor
            ]()  # used to transfer tensor between engine and worker
            layer_forward = LayerForward(
                layer_storage=layer_storage,
                h=x.to("cuda"),
                seqlen=seqlen,
                start_pos=task_info.start_pos,
                kvcaches=task_info.kvcaches,
                back=back,
            )
            self.forward_engine.add_layer_forward(layer_forward)
            h = back.get()
            task_info.start_pos += seqlen

            to_pass: torch.Tensor
            cancel = False
            if index == len(plan) - 1:
                # it's the last node in the plan, firstly generate token
                if task_info.start_pos > sampling_params.max_total_len:
                    next_token = torch.tensor([[EOS_TOKEN_ID]])
                elif sampling_params.temperature > 0:
                    probs = torch.softmax(
                        h[:, -1] / sampling_params.temperature, dim=-1
                    )
                    next_token = sample_top_p(probs, sampling_params.top_p)
                    next_token = next_token.reshape(1, -1)
                else:
                    next_token = torch.argmax(h[:, -1], dim=-1)
                    next_token = next_token.reshape(1, -1)
                to_pass = next_token.to("cpu")

                # check whether to stop
                if to_pass[0] in STOP_TOKEN_IDS:
                    cancel = True

                round += 1
                self.network_executor.submit(
                    requests.post,
                    f"{self.controller_url}/update_tasks",
                    json=[
                        {
                            "task_id": task_id,
                            "output_tokens": to_pass.tolist(),
                        }
                    ],
                )
            else:
                # pass tensor to next node
                to_pass = h

            next_index = (index + 1) % len(plan)
            next_worker_url = plan[next_index].worker_url

            if cancel:
                self.network_executor.submit(
                    requests.post,
                    f"{next_worker_url}/cancel",
                    json={
                        "task_id": task_id,
                        "plan": [step.model_dump() for step in plan],
                    },
                )
            else:
                self.network_executor.submit(
                    requests.post,
                    f"{next_worker_url}/forward",
                    data=dumps(
                        {"x": to_pass},
                        {
                            "task_id": task_id,
                            "round": round,
                            "plan": plan,
                            "sampling_params": sampling_params,
                        },
                    ),
                )
        except Exception as e:
            traceback.print_exc()

    def cancel(self, task_id: str, plan: list[PlanStep]) -> None:
        index = next(
            (i for i, x in enumerate(plan) if x.worker_id == self.worker_id), None
        )
        if index is None:
            return

        task_info = self.task_infos.pop(task_id, None)
        if task_info is not None:
            for kvcache in task_info.kvcaches.values():
                kvcache.clear()

        if index != len(plan) - 1:
            requests.post(
                f"{plan[index + 1].worker_url}/cancel",
                json={
                    "task_id": task_id,
                    "plan": plan,
                },
            )
