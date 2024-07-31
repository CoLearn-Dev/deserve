from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel

from .kvcache import KVCacheBase
from .layer_storage import LayerStorage


class PlanStep(BaseModel):
    worker_id: str
    worker_url: str
    layers: list[str]


class SamplingParams(BaseModel):
    temperature: float
    top_p: float
    max_total_len: int


class TaskInfo(BaseModel):
    task_id: str
    plan: list[PlanStep]
    round: int
    sampling_params: SamplingParams


@dataclass
class TaskData:
    task_id: str
    start_pos: int
    plan: list[PlanStep]
    round: int
    sampling_params: SamplingParams
    kvcaches: dict[int, KVCacheBase]
    """
    When flash attention is enabled, we use paged attention, otherwise the standard attention is adopted.
    """


class LayerForward:
    def __init__(
        self,
        layer_storage: LayerStorage,
        h: torch.Tensor,
        task_data: TaskData,
        need_sample: bool,
    ):
        self.layer_storage = layer_storage
        self.h = h
        self.task_info = task_data
        self.need_sample = need_sample


@dataclass
class BatchForward:
    xs: torch.Tensor
    layer_storage: LayerStorage
    task_datas: list[TaskData]
    need_sample: bool  # to be eliminated in the future, because we can infer this from LayerStorage


@dataclass
class BatchResult:
    xs: torch.Tensor
    task_ids: list[str]


@dataclass
class BatchUpdate:
    tokens: list[torch.Tensor]
    task_ids: list[str]
    cancel_ids: list[str]


@dataclass
class ResultBack:
    x: torch.Tensor
    task_id: str
