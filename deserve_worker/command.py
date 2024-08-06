from dataclasses import dataclass

import torch

from deserve_worker.kvcache.kvcache import KVCacheManager
from deserve_worker.layer_storage import LayerStorage
from deserve_worker.task import TaskData
from deserve_worker.trace import OpId


@dataclass
class BatchForward:
    xs: torch.Tensor
    layer_storage: LayerStorage
    task_datas: list[TaskData]
    kvcache_manager: KVCacheManager
    need_sample: bool  # to be eliminated in the future, because we can infer this from LayerStorage


@dataclass
class SingleTrace:
    x: torch.Tensor
    layer_storage: LayerStorage
    task_data: TaskData
    kvcache_manager: KVCacheManager
    need_sample: bool


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
class TraceResult:
    x: torch.Tensor
    task_id: str
    trace: dict[OpId, torch.Tensor]
