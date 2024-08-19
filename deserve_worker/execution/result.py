from dataclasses import dataclass

import torch

from deserve_worker.task import TaskData
from deserve_worker.trace import OpId


@dataclass
class ExecResult:
    task_datas: list[TaskData]
    pass


@dataclass
class BatchAct(ExecResult):
    xs: torch.Tensor


@dataclass
class BatchUpdate(ExecResult):
    tokens: list[torch.Tensor]
    cancel_datas: list[TaskData]


@dataclass
class BatchPersist(ExecResult):
    pass


@dataclass
class TraceResult(ExecResult):
    x: torch.Tensor
    trace: dict[OpId, torch.Tensor]
