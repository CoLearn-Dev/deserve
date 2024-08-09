from dataclasses import dataclass

import torch

from deserve_worker.trace import OpId


@dataclass
class LLMResponse:
    pass


@dataclass
class BatchResult(LLMResponse):
    xs: torch.Tensor
    task_ids: list[str]


@dataclass
class BatchUpdate(LLMResponse):
    tokens: list[torch.Tensor]
    task_ids: list[str]
    cancel_ids: list[str]


@dataclass
class TraceResult(LLMResponse):
    x: torch.Tensor
    task_id: str
    trace: dict[OpId, torch.Tensor]
