from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from deserve_worker.task import SamplingParams, main_device, main_dtype


class LLMRequest(ABC):
    @abstractmethod
    def into_safetensors(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        pass


@dataclass
class PrefillRequest(LLMRequest):
    x: torch.Tensor
    task_id: str
    sampling_params: SamplingParams

    def into_safetensors(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return {
            "x": self.x,
        }, {
            "task_id": self.task_id,
            "sampling_params": self.sampling_params.model_dump(),
        }


@dataclass
class DecodeRequest(LLMRequest):
    group_id: int
    xs: torch.Tensor
    exec_task_ids: list[str]
    cancel_task_ids: list[str]
    offload_task_ids: list[str]
    reload_task_ids: list[str]
    r2s_task_ids: list[str]
    e2s_task_ids: list[str]
    s2e_task_ids: list[str]
    r2e_task_ids: list[str]

    @staticmethod
    def empty(group_id: int, dtype: torch.dtype) -> "DecodeRequest":
        return DecodeRequest(
            group_id=group_id,
            xs=torch.empty((0,), dtype=dtype, device=main_device),
            exec_task_ids=[],
            cancel_task_ids=[],
            offload_task_ids=[],
            reload_task_ids=[],
            r2s_task_ids=[],
            e2s_task_ids=[],
            s2e_task_ids=[],
            r2e_task_ids=[],
        )

    def is_empty(self) -> bool:
        return (
            self.xs.numel() == 0
            and len(self.exec_task_ids) == 0
            and len(self.cancel_task_ids) == 0
            and len(self.offload_task_ids) == 0
            and len(self.reload_task_ids) == 0
            and len(self.r2s_task_ids) == 0
            and len(self.e2s_task_ids) == 0
            and len(self.s2e_task_ids) == 0
            and len(self.r2e_task_ids) == 0
        )

    def append_exec(self, task_id: str, x: torch.Tensor) -> None:
        self.exec_task_ids.append(task_id)
        if self.xs.numel() == 0:
            self.xs = x
        else:
            self.xs = torch.cat([self.xs, x], dim=0)

    def get_bsz(self) -> int:
        return len(self.exec_task_ids)

    def refresh(self) -> None:
        self.offload_task_ids = []
        self.reload_task_ids = []
        self.r2s_task_ids = []
        self.e2s_task_ids = []
        self.s2e_task_ids = []
        self.r2e_task_ids = []

    def into_safetensors(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return {
            "xs": self.xs,
        }, {
            "group_id": self.group_id,
            "exec_task_ids": self.exec_task_ids,
            "offload_task_ids": self.offload_task_ids,
            "reload_task_ids": self.reload_task_ids,
            "cancel_task_ids": self.cancel_task_ids,
            "r2s_task_ids": self.r2s_task_ids,
            "e2s_task_ids": self.e2s_task_ids,
            "s2e_task_ids": self.s2e_task_ids,
            "r2e_task_ids": self.r2e_task_ids,
        }


@dataclass
class JoinRequest(LLMRequest):
    x: torch.Tensor
    task_id: str

    def into_safetensors(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return {
            "x": self.x,
        }, {
            "task_id": self.task_id,
        }
