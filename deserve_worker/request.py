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
class InitRequest(LLMRequest):
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
    microbatch_id: int
    xs: torch.Tensor
    exec_task_ids: list[str]
    cancel_task_ids: list[str]
    offload_task_ids: list[str]
    reload_task_ids: list[str]

    @staticmethod
    def empty(group_id: int, dtype: torch.dtype) -> "DecodeRequest":
        return DecodeRequest(
            microbatch_id=group_id,
            xs=torch.empty((0,), dtype=dtype, device=main_device),
            exec_task_ids=[],
            cancel_task_ids=[],
            offload_task_ids=[],
            reload_task_ids=[],
        )

    def is_empty(self) -> bool:
        return (
            self.xs.numel() == 0
            and len(self.exec_task_ids) == 0
            and len(self.cancel_task_ids) == 0
            and len(self.offload_task_ids) == 0
            and len(self.reload_task_ids) == 0
        )

    def append_exec(self, task_id: str, x: torch.Tensor) -> None:
        if x.numel() == 0:
            assert False
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

    def into_safetensors(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return {
            "xs": self.xs,
        }, {
            "group_id": self.microbatch_id,
            "exec_task_ids": self.exec_task_ids,
            "cancel_task_ids": self.cancel_task_ids,
            "suspend_task_ids": self.offload_task_ids,
            "resume_task_ids": self.reload_task_ids,
        }

    def into_decode_request(self) -> "DecodeRequest":
        return DecodeRequest(
            microbatch_id=self.microbatch_id,
            xs=self.xs,
            exec_task_ids=self.exec_task_ids,
            cancel_task_ids=self.cancel_task_ids,
            offload_task_ids=self.offload_task_ids,
            reload_task_ids=self.reload_task_ids,
        )


@dataclass
class PrefillRequest(DecodeRequest):
    task_ids: list[str]
    sampling_params: list[SamplingParams]

    def into_safetensors(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        tensors, metadata = super().into_safetensors()
        metadata["task_ids"] = self.task_ids
        metadata["sampling_params"] = self.sampling_params
        return tensors, metadata

    @staticmethod
    def from_decode_request(
        decode_request: DecodeRequest,
        task_ids: list[str],
        sampling_params: list[SamplingParams],
    ) -> "PrefillRequest":
        return PrefillRequest(
            microbatch_id=decode_request.microbatch_id,
            xs=decode_request.xs,
            exec_task_ids=decode_request.exec_task_ids,
            cancel_task_ids=decode_request.cancel_task_ids,
            offload_task_ids=decode_request.offload_task_ids,
            reload_task_ids=decode_request.reload_task_ids,
            task_ids=task_ids,
            sampling_params=sampling_params,
        )


@dataclass
class TraceRequest(LLMRequest):
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
