from dataclasses import dataclass
from typing import Optional

import torch
from pydantic import BaseModel

main_dtype = torch.float16
main_device = torch.device("cuda")
torch.set_default_dtype(main_dtype)  # type: ignore


class PlanStep(BaseModel):
    worker_id: str
    worker_url: str
    layers: list[str]


class SamplingParams(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_seq_len: int = 4096
    max_new_tokens: int = 4096
    dump_probs_num: int = 0


@dataclass
class TaskData:
    task_id: str
    start_pos: int
    round: int
    seqlen: int
    initial_seqlen: int
    sampling_params: SamplingParams

    @staticmethod
    def empty(
        task_id: str, initial_seqlen: int, sampling_params: SamplingParams
    ) -> "TaskData":
        return TaskData(
            task_id=task_id,
            start_pos=0,
            round=0,
            seqlen=0,
            initial_seqlen=initial_seqlen,
            sampling_params=sampling_params,
        )

    def init(self, seqlen: int) -> None:
        self.seqlen = seqlen

    def is_prefill(self) -> bool:
        return self.start_pos >= self.initial_seqlen

    def finished_prefill(self) -> bool:
        return self.start_pos + self.seqlen >= self.initial_seqlen

    def step(self) -> None:
        # only when chunk prefill is enabled will this function changed
        assert self.seqlen > 0
        self.start_pos += self.seqlen
        if self.start_pos >= self.initial_seqlen:
            self.seqlen = 1
        else:
            self.seqlen = 0  # waiting for init
        self.round += 1


class TaskManager:
    def __init__(self, num_pages: int, page_size: int) -> None:
        self.task_datas: dict[str, TaskData] = {}
        self.page_size = page_size

    def add(self, task_data: TaskData) -> None:
        self.task_datas[task_data.task_id] = task_data

    def get(self, task_id: str) -> TaskData:
        return self.task_datas[task_id]

    def calc_occupied_space(self, task_ids: list[str]) -> int:
        task_datas = [self.get(task_id) for task_id in task_ids]
        return sum(self.calc_space(task_data.start_pos) for task_data in task_datas)

    def calc_initial_space(self, task_ids: list[str]) -> int:
        task_datas = [self.get(task_id) for task_id in task_ids]
        return sum(
            self.calc_space(task_data.initial_seqlen) for task_data in task_datas
        )

    def calc_extended_space(self, task_ids: list[str]) -> int:
        task_datas = [self.get(task_id) for task_id in task_ids]
        assert all(task_data.seqlen > 0 for task_data in task_datas)
        return sum(
            self.calc_space(task_data.start_pos + task_data.seqlen)
            - self.calc_space(task_data.start_pos)
            for task_data in task_datas
        )

    def calc_seqlens(self, task_ids: list[str]) -> int:
        task_datas = [self.get(task_id) for task_id in task_ids]
        return sum(task_data.seqlen for task_data in task_datas)

    def calc_space(self, size: int) -> int:
        return (size + self.page_size - 1) // self.page_size

    def calc_delta_space(self, task_id: str, seqlen: int) -> int:
        task_data = self.get(task_id)
        return self.calc_space(task_data.start_pos + seqlen) - self.calc_space(
            task_data.start_pos
        )

    def calc_rest_space(self, task_id: str) -> int:
        task_data = self.get(task_id)
        return self.calc_space(task_data.initial_seqlen) - self.calc_space(
            task_data.start_pos
        )
