from dataclasses import dataclass

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
    sampling_params: SamplingParams

    @staticmethod
    def empty(task_id: str, seqlen: int, sampling_params: SamplingParams) -> "TaskData":
        return TaskData(
            task_id=task_id,
            start_pos=0,
            round=0,
            seqlen=seqlen,
            sampling_params=sampling_params,
        )

    def step(self) -> None:
        # only when chunk prefill is enabled will this function changed
        self.start_pos += self.seqlen
        self.seqlen = 1
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

    def calc_extended_space(self, task_ids: list[str]) -> int:
        task_datas = [self.get(task_id) for task_id in task_ids]
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
