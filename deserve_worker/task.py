from dataclasses import dataclass

from pydantic import BaseModel

from .kvcache.kvcache import KVCache


class PlanStep(BaseModel):
    worker_id: str
    worker_url: str
    layers: list[str]


class SamplingParams(BaseModel):
    temperature: float
    top_p: float
    max_seq_len: int = 4096
    max_new_tokens: int = 4096


class TaskInfo(BaseModel):
    task_id: str
    plan: list[PlanStep]
    round: int
    seqlen: int
    sampling_params: SamplingParams

    def into_task_data_placeholder(self) -> "TaskDataPlaceholder":
        return TaskDataPlaceholder(task_id=self.task_id, seqlen=self.seqlen)


@dataclass
class TaskData:
    task_id: str
    start_pos: int
    plan: list[PlanStep]
    round: int
    seqlen: int
    sampling_params: SamplingParams
    kvcache: KVCache
    """
    When flash attention is enabled, we use paged attention, otherwise the standard attention is adopted.
    """

    def into_task_info(self) -> TaskInfo:
        return TaskInfo(
            task_id=self.task_id,
            plan=self.plan,
            round=self.round,
            seqlen=self.seqlen,
            sampling_params=self.sampling_params,
        )


@dataclass
class TaskDataPlaceholder:
    task_id: str
    seqlen: int
