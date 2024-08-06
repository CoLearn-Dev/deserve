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
    kvcaches: dict[int, KVCache]
    """
    When flash attention is enabled, we use paged attention, otherwise the standard attention is adopted.
    """
