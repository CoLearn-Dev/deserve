from dataclasses import dataclass

import torch


@dataclass
class BatchResult:
    ongoing_xs: torch.Tensor
    ongoing_task_ids: list[str]
    all_xs: torch.Tensor
    all_task_ids: list[str]
    done_task_ids: list[str]
    needed_probs: dict[str, list[tuple[int, float]]]
