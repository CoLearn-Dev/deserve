from abc import ABC
from dataclasses import dataclass
from typing import Optional

import torch

from deserve_worker.trace import OpId


@dataclass
class ForwardCtx:
    """
    Necessary data structures for forward calculation.
    """

    layer_id: int
    start_pos_list: torch.Tensor
    traces: Optional[dict[OpId, torch.Tensor]]
