from dataclasses import dataclass

import torch

from deserve_worker.kvcache.page_pool import PagePool


@dataclass
class ForwardCtx:
    page_pool: PagePool
    offsets: torch.Tensor
    bsz: int
    seqlens: torch.Tensor
    layer_id: int
