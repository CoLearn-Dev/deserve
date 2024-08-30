from dataclasses import dataclass

import torch

from deserve_worker.kvcache.paged.page_pool import GpuPagePool


@dataclass
class ForwardCtx:
    page_pool: GpuPagePool
    offsets: torch.Tensor
    bsz: int
    seqlens: torch.Tensor
    layer_id: int
