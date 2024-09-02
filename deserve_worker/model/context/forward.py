from dataclasses import dataclass

import torch

from deserve_worker.kvcache.paged.page_pool import GpuPagePool


@dataclass
class ForwardCtx:
    bsz: int
    seqlens: torch.Tensor
    layer_id: int


@dataclass
class PagedForwardCtx(ForwardCtx):
    page_pool: GpuPagePool
    offsets: torch.Tensor
