from dataclasses import dataclass

import torch

from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.task import TaskData
from deserve_worker.trace import OpId


@dataclass
class TraceForwardCtx(ForwardCtx):
    ranges: list[tuple[int, int]]
    global_freqs_cis: torch.Tensor
    traces: dict[OpId, torch.Tensor]

    @staticmethod
    def init_trace_forward_ctx(
        task_datas: list[TaskData],
        kvcaches: list[PagedKVCache[CpuPagePool]],
        global_freqs_cis: torch.Tensor,
        traces: dict[OpId, torch.Tensor],
    ) -> "TraceForwardCtx":
        raise NotImplementedError
