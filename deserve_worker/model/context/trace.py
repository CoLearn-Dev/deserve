from dataclasses import dataclass

import torch

from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import CpuPagePool
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.task import TaskData
from deserve_worker.trace import OpId


@dataclass
class TraceForwardCtx(ForwardCtx):
    global_freqs_cis: torch.Tensor
    traces: dict[OpId, torch.Tensor]
    output2input: dict[OpId, list[OpId]]
    last_op_id: OpId

    @staticmethod
    def init_trace_forward_ctx(
        task_datas: list[TaskData],
        global_freqs_cis: torch.Tensor,
        traces: dict[OpId, torch.Tensor],
        output2input: dict[OpId, list[OpId]],
    ) -> "TraceForwardCtx":
        return TraceForwardCtx(
            bsz=len(task_datas),
            seqlens=torch.tensor(
                [task.seqlen for task in task_datas], dtype=torch.int32
            ),
            layer_id=0,
            global_freqs_cis=global_freqs_cis,
            traces=traces,
            output2input=output2input,
            last_op_id=OpId(layer="tokens", component="main", op="input"),
        )
