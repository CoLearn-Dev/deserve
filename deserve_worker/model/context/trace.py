from dataclasses import dataclass
from typing import Optional, cast

import torch

from deserve_worker.kvcache.kvcache import PersistentKVCache, main_device
from deserve_worker.kvcache.packed_kvcache import PackedKVCache, PackedKVCacheManager
from deserve_worker.kvcache.page_pool import PagePool
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
        page_pool: PagePool,
        task_datas: list[TaskData],
        seqlens: list[int],
        global_freqs_cis: torch.Tensor,
        traces: dict[OpId, torch.Tensor],
    ) -> "TraceForwardCtx":
        ranges = []
        for task_data, seqlen in zip(task_datas, seqlens):
            if isinstance(task_data.kvcache, PersistentKVCache):
                print(f"Reload {task_data.task_id} to GPU")
                kvcache = task_data.kvcache.into_memory()
                assert isinstance(kvcache, PackedKVCache)
                task_data.kvcache = kvcache
            else:
                kvcache = cast(PackedKVCache, task_data.kvcache)
            kvcache.renew(task_data.start_pos + seqlen)
            kvcache = cast(PackedKVCache, task_data.kvcache)
            csct_block_table = kvcache.csct_block_table.flatten()
            begin = cast(int, csct_block_table[0].item())
            end = cast(int, csct_block_table[-1].item())
            ranges.append((begin, end))

        return TraceForwardCtx(
            page_pool=page_pool,
            offsets=torch.tensor(
                [task.start_pos for task in task_datas],
                device=main_device,
                dtype=torch.int32,
            ),
            bsz=len(task_datas),
            seqlens=torch.tensor(seqlens, dtype=torch.int32, device=main_device),
            layer_id=0,
            ranges=ranges,
            global_freqs_cis=global_freqs_cis,
            traces=traces,
        )
