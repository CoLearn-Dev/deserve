import queue
from typing import Optional

import torch

from deserve_worker.layer_storage import LayerStorage
from deserve_worker.task import TaskData
from deserve_worker.trace import OpId

from .command import BatchForward, BatchResult, BatchUpdate, SingleTrace, TraceResult
from .kvcache.kvcache import KVCacheManager, main_device

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, enable_flash_attn: bool = False
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    if enable_flash_attn:
        freqs_cis = torch.stack([freqs.cos(), freqs.sin()])  # flash_attn
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0, False).to(main_device)
flash_global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0, True).to(main_device)


class LLMEngine:
    def __init__(
        self,
        max_total_bsz: int,
        sender: queue.Queue[BatchResult | BatchUpdate | TraceResult],
    ):
        self.max_total_bsz = max_total_bsz
        self.sender = sender
        self.receiver = queue.Queue[BatchForward | SingleTrace]()

    def run(self) -> None:
        q = self.receiver
        while True:
            commands: list[BatchForward | SingleTrace] = [q.get()]
            while True:
                try:
                    new = q.get(block=False)
                    commands.append(new)
                except queue.Empty:
                    break
            traces = [
                command for command in commands if isinstance(command, SingleTrace)
            ]
            forwards = [
                command for command in commands if isinstance(command, BatchForward)
            ]
            self.handle_trace(traces)
            self.handle_forward(forwards)

    def handle_forward(self, forwards: list[BatchForward]) -> None:
        prefill_tasks = [task for task in forwards if task.xs.shape[1] > 1]
        decode_tasks = [task for task in forwards if task.xs.shape[1] == 1]

        for task in prefill_tasks:
            h = self.step_forward(
                task.xs,
                task.layer_storage,
                task.task_datas,
                task.kvcache_manager,
                flash_global_freqs_cis,
                None,
            )
            self.post_forward(h, task)

        print(
            f"prefill_tasks: {len(prefill_tasks)}, decode_tasks: {sum(task.xs.shape[0] for task in decode_tasks)}"
        )

        decode_tasks.sort(key=lambda task: task.xs.shape[0], reverse=False)
        while len(decode_tasks) > 0:
            total_bsz = 0
            todo_tasks = []
            for i in reversed(range(len(decode_tasks))):
                cur_bsz = decode_tasks[i].xs.shape[0]
                if total_bsz + cur_bsz > self.max_total_bsz:
                    continue
                total_bsz += cur_bsz
                todo_tasks.append(decode_tasks.pop(i))
            new_task_datas = []
            for task in todo_tasks:
                new_task_datas.extend(task.task_datas)
            new_xs = torch.cat([task.xs for task in todo_tasks])
            # TODO: check if all tasks share same information
            new_task = BatchForward(
                xs=new_xs,
                layer_storage=todo_tasks[0].layer_storage,
                task_datas=new_task_datas,
                need_sample=todo_tasks[0].need_sample,
                kvcache_manager=todo_tasks[0].kvcache_manager,
            )
            h = self.step_forward(
                new_task.xs,
                new_task.layer_storage,
                new_task.task_datas,
                new_task.kvcache_manager,
                flash_global_freqs_cis,
                None,
            )
            self.post_forward(h, new_task)

    def handle_trace(self, tasks: list[SingleTrace]) -> None:
        print(f"trace_tasks: {len(tasks)}")
        for task in tasks:
            traces: dict[OpId, torch.Tensor] = {}
            h = self.step_forward(
                task.x,
                task.layer_storage,
                [task.task_data],
                task.kvcache_manager,
                global_freqs_cis,
                traces,
            )
            self.post_trace(h, traces, task)

    def step_forward(
        self,
        h: torch.Tensor,
        layer_storage: LayerStorage,
        task_datas: list[TaskData],
        kvcache_manager: KVCacheManager,
        global_freqs_cis: torch.Tensor,
        traces: Optional[dict[OpId, torch.Tensor]],
    ) -> torch.Tensor:
        # we need to check that all tasks share the same layer storage
        with torch.inference_mode():
            bsz_list = [1 for _ in range(len(task_datas))]
            start_pos_list = [task.start_pos for task in task_datas]
            kvcache_list = [task.kvcaches for task in task_datas]
            result = layer_storage.forward(
                h,
                bsz_list,
                start_pos_list,
                global_freqs_cis,
                kvcache_list,
                kvcache_manager,
                traces,
            )
            return result

    def post_forward(self, merged_h: torch.Tensor, tasks: BatchForward) -> None:
        if tasks.need_sample:
            layer_storage = tasks.layer_storage
            ongoing_tokens, ongoing_ids, all_tokens, all_ids, done_ids = (
                layer_storage.sample(merged_h, tasks.task_datas)
            )
            if len(ongoing_tokens) > 0:
                self.sender.put(BatchResult(torch.cat(ongoing_tokens), ongoing_ids))
            self.sender.put(BatchUpdate(all_tokens, all_ids, done_ids))
        else:
            seqlen = tasks.xs.shape[1]
            for task in tasks.task_datas:
                task.start_pos += seqlen
            self.sender.put(
                BatchResult(merged_h, [task.task_id for task in tasks.task_datas])
            )

    def post_trace(
        self, h: torch.Tensor, traces: dict[OpId, torch.Tensor], task: SingleTrace
    ) -> None:
        task_data = task.task_data
        if task.need_sample:
            layer_storage = task.layer_storage
            ongoing_tokens, ongoing_ids, all_tokens, all_ids, done_ids = (
                layer_storage.sample(h, [task_data])
            )
            if len(ongoing_tokens) > 0:
                # at most have one
                self.sender.put(
                    TraceResult(torch.cat(ongoing_tokens), ongoing_ids[0], traces)
                )
            self.sender.put(BatchUpdate(all_tokens, all_ids, done_ids))
        else:
            seqlen = task.x.shape[1]
            task_data.start_pos += seqlen
            self.sender.put(TraceResult(h, task_data.task_id, traces))

    def add_batch_forward(self, forwards: BatchForward) -> None:
        self.receiver.put(forwards)

    def add_trace(self, trace: SingleTrace) -> None:
        self.receiver.put(trace)
