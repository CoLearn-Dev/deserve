import queue

import torch

from .command import BatchForward, BatchResult, BatchUpdate
from .kvcache.kvcache import main_device
from .model.llama import ENABLE_FLASH_ATTN

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
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
    if ENABLE_FLASH_ATTN:
        freqs_cis = torch.stack([freqs.cos(), freqs.sin()])  # flash_attn
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0).to(main_device)


class LLMEngine:
    def __init__(
        self, max_total_bsz: int, sendback_queue: queue.Queue[BatchResult | BatchUpdate]
    ):
        self.max_total_bsz = max_total_bsz
        self.sendback_queue = sendback_queue
        self.handling_queue = queue.Queue[BatchForward]()

    def run(self) -> None:
        q = self.handling_queue
        while True:
            forwards: list[BatchForward] = [q.get()]
            while True:
                try:
                    new = q.get(block=False)
                    forwards.append(new)
                except queue.Empty:
                    break
            prefill_tasks = [task for task in forwards if task.xs.shape[1] > 1]
            decode_tasks = [task for task in forwards if task.xs.shape[1] == 1]

            for task in prefill_tasks:
                h = self.forward(task)
                self.post_process(h, task)

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
                h = self.forward(new_task)
                self.post_process(h, new_task)

    def add_batch_forward(self, forwards: BatchForward) -> None:
        self.handling_queue.put(forwards)

    def forward(self, tasks: BatchForward) -> torch.Tensor:
        # we need to check that all tasks share the same layer storage
        with torch.inference_mode():
            layer_storage = tasks.layer_storage
            h = tasks.xs
            bsz_list = [1 for _ in range(len(tasks.task_datas))]
            start_pos_list = [task.start_pos for task in tasks.task_datas]
            kvcache_list = [task.kvcaches for task in tasks.task_datas]
            result = layer_storage.forward(
                h,
                bsz_list,
                start_pos_list,
                global_freqs_cis,
                kvcache_list,
                tasks.kvcache_manager,
            )
            return result

    def post_process(self, merged_h: torch.Tensor, tasks: BatchForward) -> None:
        if tasks.need_sample:
            layer_storage = tasks.layer_storage
            ongoing_tokens, ongoing_ids, all_tokens, all_ids, done_ids = (
                layer_storage.sample(merged_h, tasks.task_datas)
            )
            if len(ongoing_tokens) > 0:
                self.sendback_queue.put(
                    BatchResult(torch.cat(ongoing_tokens), ongoing_ids)
                )
            self.sendback_queue.put(BatchUpdate(all_tokens, all_ids, done_ids))
        else:
            seqlen = tasks.xs.shape[1]
            for task in tasks.task_datas:
                task.start_pos += seqlen
            self.sendback_queue.put(
                BatchResult(merged_h, [task.task_id for task in tasks.task_datas])
            )
