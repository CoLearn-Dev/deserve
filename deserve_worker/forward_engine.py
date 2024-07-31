import itertools
import queue
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .kvcache import KVCacheBase, main_device
from .layer_storage import LayerStorage
from .model import ENABLE_FLASH_ATTN
from .task import BatchForward, BatchResult, BatchUpdate, LayerForward, ResultBack

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


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class ForwardEngine:
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
                self.process(h, task)

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
                new_task = BatchForward(
                    xs=new_xs,
                    layer_storage=todo_tasks[0].layer_storage,
                    task_datas=new_task_datas,
                    need_sample=todo_tasks[0].need_sample,
                )
                h = self.forward(new_task)
                self.process(h, new_task)

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
            )
            return result

    def process(self, merged_h: torch.Tensor, tasks: BatchForward) -> None:
        if tasks.need_sample:
            ongoing_tokens = []
            ongoing_ids = []
            all_tokens = []
            all_ids = []
            done_ids = []
            for ptr, task_data in enumerate(tasks.task_datas):
                h = merged_h[ptr : ptr + 1]
                _, seqlen = h.shape[:2]
                task_data.start_pos += seqlen
                task_data.round += 1
                sampling_params = task_data.sampling_params
                if task_data.start_pos >= sampling_params.max_total_len:
                    next_token = torch.tensor([[EOS_TOKEN_ID]])
                elif sampling_params.temperature > 0:
                    probs = torch.softmax(
                        h[:, -1] / sampling_params.temperature, dim=-1
                    )
                    next_token = sample_top_p(probs, sampling_params.top_p)
                    next_token = next_token.reshape(1, -1)
                else:
                    next_token = torch.argmax(h[:, -1], dim=-1)
                    next_token = next_token.reshape(1, -1)
                next_token = next_token.to("cpu")
                all_ids.append(task_data.task_id)
                all_tokens.append(next_token)
                if next_token[0][0] in STOP_TOKEN_IDS:
                    done_ids.append(task_data.task_id)
                else:
                    ongoing_ids.append(task_data.task_id)
                    ongoing_tokens.append(next_token)
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
