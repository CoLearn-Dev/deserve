import queue
import time
from dataclasses import dataclass
from typing import Optional

import torch

from deserve_worker.task import LayerForward, ResultBack

from .kvcache import KVCacheBase
from .layer_storage import LayerStorage
from .model import ENABLE_FLASH_ATTN

EOS_TOKEN_ID = 128001  # for llama 3 only


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


global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0).to("cuda")


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


def trace(begin: float, msg: str) -> float:
    end = time.time()
    print(f"{msg}: {(end - begin)*1000:.3f}ms")
    return end


class ForwardEngine:
    def __init__(
        self, max_total_bsz: int, sendback_queue: queue.Queue[list[ResultBack]]
    ):
        self.max_total_bsz = max_total_bsz
        self.sendback_queue = sendback_queue
        self.handling_queue = queue.Queue[list[LayerForward]]()

    def run(self) -> None:
        q = self.handling_queue
        while True:
            forwards: list[LayerForward] = q.get()
            while True:
                try:
                    news = q.get(block=False)
                    forwards.extend(news)
                except queue.Empty:
                    break
            prefill_tasks = [task for task in forwards if task.h.shape[1] > 1]
            decode_tasks = [task for task in forwards if task.h.shape[1] == 1]
            # print(
            #     f"prefill_tasks: {len(prefill_tasks)}, decode_tasks: {len(decode_tasks)}"
            # )

            for task in prefill_tasks:
                h = self.forward([task])
                self.process(h, [task])

            for i in range(0, len(decode_tasks), self.max_total_bsz):
                to_decode = decode_tasks[
                    i : min(i + self.max_total_bsz, len(decode_tasks))
                ]
                h = self.forward(to_decode)
                self.process(h, to_decode)

    def add_layer_forward(self, forwards: list[LayerForward]) -> None:
        self.handling_queue.put(forwards)

    def forward(self, tasks: list[LayerForward]) -> torch.Tensor:
        # we need to check that all tasks share the same layer storage
        with torch.inference_mode():
            layer_storage = tasks[0].layer_storage
            h = torch.cat([t.h for t in tasks])
            bsz_list = []
            start_pos_list = []
            for task in tasks:
                bsz_list.append(1)
                start_pos_list.append(task.task_info.start_pos)
                for kvcache in task.task_info.kvcaches.values():
                    kvcache.renew(
                        task.h.shape[0], task.h.shape[1], task.task_info.start_pos
                    )
            return layer_storage.forward(
                h,
                bsz_list,
                start_pos_list,
                global_freqs_cis,
                [task.task_info.kvcaches for task in tasks],
            )

    def process(self, merged_h: torch.Tensor, tasks: list[LayerForward]) -> None:
        result: list[ResultBack] = []
        for ptr, task in enumerate(tasks):
            h = merged_h[ptr : ptr + 1]
            _, seqlen = h.shape[:2]
            task_info = task.task_info
            task_info.start_pos += seqlen
            if task.need_sample:
                task_info.round += 1
                sampling_params = task_info.sampling_params
                if task_info.start_pos >= sampling_params.max_total_len:
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
                result.append(ResultBack(next_token, task_info.task_id))
            else:
                result.append(ResultBack(h, task_info.task_id))
        self.sendback_queue.put(result)
