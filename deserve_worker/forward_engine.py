import queue

import torch

from .kvcache import KVCacheBase
from .layer_storage import LayerStorage
from .model import ENABLE_FLASH_ATTN


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


class LayerForward:
    def __init__(
        self,
        layer_storage: LayerStorage,
        h: torch.Tensor,
        seqlen: int,
        start_pos: int,
        kvcaches: dict[int, KVCacheBase],
        back: queue.Queue[torch.Tensor],
    ):
        self.layer_storage = layer_storage
        self.h = h
        self.seqlen = seqlen
        self.start_pos = start_pos
        self.kvcaches = kvcaches
        self.back = back


class ForwardEngine:
    def __init__(self, max_total_bsz: int):
        self.max_total_bsz = max_total_bsz

        self.queue = queue.Queue[LayerForward]()

    def run(self) -> None:
        q = self.queue
        while True:
            q_buffered: list[LayerForward] = [q.get()]
            while True:
                try:
                    tasks = q.get(block=False)
                    q_buffered.append(tasks)
                except queue.Empty:
                    break
            prefill_tasks = [task for task in q_buffered if task.seqlen > 1]
            decode_tasks = [task for task in q_buffered if task.seqlen == 1]
            print(
                f"prefill_tasks: {len(prefill_tasks)}, decode_tasks: {len(decode_tasks)}"
            )

            for task in prefill_tasks:
                h = self.forward([task])
                self.process(h, [task])

            for i in range(0, len(decode_tasks), self.max_total_bsz):
                to_decode = decode_tasks[
                    i : min(i + self.max_total_bsz, len(decode_tasks))
                ]
                h = self.forward(to_decode)
                self.process(h, to_decode)

    def add_layer_forward(self, task: LayerForward) -> None:
        self.queue.put(task)

    def forward(self, tasks: list[LayerForward]) -> torch.Tensor:
        # we need to check that all tasks share the same layer storage
        with torch.inference_mode():
            layer_storage = tasks[0].layer_storage
            h = torch.cat([t.h for t in tasks])
            bsz_list, start_pos_list = [1 for _ in tasks], [t.start_pos for t in tasks]
            return layer_storage.forward(
                h,
                bsz_list,
                start_pos_list,
                global_freqs_cis,
                [task.kvcaches for task in tasks],
            )

    def process(self, merged_h: torch.Tensor, tasks: list[LayerForward]) -> None:
        ptr = 0
        for task in tasks:
            task.back.put(merged_h[ptr : ptr + 1])
            ptr += 1
