import queue
from typing import Optional

import torch

from .kvcache import KVCacheBase, main_device, main_dtype


class PagedMemory:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.cache_k_paged = torch.randn(
            num_blocks, block_size, 8, 128, device=main_device, dtype=main_dtype
        )
        self.cache_v_paged = torch.randn(
            num_blocks, block_size, 8, 128, device=main_device, dtype=main_dtype
        )
        self.avai_blocks = queue.Queue[int]()
        for i in range(1, num_blocks):
            self.avai_blocks.put(i)


global_paged_memory = PagedMemory(11600, 256, main_device, main_dtype)


class PagedKVCache(KVCacheBase):
    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += global_paged_memory.block_size
        return cur

    def __init__(self, x: torch.Tensor, start_pos: int, main_device: torch.device):
        self.main_device = main_device
        self.is_clear = False
        bsz, seqlen = x.shape[0], x.shape[1]
        length = (
            self.get_kv_cache_length(0, start_pos + seqlen)
            // global_paged_memory.block_size
        )
        self.block_table = torch.zeros(
            (
                bsz,
                length,
            ),
            device=self.main_device,
            dtype=torch.int32,
        )
        for i in range(length):
            for j in range(bsz):
                try:
                    blk = global_paged_memory.avai_blocks.get(block=False)
                except queue.Empty:
                    assert False, "No available block"
                self.block_table[j, i] = blk

    def renew(
        self,
        bsz: int,
        seqlen: int,
        start_pos: int,
    ) -> None:
        if (
            start_pos + seqlen
            > self.block_table.shape[1] * global_paged_memory.block_size
        ):
            # enlarge block table
            length = (
                self.get_kv_cache_length(
                    self.block_table.shape[1] * global_paged_memory.block_size,
                    start_pos + seqlen,
                )
                // global_paged_memory.block_size
            )
            block_table = torch.zeros(
                (
                    bsz,
                    length,
                ),
                device=self.main_device,
                dtype=torch.int32,
            )
            block_table[:, : self.block_table.shape[1]] = self.block_table[:, :]
            for i in range(self.block_table.shape[1], length):
                for j in range(bsz):
                    block_table[j, i] = global_paged_memory.avai_blocks.get()
            self.block_table = block_table

    def clear(self) -> None:
        if self.is_clear:
            assert False, "Already cleared"
        self.is_clear = True
        for row in self.block_table.tolist():
            for item in row:
                global_paged_memory.avai_blocks.put(item)

    def shape(self) -> torch.Size:
        return self.block_table.shape
