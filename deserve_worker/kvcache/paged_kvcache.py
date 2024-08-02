import queue
from typing import Optional, cast

import torch

from .kvcache import KVCache, KVCacheManager, main_device, main_dtype


class PagedKVCacheManager(KVCacheManager):
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
        self.block_bitmap = torch.zeros(
            (num_blocks,), device=main_device, dtype=torch.bool
        )
        self.block_buffer = torch.arange(
            0, num_blocks, device=main_device, dtype=torch.int32
        )

    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += self.block_size
        return cur

    def alloc_blocks(self, size: int) -> Optional[torch.Tensor]:
        if size > self.block_buffer.shape[0]:
            block_avails = torch.nonzero(self.block_bitmap)
            self.block_bitmap[block_avails] = False
            self.block_buffer = torch.cat([self.block_buffer, block_avails])
        if size > self.block_buffer.shape[0]:
            return None
        result = self.block_buffer[:size]
        self.block_buffer = self.block_buffer[size:]
        return result

    def alloc(self, bsz: int, seqlen: int) -> Optional["PagedKVCache"]:
        len_token = self.get_kv_cache_length(0, seqlen)
        len_block = len_token // self.block_size
        total_block = len_block * bsz
        blocks = self.alloc_blocks(total_block)
        if blocks is None:
            return None
        else:
            return PagedKVCache(blocks.reshape(bsz, -1), self)

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PagedKVCache, kvcache)
        self.block_bitmap[kvcache.block_table.flatten()] = True
        kvcache.block_table = torch.empty((0, 0), device=main_device, dtype=torch.int32)

    def renew(self, kvcache: KVCache, bsz: int, seqlen: int, start_pos: int) -> bool:
        kvcache = cast(PagedKVCache, kvcache)
        if start_pos + seqlen > kvcache.block_table.shape[1] * self.block_size:
            len_block = (
                self.get_kv_cache_length(
                    kvcache.block_table.shape[1] * self.block_size, start_pos + seqlen
                )
                // self.block_size
            )
            total_block = (len_block - kvcache.block_table.shape[1]) * bsz
            blocks = self.alloc_blocks(total_block)
            if blocks is None:
                return False
            else:
                new_block_table = torch.zeros(
                    (
                        bsz,
                        len_block,
                    ),
                    device=main_device,
                    dtype=torch.int32,
                )
                new_block_table[:, : kvcache.block_table.shape[1]] = (
                    kvcache.block_table[:, :]
                )
                new_block_table[:, kvcache.block_table.shape[1] :] = blocks.reshape(
                    bsz, -1
                )
                kvcache.block_table = new_block_table
        return True


class PagedKVCache(KVCache):
    def __init__(
        self,
        block_table: torch.Tensor,
        manager: PagedKVCacheManager,
    ):
        self.block_table = block_table
        self.manager = manager

    def renew(
        self,
        bsz: int,
        seqlen: int,
        start_pos: int,
    ) -> bool:
        return self.manager.renew(self, bsz, seqlen, start_pos)

    def clear(self) -> None:
        self.manager.recycle(self)

    def shape(self) -> torch.Size:
        return self.block_table.shape
