from typing import Optional, cast

import torch

from deserve_worker.kvcache.block_pool import BlockPool
from deserve_worker.kvcache.kvcache import KVCache, KVCacheManager, main_device


def del_tensor(t: torch.Tensor) -> None:
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


class PackedKVCacheManager(KVCacheManager):
    def __init__(self, block_pool: BlockPool):
        self.block_pool = block_pool
        self.block_size = block_pool.block_size

    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += self.block_size
        return cur

    def alloc(self, bsz: int, seqlen: int) -> Optional[KVCache]:
        len_token = self.get_kv_cache_length(0, seqlen)
        len_block = len_token // self.block_size
        total_block = len_block * bsz
        blocks = self.block_pool.alloc(total_block)
        # the consecutive block table is in shape of [bsz, len_block], which corresponds to [bsz, len_block * block_size, 8, 128] in memory
        if blocks is None:
            return None
        else:
            return PackedKVCache(blocks.reshape(bsz, -1), self)

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PackedKVCache, kvcache)
        self.block_pool.recycle(kvcache.csct_block_table.flatten())
        kvcache.csct_block_table = torch.empty(
            (0, 0), device=main_device, dtype=torch.int32
        )

    def renew(self, kvcache: KVCache, bsz: int, seqlen: int, start_pos: int) -> bool:
        kvcache = cast(PackedKVCache, kvcache)
        if (
            start_pos + seqlen
            > kvcache.csct_block_table.shape[1] * self.block_pool.block_size
        ):
            len_token = self.get_kv_cache_length(
                kvcache.csct_block_table.shape[1] * self.block_size, start_pos + seqlen
            )
            len_block = len_token // self.block_size
            total_block = len_block * bsz
            blocks = self.block_pool.alloc(total_block)
            if blocks is None:
                return False
            else:
                # the original blocks are viewed as [bsz, old_len_block * block_size, 8, 128]
                # the new blocks are viewed as [bsz, len_block * block_size, 8, 128]
                # we need to copy the old blocks to the new blocks
                old_len_block = kvcache.csct_block_table.shape[1]
                old_blocks = kvcache.csct_block_table.flatten()
                old_block_ks = self.block_pool.block_ks[
                    old_blocks[0] : old_blocks[-1] + 1
                ].view(bsz, old_len_block * self.block_size, 8, 128)
                new_block_ks = self.block_pool.block_ks[
                    blocks[0] : blocks[-1] + 1
                ].view(bsz, len_block * self.block_size, 8, 128)
                new_block_ks[:, :start_pos, :, :] = old_block_ks[:, :start_pos, :, :]

                old_block_vs = self.block_pool.block_vs[
                    old_blocks[0] : old_blocks[-1] + 1
                ].view(bsz, old_len_block * self.block_size, 8, 128)
                new_block_vs = self.block_pool.block_vs[
                    blocks[0] : blocks[-1] + 1
                ].view(bsz, len_block * self.block_size, 8, 128)
                new_block_vs[:, :start_pos, :, :] = old_block_vs[:, :start_pos, :, :]

                self.block_pool.recycle(old_blocks)
                kvcache.csct_block_table = blocks.reshape(bsz, -1)

        return True


class PackedKVCache(KVCache):
    def __init__(
        self,
        csct_block_table: torch.Tensor,
        manager: PackedKVCacheManager,
    ):
        self.csct_block_table = csct_block_table  # consecutive block table
        self.manager = manager

    def renew(self, bsz: int, seqlen: int, start_pos: int) -> bool:
        return self.manager.renew(self, bsz, seqlen, start_pos)

    def clear(self) -> None:
        return self.manager.recycle(self)
