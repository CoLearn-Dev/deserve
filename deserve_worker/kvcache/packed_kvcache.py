from dataclasses import dataclass
from typing import Optional, cast

import torch

from deserve_worker.kvcache.kvcache import KVCache, KVCacheManager, main_device
from deserve_worker.kvcache.page_pool import PagePool


def del_tensor(t: torch.Tensor) -> None:
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


class PackedKVCacheManager(KVCacheManager):
    def __init__(self, page_pool: PagePool):
        self.page_pool = page_pool
        self.block_size = page_pool.block_size

    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += self.block_size
        return cur

    def alloc(self, total_len: int) -> Optional[KVCache]:
        len_block = (total_len + self.block_size - 1) // self.page_pool.block_size
        blocks = self.page_pool.alloc(len_block)
        # the consecutive block table is in shape of [bsz, len_block], which corresponds to [bsz, len_block * block_size, 8, 128] in memory
        if blocks is None:
            return None
        else:
            return PackedKVCache(blocks.view(1, -1), self)

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PackedKVCache, kvcache)
        self.page_pool.recycle(kvcache.csct_block_table.flatten())
        kvcache.csct_block_table = torch.empty(
            (0, 0), device=main_device, dtype=torch.int32
        )

    def renew(self, kvcache: KVCache, total_len: int) -> bool:
        kvcache = cast(PackedKVCache, kvcache)
        if total_len > kvcache.csct_block_table.shape[1] * self.page_pool.block_size:
            len_block = (total_len + self.block_size - 1) // self.page_pool.block_size
            blocks = self.page_pool.alloc(len_block)
            if blocks is None:
                return False
            else:
                # the original blocks are viewed as [bsz, old_len_block * block_size, 8, 128]
                # the new blocks are viewed as [bsz, len_block * block_size, 8, 128]
                # we need to copy the old blocks to the new blocks
                # treat every layer separately
                old_len_block = kvcache.csct_block_table.shape[1]
                old_len_token = old_len_block * self.block_size
                old_blocks = kvcache.csct_block_table.flatten()
                for i in range(self.page_pool.num_layers):
                    old_block_ks = self.page_pool.pages_k[i][
                        old_blocks[0] : old_blocks[-1] + 1
                    ].view(1, old_len_token, 8, 128)
                    new_block_ks = self.page_pool.pages_k[i][
                        blocks[0] : blocks[-1] + 1
                    ].view(1, len_block * self.block_size, 8, 128)
                    new_block_ks[:, :old_len_token, :, :] = old_block_ks[
                        :, :old_len_token, :, :
                    ]
                    old_block_vs = self.page_pool.pages_v[i][
                        old_blocks[0] : old_blocks[-1] + 1
                    ].view(1, old_len_token, 8, 128)
                    new_block_vs = self.page_pool.pages_v[i][
                        blocks[0] : blocks[-1] + 1
                    ].view(1, len_block * self.block_size, 8, 128)
                    new_block_vs[:, :old_len_token, :, :] = old_block_vs[
                        :, :old_len_token, :, :
                    ]

                self.page_pool.recycle(old_blocks)
                kvcache.csct_block_table = blocks.view(1, -1)

        return True


class PackedKVCache(KVCache):
    def __init__(
        self,
        csct_block_table: torch.Tensor,
        manager: PackedKVCacheManager,
    ):
        self.csct_block_table = csct_block_table  # consecutive block table
        self.manager = manager

    def renew(self, total_len: int) -> bool:
        return self.manager.renew(self, total_len)

    def clear(self) -> None:
        return self.manager.recycle(self)
