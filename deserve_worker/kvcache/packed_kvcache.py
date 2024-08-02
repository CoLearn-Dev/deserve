from typing import Optional, cast

import torch

from deserve_worker.kvcache.kvcache import KVCache, KVCacheManager


def del_tensor(t: torch.Tensor) -> None:
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


class PackedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.main_device = main_device
        self.main_dtype = main_dtype

    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += self.block_size
        return cur

    def alloc(self, bsz: int, seqlen: int) -> Optional[KVCache]:
        len_token = self.get_kv_cache_length(0, seqlen)
        len_block = len_token // self.block_size
        if bsz * len_block <= self.num_blocks:
            self.num_blocks -= bsz * len_block
            cache_k = torch.zeros(
                (bsz, len_token, 8, 128),
                device=self.main_device,
                dtype=self.main_dtype,
            )
            cache_v = torch.zeros(
                (bsz, len_token, 8, 128),
                device=self.main_device,
                dtype=self.main_dtype,
            )
            return PackedKVCache(cache_k, cache_v, self)
        else:
            return None

    def recycle(self, kvcache: KVCache) -> None:
        kvcache = cast(PackedKVCache, kvcache)
        bsz, seqlen = kvcache.cache_k.shape[:2]
        self.num_blocks += bsz * seqlen

        del_tensor(kvcache.cache_k)
        del_tensor(kvcache.cache_v)
        kvcache.cache_k = torch.empty(
            (0, 0), device=self.main_device, dtype=self.main_dtype
        )
        kvcache.cache_v = torch.empty(
            (0, 0), device=self.main_device, dtype=self.main_dtype
        )
        torch.cuda.empty_cache()

    def renew(self, kvcache: KVCache, bsz: int, seqlen: int, start_pos: int) -> bool:
        kvcache = cast(PackedKVCache, kvcache)
        if start_pos + seqlen > kvcache.cache_k.shape[1]:
            len_token = self.get_kv_cache_length(
                kvcache.cache_k.shape[1], start_pos + seqlen
            )
            len_block = len_token // self.block_size
            if bsz * len_block <= self.num_blocks:
                self.num_blocks -= bsz * len_token
                cache_k = torch.zeros(
                    (bsz, len_token, 8, 128),
                    device=self.main_device,
                    dtype=self.main_dtype,
                )
                cache_v = torch.zeros(
                    (bsz, len_token, 8, 128),
                    device=self.main_device,
                    dtype=self.main_dtype,
                )
                cache_k[:, :start_pos, :, :], cache_v[:, :start_pos, :, :] = (
                    kvcache.cache_k[:, :start_pos, :, :],
                    kvcache.cache_v[:, :start_pos, :, :],
                )
                original_shape = bsz * kvcache.cache_k.shape[1]
                del_tensor(kvcache.cache_k)
                del_tensor(kvcache.cache_v)
                self.num_blocks += original_shape
                kvcache.cache_k = cache_k
                kvcache.cache_v = cache_v
                return True
        return False


class PackedKVCache(KVCache):
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        manager: PackedKVCacheManager,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.manager = manager

    def renew(self, bsz: int, seqlen: int, start_pos: int) -> bool:
        return self.manager.renew(self, bsz, seqlen, start_pos)

    def clear(self) -> None:
        return self.manager.recycle(self)
