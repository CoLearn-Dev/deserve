from abc import ABC, abstractmethod

import torch

main_dtype = torch.float16
main_device = torch.device("cuda")
torch.set_default_dtype(main_dtype)  # type: ignore


def del_tensor(t: torch.Tensor) -> None:
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


class KVCacheBase(ABC):
    @abstractmethod
    def renew(self, x: torch.Tensor, start_pos: int) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


KV_CACHE_BLOCK_SIZE = 256


class KVCache(KVCacheBase):
    def get_kv_cache_length(self, cur: int, seqlen: int) -> int:
        while cur < seqlen:
            cur += KV_CACHE_BLOCK_SIZE
        return cur

    def __init__(
        self,
        x: torch.Tensor,
        start_pos: int,
        n_local_kv_heads: int,
        head_dim: int,
    ):
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim

        bsz, seqlen = x.shape[0], x.shape[1]
        length = self.get_kv_cache_length(0, start_pos + seqlen)
        self.cache_k = torch.zeros(
            (
                bsz,
                length,
                n_local_kv_heads,
                head_dim,
            ),
            device=main_device,
            dtype=main_dtype,
        )
        self.cache_v = torch.zeros(
            (
                bsz,
                length,
                n_local_kv_heads,
                head_dim,
            ),
            device=main_device,
            dtype=main_dtype,
        )
        self.main_device = main_device

    def renew(self, x: torch.Tensor, start_pos: int) -> None:
        bsz, seqlen = x.shape[0], x.shape[1]
        if start_pos + seqlen > self.cache_k.shape[1]:
            length = self.get_kv_cache_length(self.cache_k.shape[1], start_pos + seqlen)
            cache_k = torch.zeros(
                (
                    bsz,
                    length,
                    self.n_local_kv_heads,
                    self.head_dim,
                ),
                device=self.main_device,
            )
            cache_v = torch.zeros(
                (
                    bsz,
                    length,
                    self.n_local_kv_heads,
                    self.head_dim,
                ),
                device=self.main_device,
            )
            cache_k[:, :start_pos, :, :], cache_v[:, :start_pos, :, :] = (
                self.cache_k[:, :start_pos, :, :],
                self.cache_v[:, :start_pos, :, :],
            )
            del_tensor(self.cache_k)
            del_tensor(self.cache_v)
            self.cache_k = cache_k
            self.cache_v = cache_v

    def clear(self) -> None:
        del_tensor(self.cache_k)
        del_tensor(self.cache_v)
        torch.cuda.empty_cache()
