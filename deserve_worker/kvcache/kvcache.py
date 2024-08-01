from abc import ABC, abstractmethod
from typing import Optional

import torch

KV_CACHE_BLOCK_SIZE = 256

main_dtype = torch.float16
main_device = torch.device("cuda")
torch.set_default_dtype(main_dtype)  # type: ignore


def del_tensor(t: torch.Tensor) -> None:
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


class KVCache(ABC):
    @abstractmethod
    def renew(self, bsz: int, seqlen: int, start_pos: int) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class KVCacheManager(ABC):
    @abstractmethod
    def alloc(self, bsz: int, seqlen: int) -> Optional[KVCache]:
        pass

    @abstractmethod
    def recycle(self, kvcache: KVCache) -> None:
        pass

    @abstractmethod
    def renew(self, kvcache: KVCache, bsz: int, seqlen: int, start_pos: int) -> bool:
        pass
