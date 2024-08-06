from abc import ABC, abstractmethod
from typing import Optional

import torch

KV_CACHE_BLOCK_SIZE = 256

main_dtype = torch.float16
main_device = torch.device("cuda")
torch.set_default_dtype(main_dtype)  # type: ignore


class KVCache(ABC):
    @abstractmethod
    def renew(self, total_len: int) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class KVCacheManager(ABC):
    @abstractmethod
    def alloc(self, total_len: int) -> Optional[KVCache]:
        pass

    @abstractmethod
    def recycle(self, kvcache: KVCache) -> None:
        pass

    @abstractmethod
    def renew(self, kvcache: KVCache, total_len: int) -> bool:
        pass
