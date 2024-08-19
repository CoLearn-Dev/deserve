from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

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

    @abstractmethod
    def into_persistent(self) -> "PersistentKVCache":
        pass


class KVCacheManager(ABC):
    @abstractmethod
    def new(self) -> KVCache:
        pass

    @abstractmethod
    def recycle(self, kvcache: KVCache) -> None:
        pass

    @abstractmethod
    def renew(self, kvcache: KVCache, total_len: int) -> bool:
        pass

    @abstractmethod
    def reinsert(self, kvcache: "PersistentKVCache") -> Optional[KVCache]:
        pass


def del_tensor(t: torch.Tensor) -> None:
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


@dataclass
class PersistentKVCache:
    storage_k: list[torch.Tensor]
    storage_v: list[torch.Tensor]
    manager: KVCacheManager

    def into_memory(self) -> Optional[KVCache]:
        return self.manager.reinsert(self)

    def clear(self) -> None:
        for t in self.storage_k:
            del_tensor(t)
        for t in self.storage_v:
            del_tensor(t)
        self.storage_k.clear()
        self.storage_v.clear()

    def get_num_pages(self) -> int:
        return self.storage_k[0].shape[0]
