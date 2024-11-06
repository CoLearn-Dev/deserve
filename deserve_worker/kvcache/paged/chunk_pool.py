import bisect
import math
import threading
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChunkHandle:
    id: int
    level: int
    initial_size: int
    pool: "ChunkPool"

    def free(self) -> None:
        self.pool.free(self.id, self.level)

    def retrieve(self) -> torch.Tensor:
        begin = self.id * (2**self.level) * self.pool.num_layers
        end = (self.id + 1) * (2**self.level) * self.pool.num_layers
        return self.pool.chunks[begin:end]


class ChunkPool:
    def __init__(
        self,
        num_layers: int,
        height: int,
        page_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.height = height
        self.page_size = page_size
        self.main_device = main_device
        self.main_dtype = main_dtype
        self.mutex = threading.Lock()
        self.per_token_shape = (2, page_size, 8, 128)

        self.chunks = torch.empty(
            ((2 ** (height - 1)) * num_layers, 2, page_size, 8, 128),
            device=main_device,
            dtype=main_dtype,
            pin_memory=True,
        )
        self.available_ids: list[list[int]] = [[] for _ in range(height)]
        self.available_ids[height - 1] = [0]

    def split(self, begin: int, end: int) -> None:
        for level in range(end, begin, -1):
            parent_id = self.available_ids[level].pop(0)
            left_child = parent_id * 2
            right_child = parent_id * 2 + 1
            self.available_ids[level - 1].extend([left_child, right_child])

    def alloc(self, size: int) -> ChunkHandle:
        with self.mutex:
            target_level = math.ceil(math.log2(size))
            if len(self.available_ids[target_level]) == 0:
                for i in range(target_level + 1, self.height):
                    if len(self.available_ids[i]) != 0:
                        self.split(target_level, i)
                        break
            id = self.available_ids[target_level].pop(0)
            return ChunkHandle(id, target_level, size, self)

    def free(self, id: int, level: int) -> None:
        def insert_or_pop(target_ids: list[int], id: int, pos: int) -> Optional[int]:
            if pos != 0 and (target_ids[pos - 1] ^ 1) == id:
                target_ids.pop(pos - 1)
                return id // 2
            elif pos != len(target_ids) and (target_ids[pos] ^ 1) == id:
                target_ids.pop(pos)
                return id // 2
            else:
                target_ids.insert(pos, id)
                return None

        with self.mutex:
            for curr_level in range(level, self.height):
                target_ids = self.available_ids[curr_level]
                pos = bisect.bisect_left(target_ids, id)
                if ret := insert_or_pop(target_ids, id, pos):
                    id = ret
                else:
                    break


class CpuChunkPool(ChunkPool):
    def __init__(
        self,
        num_layers: int,
        height: int,
        page_size: int,
        main_dtype: torch.dtype,
    ):
        super().__init__(
            num_layers,
            height,
            page_size,
            torch.device("cpu"),
            main_dtype,
        )
