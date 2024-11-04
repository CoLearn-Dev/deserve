import threading
from dataclasses import dataclass
from typing import Optional

import torch


class ChunkPool:
    def __init__(
        self,
        num_layers: int,
        num_chunks: int,
        num_pages_per_chunk: int,
        page_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_chunks = num_chunks
        self.num_pages_per_chunk = num_pages_per_chunk
        self.page_size = page_size
        self.main_device = main_device
        self.main_dtype = main_dtype
        self.fetch_size = 64
        self.mutex = threading.Lock()
        self.per_token_shape = (2, page_size, 8, 128)

        self.chunks = torch.empty(
            (num_chunks, num_layers * num_pages_per_chunk, 2, page_size, 8, 128),
            device=main_device,
            dtype=main_dtype,
            pin_memory=True,
        )
        self.recycled_queue = [i for i in range(num_chunks)]

    def alloc(self) -> int:
        with self.mutex:
            if len(self.recycled_queue) == 0:
                raise RuntimeError("Failed to allocate chunks on CPU")
            id = self.recycled_queue.pop(0)
            return id

    def retrieve(self, id: int) -> torch.Tensor:
        return self.chunks[id]

    def free(self, id: int) -> None:
        with self.mutex:
            self.recycled_queue.append(id)


@dataclass
class ChunkHandle:
    id: int
    size: int
    pool: ChunkPool

    def free(self) -> None:
        self.pool.free(self.id)


class GpuChunkPool(ChunkPool):
    def __init__(
        self,
        num_layers: int,
        num_chunks: int,
        num_pages_per_chunk: int,
        page_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        super().__init__(
            num_layers,
            num_chunks,
            num_pages_per_chunk,
            page_size,
            main_device,
            main_dtype,
        )


class CpuChunkPool(ChunkPool):
    def __init__(
        self,
        num_layers: int,
        num_chunks: int,
        num_pages_per_chunk: int,
        page_size: int,
        main_dtype: torch.dtype,
    ):
        super().__init__(
            num_layers,
            num_chunks,
            num_pages_per_chunk,
            page_size,
            torch.device("cpu"),
            main_dtype,
        )
