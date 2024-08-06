import threading
from typing import Optional

import torch


class PagePool:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.main_device = main_device
        self.main_dtype = main_dtype
        self.fetch_size = 64
        self.mutex = threading.Lock()

        self.pages_k = [
            torch.empty(
                (num_blocks, block_size, 8, 128), device=main_device, dtype=main_dtype
            )
            for _ in range(num_layers)
        ]
        self.pages_v = [
            torch.empty(
                (num_blocks, block_size, 8, 128), device=main_device, dtype=main_dtype
            )
            for _ in range(num_layers)
        ]
        self.page_bitmap = torch.ones(
            (num_blocks,), device=main_device, dtype=torch.bool
        )
        self.page_buffer = torch.empty(0, device=main_device, dtype=torch.int32)

    def alloc(self, size: int) -> Optional[torch.Tensor]:
        with self.mutex:
            if size > self.page_buffer.shape[0]:
                fetch_size = max(self.fetch_size, size - self.page_buffer.shape[0])
                block_avails = torch.nonzero(self.page_bitmap)[:fetch_size]
                self.page_bitmap[block_avails] = False
                self.page_buffer = torch.cat([self.page_buffer, block_avails])
            if size > self.page_buffer.shape[0]:
                return None
            result = self.page_buffer[:size]
            self.page_buffer = self.page_buffer[size:]
            return result

    def alloc_consecutive(self, size: int) -> Optional[torch.Tensor]:
        output, invert_indices, counts = torch.unique_consecutive(
            self.page_bitmap, return_counts=True, return_inverse=True
        )
        avail_bitmap: torch.Tensor = (counts >= size) & output
        avail_indices = avail_bitmap.nonzero().flatten()
        if avail_indices.shape[0] == 0:
            return None
        else:
            index = avail_indices[0]
            return (invert_indices == index).nonzero().flatten()

    def recycle(self, blocks: torch.Tensor) -> None:
        self.page_bitmap[blocks] = True
