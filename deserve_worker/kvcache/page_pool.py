import threading
from queue import Queue
from typing import Optional

import torch

from deserve_worker.engine.event.base import EngineEvent, MoreSpaceEvent


def calc_pages_needed_num(total_len: int, page_size: int) -> int:
    return (total_len + page_size - 1) // page_size


class PagePool:
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        sender: Queue[EngineEvent],
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.num_avails = num_pages
        self.page_size = page_size
        self.sender = sender
        self.main_device = main_device
        self.main_dtype = main_dtype
        self.fetch_size = 64
        self.mutex = threading.Lock()

        self.pages_k = [
            torch.empty(
                (num_pages, page_size, 8, 128),
                device=main_device,
                dtype=main_dtype,
            )
            for _ in range(num_layers)
        ]
        self.pages_v = [
            torch.empty(
                (num_pages, page_size, 8, 128),
                device=main_device,
                dtype=main_dtype,
            )
            for _ in range(num_layers)
        ]
        self.page_bitmap = torch.ones(
            (num_pages,), device=main_device, dtype=torch.bool
        )
        self.page_buffer = torch.empty(0, device=main_device, dtype=torch.int32)

    def alloc(self, size: int) -> Optional[torch.Tensor]:
        with self.mutex:
            if size > self.num_avails:
                return None
            self.num_avails -= size
            if size > self.page_buffer.shape[0]:
                fetch_size = max(self.fetch_size, size - self.page_buffer.shape[0])
                block_avails = (
                    torch.nonzero(self.page_bitmap)[:fetch_size]
                    .to(torch.int32)
                    .flatten()
                )
                self.page_bitmap[block_avails] = False
                self.page_buffer = torch.cat([self.page_buffer, block_avails])
            result = self.page_buffer[:size]
            self.page_buffer = self.page_buffer[size:]
            return result

    def alloc_consecutive(self, size: int) -> Optional[torch.Tensor]:
        with self.mutex:
            if size > self.num_avails:
                return None
            output, invert_indices, counts = torch.unique_consecutive(
                self.page_bitmap, return_counts=True, return_inverse=True
            )
            avail_bitmap: torch.Tensor = (counts >= size) & output
            avail_indices = avail_bitmap.nonzero().flatten()
            if avail_indices.shape[0] == 0:
                return None
            else:
                self.num_avails -= size
                index: torch.Tensor = avail_indices[0]
                return (invert_indices == index).nonzero().flatten()  # type: ignore

    def recycle(self, blocks: torch.Tensor) -> None:
        with self.mutex:
            self.sender.put(MoreSpaceEvent())
            self.num_avails += blocks.shape[0]
            self.page_bitmap[blocks] = True

    def retrieve(
        self, indices: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        blocks_k = [page_k[indices] for page_k in self.pages_k]
        blocks_v = [page_v[indices] for page_v in self.pages_v]
        return blocks_k, blocks_v

    def get_pages_k(self, layer: int) -> torch.Tensor:
        return self.pages_k[layer]

    def get_pages_v(self, layer: int) -> torch.Tensor:
        return self.pages_v[layer]
