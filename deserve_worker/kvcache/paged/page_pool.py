import threading
from typing import Optional

import torch


class PagePool:
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.num_avails = num_pages
        self.page_size = page_size
        self.main_device = main_device
        self.main_dtype = main_dtype
        self.fetch_size = 64
        self.mutex = threading.Lock()

        self.pages = torch.empty(
            (num_layers, num_pages, 2, page_size, 8, 128),
            device=main_device,
            dtype=main_dtype,
        )
        self.page_bitmap = torch.ones(
            (num_pages,), device=main_device, dtype=torch.bool
        )
        self.page_buffer = torch.empty(0, device=main_device, dtype=torch.int32)

    def calc_num_pages(self, size: int) -> int:
        return (size + self.page_size - 1) // self.page_size

    def alloc(self, size: int) -> Optional[torch.Tensor]:
        with self.mutex:
            if size > self.num_avails:
                raise ValueError(f"Not enough pages {size} > {self.num_avails}")
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

    def free(self, blocks: torch.Tensor) -> None:
        with self.mutex:
            self.num_avails += blocks.shape[0]
            self.page_bitmap[blocks] = True

    def retrieve(self, indices: torch.Tensor) -> torch.Tensor:
        return self.pages[:, indices, :, :, :, :]

    def get_layered_pages(self, layer: int) -> torch.Tensor:
        return self.pages[layer]


class GpuPagePool(PagePool):
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        super().__init__(num_layers, num_pages, page_size, main_device, main_dtype)


class CpuPagePool(PagePool):
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        main_dtype: torch.dtype,
    ):
        super().__init__(
            num_layers, num_pages, page_size, torch.device("cpu"), main_dtype
        )
