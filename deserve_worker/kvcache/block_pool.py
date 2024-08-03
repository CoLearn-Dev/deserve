from typing import Optional
import torch


class BlockPool:
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
        self.fetch_size = 1024

        self.block_ks = torch.randn(
            num_blocks, block_size, 8, 128, device=main_device, dtype=main_dtype
        )
        self.block_vs = torch.randn(
            num_blocks, block_size, 8, 128, device=main_device, dtype=main_dtype
        )
        self.block_bitmap = torch.ones(
            (num_blocks,), device=main_device, dtype=torch.bool
        )
        self.block_buffer = torch.empty(0, device=main_device, dtype=torch.int32)

    def alloc(self, size: int) -> Optional[torch.Tensor]:
        if size > self.block_buffer.shape[0]:
            fetch_size = max(self.fetch_size, size - self.block_buffer.shape[0])
            block_avails = torch.nonzero(self.block_bitmap)[:fetch_size]
            self.block_bitmap[block_avails] = False
            self.block_buffer = torch.cat([self.block_buffer, block_avails])
        if size > self.block_buffer.shape[0]:
            return None
        result = self.block_buffer[:size]
        self.block_buffer = self.block_buffer[size:]
        return result

    def alloc_consecutive(self, size: int) -> Optional[torch.Tensor]:
        output, invert_indices, counts = torch.unique_consecutive(
            self.block_bitmap, return_counts=True, return_inverse=True
        )
        avail_bitmap: torch.Tensor = (counts >= size) & output
        avail_indices = avail_bitmap.nonzero().flatten()
        if avail_indices.shape[0] == 0:
            return None
        else:
            index = avail_indices[0]
            return (invert_indices == index).nonzero().flatten()

    def recycle(self, blocks: torch.Tensor) -> None:
        self.block_bitmap[blocks] = True
