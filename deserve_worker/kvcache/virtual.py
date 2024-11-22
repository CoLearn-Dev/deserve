import torch

from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import GpuPagePool
from deserve_worker.kvcache.paged.page_table import PageTableAllocator, PageTableHandle
from deserve_worker.kvcache.pinned.pinned_memory import PinnedMemory


class CopiedPinnedMemory(PinnedMemory):
    def __init__(self, size: int, num_pages_swap: int, device: torch.device) -> None:
        super().__init__(size)
        self.bitmap = torch.ones((num_pages_swap,), device=device, dtype=torch.int8)
        self.count = num_pages_swap


class VirtualPagePool(GpuPagePool):
    def __init__(
        self,
        num_layers: int,
        num_pages_main: int,
        num_pages_swap: int,
        page_size: int,
        main_device: torch.device,
        main_dtype: torch.dtype,
    ):
        super().__init__(
            num_layers,
            num_pages_main + num_pages_swap * 2,
            page_size,
            main_device,
            main_dtype,
        )
        self.slices = [
            (num_pages_main, num_pages_main + num_pages_swap),
            (num_pages_main + num_pages_swap, num_pages_main + num_pages_swap * 2),
        ]

        # adjust page pool
        self.page_bitmap[self.slices[1][0] : self.slices[1][1]] = False
        self.swap_bitmap = torch.ones(
            (num_pages_swap,), device=self.main_device, dtype=torch.bool
        )
        self.swap_count = num_pages_swap
        self.num_avails = num_pages_main + num_pages_swap
        self.num_pages_main = num_pages_main
        self.num_pages_swap = num_pages_swap
        self.on = 0
        self.stream = torch.cuda.Stream(device=self.main_device)  # type: ignore
        self.stream2 = torch.cuda.Stream(device=self.main_device)  # type: ignore

    def switch(self) -> None:
        with self.mutex:
            self.page_bitmap[self.page_buffer] = True
            self.page_buffer = torch.empty(
                0, device=self.main_device, dtype=torch.int32
            )
            on = self.on
            off = 1 - self.on

            remained = (
                self.page_bitmap[self.slices[on][0] : self.slices[on][1]].sum().item()
            )
            assert isinstance(remained, int)
            self.num_avails -= remained
            self.page_bitmap[self.slices[off][0] : self.slices[off][1]] = (
                self.swap_bitmap
            )
            self.num_avails += self.swap_count
            self.swap_bitmap = self.page_bitmap[
                self.slices[on][0] : self.slices[on][1]
            ].clone()
            self.swap_count = remained
            self.page_bitmap[self.slices[on][0] : self.slices[on][1]] = False
            self.on = off

    def swap2(
        self, from_memory: CopiedPinnedMemory, to_memory: CopiedPinnedMemory
    ) -> None:
        off = 1 - self.on
        to_memory.bitmap = self.swap_bitmap
        to_memory.count = self.swap_count
        self.swap_bitmap = from_memory.bitmap
        self.swap_count = from_memory.count
        slice = self.slices[off]
        begin, middle, end = 0, self.num_layers // 2, self.num_layers
        interval = 1

        with torch.cuda.stream(self.stream):
            to_memory_view = to_memory.memory.view(
                self.num_layers, self.num_pages_swap, 2, self.page_size, 8, 128
            )
            from_memory_view = from_memory.memory.view(
                self.num_layers,
                self.num_pages_swap,
                2,
                self.page_size,
                8,
                128,
            )
            for i in range(begin, middle, interval):
                start_ptr = i
                end_ptr = min(middle, i + interval)
                to_memory_view[start_ptr:end_ptr, :, :, :, :, :].copy_(
                    self.pages[start_ptr:end_ptr, slice[0] : slice[1], :, :, :, :],
                    non_blocking=True,
                )
                self.pages[start_ptr:end_ptr, slice[0] : slice[1], :, :, :, :].copy_(
                    from_memory_view[start_ptr:end_ptr, :, :, :, :, :],
                    non_blocking=True,
                )

        with torch.cuda.stream(self.stream2):
            to_memory_view = to_memory.memory.view(
                self.num_layers, self.num_pages_swap, 2, self.page_size, 8, 128
            )
            from_memory_view = from_memory.memory.view(
                self.num_layers,
                self.num_pages_swap,
                2,
                self.page_size,
                8,
                128,
            )
            for i in range(middle, end, interval):
                start_ptr = i
                end_ptr = min(end, i + interval)
                to_memory_view[start_ptr:end_ptr, :, :, :, :, :].copy_(
                    self.pages[start_ptr:end_ptr, slice[0] : slice[1], :, :, :, :],
                    non_blocking=True,
                )
                self.pages[start_ptr:end_ptr, slice[0] : slice[1], :, :, :, :].copy_(
                    from_memory_view[start_ptr:end_ptr, :, :, :, :, :],
                    non_blocking=True,
                )

    def swap(
        self, from_memory: CopiedPinnedMemory, to_memory: CopiedPinnedMemory
    ) -> None:
        with torch.cuda.stream(self.stream):
            off = 1 - self.on
            to_memory.bitmap = self.swap_bitmap.clone()
            self.swap_bitmap = from_memory.bitmap
            slice = self.slices[off]
            to_memory.memory.view(
                self.num_layers, self.num_pages_swap, 2, self.page_size, 8, 128
            ).copy_(self.pages[:, slice[0] : slice[1], :, :, :, :], non_blocking=True)
            self.pages[:, slice[0] : slice[1], :, :, :, :].copy_(
                from_memory.memory.view(
                    self.num_layers, self.num_pages_swap, 2, self.page_size, 8, 128
                ),
                non_blocking=True,
            )

    def calc_swap_size(self) -> int:
        return self.num_layers * self.num_pages_swap * 2 * self.page_size * 8 * 128


class VirtualPagedKVCache(PagedKVCache[VirtualPagePool]):
    def __init__(
        self, page_table: PageTableHandle, isswap: torch.Tensor, pool: VirtualPagePool
    ) -> None:
        super().__init__(page_table, pool)
        self.isswap = isswap

    @staticmethod
    def empty(
        allocator: PageTableAllocator, pool: VirtualPagePool
    ) -> "VirtualPagedKVCache":
        return VirtualPagedKVCache(
            allocator.alloc(),
            torch.zeros(
                (allocator.max_context_len,), device=pool.main_device, dtype=torch.bool
            ),
            pool,
        )

    def extend(self, num_tokens: int) -> bool:
        if num_tokens > self.page_table.occupied * self.pool.page_size:
            new_num_pages = self.pool.calc_num_pages(num_tokens)
            delta = new_num_pages - self.page_table.occupied
            page_indices = self.pool.alloc(delta)
            if page_indices is None:
                return False
            self.isswap[self.page_table.occupied : new_num_pages] = (
                page_indices >= self.pool.num_pages_main
            )
            self.page_table.extend(page_indices)
        return True

    def adjust(self, offset: int) -> None:
        self.page_table.retrieve()[self.isswap[: self.page_table.occupied]] += offset

    def free(self) -> None:
        self.pool.free(self.page_table.retrieve())
        self.page_table.free()
        self.isswap = torch.empty(
            (0,),
            device=self.pool.main_device,
            dtype=torch.bool,
        )
