import threading
from dataclasses import dataclass

import torch


@dataclass
class PageTableHandle:
    offset: int
    occupied: int
    allocator: "PageTableAllocator"

    def retrieve(self) -> torch.Tensor:
        return self.allocator.tables[self.offset, : self.occupied]

    def retrieve_all(self) -> torch.Tensor:
        return self.allocator.tables[self.offset, :]

    def extend(self, fill: torch.Tensor) -> None:
        self.allocator.tables[
            self.offset, self.occupied : self.occupied + fill.shape[0]
        ] = fill
        self.occupied += fill.shape[0]

    def free(self) -> None:
        self.allocator.free(self)


class PageTableAllocator:
    def __init__(
        self,
        max_requests_num: int,
        max_context_len: int,
        page_size: int,
        main_device: torch.device,
    ) -> None:
        self.max_requests_num = max_requests_num
        self.max_context_len = max_context_len
        self.page_size = page_size
        self.main_device = main_device
        self.mutex = threading.Lock()

        self.tables = torch.empty(
            (max_requests_num, max_context_len // page_size),
            device=main_device,
            dtype=torch.int32,
        )
        self.slots = list(range(max_requests_num))

    def alloc(self) -> PageTableHandle:
        with self.mutex:
            if len(self.slots) == 0:
                raise ValueError("No available slots")
            num = self.slots.pop()
        # self.tables[num, :].fill_(0)
        return PageTableHandle(num, 0, self)

    def free(self, handle: PageTableHandle) -> None:
        with self.mutex:
            self.slots.append(handle.offset)
