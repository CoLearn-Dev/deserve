import torch


class PinnedMemory:
    def __init__(self, size: int) -> None:
        self.size = 0
        self.memory = torch.empty(
            (size,), device=torch.device("cpu"), dtype=torch.float16, pin_memory=True
        )
