import os

import torch


class ResourceCollector:
    def __init__(self) -> None:
        # for cpu
        self.cpu_count = os.cpu_count()

        # for gpu
        self.gpu_mem, _ = torch.cuda.mem_get_info()
