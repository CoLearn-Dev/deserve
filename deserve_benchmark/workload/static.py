import random

from src.workload.utils import Workload


class StaticWorkload(Workload):
    def __init__(self, size: int, length: int, variance: int) -> None:
        self._size = size
        self.length = length
        self.variance = variance

    def get(self, offset: int, length: int) -> list[str]:
        if offset >= self._size:
            return []
        result_length = self.length + random.randint(-self.variance, self.variance)
        return ["a" * 8 * (result_length - 1)] * min(length, self._size - offset)

    def size(self) -> int:
        return self._size
