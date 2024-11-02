from abc import ABC, abstractmethod

from deserve_worker.request import StepRequest
from deserve_worker.task import TaskManager


class Stage(ABC):
    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager

    @abstractmethod
    def process(self, rest_pages: int, request: StepRequest) -> tuple[int, StepRequest]:
        pass
