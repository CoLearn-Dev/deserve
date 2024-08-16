from dataclasses import dataclass

from deserve_worker.engine.event.base import EngineEvent
from deserve_worker.execution.exec import BatchExec


@dataclass
class NewExecEvent(EngineEvent):
    exec: BatchExec
