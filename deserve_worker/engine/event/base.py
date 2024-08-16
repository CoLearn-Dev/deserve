from dataclasses import dataclass


@dataclass
class EngineEvent:
    pass


@dataclass
class MoreSpaceEvent(EngineEvent):
    pass
