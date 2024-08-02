from dataclasses import dataclass


@dataclass
class LayerId:
    layer: str

    def with_component(self, component: str) -> "ComponentId":
        return ComponentId(self.layer, component)


@dataclass
class ComponentId:
    layer: str
    component: str

    def with_op(self, op: str) -> "OpId":
        return OpId(self.layer, self.component, op)


@dataclass
class OpId:
    layer: str
    component: str
    op: str

    def __hash__(self) -> int:
        return hash((self.layer, self.component, self.op))
