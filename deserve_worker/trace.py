from dataclasses import dataclass


@dataclass
class LayerId:
    layer: str

    def with_component(self, component: str) -> "ComponentId":
        return ComponentId(self.layer, component)

    def __str__(self) -> str:
        return self.layer

    @staticmethod
    def from_str(s: str) -> "LayerId":
        return LayerId(s)


@dataclass
class ComponentId:
    layer: str
    component: str

    def with_op(self, op: str) -> "OpId":
        return OpId(self.layer, self.component, op)

    def __str__(self) -> str:
        return f"{self.layer}.{self.component}"

    @staticmethod
    def from_str(s: str) -> "ComponentId":
        layer, component = s.split(".")
        return ComponentId(layer, component)


@dataclass
class OpId:
    layer: str
    component: str
    op: str

    def __hash__(self) -> int:
        return hash((self.layer, self.component, self.op))

    def __str__(self) -> str:
        return f"{self.layer}.{self.component}.{self.op}"

    @staticmethod
    def from_str(s: str) -> "OpId":
        layer, component, op = s.split(".")
        return OpId(layer, component, op)
