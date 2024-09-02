from dataclasses import dataclass

from pydantic import BaseModel


class LayerId(BaseModel):
    layer: str

    def with_component(self, component: str) -> "ComponentId":
        return ComponentId(layer=self.layer, component=component)

    def __str__(self) -> str:
        return self.layer

    @staticmethod
    def from_str(s: str) -> "LayerId":
        return LayerId(layer=s)


class ComponentId(BaseModel):
    layer: str
    component: str

    def with_op(self, op: str) -> "OpId":
        return OpId(layer=self.layer, component=self.component, op=op)

    def __str__(self) -> str:
        return f"{self.layer}.{self.component}"

    @staticmethod
    def from_str(s: str) -> "ComponentId":
        layer, component = s.split(".")
        return ComponentId(layer=layer, component=component)


class OpId(BaseModel):
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
        return OpId(layer=layer, component=component, op=op)
