from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from deserve_utils.hash import BatchMerkleTree


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

    def transpose(self) -> "OpId":
        return OpId(layer=self.layer, component=self.component, op=f"{self.op}_t")

    @staticmethod
    def from_str(s: str) -> "OpId":
        layer, component, op = s.split(".")
        return OpId(layer=layer, component=component, op=op)


@dataclass
class BaseOp:
    op_id: OpId
    result: torch.Tensor

    def calc_size(self) -> int:
        return self.result.numel() * self.result.element_size()

    def verify(self) -> bool:
        return True


@dataclass
class MulOp(BaseOp):
    lhs: torch.Tensor
    lhs_merkle_root: bytes
    lhs_merkle_path: list[tuple[bytes, bytes]]
    rhs: torch.Tensor
    rhs_merkle_root: bytes
    rhs_merkle_path: list[tuple[bytes, bytes]]
    output_merkle_root: bytes
    output_merkle_path: list[tuple[bytes, bytes]]
    output_index: int

    def calc_size(self) -> int:
        return (
            super().calc_size()
            + self.lhs.numel() * self.lhs.element_size()
            + self.rhs.numel() * self.rhs.element_size()
            + len(self.lhs_merkle_root)
            + len(self.rhs_merkle_root)
            + len(self.output_merkle_root)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.lhs_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.rhs_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.output_merkle_path)
        )

    def verify(self) -> bool:
        if (
            not BatchMerkleTree.verify_membership(
                self.lhs, self.lhs_merkle_path, self.lhs_merkle_root
            )
            or not BatchMerkleTree.verify_membership(
                self.rhs, self.rhs_merkle_path, self.rhs_merkle_root
            )
            or not BatchMerkleTree.verify_membership(
                self.result, self.output_merkle_path, self.output_merkle_root
            )
        ):
            raise ValueError("Mul op verification failed")
        return torch.equal(self.result[self.output_index], self.lhs * self.rhs)

    def __str__(self) -> str:
        return f"Matmul({self.op_id}): {self.lhs.shape} * {self.rhs.shape} -> {self.result.shape}"


@dataclass
class SumOp(BaseOp):
    input: torch.Tensor
    input_merkle_root: bytes
    input_merkle_path: list[tuple[bytes, bytes]]
    output_merkle_root: bytes
    output_merkle_path: list[tuple[bytes, bytes]]

    def calc_size(self) -> int:
        return (
            super().calc_size()
            + self.input.numel() * self.input.element_size()
            + len(self.input_merkle_root)
            + len(self.output_merkle_root)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.input_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.output_merkle_path)
        )

    def verify(self) -> bool:
        if not BatchMerkleTree.verify_membership(
            self.input, self.input_merkle_path, self.input_merkle_root
        ) or not BatchMerkleTree.verify_membership(
            self.result, self.output_merkle_path, self.output_merkle_root
        ):
            raise ValueError("Sum op verification failed")
        return torch.equal(self.result, self.input.sum())

    def __str__(self) -> str:
        return f"Sum({self.op_id}): {self.input.shape} -> {self.result.shape}"


@dataclass
class LinearOp(BaseOp):
    input: torch.Tensor
    input_merkle_root: bytes
    input_merkle_path: list[tuple[bytes, bytes]]
    weight: torch.Tensor
    weight_merkle_root: bytes
    weight_merkle_path: list[tuple[bytes, bytes]]
    bias: torch.Tensor
    bias_merkle_root: bytes
    bias_merkle_path: list[tuple[bytes, bytes]]
    output_merkle_root: bytes
    output_merkle_path: list[tuple[bytes, bytes]]
    index: int

    def calc_size(self) -> int:
        return (
            super().calc_size()
            + self.input.numel() * self.input.element_size()
            + self.weight.numel() * self.weight.element_size()
            + self.bias.numel() * self.bias.element_size()
            + len(self.input_merkle_root)
            + len(self.weight_merkle_root)
            + len(self.bias_merkle_root)
            + len(self.output_merkle_root)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.input_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.weight_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.bias_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.output_merkle_path)
        )

    def verify(self) -> bool:
        if (
            not BatchMerkleTree.verify_membership(
                self.input, self.input_merkle_path, self.input_merkle_root
            )
            or not BatchMerkleTree.verify_membership(
                self.weight, self.weight_merkle_path, self.weight_merkle_root
            )
            or not BatchMerkleTree.verify_membership(
                self.bias, self.bias_merkle_path, self.bias_merkle_root
            )
            or not BatchMerkleTree.verify_membership(
                self.result, self.output_merkle_path, self.output_merkle_root
            )
        ):
            raise ValueError("Linear op verification failed")
        return torch.equal(self.result, self.input * self.weight + self.bias)

    def __str__(self) -> str:
        return f"Linear({self.op_id}): {self.input.shape} * {self.weight.shape} + {self.bias.shape} -> {self.result.shape}"


@dataclass
class FuncOp(BaseOp):
    input: torch.Tensor
    input_merkle_root: bytes
    input_merkle_path: list[tuple[bytes, bytes]]
    output_merkle_root: bytes
    output_merkle_path: list[tuple[bytes, bytes]]
    func: str

    def calc_size(self) -> int:
        return (
            super().calc_size()
            + self.input.numel() * self.input.element_size()
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.input_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.output_merkle_path)
            + len(self.input_merkle_root)
            + len(self.output_merkle_root)
        )

    def verify(self) -> bool:
        if not BatchMerkleTree.verify_membership(
            self.input, self.input_merkle_path, self.input_merkle_root
        ) or not BatchMerkleTree.verify_membership(
            self.result, self.output_merkle_path, self.output_merkle_root
        ):
            raise ValueError("Func op verification failed")
        if self.func == "rsqrt":
            return torch.equal(self.result, torch.rsqrt(self.input))
        elif self.func == "silu":
            return torch.equal(self.result, F.silu(self.input))
        elif self.func == "softmax":
            return torch.equal(self.result, F.softmax(self.input, dim=-1))
        else:
            raise ValueError(f"Unknown function: {self.func}")

    def __str__(self) -> str:
        return f"Func({self.op_id}): {self.func}({self.input.shape}) -> {self.result.shape}"


@dataclass
class TransposeOp(BaseOp):
    input: torch.Tensor
    input_merkle_root: bytes
    input_merkle_path: list[tuple[bytes, bytes]]
    input_index: list[int]
    output_merkle_root: bytes
    output_merkle_path: list[tuple[bytes, bytes]]
    output_index: list[int]
    rule: list[int]

    def calc_size(self) -> int:
        return (
            super().calc_size()
            + self.input.numel() * self.input.element_size()
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.input_merkle_path)
            + sum(len(lhs) + len(rhs) for lhs, rhs in self.output_merkle_path)
            + len(self.input_merkle_root)
            + len(self.output_merkle_root)
        )

    def verify(self) -> bool:
        if BatchMerkleTree.verify_membership(
            self.input, self.input_merkle_path, self.input_merkle_root
        ) and BatchMerkleTree.verify_membership(
            self.result, self.output_merkle_path, self.output_merkle_root
        ):
            raise ValueError("Transpose op verification failed")
        return torch.equal(self.result, self.input)  # todo: verify index transformation

    def __str__(self) -> str:
        return f"Transpose({self.op_id}): {self.input.shape} -> {self.result.shape}"
