from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from deserve_utils.hash import BatchMerkleTree
from deserve_utils.trace import OpId


class VerifyResult(Enum):
    VERIFIED = 0
    DIFFERENT_STRUCT = 1
    WRONG_MEMBERSHIP = 2
    SAME_ANSWER = 3


@dataclass
class TraceTensorInfo:
    opid: OpId
    tensor: torch.Tensor
    keep_last_dim: bool
    merkle_root_sig: bytes

    def to_verify_tensor_info(self, index: list[int]) -> "VerifyTensorInfo":
        if len(index) == len(self.tensor.shape):
            if self.keep_last_dim:
                index = index[:-1]
            merkle_tree = BatchMerkleTree(self.tensor, self.keep_last_dim)
            merkle_path = merkle_tree.generate_membership_proof(index)

            return VerifyTensorInfo(
                opid=self.opid,
                tensor=self.tensor[*index],
                merkle_root_sig=self.merkle_root_sig,
                merkle_path=merkle_path,
            )
        else:
            raise ValueError(
                f"Index {index} does not match tensor shape {self.tensor.shape}"
            )


@dataclass
class VerifyTensorInfo:
    opid: OpId
    tensor: torch.Tensor
    merkle_root_sig: bytes
    merkle_path: list[tuple[bytes, bytes]]

    def check_membership(self) -> bool:
        return BatchMerkleTree.verify_membership(
            self.tensor, self.merkle_path, self.merkle_root_sig
        )


def verify_struct_sig(
    output: OpId, inputs: list[OpId], struct_sig: bytes, pubkey: rsa.RSAPublicKey
) -> bool:
    ops = [output] + inputs
    ops_str = "".join(str(op) for op in ops)
    message = ops_str.encode()

    try:
        pubkey.verify(
            struct_sig,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except:
        return False


def generate_struct_sig(
    output: OpId, input_opids: list[OpId], privkey: rsa.RSAPrivateKey
) -> bytes:
    ops = [output] + input_opids
    ops_str = "".join(str(op) for op in ops)
    message = ops_str.encode()
    return privkey.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


@dataclass
class VerifyBaseOp:
    output: VerifyTensorInfo
    inputs: list[VerifyTensorInfo]
    struct_sig: bytes

    def verify_sigs(self, pubkey: rsa.RSAPublicKey) -> bool:
        return verify_struct_sig(
            self.output.opid,
            [input.opid for input in self.inputs],
            self.struct_sig,
            pubkey,
        )

    def verify_output(self) -> VerifyResult:
        raise NotImplementedError()


@dataclass
class TraceBaseOp:
    output: TraceTensorInfo
    input_opids: list[OpId]
    struct_sig: bytes

    def verify_sig(self, pubkey: rsa.RSAPublicKey) -> bool:
        return verify_struct_sig(
            self.output.opid,
            [input_opid for input_opid in self.input_opids],
            self.struct_sig,
            pubkey,
        )

    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyBaseOp]:
        raise NotImplementedError()


@dataclass
class TraceRegisterOp(TraceBaseOp):
    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyBaseOp]:
        return None  # currently no verification for register op, because normally it just comes from model weight


@dataclass
class VerifyMatmulOp(VerifyBaseOp):
    index: list[int]

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 2:
            return VerifyResult.DIFFERENT_STRUCT
        lhs, rhs = self.inputs
        output = self.output
        if (
            not lhs.check_membership()
            or not rhs.check_membership()
            or not output.check_membership()
        ):
            return VerifyResult.WRONG_MEMBERSHIP
        candidate = lhs.tensor * rhs.tensor
        if not torch.equal(candidate, output.tensor[self.index[-1]]):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TraceMatmulOp(TraceBaseOp):
    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyMatmulOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 2
        lhs, rhs = inputs  # the rhs should be transposed before being passed in
        output = self.output
        candidate = lhs.output.tensor @ rhs.output.tensor.mT
        if not torch.equal(candidate, output.tensor):
            index = (candidate != output.tensor).nonzero()[0].tolist()
            lhs_v = lhs.output.to_verify_tensor_info(index)
            rhs_v = rhs.output.to_verify_tensor_info(
                index[-len(rhs.output.tensor.shape) :]
            )
            output_v = self.output.to_verify_tensor_info(index)
            return VerifyMatmulOp(
                output=output_v,
                inputs=[lhs_v, rhs_v],
                struct_sig=self.struct_sig,
                index=index,
            )
        else:
            return None


@dataclass
class VerifySumOp(VerifyBaseOp):
    index: list[int]

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 1:
            return VerifyResult.DIFFERENT_STRUCT
        input = self.inputs[0]
        output = self.output
        if not input.check_membership() or not output.check_membership():
            return VerifyResult.WRONG_MEMBERSHIP
        candidate = input.tensor.sum(dim=-1)
        if not torch.equal(candidate, output.tensor[self.index[-1]]):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TraceSumOp(TraceBaseOp):
    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifySumOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 1
        input = inputs[0]
        output = self.output.tensor
        candidate = input.output.tensor.sum(dim=-1)
        if not torch.equal(candidate, output):
            index = (candidate != output).nonzero()[0].tolist()
            input_v = input.output.to_verify_tensor_info(index)
            output_v = self.output.to_verify_tensor_info(index)
            return VerifySumOp(
                output=output_v,
                inputs=[input_v],
                struct_sig=self.struct_sig,
                index=index,
            )
        else:
            return None


@dataclass
class VerifyBinaryOp(VerifyBaseOp):
    index: list[int]
    binary: str

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 2:
            return VerifyResult.DIFFERENT_STRUCT
        lhs, rhs = self.inputs
        output = self.output
        if (
            not lhs.check_membership()
            or not rhs.check_membership()
            or not output.check_membership()
        ):
            print(
                lhs.check_membership(),
                rhs.check_membership(),
                output.check_membership(),
            )
            return VerifyResult.WRONG_MEMBERSHIP
        if self.binary == "mul":
            candidate = lhs.tensor * rhs.tensor
        elif self.binary == "add":
            candidate = lhs.tensor + rhs.tensor
        elif self.binary == "complex_mul":
            candidate = torch.view_as_real(
                torch.view_as_complex(lhs.tensor.reshape(*lhs.tensor.shape[:-1], -1, 2))
                * rhs.tensor
            ).flatten(start_dim=-2)
        else:
            raise ValueError(f"Unknown binary operation: {self.binary}")
        if not torch.equal(candidate, output.tensor[self.index[-1]]):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TraceBinaryOp(TraceBaseOp):
    binary: str

    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyBinaryOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 2
        lhs, rhs = inputs
        output = self.output.tensor
        if self.binary == "mul":
            candidate = lhs.output.tensor * rhs.output.tensor
        elif self.binary == "add":
            candidate = lhs.output.tensor + rhs.output.tensor
        elif self.binary == "complex_mul":
            candidate = torch.view_as_real(
                torch.view_as_complex(
                    lhs.output.tensor.reshape(*lhs.output.tensor.shape[:-1], -1, 2)
                )
                * rhs.output.tensor
            ).flatten(start_dim=-2)
        else:
            raise ValueError(f"Unknown binary operation: {self.binary}")
        if not torch.equal(candidate, output):
            index = (candidate != output).nonzero()[0].tolist()
            print(
                lhs.output.merkle_root_sig.hex(),
                BatchMerkleTree(lhs.output.tensor, lhs.output.keep_last_dim)
                .get_root()
                .hex(),
            )
            lhs_v = lhs.output.to_verify_tensor_info(index)
            print(lhs_v.check_membership())
            if len(rhs.output.tensor.shape) > 1:
                rhs_shape = list(rhs.output.tensor.shape)
                rhs_index = index.copy()[-len(rhs_shape) :]
                lhs_shape = list(lhs.output.tensor.shape)[-len(rhs_shape) :]
                for i, (lhs_dim, rhs_dim) in enumerate(zip(lhs_shape, rhs_shape)):
                    if lhs_dim != rhs_dim:
                        rhs_index[i] = index[i] % rhs_dim
            else:
                rhs_index = [0]
            print(lhs.output.tensor.shape, rhs.output.tensor.shape)
            rhs_v = rhs.output.to_verify_tensor_info(rhs_index)
            print(rhs.output.tensor.shape, rhs_index)
            output_v = self.output.to_verify_tensor_info(index)
            return VerifyBinaryOp(
                output=output_v,
                inputs=[lhs_v, rhs_v],
                struct_sig=self.struct_sig,
                index=index,
                binary=self.binary,
            )
        else:
            return None


@dataclass
class VerifyFuncOp(VerifyBaseOp):
    index: list[int]
    func: str

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 1:
            return VerifyResult.DIFFERENT_STRUCT
        input = self.inputs[0]
        output = self.output
        if not input.check_membership() or not output.check_membership():
            return VerifyResult.WRONG_MEMBERSHIP
        last_index = self.index[-1]
        candidate = F.silu(input.tensor[last_index])
        if not torch.equal(candidate, output.tensor[last_index]):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TraceFuncOp(TraceBaseOp):
    func: str

    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyFuncOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 1
        input = inputs[0]
        output = self.output.tensor
        if self.func == "rsqrt":
            candidate = input.output.tensor.rsqrt()
        elif self.func == "silu":
            candidate = F.silu(input.output.tensor)
        elif self.func == "softmax":
            candidate = F.softmax(input.output.tensor, dim=-1)
        else:
            raise ValueError(f"Unknown function: {self.func}")
        if not torch.equal(candidate, output):
            index = (candidate != output).nonzero()[0].tolist()
            input_v = input.output.to_verify_tensor_info(index)
            output_v = self.output.to_verify_tensor_info(index)
            return VerifyFuncOp(
                output=output_v,
                inputs=[input_v],
                struct_sig=self.struct_sig,
                index=index,
                func=self.func,
            )
        else:
            return None


@dataclass
class VerifyPermuteOp(VerifyBaseOp):
    input_index: list[int]
    output_index: list[int]
    rule: list[int]

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 1:
            return VerifyResult.DIFFERENT_STRUCT
        input = self.inputs[0]
        output = self.output
        if not input.check_membership() or not output.check_membership():
            return VerifyResult.WRONG_MEMBERSHIP
        candidate = input.tensor[self.input_index[-1]]
        if not torch.equal(candidate, output.tensor[self.output_index[-1]]):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TracePermuteOp(TraceBaseOp):
    rule: list[int]

    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyPermuteOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 1
        input = inputs[0]
        output = self.output.tensor
        candidate = input.output.tensor.permute(*self.rule)
        if not torch.equal(candidate, output):
            output_index = (candidate != output).nonzero()[0].tolist()
            print(output_index)
            print(candidate[*output_index])
            print(output[*output_index])
            input_index = [
                output_index[self.rule.index(i)] for i in range(len(self.rule))
            ]
            input_v = input.output.to_verify_tensor_info(input_index)
            output_v = self.output.to_verify_tensor_info(output_index)
            return VerifyPermuteOp(
                output=output_v,
                inputs=[input_v],
                struct_sig=self.struct_sig,
                input_index=input_index,
                output_index=output_index,
                rule=self.rule,
            )
        else:
            return None


@dataclass
class VerifyRepeatOp(VerifyBaseOp):
    start_dim: int
    repeat_times: int
    input_index: list[int]
    output_index: list[int]

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 1:
            return VerifyResult.DIFFERENT_STRUCT
        input = self.inputs[0]
        output = self.output
        if not input.check_membership() or not output.check_membership():
            return VerifyResult.WRONG_MEMBERSHIP
        # FIXME: verify the input_index and output_index
        if not torch.equal(
            input.tensor[self.input_index[-1]], output.tensor[self.output_index[-1]]
        ):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TraceRepeatOp(TraceBaseOp):
    start_dim: int
    repeat_times: int

    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyRepeatOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 1
        input = inputs[0]
        input_shape = input.output.tensor.shape
        output = self.output.tensor
        candidate = (
            input.output.tensor.view(*input_shape[: self.start_dim], -1)
            .repeat_interleave(self.repeat_times, dim=-1)
            .view(
                *input_shape[: self.start_dim], -1, *input_shape[self.start_dim + 1 :]
            )
        )
        if not torch.equal(candidate, output):
            index = (candidate != output).nonzero()[0].tolist()
            return VerifyRepeatOp(
                output=self.output.to_verify_tensor_info(index),
                inputs=[input.output.to_verify_tensor_info(index)],
                struct_sig=self.struct_sig,
                input_index=index,
                output_index=index,
                repeat_times=self.repeat_times,
                start_dim=self.start_dim,
            )
        else:
            return None


@dataclass
class VerifyReshapeOp(VerifyBaseOp):
    old_shape: list[int]
    new_shape: list[int]
    old_index: list[int]
    new_index: list[int]

    def verify_output(self) -> VerifyResult:
        if len(self.inputs) != 1:
            return VerifyResult.DIFFERENT_STRUCT
        input = self.inputs[0]
        output = self.output
        if not input.check_membership() or not output.check_membership():
            return VerifyResult.WRONG_MEMBERSHIP
        candidate = input.tensor[self.old_index[-1]]
        if not torch.equal(candidate, output.tensor[self.new_index[-1]]):
            return VerifyResult.VERIFIED
        else:
            return VerifyResult.SAME_ANSWER


@dataclass
class TraceReshapeOp(TraceBaseOp):
    old_shape: list[int]
    new_shape: list[int]

    def try_verify_output(self, ctx: "TraceCtx") -> Optional[VerifyReshapeOp]:
        inputs = [ctx.traces[opid] for opid in self.input_opids]
        assert len(inputs) == 1
        input = inputs[0]
        output = self.output.tensor
        candidate = input.output.tensor.reshape(*self.new_shape)
        if not torch.equal(candidate, output):
            new_index = (candidate != output).nonzero()[0].tolist()
            absolute_index = 0
            for i, dim in enumerate(self.new_shape):
                absolute_index = absolute_index * dim + new_index[i]
            old_index = []
            for i, dim in reversed(list(enumerate(self.old_shape))):
                old_index.append(absolute_index % dim)
                absolute_index //= dim
            old_index.reverse()
            print(old_index, self.old_shape, new_index, self.new_shape)
            return VerifyReshapeOp(
                output=self.output.to_verify_tensor_info(new_index),
                inputs=[input.output.to_verify_tensor_info(old_index)],
                struct_sig=self.struct_sig,
                old_shape=self.old_shape,
                new_shape=self.new_shape,
                old_index=old_index,
                new_index=new_index,
            )
        else:
            return None


@dataclass
class TraceCtx:
    traces: dict[OpId, TraceBaseOp]
    private_key: rsa.RSAPrivateKey

    def register(self, opid: OpId, tensor: torch.Tensor, keep_last_dim: bool) -> OpId:
        self.traces[opid] = TraceRegisterOp(
            output=TraceTensorInfo(
                opid=opid,
                tensor=tensor,
                merkle_root_sig=BatchMerkleTree(tensor, keep_last_dim).get_root(),
                keep_last_dim=keep_last_dim,
            ),
            input_opids=[],
            struct_sig=generate_struct_sig(opid, [], self.private_key),
        )
        return opid

    def get(self, opid: OpId) -> torch.Tensor:
        return self.traces[opid].output.tensor

    def matmul(self, output_opid: OpId, lhs_opid: OpId, rhs_opid: OpId) -> OpId:
        lhs_op = self.traces[lhs_opid]
        rhs_op = self.traces[rhs_opid]
        output = lhs_op.output.tensor @ rhs_op.output.tensor.mT
        keep_last_dim = lhs_op.output.keep_last_dim or rhs_op.output.keep_last_dim

        self.traces[output_opid] = TraceMatmulOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, keep_last_dim).get_root(),
                keep_last_dim=keep_last_dim,
            ),
            input_opids=[lhs_opid, rhs_opid],
            struct_sig=generate_struct_sig(
                output_opid, [lhs_opid, rhs_opid], self.private_key
            ),
        )
        return output_opid

    def sum(self, output_opid: OpId, input_opid: OpId) -> OpId:
        input_op = self.traces[input_opid]
        output = input_op.output.tensor.sum(dim=-1)
        self.traces[output_opid] = TraceSumOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, False).get_root(),
                keep_last_dim=False,
            ),
            input_opids=[input_opid],
            struct_sig=generate_struct_sig(output_opid, [input_opid], self.private_key),
        )
        return output_opid

    def binary(
        self,
        output_opid: OpId,
        lhs_opid: OpId,
        rhs_opid: OpId,
        binary: str,
    ) -> OpId:
        lhs = self.traces[lhs_opid]
        rhs = self.traces[rhs_opid]

        if binary == "mul":
            output = lhs.output.tensor * rhs.output.tensor
        elif binary == "add":
            output = lhs.output.tensor + rhs.output.tensor
        elif binary == "complex_mul":
            output = torch.view_as_real(
                torch.view_as_complex(
                    lhs.output.tensor.reshape(*lhs.output.tensor.shape[:-1], -1, 2)
                )
                * rhs.output.tensor
            ).flatten(start_dim=-2)
        else:
            raise ValueError(f"Unknown binary operation: {binary}")
        keep_last_dim = lhs.output.keep_last_dim or rhs.output.keep_last_dim
        self.traces[output_opid] = TraceBinaryOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, keep_last_dim).get_root(),
                keep_last_dim=keep_last_dim,
            ),
            input_opids=[lhs_opid, rhs_opid],
            struct_sig=generate_struct_sig(
                output_opid, [lhs_opid, rhs_opid], self.private_key
            ),
            binary=binary,
        )
        return output_opid

    def unary(self, output_opid: OpId, input_opid: OpId, func: str) -> OpId:
        input_op = self.traces[input_opid]
        output: torch.Tensor
        if func == "rsqrt":
            output = input_op.output.tensor.rsqrt()
        elif func == "silu":
            output = F.silu(input_op.output.tensor)
        elif func == "softmax":
            output = F.softmax(input_op.output.tensor, dim=-1)
        else:
            raise ValueError(f"Unknown function: {func}")
        keep_last_dim = input_op.output.keep_last_dim
        self.traces[output_opid] = TraceFuncOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, keep_last_dim).get_root(),
                keep_last_dim=keep_last_dim,
            ),
            input_opids=[input_opid],
            struct_sig=generate_struct_sig(output_opid, [input_opid], self.private_key),
            func=func,
        )
        return output_opid

    def permute(self, output_opid: OpId, input_opid: OpId, rule: list[int]) -> OpId:
        input_op = self.traces[input_opid]
        output = input_op.output.tensor.permute(*rule).clone()
        keep_last_dim = input_op.output.keep_last_dim
        self.traces[output_opid] = TracePermuteOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, keep_last_dim).get_root(),
                keep_last_dim=keep_last_dim,
            ),
            input_opids=[input_opid],
            struct_sig=generate_struct_sig(output_opid, [input_opid], self.private_key),
            rule=rule,
        )
        return output_opid

    def repeat(
        self, output_opid: OpId, input_opid: OpId, start_dim: int, repeat_times: int
    ) -> OpId:
        input_op = self.traces[input_opid]
        input_shape = input_op.output.tensor.shape
        output = (
            input_op.output.tensor.view(*input_shape[:start_dim], -1)
            .repeat_interleave(repeat_times, dim=-1)
            .view(*input_shape[:start_dim], -1, *input_shape[start_dim + 1 :])
        )
        self.traces[output_opid] = TraceRepeatOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, True).get_root(),
                keep_last_dim=True,
            ),
            input_opids=[input_opid],
            struct_sig=generate_struct_sig(output_opid, [input_opid], self.private_key),
            start_dim=start_dim,
            repeat_times=repeat_times,
        )
        return output_opid

    def reshape(
        self, output_opid: OpId, input_opid: OpId, new_shape: list[int]
    ) -> OpId:
        input_op = self.traces[input_opid]
        output = input_op.output.tensor.reshape(*new_shape).clone()
        self.traces[output_opid] = TraceReshapeOp(
            output=TraceTensorInfo(
                opid=output_opid,
                tensor=output,
                merkle_root_sig=BatchMerkleTree(output, True).get_root(),
                keep_last_dim=True,
            ),
            input_opids=[input_opid],
            struct_sig=generate_struct_sig(output_opid, [input_opid], self.private_key),
            old_shape=list(input_op.output.tensor.shape),
            new_shape=new_shape,
        )
        return output_opid
