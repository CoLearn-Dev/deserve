import argparse
import hashlib
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from torch import nn
from transformers import AutoTokenizer  # type: ignore

from deserve_utils.hash import BatchMerkleTree
from deserve_utils.serde import dumps, loads
from deserve_utils.trace import (
    BaseOp,
    ComponentId,
    FuncOp,
    LayerId,
    LinearOp,
    MulOp,
    OpId,
    SumOp,
    TransposeOp,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


llama_3_8b_args = ModelArgs(
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000.0,
)

llama_3_70b_args = ModelArgs(
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-05,
    rope_theta=500000.0,
    max_batch_size=32,
    max_seq_len=2048,
)

model2args = {
    "meta-llama/Meta-Llama-3-8B-Instruct": llama_3_8b_args,
    "meta-llama/Meta-Llama-3-70B-Instruct": llama_3_70b_args,
}


@dataclass
class ForwardCtx:
    intermediate_dtype: torch.dtype
    result_dtype: torch.dtype


@dataclass
class CheckCtx(ForwardCtx):
    traces: dict[OpId, torch.Tensor]
    diffs: dict[OpId, float]
    device: torch.device

    def check(self, op_id: OpId, x: torch.Tensor) -> torch.Tensor:
        y = self.traces[op_id].to(self.device)
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        self.diffs[op_id] = torch.max(torch.abs(x - y)).item()
        return y


@dataclass
class PickCtx(ForwardCtx):
    traces: dict[OpId, torch.Tensor]
    device: torch.device
    op: Optional[BaseOp]

    def pick_matmul(self, op_id: OpId, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
        x = lhs @ rhs
        y = self.traces[op_id].to(self.device)
        if x.numel() != y.numel():
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        y = y.view_as(x)
        if self.op is None and not torch.equal(x, y):
            index = (x != y).nonzero()[0].tolist()
            if len(rhs.shape) == 2:
                rhs = rhs.transpose(-1, -2)
                lhs_merkle_tree = BatchMerkleTree(lhs, keep_last_dim=True)
                rhs_merkle_tree = BatchMerkleTree(rhs, keep_last_dim=True)
                lhs_merkle_root = lhs_merkle_tree.get_root()
                rhs_merkle_root = rhs_merkle_tree.get_root()
                lhs_merkle_path = lhs_merkle_tree.generate_membership_proof(index[:-1])
                rhs_merkle_path = rhs_merkle_tree.generate_membership_proof([index[-1]])
                output_merkle_tree = BatchMerkleTree(y, keep_last_dim=True)
                output_merkle_root = output_merkle_tree.get_root()
                output_merkle_path = output_merkle_tree.generate_membership_proof(
                    index[:-1]
                )
                self.op = MulOp(
                    op_id=op_id,
                    lhs=lhs[*index[:-1]],
                    lhs_merkle_root=lhs_merkle_root,
                    lhs_merkle_path=lhs_merkle_path,
                    rhs=rhs[index[-1]],
                    rhs_merkle_root=rhs_merkle_root,
                    rhs_merkle_path=rhs_merkle_path,
                    result=y[*index[:-1]],
                    output_merkle_root=output_merkle_root,
                    output_merkle_path=output_merkle_path,
                    output_index=index[-1],
                )
            elif len(lhs.shape) == len(rhs.shape):
                shared_index = index[:-2]
                rhs = rhs.transpose(-1, -2)
                lhs_merkle_tree = BatchMerkleTree(lhs, keep_last_dim=True)
                rhs_merkle_tree = BatchMerkleTree(rhs, keep_last_dim=True)
                lhs_merkle_root = lhs_merkle_tree.get_root()
                rhs_merkle_root = rhs_merkle_tree.get_root()
                lhs_merkle_path = lhs_merkle_tree.generate_membership_proof(
                    shared_index + [index[-2]]
                )
                rhs_merkle_path = rhs_merkle_tree.generate_membership_proof(
                    shared_index + [index[-1]]
                )
                output_merkle_tree = BatchMerkleTree(y, keep_last_dim=True)
                output_merkle_root = output_merkle_tree.get_root()
                output_merkle_path = output_merkle_tree.generate_membership_proof(
                    index[:-1]
                )
                partial_lhs = lhs[*shared_index, index[-2]]
                partial_rhs = rhs[*shared_index, index[-1]]
                self.op = MulOp(
                    op_id=op_id,
                    lhs=partial_lhs,
                    lhs_merkle_root=lhs_merkle_root,
                    lhs_merkle_path=lhs_merkle_path,
                    rhs=partial_rhs,
                    rhs_merkle_root=rhs_merkle_root,
                    rhs_merkle_path=rhs_merkle_path,
                    result=y[*index[:-1]],
                    output_merkle_root=output_merkle_root,
                    output_merkle_path=output_merkle_path,
                    output_index=index[:-1],
                )

    def pick_sum(self, op_id: OpId, input: torch.Tensor) -> None:
        x = input.sum(dim=-1)
        y = self.traces[op_id].to(self.device)
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        if self.op is None and not torch.equal(x, y):
            index = list((x != y).nonzero()[0])
            partial_input = input[index]
            input_merkle_tree = BatchMerkleTree(input, keep_last_dim=True)
            input_merkle_root = input_merkle_tree.get_root()
            input_merkle_path = input_merkle_tree.generate_membership_proof(index)
            output_merkle_tree = BatchMerkleTree(y, keep_last_dim=True)
            output_merkle_root = output_merkle_tree.get_root()
            output_merkle_path = output_merkle_tree.generate_membership_proof(index)
            self.op = SumOp(
                op_id=op_id,
                input=partial_input,
                input_merkle_root=input_merkle_root,
                input_merkle_path=input_merkle_path,
                result=y[index],
                output_merkle_root=output_merkle_root,
                output_merkle_path=output_merkle_path,
            )

    def pick_linear(
        self, op_id: OpId, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> None:
        weight = weight.expand_as(input)
        bias = bias.expand_as(input)
        x = input * weight + bias
        y = self.traces[op_id].to(self.device)
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        if self.op is None and not torch.equal(x, y):
            index = list((x != y).nonzero()[0])
            input_merkle_tree = BatchMerkleTree(input, keep_last_dim=True)
            input_merkle_root = input_merkle_tree.get_root()
            input_merkle_path = input_merkle_tree.generate_membership_proof(index[:-1])
            weight_merkle_tree = BatchMerkleTree(weight, keep_last_dim=True)
            weight_merkle_root = weight_merkle_tree.get_root()
            weight_merkle_path = weight_merkle_tree.generate_membership_proof(
                index[:-1]
            )
            bias_merkle_tree = BatchMerkleTree(bias, keep_last_dim=True)
            bias_merkle_root = bias_merkle_tree.get_root()
            bias_merkle_path = bias_merkle_tree.generate_membership_proof(index[:-1])
            output_merkle_tree = BatchMerkleTree(y, keep_last_dim=True)
            output_merkle_root = output_merkle_tree.get_root()
            output_merkle_path = output_merkle_tree.generate_membership_proof(
                index[:-1]
            )
            self.op = LinearOp(
                op_id=op_id,
                input=input[*index[:-1]],
                weight=weight[*index[:-1]],
                bias=bias[*index[:-1]],
                result=y[*index[:-1]],
                index=index[-1],
                input_merkle_root=input_merkle_root,
                input_merkle_path=input_merkle_path,
                weight_merkle_root=weight_merkle_root,
                weight_merkle_path=weight_merkle_path,
                bias_merkle_root=bias_merkle_root,
                bias_merkle_path=bias_merkle_path,
                output_merkle_root=output_merkle_root,
                output_merkle_path=output_merkle_path,
            )

    def pick_func(self, op_id: OpId, input: torch.Tensor, func: str) -> None:
        if func == "rsqrt":
            x = input.rsqrt()
        elif func == "silu":
            x = F.silu(input)
        elif func == "softmax":
            x = F.softmax(input, dim=-1)
        else:
            raise ValueError(f"Unknown function: {func}")
        y = self.traces[op_id].to(self.device)
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        if self.op is None and not torch.equal(x, y):
            index = list((x != y).nonzero()[0])
            input_merkle_tree = BatchMerkleTree(input)
            input_merkle_root = input_merkle_tree.get_root()
            input_merkle_path = input_merkle_tree.generate_membership_proof(index[:-1])
            output_merkle_tree = BatchMerkleTree(y, keep_last_dim=True)
            output_merkle_root = output_merkle_tree.get_root()
            output_merkle_path = output_merkle_tree.generate_membership_proof(
                index[:-1]
            )
            self.op = FuncOp(
                op_id=op_id,
                input=input[*index[:-1]],
                input_merkle_root=input_merkle_root,
                input_merkle_path=input_merkle_path,
                result=y[*index[:-1]],
                output_merkle_root=output_merkle_root,
                output_merkle_path=output_merkle_path,
                func=func,
            )

    def pick_permute(
        self,
        output_op_id: OpId,
        input: torch.Tensor,
        rule: list[int],
    ) -> None:
        x = input.permute(*rule)
        y = self.traces[output_op_id].to(self.device)
        if x.numel() != y.numel():
            raise ValueError(f"Shape mismatch: {x.shape} != {y.shape}")
        y = y.view_as(x)
        if self.op is None and not torch.equal(x, y):
            index = list((x != y).nonzero()[0])
            new_index = list(range(len(rule)))
            for i, j in enumerate(rule):
                new_index[j] = index[i]
            input_merkle_tree = BatchMerkleTree(input, keep_last_dim=True)
            input_merkle_root = input_merkle_tree.get_root()
            input_merkle_path = input_merkle_tree.generate_membership_proof(
                new_index[:-1]
            )
            output_merkle_tree = BatchMerkleTree(y, keep_last_dim=True)
            output_merkle_root = output_merkle_tree.get_root()
            output_merkle_path = output_merkle_tree.generate_membership_proof(
                index[:-1]
            )
            self.op = TransposeOp(
                op_id=output_op_id,
                input=input[new_index],
                input_merkle_root=input_merkle_root,
                input_merkle_path=input_merkle_path,
                result=y[index],
                output_merkle_root=output_merkle_root,
                output_merkle_path=output_merkle_path,
                rule=rule,
                input_index=new_index,
                output_index=index,
            )


@dataclass
class VerifyCtx(ForwardCtx):
    op_id: OpId
    threshold: float
    traces: dict[OpId, torch.Tensor]
    device: torch.device

    def try_get_trace(self, op_id: OpId) -> Optional[torch.Tensor]:
        if op_id in self.traces:
            return self.traces[op_id].to(self.device)
        else:
            return None

    def get_trace(self, op_id: OpId) -> torch.Tensor:
        return self.traces[op_id].to(self.device)

    def verify(self, x: torch.Tensor) -> bool:
        y = self.traces[self.op_id].to(self.device)
        return torch.allclose(x, y, atol=self.threshold)


@dataclass
class TraceCtx(ForwardCtx):
    traces: dict[OpId, torch.Tensor]
    output2input: dict[OpId, list[OpId]]
    last_op_id: OpId
    device: torch.device
    signatures: dict[OpId, bytes]

    def trace(
        self, op_id: OpId, x: torch.Tensor, op_inputs: Optional[list[OpId]]
    ) -> None:
        if op_inputs is None:
            op_inputs = [self.last_op_id]
        self.traces[op_id] = x.clone()
        self.output2input[op_id] = op_inputs
        self.last_op_id = op_id
        self.signatures[op_id] = BatchMerkleTree(x, keep_last_dim=True).get_root()


class RMSNorm(nn.Module):
    def __init__(self, component_id: ComponentId, dim: int, eps: float = 1e-6):
        super().__init__()
        self.component_id = component_id
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def verify(self, x: Optional[torch.Tensor], ctx: VerifyCtx) -> bool:
        op = ctx.op_id.op
        if op == "output":
            assert x is not None
            return ctx.verify(self.norm(x.float()).type_as(x))
        else:
            output = ctx.get_trace(self.component_id.with_op("weighted_output"))
            return ctx.verify(output * self.weight)

    def pick(self, x: torch.Tensor, ctx: PickCtx) -> torch.Tensor:
        ctx.pick_linear(
            self.component_id.with_op("pow"),
            x,
            x,
            torch.zeros_like(x),
        )
        p = x.pow(2)
        cnt = x.shape[-1]
        ctx.pick_linear(
            self.component_id.with_op("mean"),
            p,
            torch.tensor(1.0 / cnt, dtype=x.dtype, device=ctx.device),
            torch.tensor(self.eps, dtype=x.dtype, device=ctx.device),
        )
        denom = p / torch.tensor(cnt, dtype=x.dtype, device=ctx.device) + torch.tensor(
            self.eps, dtype=x.dtype, device=ctx.device
        )
        ctx.pick_func(self.component_id.with_op("rsqrt"), denom, "rsqrt")
        rsqrt = denom.rsqrt()
        ctx.pick_linear(
            self.component_id.with_op("output"),
            x,
            rsqrt,
            torch.zeros_like(x.to(ctx.intermediate_dtype)),
        )
        output = (x * rsqrt).type_as(x)
        ctx.pick_linear(
            self.component_id.with_op("weighted_output"),
            output,
            self.weight,
            torch.zeros_like(output),
        )
        result = output * self.weight
        return result

    def check(self, x: torch.Tensor, ctx: CheckCtx) -> torch.Tensor:
        output = self.norm(x.float()).type_as(x)
        output = ctx.check(self.component_id.with_op("output"), output)
        result = output * self.weight
        result = ctx.check(self.component_id.with_op("weighted_output"), result)
        return result

    def forward(self, x: torch.Tensor, ctx: TraceCtx) -> torch.Tensor:
        p = x.to(ctx.intermediate_dtype).pow(2)
        ctx.trace(self.component_id.with_op("pow"), p, None)
        cnt = x.shape[-1]
        denom = p / torch.tensor(cnt, dtype=x.dtype, device=ctx.device) + torch.tensor(
            self.eps, dtype=x.dtype, device=ctx.device
        )
        ctx.trace(self.component_id.with_op("mean"), denom, None)
        denom = denom.rsqrt()
        ctx.trace(self.component_id.with_op("rsqrt"), denom, None)
        output = x * denom
        ctx.trace(self.component_id.with_op("output"), output, None)
        result = (
            output.to(ctx.intermediate_dtype) * self.weight.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(
            self.component_id.with_op("weighted_output"),
            result,
            [self.component_id.with_op("output")],
        )
        return result


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    print("xq", xq.shape)
    print("xk", xk.shape)
    print("freqs_cis", freqs_cis.shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    print("xq_", xq_.shape)
    print("xk_", xk_.shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    print("freqs_cis_after", freqs_cis.shape)
    print("xq_real", torch.view_as_real(xq_).flatten(-2).shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    print("xq_out", xq_out.shape)
    print("xk_out", xk_out.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, component_id: ComponentId, args: ModelArgs):
        super().__init__()
        self.component_id = component_id
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def verify(
        self,
        x: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: VerifyCtx,
    ) -> bool:
        op = ctx.op_id.op
        if op == "xq":
            assert x is not None
            bsz, seqlen, _ = x.shape
            xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            return ctx.verify(xq)
        elif op == "xk":
            assert x is not None
            bsz, seqlen, _ = x.shape
            xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            return ctx.verify(xk)
        elif op == "xv":
            assert x is not None
            bsz, seqlen, _ = x.shape
            xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            return ctx.verify(xv)
        elif op == "xq_rotary" or op == "xk_rotary":
            xq = ctx.get_trace(self.component_id.with_op("xq"))
            xk = ctx.get_trace(self.component_id.with_op("xk"))
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            if op == "xq_rotary":
                return ctx.verify(xq)
            else:
                return ctx.verify(xk)
        elif op == "scores":
            xq = ctx.get_trace(self.component_id.with_op("xq_rotary"))
            keys = ctx.get_trace(self.component_id.with_op("xk_rotary"))
            keys = repeat_kv(
                keys, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(
                1, 2
            )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            return ctx.verify(scores)
        elif op == "output":
            scores = ctx.get_trace(self.component_id.with_op("scores"))
            values = ctx.get_trace(self.component_id.with_op("xv"))
            bsz, seqlen = values.shape[:2]
            values = repeat_kv(
                values, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            values = values.transpose(
                1, 2
            )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            output = torch.matmul(
                scores, values
            )  # (bs, n_local_heads, seqlen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return ctx.verify(output)
        elif op == "weighted_output":
            output = ctx.get_trace(self.component_id.with_op("output"))
            return ctx.verify(self.wo(output))
        assert False

    def check(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: CheckCtx,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = ctx.check(self.component_id.with_op("xq"), xq)

        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xk = ctx.check(self.component_id.with_op("xk"), xk)

        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = ctx.check(self.component_id.with_op("xv"), xv)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = ctx.check(self.component_id.with_op("xq_rotary"), xq)
        xk = ctx.check(self.component_id.with_op("xk_rotary"), xk)

        keys = xk.clone()
        values = xv.clone()

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # check scores
        scores = ctx.check(self.component_id.with_op("scores"), scores)

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = ctx.check(self.component_id.with_op("output"), output)

        result = self.wo(output)
        result = ctx.check(self.component_id.with_op("weighted_output"), result)
        return result  # type: ignore

    def pick(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: PickCtx,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        ctx.pick_matmul(self.component_id.with_op("xq"), x, self.wq.weight.T)
        xq = (x @ self.wq.weight.T).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        ctx.pick_matmul(self.component_id.with_op("xk"), x, self.wk.weight.T)
        xk = (x @ self.wk.weight.T).view(
            bsz, seqlen, self.n_local_kv_heads, self.head_dim
        )
        ctx.pick_matmul(self.component_id.with_op("xv"), x, self.wv.weight.T)
        xv = (x @ self.wv.weight.T).view(
            bsz, seqlen, self.n_local_kv_heads, self.head_dim
        )

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = xk
        values = xv
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        ctx.pick_permute(self.component_id.with_op("xq_transpose"), xq, [0, 2, 1, 3])
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        ctx.pick_permute(
            self.component_id.with_op("keys_transpose"), keys, [0, 2, 3, 1]
        )
        keys = keys.permute(
            0, 2, 3, 1
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        ctx.pick_permute(
            self.component_id.with_op("values_transpose"), values, [0, 2, 1, 3]
        )
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        ctx.pick_matmul(self.component_id.with_op("scores_matmul"), xq, keys)
        scores = torch.matmul(xq, keys)
        ctx.pick_linear(
            self.component_id.with_op("scores_div"),
            scores,
            torch.tensor([1.0 / math.sqrt(self.head_dim)], dtype=scores.dtype),
            torch.zeros_like(scores),
        )
        scores = scores * torch.tensor(
            [1.0 / math.sqrt(self.head_dim)], dtype=scores.dtype
        )
        if mask is not None:
            ctx.pick_linear(
                self.component_id.with_op("scores_mask"),
                scores,
                torch.ones_like(scores),
                mask,
            )
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        ctx.pick_func(self.component_id.with_op("scores_softmax"), scores, "softmax")
        scores = F.softmax(scores, dim=-1)

        ctx.pick_matmul(self.component_id.with_op("output_matmul"), scores, values)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        ctx.pick_permute(
            self.component_id.with_op("output_transpose"), output, [0, 2, 1, 3]
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        ctx.pick_matmul(
            self.component_id.with_op("weighted_output"),
            output,
            self.wo.weight.T,
        )
        result = output @ self.wo.weight.T
        return result  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: TraceCtx,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        op_input = ctx.last_op_id
        xq = (
            (x.to(ctx.intermediate_dtype) @ self.wq.weight.T.to(ctx.intermediate_dtype))
            .view(bsz, seqlen, self.n_local_heads, self.head_dim)
            .to(ctx.result_dtype)
        )
        ctx.trace(self.component_id.with_op("xq"), xq, [op_input])

        xk = (
            (x.to(ctx.intermediate_dtype) @ self.wk.weight.T.to(ctx.intermediate_dtype))
            .view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            .to(ctx.result_dtype)
        )
        ctx.trace(self.component_id.with_op("xk"), xk, [op_input])

        xv = (
            (x.to(ctx.intermediate_dtype) @ self.wv.weight.T.to(ctx.intermediate_dtype))
            .view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            .to(ctx.result_dtype)
        )
        ctx.trace(self.component_id.with_op("xv"), xv, [op_input])

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        ctx.trace(
            self.component_id.with_op("xq_rotary"),
            xq,
            [self.component_id.with_op("xq"), self.component_id.with_op("xk")],
        )
        ctx.trace(
            self.component_id.with_op("xk_rotary"),
            xk,
            [self.component_id.with_op("xq"), self.component_id.with_op("xk")],
        )

        keys = xk.clone()
        values = xv.clone()

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        ctx.trace(
            self.component_id.with_op("xq_transpose"),
            xq,
            [self.component_id.with_op("xq_rotary")],
        )
        keys = keys.permute(0, 2, 3, 1)
        ctx.trace(
            self.component_id.with_op("keys_transpose"),
            keys,
            [self.component_id.with_op("xk_rotary")],
        )
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        ctx.trace(
            self.component_id.with_op("values_transpose"),
            values,
            [self.component_id.with_op("xv")],
        )
        scores = torch.matmul(
            xq.to(ctx.intermediate_dtype),
            keys.to(ctx.intermediate_dtype),
        ).to(ctx.result_dtype)
        ctx.trace(
            self.component_id.with_op("scores_matmul"),
            scores,
            [
                self.component_id.with_op("xq_transpose"),
                self.component_id.with_op("keys_transpose"),
            ],
        )
        scores = scores * torch.tensor(
            [1.0 / math.sqrt(self.head_dim)], dtype=scores.dtype
        )
        ctx.trace(
            self.component_id.with_op("scores_div"),
            scores,
            [self.component_id.with_op("scores_matmul")],
        )
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            ctx.trace(
                self.component_id.with_op("scores_mask"),
                scores,
                [self.component_id.with_op("scores_matmul")],
            )
        scores = F.softmax(scores.to(ctx.intermediate_dtype), dim=-1).to(
            ctx.result_dtype
        )
        ctx.trace(
            self.component_id.with_op("scores_softmax"),
            scores,
            [self.component_id.with_op("scores_div")],
        )

        output = torch.matmul(
            scores.to(ctx.intermediate_dtype), values.to(ctx.intermediate_dtype)
        ).to(
            ctx.result_dtype
        )  # (bs, n_local_heads, seqlen, head_dim)
        ctx.trace(
            self.component_id.with_op("output_matmul"),
            output,
            [self.component_id.with_op("scores"), self.component_id.with_op("xv")],
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        ctx.trace(
            self.component_id.with_op("output_transpose"),
            output,
            [self.component_id.with_op("output_matmul")],
        )

        result = (
            output.to(ctx.intermediate_dtype)
            @ self.wo.weight.T.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(
            self.component_id.with_op("weighted_output"),
            result,
            [self.component_id.with_op("output")],
        )
        return result  # type: ignore


class FeedForward(nn.Module):
    def __init__(
        self,
        component_id: ComponentId,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        self.component_id = component_id
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = torch.nn.utils.skip_init(  # type: ignore
            nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )

    def verify(self, x: Optional[torch.Tensor], ctx: VerifyCtx) -> bool:
        op = ctx.op_id.op
        if op == "w1":
            assert x is not None
            return ctx.verify(F.silu(self.w1(x)))
        elif op == "w3":
            assert x is not None
            return ctx.verify(self.w3(x))
        elif op == "w2":
            w1 = ctx.get_trace(self.component_id.with_op("w1"))
            w3 = ctx.get_trace(self.component_id.with_op("w3"))
            return ctx.verify(self.w2(w1 * w3))
        assert False

    def check(
        self,
        x: torch.Tensor,
        ctx: CheckCtx,
    ) -> torch.Tensor:
        w1 = F.silu(self.w1(x))
        w1 = ctx.check(self.component_id.with_op("w1"), w1)
        w3 = self.w3(x)
        w3 = ctx.check(self.component_id.with_op("w3"), w3)
        w2 = self.w2(w1 * w3)
        w2 = ctx.check(self.component_id.with_op("w2"), w2)
        return w2  # type: ignore

    def pick(self, x: torch.Tensor, ctx: PickCtx) -> torch.Tensor:
        ctx.pick_matmul(
            self.component_id.with_op("w1"),
            x,
            self.w1.weight.T,
        )
        w1 = x @ self.w1.weight.T
        ctx.pick_func(self.component_id.with_op("w1_silu"), w1, "silu")
        w1_silu = F.silu(w1)
        ctx.pick_matmul(
            self.component_id.with_op("w3"),
            x,
            self.w3.weight.T,
        )
        w3 = x @ self.w3.weight.T
        ctx.pick_linear(
            self.component_id.with_op("w1_silu_w3"), w1_silu, w3, torch.zeros_like(w3)
        )
        w1_silu_w3 = w1_silu * w3
        ctx.pick_matmul(self.component_id.with_op("w2"), w1_silu_w3, self.w2.weight.T)
        w2 = w1_silu_w3 @ self.w2.weight.T
        return w2  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        ctx: TraceCtx,
    ) -> torch.Tensor:
        op_input = ctx.last_op_id
        w1 = (
            x.to(ctx.intermediate_dtype) @ self.w1.weight.T.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(self.component_id.with_op("w1"), w1, [op_input])
        w1_silu = F.silu(w1.to(ctx.intermediate_dtype)).to(ctx.result_dtype)
        ctx.trace(
            self.component_id.with_op("w1_silu"),
            w1_silu,
            [self.component_id.with_op("w1")],
        )
        w3 = (
            x.to(ctx.intermediate_dtype) @ self.w3.weight.T.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(self.component_id.with_op("w3"), w3, [op_input])
        w1_silu_w3 = (
            w1_silu.to(ctx.intermediate_dtype) * w3.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(
            self.component_id.with_op("w1_silu_w3"),
            w1_silu_w3,
            [self.component_id.with_op("w1_silu"), self.component_id.with_op("w3")],
        )
        w2 = (
            w1_silu_w3.to(ctx.intermediate_dtype)
            @ self.w2.weight.T.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(
            self.component_id.with_op("w2"),
            w2,
            [self.component_id.with_op("w1_silu"), self.component_id.with_op("w3")],
        )
        return w2  # type: ignore


class TraceLinear(nn.Module):
    def __init__(
        self,
        component_id: ComponentId,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.component_id = component_id
        self.linear = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

    @torch.inference_mode()
    def verify(self, x: Optional[torch.Tensor], ctx: VerifyCtx) -> bool:
        assert x is not None
        result = self.linear(x)
        return ctx.verify(result)

    @torch.inference_mode()
    def check(self, x: torch.Tensor, ctx: CheckCtx) -> torch.Tensor:
        result = self.linear(x)
        result = ctx.check(self.component_id.with_op("output"), result)
        return result  # type: ignore

    @torch.inference_mode()
    def pick(self, x: torch.Tensor, ctx: PickCtx) -> torch.Tensor:
        ctx.pick_matmul(self.component_id.with_op("output"), x, self.linear.weight.T)
        result = x @ self.linear.weight.T
        return result  # type: ignore

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, ctx: TraceCtx) -> torch.Tensor:
        result = (
            x.to(ctx.intermediate_dtype)
            @ self.linear.weight.T.to(ctx.intermediate_dtype)
        ).to(ctx.result_dtype)
        ctx.trace(self.component_id.with_op("output"), result, None)
        return result

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.linear.load_state_dict(state_dict, strict, assign)  # type: ignore


class TraceEmbedding(nn.Module):
    def __init__(
        self,
        component_id: ComponentId,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.component_id = component_id
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )

    @torch.inference_mode()
    def verify(self, x: Optional[torch.Tensor], ctx: VerifyCtx) -> bool:
        assert x is not None
        result = self.embedding(x)
        return ctx.verify(result)

    @torch.inference_mode()
    def check(self, x: torch.Tensor, ctx: CheckCtx) -> torch.Tensor:
        result = self.embedding(x)
        result = ctx.check(self.component_id.with_op("output"), result)
        return result  # type: ignore

    @torch.inference_mode()
    def pick(self, x: torch.Tensor, ctx: PickCtx) -> torch.Tensor:
        # ctx.pick_embedding(self.component_id.with_op("output"), x, self.embedding.weight)
        result = self.embedding(x)
        return result  # type: ignore

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, ctx: TraceCtx) -> torch.Tensor:
        result = self.embedding(x)
        ctx.trace(self.component_id.with_op("output"), result, None)
        return result  # type: ignore

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.embedding.load_state_dict(state_dict, strict, assign)  # type: ignore


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: LayerId, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id.with_component("attention"), args)
        self.feed_forward = FeedForward(
            layer_id.with_component("feed_forward"),
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(
            layer_id.with_component("attention_norm"), args.dim, eps=args.norm_eps
        )
        self.ffn_norm = RMSNorm(
            layer_id.with_component("ffn_norm"), args.dim, eps=args.norm_eps
        )

    def verify(
        self,
        x: Optional[torch.Tensor],
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: VerifyCtx,
    ) -> bool:
        layer = ctx.op_id.layer
        component = ctx.op_id.component
        op = ctx.op_id.op
        if component == "feed_forward":
            if op == "res":
                return ctx.verify(
                    ctx.get_trace(OpId(layer=layer, component="attention", op="res"))
                    + ctx.get_trace(
                        OpId(layer=layer, component="feed_forward", op="w2")
                    )
                )
            else:
                return self.feed_forward.verify(
                    ctx.try_get_trace(
                        OpId(layer=layer, component="ffn_norm", op="weighted_output")
                    ),
                    ctx,
                )
        elif component == "ffn_norm":
            return self.ffn_norm.verify(
                ctx.try_get_trace(OpId(layer=layer, component="attention", op="res")),
                ctx,
            )
        elif component == "attention_norm":
            return self.attention_norm.verify(x, ctx)
        elif component == "attention":
            if op == "res":
                assert x is not None
                return ctx.verify(
                    x
                    + ctx.get_trace(
                        OpId(layer=layer, component="attention", op="weighted_output")
                    )
                )
            else:
                return self.attention.verify(
                    ctx.try_get_trace(
                        OpId(
                            layer=layer,
                            component="attention_norm",
                            op="weighted_output",
                        )
                    ),
                    freqs_cis,
                    mask,
                    ctx,
                )
        assert False

    def check(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: CheckCtx,
    ) -> torch.Tensor:
        attn_norm = self.attention_norm.check(x, ctx)
        attn = self.attention.check(attn_norm, freqs_cis, mask, ctx)
        h = x + attn
        h = ctx.check(self.layer_id.with_component("attention").with_op("res"), h)

        ffn_norm = self.ffn_norm.check(h, ctx)
        ffn = self.feed_forward.check(ffn_norm, ctx)
        h = h + ffn
        h = ctx.check(self.layer_id.with_component("feed_forward").with_op("res"), h)
        return h

    def pick(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: PickCtx,
    ) -> torch.Tensor:
        attn_norm = self.attention_norm.pick(x, ctx)
        attn = self.attention.pick(attn_norm, freqs_cis, mask, ctx)
        ctx.pick_linear(
            self.layer_id.with_component("attention").with_op("res"),
            x,
            torch.ones_like(x),
            attn,
        )
        h = x + attn

        ffn_norm = self.ffn_norm.pick(h, ctx)
        ffn = self.feed_forward.pick(ffn_norm, ctx)
        ctx.pick_linear(
            self.layer_id.with_component("feed_forward").with_op("res"),
            x,
            torch.ones_like(x),
            ffn,
        )
        h = x + ffn
        return h

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: TraceCtx,
    ) -> torch.Tensor:
        init_input_op = ctx.last_op_id
        attn_norm = self.attention_norm.forward(x, ctx)
        attn = self.attention.forward(attn_norm, freqs_cis, mask, ctx)
        h = x + attn
        ctx.trace(
            self.layer_id.with_component("attention").with_op("res"),
            h,
            [init_input_op, ctx.last_op_id],
        )

        ffn_norm = self.ffn_norm.forward(h, ctx)
        ffn = self.feed_forward.forward(ffn_norm, ctx)
        h = h + ffn
        ctx.trace(
            self.layer_id.with_component("feed_forward").with_op("res"),
            h,
            [self.layer_id.with_component("attention").with_op("res"), ctx.last_op_id],
        )
        return h


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        cache_dir = "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
        cache_dir = os.path.expanduser(cache_dir)
        self.device = device

        self.tok_embeddings = torch.nn.utils.skip_init(  # type: ignore
            TraceEmbedding,
            ComponentId(layer="tok_embeddings", component="main"),
            params.vocab_size,
            params.dim,
        )
        self.tok_embeddings.load_state_dict(
            torch.load(
                cache_dir + "tok_embeddings.pt", map_location="cpu", weights_only=True
            )
        )
        self.tok_embeddings.to(device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            layer = TransformerBlock(LayerId.from_str(f"{layer_id:02}"), params)
            layer.load_state_dict(
                torch.load(
                    cache_dir + f"layers.{layer_id}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            layer.to(device)
            self.layers.append(layer)

        self.norm = RMSNorm(
            ComponentId(layer="norm", component="main"), params.dim, eps=params.norm_eps
        )
        self.norm.load_state_dict(
            torch.load(cache_dir + "norm.pt", map_location="cpu", weights_only=True)
        )
        self.norm.to(device)
        self.output = torch.nn.utils.skip_init(  # type: ignore
            TraceLinear,
            ComponentId(layer="output", component="main"),
            params.dim,
            params.vocab_size,
        )
        self.output.load_state_dict(
            torch.load(cache_dir + "output.pt", map_location="cpu", weights_only=True)
        )
        self.output.to(device)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def verify(self, tokens: torch.Tensor, ctx: VerifyCtx) -> bool:
        _bsz, seqlen = tokens.shape
        layer = ctx.op_id.layer
        if layer.isdigit():
            freqs_cis = self.freqs_cis[0:seqlen]
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)
                mask = torch.hstack(
                    [torch.zeros((seqlen, 0), device=tokens.device), mask]
                ).type_as(tokens)

            layer_int = int(layer)
            if layer_int < 0 or layer_int >= self.n_layers:
                assert False

            if layer_int == 0:
                input = ctx.try_get_trace(
                    OpId(layer="tok_embeddings", component="main", op="output")
                )
            else:
                input = ctx.try_get_trace(
                    OpId(
                        layer=f"{layer_int - 1:02}", component="feed_forward", op="res"
                    )
                )

            return self.layers[layer_int].verify(input, freqs_cis, mask, ctx)  # type: ignore
        elif layer == "tok_embeddings":
            return self.tok_embeddings.verify(tokens, ctx)  # type: ignore
        elif layer == "norm":
            num_layers = self.n_layers
            return self.norm.verify(
                ctx.try_get_trace(
                    OpId(
                        layer=f"{num_layers - 1:02}", component="feed_forward", op="res"
                    )
                ),
                ctx,
            )
        elif layer == "output":
            return self.output.verify(  # type: ignore
                ctx.try_get_trace(
                    OpId(layer="norm", component="main", op="weighted_output")
                ),
                ctx,
            )
        assert False

    @torch.inference_mode()
    def check(self, tokens: torch.Tensor, ctx: CheckCtx) -> torch.Tensor:
        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings.check(tokens, ctx)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[0:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, 0), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer.check(h, freqs_cis, mask, ctx)

        h = self.norm.check(h, ctx)

        output = self.output.check(h, ctx)
        return output  # type: ignore

    @torch.inference_mode()
    def pick(self, tokens: torch.Tensor, ctx: PickCtx) -> torch.Tensor:
        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings.pick(tokens, ctx)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[0:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, 0), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer.pick(h, freqs_cis, mask, ctx)

        h = self.norm.pick(h, ctx)

        output = self.output.pick(h, ctx)
        return output  # type: ignore

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, ctx: TraceCtx) -> torch.Tensor:
        # ctx.traces[OpId(layer="tokens", component="main", op="input")] = tokens
        ctx.trace(OpId(layer="tokens", component="main", op="input"), tokens, [])
        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings.forward(tokens, ctx)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[0:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, 0), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer.forward(h, freqs_cis, mask, ctx)

        h = self.norm.forward(h, ctx)

        output = self.output.forward(h, ctx)
        return output  # type: ignore


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
torch.set_default_dtype(torch.float16)  # type: ignore
app = FastAPI()
device: torch.device


@app.post("/check")
async def check(request: Request) -> dict[str, float]:
    body = await request.body()
    tensors, metadata = loads(body)

    messages = metadata["messages"]
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    _, seqlen = tokens.shape
    tokens = tokens[:, : seqlen - 1]
    traces = {OpId.from_str(k): v.to(device) for k, v in tensors.items()}
    diffs: dict[OpId, float] = {}
    dtype = torch.float16
    model.check(tokens, CheckCtx(dtype, dtype, traces, diffs, device))
    return {str(k): v for k, v in diffs.items()}


@app.post("/verify")
async def verify(request: Request) -> tuple[bool, float]:
    begin = time.time()
    body = await request.body()
    tensors, metadata = loads(body)

    messages = metadata["messages"]
    op_id = OpId.from_str(metadata["op_id"])
    threshold = metadata["threshold"]
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    _, seqlen = tokens.shape
    tokens = tokens[:, : seqlen - 1]
    traces = {OpId.from_str(k): v.to(device) for k, v in tensors.items()}
    dtype = torch.float16
    result = model.verify(
        tokens, VerifyCtx(dtype, dtype, op_id, threshold, traces, device)
    )
    return result, (time.time() - begin) * 1000


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    intermediate_dtype: str = "float16"
    result_dtype: str = "float16"


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}")


@app.post("/forward")
async def forward(request: ChatRequest) -> str:
    messages = request.messages
    intermediate_dtype = request.intermediate_dtype
    result_dtype = request.result_dtype
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    _, seqlen = tokens.shape
    tokens = tokens[:, : seqlen - 1]
    traces: dict[OpId, torch.Tensor] = {}
    output2input: dict[OpId, list[OpId]] = {}
    signatures: dict[OpId, bytes] = {}
    tensor = model.forward(
        tokens,
        TraceCtx(
            get_dtype(intermediate_dtype),
            get_dtype(result_dtype),
            traces,
            output2input,
            OpId(layer="tokens", component="main", op="input"),
            device,
            signatures,
        ),
    )
    next_token = torch.argmax(tensor[-1, -1], dim=-1).reshape(1)
    return tokenizer.decode(next_token)  # type: ignore


@app.post("/trace")
async def trace(request: ChatRequest) -> Response:
    begin = time.time()
    messages = request.messages
    intermediate_dtype_str = request.intermediate_dtype
    result_dtype_str = request.result_dtype
    intermediate_dtype = get_dtype(intermediate_dtype_str)
    result_dtype = get_dtype(result_dtype_str)
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    _, seqlen = tokens.shape
    tokens = tokens[:, : seqlen - 1]
    traces: dict[OpId, torch.Tensor] = {}
    output2input: dict[OpId, list[OpId]] = {}
    signatures: dict[OpId, bytes] = {}
    tensor = model.forward(
        tokens,
        TraceCtx(
            intermediate_dtype,
            result_dtype,
            traces,
            output2input,
            OpId(layer="tokens", component="main", op="input"),
            device,
            signatures,
        ),
    )
    probs = torch.softmax(
        tensor[-1, -1].to(intermediate_dtype),
        dim=-1,
    ).to(result_dtype)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_map = {i: p for i, p in zip(probs_idx.tolist(), probs_sort.tolist())}

    str_traces = {str(k): v.cpu() for k, v in traces.items()}
    return Response(
        content=dumps(
            str_traces,
            {
                "probs": probs_map,
                "output2input": output2input,
                "time": (time.time() - begin) * 1000,
            },
        ),
        media_type="application/octet-stream",
    )


def setup():
    seed = 42
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def test_norm(seqlen: int) -> None:
    cache_dir = "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
    cache_dir = os.path.expanduser(cache_dir)
    norm = RMSNorm(ComponentId(layer="norm", component="main"), 4096, eps=1e-5)
    norm.load_state_dict(
        torch.load(cache_dir + "norm.pt", map_location="cpu", weights_only=True)
    )
    x = torch.randn(1, seqlen, 4096, device="cpu")
    signatures: dict[OpId, bytes] = {}
    trace_ctx = TraceCtx(
        torch.float16,
        torch.float16,
        {},
        {},
        OpId(layer="tokens", component="main", op="input"),
        device=torch.device("cpu"),
        signatures=signatures,
    )
    norm.forward(x, trace_ctx)
    for op_id in trace_ctx.traces:
        print("modify", op_id)
        backup = trace_ctx.traces[op_id].clone()
        trace_ctx.traces[op_id][..., -1] += 1.0
        pick_ctx = PickCtx(
            torch.float16, torch.float16, trace_ctx.traces, torch.device("cpu"), None
        )
        norm.pick(x, pick_ctx)
        trace_ctx.traces[op_id] = backup
        if pick_ctx.op is not None:
            print(pick_ctx.op, flush=True)
            torch.cpu.synchronize()
            begin = time.time()
            print(pick_ctx.op.verify())
            torch.cpu.synchronize()
            print((time.time() - begin) * 1000, "ms")
            print(pick_ctx.op.calc_size() / 1024, "kb")


def test_ffn(seqlen: int) -> None:
    cache_dir = "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
    cache_dir = os.path.expanduser(cache_dir)
    layer = TransformerBlock(LayerId.from_str("00"), llama_3_8b_args)
    layer.load_state_dict(
        torch.load(
            cache_dir + f"layers.0.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    layer.to("cpu")
    ffn = layer.feed_forward
    x = torch.randn(1, seqlen, 4096, device="cpu")
    signatures: dict[OpId, bytes] = {}
    trace_ctx = TraceCtx(
        torch.float16,
        torch.float16,
        {},
        {},
        OpId(layer="tokens", component="main", op="input"),
        device=torch.device("cpu"),
        signatures=signatures,
    )
    ffn.forward(x, trace_ctx)
    for op_id in trace_ctx.traces:
        print("modify", op_id)
        backup = trace_ctx.traces[op_id].clone()
        trace_ctx.traces[op_id][..., -1] += 1.0
        pick_ctx = PickCtx(
            torch.float16, torch.float16, trace_ctx.traces, torch.device("cpu"), None
        )
        ffn.pick(x, pick_ctx)
        trace_ctx.traces[op_id] = backup
        if pick_ctx.op is not None:
            print(pick_ctx.op, flush=True)
            torch.cpu.synchronize()
            begin = time.time()
            print(pick_ctx.op.verify())
            torch.cpu.synchronize()
            print((time.time() - begin) * 1000, "ms")
            print(pick_ctx.op.calc_size() / 1024, "kb")


def test_attention(seqlen: int) -> None:
    cache_dir = "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
    cache_dir = os.path.expanduser(cache_dir)
    layer = TransformerBlock(LayerId.from_str("00"), llama_3_8b_args)
    layer.load_state_dict(
        torch.load(
            cache_dir + f"layers.0.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    signatures: dict[OpId, bytes] = {}
    layer.to("cpu")
    attention = layer.attention
    x = torch.randn(1, seqlen, 4096, device="cpu")
    trace_ctx = TraceCtx(
        torch.float16,
        torch.float16,
        {},
        {},
        OpId(layer="tokens", component="main", op="input"),
        device=torch.device("cpu"),
        signatures=signatures,
    )
    params = llama_3_8b_args
    freqs_cis = precompute_freqs_cis(
        params.dim // params.n_heads,
        params.max_seq_len * 2,
        params.rope_theta,
    )
    freqs_cis = freqs_cis[0:seqlen]
    print(freqs_cis.shape)

    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device="cpu")

        mask = torch.triu(mask, diagonal=1)

        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack([torch.zeros((seqlen, 0), device="cpu"), mask]).to(
            torch.float16
        )

    attention.forward(x, freqs_cis, mask, trace_ctx)
    for op_id in trace_ctx.traces:
        backup = trace_ctx.traces[op_id].clone()
        print("modify", op_id)
        trace_ctx.traces[op_id][..., -1] += 1.0
        pick_ctx = PickCtx(
            torch.float16, torch.float16, trace_ctx.traces, torch.device("cpu"), None
        )
        attention.pick(x, freqs_cis, mask, pick_ctx)
        trace_ctx.traces[op_id] = backup
        if pick_ctx.op is not None:
            print(pick_ctx.op, flush=True)
            torch.cpu.synchronize()
            begin = time.time()
            print(pick_ctx.op.verify())
            torch.cpu.synchronize()
            print((time.time() - begin) * 1000, "ms")
            print(pick_ctx.op.calc_size() / 1024, "kb")


@torch.inference_mode()
def test_linear(seqlen: int) -> None:
    cache_dir = "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
    cache_dir = os.path.expanduser(cache_dir)
    output = torch.nn.utils.skip_init(  # type: ignore
        TraceLinear,
        ComponentId(layer="output", component="main"),
        llama_3_8b_args.dim,
        llama_3_8b_args.vocab_size,
    )
    output.load_state_dict(
        torch.load(cache_dir + "output.pt", map_location="cpu", weights_only=True)
    )
    output.to("cpu")
    x = torch.randn(1, seqlen, 4096, device="cpu")
    signatures: dict[OpId, bytes] = {}
    trace_ctx = TraceCtx(
        torch.float16,
        torch.float16,
        {},
        {},
        OpId(layer="tokens", component="main", op="input"),
        device=torch.device("cpu"),
        signatures=signatures,
    )
    output.forward(x, trace_ctx)
    for op_id in trace_ctx.traces:
        print("modify", op_id)
        backup = trace_ctx.traces[op_id].clone()
        trace_ctx.traces[op_id][..., -1] += 1.0
        pick_ctx = PickCtx(
            torch.float16,
            torch.float16,
            trace_ctx.traces,
            torch.device("cpu"),
            None,
        )
        output.pick(x, pick_ctx)
        trace_ctx.traces[op_id] = backup
        if pick_ctx.op is not None:
            print(pick_ctx.op, flush=True)
            torch.cpu.synchronize()
            begin = time.time()
            print(pick_ctx.op.verify())
            torch.cpu.synchronize()
            print((time.time() - begin) * 1000, "ms")
            print(pick_ctx.op.calc_size() / 1024, "kb")


@torch.inference_mode()
def test_whole() -> None:
    model = Transformer(
        model2args["meta-llama/Meta-Llama-3-8B-Instruct"], device=torch.device("cpu")
    )
    content_8 = "Hello But What"
    content_32 = "Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What"
    content_128 = "Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What Hello But What"
    messages = [{"role": "user", "content": content_8}]
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
    signatures: dict[OpId, bytes] = {}
    trace_ctx = TraceCtx(
        torch.float16,
        torch.float16,
        {},
        {},
        OpId(layer="tokens", component="main", op="input"),
        device=torch.device("cpu"),
        signatures=signatures,
    )
    model.forward(tokens, trace_ctx)
    for op_id in trace_ctx.traces:
        backup = trace_ctx.traces[op_id].clone()
        if backup.dtype != torch.float16:
            continue
        print("modify", op_id)
        trace_ctx.traces[op_id][..., -1] += 1.0
        pick_ctx = PickCtx(
            torch.float16, torch.float16, trace_ctx.traces, torch.device("cpu"), None
        )
        model.pick(tokens, pick_ctx)
        trace_ctx.traces[op_id] = backup
        if pick_ctx.op is not None:
            print(pick_ctx.op.op_id)
            begin = time.time()
            print(pick_ctx.op.verify())
            print((time.time() - begin) * 1000, "ms")
            print(pick_ctx.op.calc_size() / 1024, "kb")


if __name__ == "__main__":
    setup()
    print(f"New number of threads: {torch.get_num_threads()}")
    print(f"New number of inter-op threads: {torch.get_num_interop_threads()}")

    for seqlen in [8, 32, 128]:
        print("seqlen", seqlen)
        test_linear(seqlen)
        test_attention(seqlen)
        test_norm(seqlen)
        test_ffn(seqlen)
    # test_whole()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--port", type=int, default=19001)
    # args = parser.parse_args()

    # device = torch.device(args.device)
    # model = Transformer(model2args[args.model], device)
    # port = args.port
    # import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=port)
