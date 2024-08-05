import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer  # type: ignore

from deserve_worker.trace import ComponentId, LayerId, OpId

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
torch.set_default_dtype(torch.float16)
main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


@dataclass
class Diff:
    op_id: OpId
    diff: float
    
@dataclass
class CheckCtx: 
    threshold: float 
    traces: dict[OpId, torch.Tensor]
    
    def check(self, op_id: OpId, x: torch.Tensor) -> torch.Tensor | Diff:
        y = self.traces[op_id].to(main_device)
        if torch.allclose(x, y, atol=self.threshold):
            return y
        else: 
            return Diff(op_id, torch.max(torch.abs(x - y)).item())
    
@dataclass 
class VerifyCtx: 
    op_id: OpId 
    threshold: float
    traces: dict[OpId, torch.Tensor] 
    
    def get_trace(self, op_id: OpId) -> torch.Tensor: 
        return self.traces[op_id].to(main_device)

    def verify(self, x: torch.Tensor) -> bool:
        y = self.traces[self.op_id].to(main_device)
        return torch.allclose(x, y, atol=self.threshold)
    

class RMSNorm(nn.Module):
    def __init__(self, component_id: ComponentId, dim: int, eps: float = 1e-6):
        super().__init__()
        self.component_id = component_id
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def verify(self, x: torch.Tensor, ctx: VerifyCtx) -> bool:
        op = ctx.op_id.op
        if op == "output":
            return ctx.verify(self._norm(x.float()).type_as(x))
        else: 
            output = ctx.get_trace(self.component_id.with_op("weighted_output"))
            return ctx.verify(output * self.weight)    
        

    def forward(
        self,
        x: torch.Tensor,
        ctx: CheckCtx
    ) -> torch.Tensor | Diff:
        output = ctx.check(self.component_id.with_op("output"), self._norm(x.float()).type_as(x))
        if isinstance(output, Diff):
            return output
        return ctx.check(self.component_id.with_op("weighted_output"), output * self.weight)


 
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
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
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
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

        self.wq = torch.nn.utils.skip_init(
            nn.Linear,
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.utils.skip_init(
            nn.Linear,
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.utils.skip_init(
            nn.Linear,
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.utils.skip_init(
            nn.Linear,
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def verify(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: VerifyCtx, 
    ) -> bool:
        bsz, seqlen, _ = x.shape
        op = ctx.op_id.op
        if op == "xq":
            xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            return ctx.verify(xq)
        elif op == "xk":
            xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            return ctx.verify(xk)
        elif op == "xv":
            xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            return ctx.verify(xv)
        elif op == "xq_rotary" or op == "xk_rotary":
            xq, xk = self.wq(x), self.wk(x)
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
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

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: CheckCtx,
    ) -> torch.Tensor | Diff:
        bsz, seqlen, _ = x.shape

        xq = ctx.check(self.component_id.with_op("xq"), self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim))
        if isinstance(xq, Diff):
            return xq

        xk = ctx.check(self.component_id.with_op("xk"), self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        if isinstance(xk, Diff):
            return xk

        xv = ctx.check(self.component_id.with_op("xv"), self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim))
        if isinstance(xv, Diff):
            return xv

        xq_new, xk_new = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = ctx.check(self.component_id.with_op("xq_rotary"), xq_new)
        if isinstance(xq, Diff):
            return xq 
        
        xk = ctx.check(self.component_id.with_op("xk_rotary"), xk_new)
        if isinstance(xk, Diff):
            return xk

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
        scores_new = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores_new = scores_new + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores_new = F.softmax(scores_new.float(), dim=-1).type_as(xq)

        # check scores
        scores = ctx.check(self.component_id.with_op("scores"), scores_new)
        if isinstance(scores, Diff):
            return scores

        output_new = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output_new = output_new.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = ctx.check(self.component_id.with_op("output"), output_new)
        if isinstance(output, Diff):
            return output

        return ctx.check(self.component_id.with_op("weighted_output"), self.wo(output))


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

        self.w1 = torch.nn.utils.skip_init(
            nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = torch.nn.utils.skip_init(
            nn.Linear,
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = torch.nn.utils.skip_init(
            nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )

    def verify(self, x: torch.Tensor, ctx: VerifyCtx) -> bool:
        op = ctx.op_id.op
        if op == "w1":
            return ctx.verify(F.silu(self.w1(x)))
        elif op == "w3":
            return ctx.verify(self.w3(x))
        elif op == "w2":
            w1 = ctx.get_trace(self.component_id.with_op("w1"))
            w3 = ctx.get_trace(self.component_id.with_op("w3"))
            return ctx.verify(self.w2(w1 * w3))
        assert False

    def forward(
        self, x: torch.Tensor, ctx: CheckCtx,
    ) -> torch.Tensor | Diff:
        # check w1, w3, w2
        w1 = ctx.check(self.component_id.with_op("w1"), F.silu(self.w1(x)))
        if isinstance(w1, Diff):
            return w1
        
        w3 = ctx.check(self.component_id.with_op("w3"), self.w3(x))
        if isinstance(w3, Diff):
            return w3
        
        return ctx.check(self.component_id.with_op("w2"), self.w2(w1 * w3))

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
    def forward(
        self, x: torch.Tensor, ctx: CheckCtx
    ) -> torch.Tensor | Diff:
        return ctx.check(self.component_id.with_op("output"), self.linear(x))

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
    def forward(
        self, x: torch.Tensor, ctx: CheckCtx
    ) -> torch.Tensor | Diff:
        return ctx.check(self.component_id.with_op("output"), self.embedding(x))

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
        self.ffn_norm = RMSNorm(layer_id.with_component("ffn_norm"), args.dim, eps=args.norm_eps)

    def verify(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: VerifyCtx
    ) -> bool:
        layer = ctx.op_id.layer
        component = ctx.op_id.component 
        op = ctx.op_id.op
        if component == "feed_forward":
            if op == "res":
                return ctx.verify(ctx.get_trace(OpId(layer, "attention", "res")) + ctx.get_trace(OpId(layer, "feed_forward", "w2")))
            else: 
                return self.feed_forward.verify(
                    ctx.get_trace(OpId(layer, "ffn_norm", "weighted_output")), ctx
                )
        elif component == "ffn_norm":
            return self.ffn_norm.verify(ctx.get_trace(OpId(layer, "attention", "res")), ctx)
        elif component == "attention_norm":
            return self.attention_norm.verify(x, ctx)
        elif component == "attention":
            if op == "res": 
                return ctx.verify(x + ctx.get_trace(OpId(layer, "attention", "weighted_output")))
            else: 
                return self.attention.verify(
                    ctx.get_trace(OpId(layer, "attention_norm", "weighted_output")),
                    freqs_cis,
                    mask,
                    ctx
                )
        assert False

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ctx: CheckCtx,
    ) -> torch.Tensor | Diff:
        attn_norm = self.attention_norm.forward(x, ctx)
        if isinstance(attn_norm, Diff):
            return attn_norm

        attn = self.attention.forward(attn_norm, freqs_cis, mask, ctx)
        if isinstance(attn, Diff):
            return attn

        h = ctx.check(self.layer_id.with_component("attention").with_op("res"), x + attn)
        if isinstance(h, Diff):
            return h

        ffn_norm = self.ffn_norm.forward(h, ctx)
        if isinstance(ffn_norm, Diff):
            return ffn_norm

        ffn = self.feed_forward.forward(ffn_norm, ctx)
        if isinstance(ffn, Diff):
            return ffn

        return ctx.check(self.layer_id.with_component("feed_forward").with_op("res"), h + ffn) 


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        cache_dir = "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
        cache_dir = os.path.expanduser(cache_dir)

        self.tok_embeddings = torch.nn.utils.skip_init(
            TraceEmbedding, ComponentId("tok_embeddings", "main"), params.vocab_size, params.dim
        )
        self.tok_embeddings.load_state_dict(
            torch.load(cache_dir + "tok_embeddings.pt", map_location="cpu")
        )
        self.tok_embeddings.to(main_device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            layer = TransformerBlock(LayerId.from_str(f"{layer_id:02}"), params)
            layer.load_state_dict(
                torch.load(cache_dir + f"layers.{layer_id}.pt", map_location="cpu")
            )
            layer.to(main_device)
            self.layers.append(layer)

        self.norm = RMSNorm(ComponentId("norm", "main"), params.dim, eps=params.norm_eps)
        self.norm.load_state_dict(torch.load(cache_dir + "norm.pt", map_location="cpu"))
        self.norm.to(main_device)
        self.output = torch.nn.utils.skip_init(
            TraceLinear, ComponentId("output", "main"), params.dim, params.vocab_size
        )
        self.output.load_state_dict(
            torch.load(cache_dir + "output.pt", map_location="cpu")
        )
        self.output.to(main_device)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def verify(
        self, tokens: torch.Tensor, ctx: VerifyCtx
    ) -> bool:
        _bsz, seqlen = tokens.shape
        layer = ctx.op_id.layer
        if layer.isdigit():
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
                input = ctx.get_trace(OpId("tok_embeddings", "main", "output"))
            else: 
                input = ctx.get_trace(OpId(f"{layer_int - 1:02}", "feed_forward", "res"))
                
            return self.layers[layer_int].verify(input, self.freqs_cis, mask, ctx)
        elif layer == "tok_embeddings":
            return self.tok_embeddings.verify(tokens, ctx)
        elif layer == "norm":
            num_layers = self.n_layers
            return self.norm.verify(ctx.get_trace(OpId(f"{num_layers - 1:02}", "feed_forward", "res")), ctx)
        elif layer == "output":
            return self.output.verify(ctx.get_trace(OpId("norm", "main", "output")), ctx)
        assert False

    @torch.inference_mode()
    def forward(
        self, tokens: torch.Tensor, ctx: CheckCtx
    ) -> torch.Tensor | Diff:
        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings.forward(tokens, ctx)
        if isinstance(h, Diff):
            return h

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
            if isinstance(h, Diff):
                return h
        h = self.norm.forward(h, ctx)
        if isinstance(h, Diff):
            return h

        output = self.output.forward(h, ctx)
        if isinstance(output, Diff):
            return output
        return output.float()
