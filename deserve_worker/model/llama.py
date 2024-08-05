# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import math
import pickle
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple, cast

import safetensors.torch
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache  # type: ignore
from torch import nn

from deserve_worker.kvcache.paged_kvcache import PagedKVCache, PagedKVCacheManager
from deserve_worker.trace import ComponentId, LayerId, OpId

from ..kvcache.kvcache import KVCache, KVCacheManager
from ..kvcache.packed_kvcache import PackedKVCache, PackedKVCacheManager


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


def trace_op(
    traces: Optional[dict[OpId, torch.Tensor]], op_id: OpId, op_value: torch.Tensor
) -> None:
    if traces is not None:
        traces[op_id] = op_value


class RMSNorm(torch.nn.Module):
    def __init__(self, component_id: ComponentId, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.component_id = component_id
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        traces: Optional[dict[OpId, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        trace_op(traces, self.component_id.with_op("output"), output)
        result = output * self.weight
        trace_op(traces, self.component_id.with_op("weighted_output"), result)
        return result


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.

    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
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
    """Multi-head attention module."""

    def __init__(self, component_id: ComponentId, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.component_id = component_id
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads  # // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads  # // model_parallel_size
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

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        bsz_list: List[int],
        start_pos_list: List[int],
        global_freqs_cis: torch.Tensor,
        kvcache_list: list[KVCache],
        kvcache_manager: KVCacheManager,
        traces: Optional[dict[OpId, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        _, seqlen, _ = x.shape
        xq_, xk_, xv_ = self.wq(x), self.wk(x), self.wv(x)

        if isinstance(kvcache_manager, PagedKVCacheManager):
            cache_seqlens = []
            for i, bsz in enumerate(bsz_list):
                cache_seqlens += [start_pos_list[i]] * bsz
            cache_seqlens_tch = torch.tensor(
                cache_seqlens, dtype=torch.int32, device=x.device
            )
            bsz = cache_seqlens_tch.shape[0]
            paged_kv_cache_list = cast(list[PagedKVCache], kvcache_list)

            max_len = max([kvcache.shape()[1] for kvcache in paged_kv_cache_list])
            block_table = torch.zeros(
                (bsz, max_len), dtype=torch.int32, device=x.device
            )
            start = 0
            for i, bsz in enumerate(bsz_list):
                block_table[
                    start : start + bsz, : paged_kv_cache_list[i].shape()[1]
                ] = paged_kv_cache_list[i].block_table
                start += bsz

            bsz = cache_seqlens_tch.shape[0]
            xq = xq_.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk_.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv_.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            cos = global_freqs_cis[0].type_as(xq)
            sin = global_freqs_cis[1].type_as(xq)
            output = flash_attn_with_kvcache(
                xq,
                kvcache_manager.block_pool.block_ks,
                kvcache_manager.block_pool.block_vs,
                xk,
                xv,
                rotary_cos=cos,
                rotary_sin=sin,
                cache_seqlens=cache_seqlens_tch,
                block_table=block_table,
                causal=True,
                rotary_interleaved=True,
            )
            output = output.view(bsz, seqlen, -1)
            return self.wo(output)  # type: ignore
        else:
            kvcache_manager = cast(PackedKVCacheManager, kvcache_manager)
            start = 0
            output_list = []
            for i, bsz in enumerate(bsz_list):
                xq = xq_[start : start + bsz].view(
                    bsz, seqlen, self.n_local_heads, self.head_dim
                )
                xk = xk_[start : start + bsz].view(
                    bsz, seqlen, self.n_local_kv_heads, self.head_dim
                )
                xv = xv_[start : start + bsz].view(
                    bsz, seqlen, self.n_local_kv_heads, self.head_dim
                )
                trace_op(traces, self.component_id.with_op("xq"), xq)
                trace_op(traces, self.component_id.with_op("xk"), xk)
                trace_op(traces, self.component_id.with_op("xv"), xv)
                start += bsz

                start_pos = start_pos_list[i]
                # remember consecutive block table [bsz, len] corresponds to memory [bsz, len * block_size, 8, 128]
                kv_cache: PackedKVCache = cast(PackedKVCache, kvcache_list[i])
                csct_block_table = kv_cache.csct_block_table.flatten()
                block_bsz, block_len = kv_cache.csct_block_table.shape[:2]
                cache_k = kvcache_manager.block_pool.block_ks[
                    csct_block_table[0] : csct_block_table[-1] + 1
                ].view(block_bsz, block_len * kvcache_manager.block_size, 8, 128)
                cache_v = kvcache_manager.block_pool.block_vs[
                    csct_block_table[0] : csct_block_table[-1] + 1
                ].view(block_bsz, block_len * kvcache_manager.block_size, 8, 128)

                freqs_cis = global_freqs_cis[start_pos : start_pos + seqlen]
                mask = None
                if seqlen > 1:
                    mask = torch.full(
                        (1, 1, seqlen, seqlen), float("-inf"), device=x.device
                    )
                    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)

                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
                trace_op(traces, self.component_id.with_op("xq_rotary"), xq)
                trace_op(traces, self.component_id.with_op("xk_rotary"), xk)

                cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                cache_v[:bsz, start_pos : start_pos + seqlen] = xv

                keys = cache_k[:bsz, : start_pos + seqlen]
                values = cache_v[:bsz, : start_pos + seqlen]

                # repeat k/v heads if n_kv_heads < n_heads
                keys = repeat_kv(
                    keys, self.n_rep
                )  # (bs, seqlen, n_local_heads, head_dim)
                values = repeat_kv(
                    values, self.n_rep
                )  # (bs, seqlen, n_local_heads, head_dim)

                xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)
                scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
                    self.head_dim
                )
                if mask is not None:
                    scores = (
                        scores + mask
                    )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                trace_op(traces, self.component_id.with_op("scores"), scores)
                output = torch.matmul(
                    scores, values
                )  # (bs, n_local_heads, seqlen, head_dim)

                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                trace_op(traces, self.component_id.with_op("output"), output)
                output_list.append(output)
            output = torch.cat([x for x in output_list])
            result = self.wo(output)
            trace_op(traces, self.component_id.with_op("weighted_output"), result)
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
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
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

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, traces: Optional[dict[OpId, torch.Tensor]]
    ) -> torch.Tensor:
        w1 = F.silu(self.w1(x))
        w3 = self.w3(x)
        w2 = self.w2(w1 * w3)
        trace_op(traces, self.component_id.with_op("w1"), w1)
        trace_op(traces, self.component_id.with_op("w3"), w3)
        trace_op(traces, self.component_id.with_op("w2"), w2)

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
    def forward(
        self, x: torch.Tensor, traces: Optional[dict[OpId, torch.Tensor]]
    ) -> torch.Tensor:
        out = self.linear(x)
        trace_op(traces, self.component_id.with_op("output"), out)
        return out  # type: ignore

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
        self, x: torch.Tensor, traces: Optional[dict[OpId, torch.Tensor]]
    ) -> torch.Tensor:
        out = self.embedding(x)
        trace_op(traces, self.component_id.with_op("output"), out)
        return out  # type: ignore

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.embedding.load_state_dict(state_dict, strict, assign)  # type: ignore


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: LayerId, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
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

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        bsz_list: List[int],
        start_pos_list: List[int],
        global_freqs_cis: torch.Tensor,
        kvcache_list: list[KVCache],
        kvcache_manager: KVCacheManager,
        traces: Optional[dict[OpId, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention.forward(
            self.attention_norm(x, traces),
            bsz_list,
            start_pos_list,
            global_freqs_cis,
            kvcache_list,
            kvcache_manager,
            traces,
        )
        trace_op(traces, self.layer_id.with_component("attention").with_op("res"), h)
        out = h + self.feed_forward.forward(self.ffn_norm(h, traces), traces)
        trace_op(
            traces, self.layer_id.with_component("feed_forward").with_op("res"), out
        )
        return out


def dumps(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> bytes:
    """
    Dump tensors and metadata into bytes
    """

    metadata_bytes = pickle.dumps(metadata)
    tensors_bytes = safetensors.torch.save(tensors)
    return (
        len(metadata_bytes).to_bytes(4, byteorder="big")
        + metadata_bytes
        + tensors_bytes
    )


def loads(b: bytes) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Load tensors and metadata from bytes
    """

    metadata_length = int.from_bytes(b[:4], byteorder="big")
    metadata = pickle.loads(b[4 : 4 + metadata_length])
    tensors = safetensors.torch.load(b[4 + metadata_length :])
    return tensors, metadata
