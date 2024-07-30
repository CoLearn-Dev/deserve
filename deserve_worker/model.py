# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import math
import pickle
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn

from deserve_worker.paged_kvcache import PagedKVCache

from .kvcache import KVCache, KVCacheBase

ENABLE_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_with_kvcache  # type: ignore

    from .paged_kvcache import global_paged_memory

    ENABLE_FLASH_ATTN = True
except ImportError as e:
    print(
        "Package flash-attn is not found. Please install it for better performance. https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features"
    )


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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    if ENABLE_FLASH_ATTN:
        freqs_cis = torch.stack([freqs.cos(), freqs.sin()])  # flash_attn
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


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

    def __init__(self, args: ModelArgs):
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
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads  # // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads  # // model_parallel_size
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

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        bsz_list: List[int],
        start_pos_list: List[int],
        global_freqs_cis: torch.Tensor,
        kv_cache_list: list[KVCacheBase],
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

        if ENABLE_FLASH_ATTN:
            cache_seqlens = []
            for i, bsz in enumerate(bsz_list):
                cache_seqlens += [start_pos_list[i]] * bsz
            cache_seqlens_tch = torch.tensor(
                cache_seqlens, dtype=torch.int32, device=x.device
            )
            bsz = cache_seqlens_tch.shape[0]
            paged_kv_cache_list: list[PagedKVCache] = kv_cache_list  # type: ignore

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
                global_paged_memory.cache_k_paged,
                global_paged_memory.cache_v_paged,
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
        else:
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
                start += bsz

                start_pos = start_pos_list[i]
                kv_cache: KVCache = cast(KVCache, kv_cache_list[i])
                cache_k, cache_v = kv_cache.cache_k, kv_cache.cache_v

                freqs_cis = global_freqs_cis[start_pos : start_pos + seqlen]
                mask = None
                if seqlen > 1:
                    mask = torch.full(
                        (1, 1, seqlen, seqlen), float("-inf"), device=x.device
                    )
                    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)

                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

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
                output = torch.matmul(
                    scores, values
                )  # (bs, n_local_heads, seqlen, head_dim)
                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                output_list.append(output)
            output = torch.cat([x for x in output_list])
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
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

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
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
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        # self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        bsz_list: List[int],
        start_pos_list: List[int],
        global_freqs_cis: torch.Tensor,
        kv_cache_list: list[KVCacheBase],
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
            self.attention_norm(x),
            bsz_list,
            start_pos_list,
            global_freqs_cis,
            kv_cache_list,
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
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
