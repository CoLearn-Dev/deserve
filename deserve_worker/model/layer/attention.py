import math

import torch
import torch.nn.functional as F
from flashinfer import append_paged_kv_cache, apply_rope  # type: ignore

from deserve_worker.model.args import ModelArgs
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.model.context.paged import (
    PagedDecodeCtx,
    PagedForwardCtx,
    PagedPrefillCtx,
)
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.model.utils import trace_op
from deserve_worker.trace import ComponentId


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
) -> tuple[torch.Tensor, torch.Tensor]:
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


class Attention(torch.nn.Module):
    """Multi-head attention module."""

    def __init__(self, component_id: ComponentId, args: ModelArgs):
        super().__init__()
        self.component_id = component_id
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.utils.skip_init(
            torch.nn.Linear,
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )  # type: ignore
        self.wk = torch.nn.utils.skip_init(
            torch.nn.Linear,
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )  # type: ignore
        self.wv = torch.nn.utils.skip_init(
            torch.nn.Linear,
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )  # type: ignore
        self.wo = torch.nn.utils.skip_init(
            torch.nn.Linear,
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )  # type: ignore

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        ctx: ForwardCtx,
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
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if isinstance(ctx, PagedForwardCtx):
            xq = xq.view(-1, self.n_local_heads, self.head_dim)
            xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)
            return self.paged_forward(xq, xk, xv, ctx)
        else:
            assert isinstance(ctx, TraceForwardCtx)
            bsz, seqlen = ctx.bsz, ctx.seqlens[0]
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            return self.traced_forward(xq, xk, xv, ctx).view(bsz, seqlen, -1)  # type: ignore

    def paged_forward(
        self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, ctx: PagedForwardCtx
    ) -> torch.Tensor:
        """
        The return tensor is in ragged format.
        """
        xq, xk = apply_rope(
            xq,
            xk,
            ctx.indptr,
            ctx.offsets,
            interleave=True,
            rope_theta=500000,
        )
        pages = ctx.page_pool.get_layered_pages(ctx.layer_id)
        append_paged_kv_cache(
            xk,
            xv,
            ctx.indptr,
            pages,
            ctx.kv_page_indices,
            ctx.kv_page_indptr,
            ctx.kv_last_page_lens,
        )
        if isinstance(ctx, PagedDecodeCtx):
            output = ctx.decode_wrapper.forward(xq, pages).view(-1, self.dim)
        else:
            assert isinstance(ctx, PagedPrefillCtx)
            output = ctx.prefill_wrapper.forward(
                xq,
                pages,
                causal=True,
            ).view(-1, self.dim)
        return self.wo(output)  # type: ignore

    def traced_forward(
        self,
        xqs: torch.Tensor,
        xks: torch.Tensor,
        xvs: torch.Tensor,
        ctx: TraceForwardCtx,
    ) -> torch.Tensor:
        seqlen = xqs.shape[1]
        output_list = []
        pages = ctx.page_pool.get_layered_pages(ctx.layer_id)
        pages_k = pages[:, 0, :, :, :].flatten(0, 1)
        pages_v = pages[:, 1, :, :, :].flatten(0, 1)
        for i, start_pos in enumerate(ctx.offsets):
            xq = xqs[i].view(1, seqlen, self.n_local_heads, self.head_dim)
            xk = xks[i].view(1, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xvs[i].view(1, seqlen, self.n_local_kv_heads, self.head_dim)
            trace_op(ctx, self.component_id.with_op("xq"), xq)
            trace_op(ctx, self.component_id.with_op("xk"), xk)
            trace_op(ctx, self.component_id.with_op("xv"), xv)

            # remember consecutive block table [bsz, len] corresponds to memory [bsz, len * block_size, 8, 128]
            begin, end = ctx.ranges[i]
            cache_k = pages_k[begin : end + 1].view(
                1, (end - begin + 1) * ctx.page_pool.page_size, 8, 128
            )
            cache_v = pages_v[begin : end + 1].view(
                1, (end - begin + 1) * ctx.page_pool.page_size, 8, 128
            )

            freqs_cis = ctx.global_freqs_cis[start_pos : start_pos + seqlen]
            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=xq.device
                )
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(xq)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            trace_op(ctx, self.component_id.with_op("xq_rotary"), xq)
            trace_op(ctx, self.component_id.with_op("xk_rotary"), xk)

            cache_k[:, start_pos : start_pos + seqlen] = xk
            cache_v[:, start_pos : start_pos + seqlen] = xv

            keys = cache_k[:, : start_pos + seqlen]
            values = cache_v[:, : start_pos + seqlen]

            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            values = repeat_kv(
                values, self.n_rep
            )  # (bs, seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = (
                    scores + mask
                )  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            trace_op(ctx, self.component_id.with_op("scores"), scores)
            output = torch.matmul(
                scores, values
            )  # (bs, n_local_heads, seqlen, head_dim)

            output = output.transpose(1, 2).contiguous().view(1, seqlen, -1)
            trace_op(ctx, self.component_id.with_op("output"), output)
            output_list.append(output)
        output = torch.cat([x for x in output_list])
        result = self.wo(output)
        trace_op(ctx, self.component_id.with_op("weighted_output"), result)
        return result  # type: ignore
