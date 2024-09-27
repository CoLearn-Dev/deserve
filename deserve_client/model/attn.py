import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deserve_client.model.context import TraceCtx
from deserve_utils.trace import ComponentId, OpId
from deserve_worker.model.args import ModelArgs


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

    def forward(
        self,
        x: OpId,
        freqs_cis: OpId,
        mask: OpId,
        ctx: TraceCtx,
    ) -> OpId:
        cid = self.component_id
        xq = ctx.matmul(
            cid.with_op("xq"),
            x,
            ctx.register(
                cid.with_op("wq_weight"),
                self.wq.weight,
                keep_last_dim=True,
            ),
        )
        bsz, seqlen = ctx.get(x).shape[:2]
        xq = ctx.reshape(
            cid.with_op("xq_splitted"),
            xq,
            [bsz, seqlen, self.n_local_heads, self.head_dim],
        )
        xk = ctx.matmul(
            cid.with_op("xk"),
            x,
            ctx.register(
                cid.with_op("wk_weight"),
                self.wk.weight,
                keep_last_dim=True,
            ),
        )
        xk = ctx.reshape(
            cid.with_op("xk_splitted"),
            xk,
            [bsz, seqlen, self.n_local_kv_heads, self.head_dim],
        )
        xv = ctx.matmul(
            cid.with_op("xv"),
            x,
            ctx.register(
                cid.with_op("wv_weight"),
                self.wv.weight,
                keep_last_dim=True,
            ),
        )
        xv = ctx.reshape(
            cid.with_op("xv_splitted"),
            xv,
            [bsz, seqlen, self.n_local_kv_heads, self.head_dim],
        )

        # apply rotary embedding
        xq = ctx.binary(cid.with_op("xq_rotary"), xq, freqs_cis, "complex_mul")
        xk = ctx.binary(cid.with_op("xk_rotary"), xk, freqs_cis, "complex_mul")
        keys = ctx.repeat(cid.with_op("keys"), xk, -2, self.n_rep)
        values = ctx.repeat(cid.with_op("values"), xv, -2, self.n_rep)
        xq = ctx.permute(cid.with_op("xq_transpose"), xq, [0, 2, 1, 3])
        keys = ctx.permute(cid.with_op("keys_transpose"), keys, [0, 2, 1, 3])
        values = ctx.permute(cid.with_op("values_transpose"), values, [0, 2, 3, 1])
        scores = ctx.matmul(cid.with_op("scores_matmul"), xq, keys)
        scores = ctx.binary(
            cid.with_op("scores_div"),
            scores,
            ctx.register(
                cid.with_op("rhead_dim"),
                torch.tensor(
                    [1.0 / math.sqrt(self.head_dim)],
                    dtype=ctx.get(scores).dtype,
                    device=ctx.get(scores).device,
                ),
                keep_last_dim=False,
            ),
            "mul",
        )
        if mask is not None:
            scores = ctx.binary(cid.with_op("scores_mask"), scores, mask, "add")
        scores = ctx.unary(cid.with_op("scores_softmax"), scores, "softmax")
        output = ctx.matmul(cid.with_op("output_matmul"), scores, values)
        output = ctx.permute(cid.with_op("output_transpose"), output, [0, 2, 1, 3])
        output = ctx.reshape(
            cid.with_op("output_view"),
            output,
            [bsz, seqlen, self.n_local_heads * self.head_dim],
        )
        output = ctx.matmul(
            cid.with_op("weighted_output"),
            output,
            ctx.register(
                cid.with_op("wo_weight"),
                self.wo.weight,
                keep_last_dim=True,
            ),
        )
        return output
