from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from deserve_client.model.context import TraceCtx
from deserve_utils.trace import ComponentId, OpId


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

    def forward(
        self,
        x: OpId,
        ctx: TraceCtx,
    ) -> OpId:
        cid = self.component_id
        w1 = ctx.matmul(
            cid.with_op("w1"),
            x,
            ctx.register(cid.with_op("w1_weight"), self.w1.weight, keep_last_dim=True),
        )
        w1_silu = ctx.unary(cid.with_op("w1_silu"), w1, "silu")
        w3 = ctx.matmul(
            cid.with_op("w3"),
            x,
            ctx.register(cid.with_op("w3_weight"), self.w3.weight, keep_last_dim=True),
        )
        w1_silu_w3 = ctx.binary(
            cid.with_op("w1_silu_w3"),
            w1_silu,
            w3,
            "mul",
        )
        w2 = ctx.matmul(
            cid.with_op("w2"),
            w1_silu_w3,
            ctx.register(cid.with_op("w2_weight"), self.w2.weight, keep_last_dim=True),
        )
        return w2
