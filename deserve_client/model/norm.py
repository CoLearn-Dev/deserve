from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from deserve_client.model.context import TraceCtx
from deserve_utils.trace import ComponentId, OpId


class RMSNorm(nn.Module):
    def __init__(self, component_id: ComponentId, dim: int, eps: float = 1e-6):
        super().__init__()
        self.component_id = component_id
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: OpId, ctx: TraceCtx) -> OpId:
        cid = self.component_id
        pow = ctx.binary(cid.with_op("pow"), x, x, "mul")
        mean = ctx.binary(
            cid.with_op("mean"),
            pow,
            ctx.register(
                cid.with_op("rcnt"),
                torch.tensor(
                    [1.0 / ctx.get(x).shape[-1]],
                    dtype=ctx.get(x).dtype,
                    device=ctx.get(x).device,
                ),
                keep_last_dim=False,
            ),
            "mul",
        )
        offset = ctx.binary(
            cid.with_op("offset"),
            mean,
            ctx.register(
                cid.with_op("eps"),
                torch.tensor(
                    [self.eps], dtype=ctx.get(x).dtype, device=ctx.get(x).device
                ),
                keep_last_dim=False,
            ),
            "add",
        )
        rsqrt = ctx.unary(cid.with_op("rsqrt"), offset, "rsqrt")
        output = ctx.binary(cid.with_op("output"), x, rsqrt, "mul")
        result = ctx.binary(
            cid.with_op("weighted_output"),
            output,
            ctx.register(
                cid.with_op("weight"),
                self.weight,
                keep_last_dim=True,
            ),
            "mul",
        )
        return result
