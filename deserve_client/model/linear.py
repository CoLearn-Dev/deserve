from typing import Any, Mapping, Optional

import torch
from torch import nn

from deserve_client.model.context import TraceCtx
from deserve_utils.trace import ComponentId, OpId


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
    def forward(self, x: OpId, ctx: TraceCtx) -> OpId:
        return ctx.matmul(
            self.component_id.with_op("output"),
            x,
            ctx.register(
                self.component_id.with_op("weight"),
                self.linear.weight,
                keep_last_dim=True,
            ),
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.linear.load_state_dict(state_dict, strict, assign)  # type: ignore
