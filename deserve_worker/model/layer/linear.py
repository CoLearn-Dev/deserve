from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.model.utils import trace_op
from deserve_worker.trace import ComponentId, OpId


class TraceLinear(torch.nn.Module):
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
        self.linear = torch.nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, ctx: ForwardCtx) -> torch.Tensor:
        out = self.linear(x)
        trace_op(ctx, self.component_id.with_op("output"), out)
        return out  # type: ignore

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.linear.load_state_dict(state_dict, strict, assign)  # type: ignore


class TraceEmbedding(torch.nn.Module):
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
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        ctx: ForwardCtx,
    ) -> torch.Tensor:
        out = self.embedding(x)
        trace_op(ctx, self.component_id.with_op("output"), out)
        return out  # type: ignore

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.embedding.load_state_dict(state_dict, strict, assign)  # type: ignore


class FeedForward(torch.nn.Module):
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
            torch.nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )  # type: ignore
        self.w2 = torch.nn.utils.skip_init(
            torch.nn.Linear,
            hidden_dim,
            dim,
            bias=False,
        )  # type: ignore
        self.w3 = torch.nn.utils.skip_init(
            torch.nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )  # type: ignore

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        ctx: ForwardCtx,
    ) -> torch.Tensor:
        w1 = F.silu(self.w1(x))
        w3 = self.w3(x)
        w2 = self.w2(w1 * w3)
        trace_op(ctx, self.component_id.with_op("w1"), w1)
        trace_op(ctx, self.component_id.with_op("w3"), w3)
        trace_op(ctx, self.component_id.with_op("w2"), w2)

        return w2  # type: ignore
