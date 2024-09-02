import torch
from flashinfer.norm import rmsnorm  # type: ignore

from deserve_worker.model.context.flash import FlashForwardCtx
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.model.utils import trace_op
from deserve_worker.trace import ComponentId


class RMSNorm(torch.nn.Module):
    def __init__(self, component_id: ComponentId, dim: int, eps: float = 1e-6):
        super().__init__()
        self.component_id = component_id
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        ctx: ForwardCtx,
    ) -> torch.Tensor:
        if isinstance(ctx, FlashForwardCtx):
            return self.paged_forward(x, ctx)
        else:
            assert isinstance(ctx, TraceForwardCtx)
            return self.trace_forward(x, ctx)

    def paged_forward(self, x: torch.Tensor, ctx: FlashForwardCtx) -> torch.Tensor:
        return rmsnorm(x, self.weight.data, self.eps)  # type: ignore

    def trace_forward(self, x: torch.Tensor, ctx: TraceForwardCtx) -> torch.Tensor:
        output = self.norm(x.float()).type_as(x)
        trace_op(ctx, self.component_id.with_op("output"), output, None)
        result = output * self.weight
        trace_op(
            ctx,
            self.component_id.with_op("weighted_output"),
            result,
            [self.component_id.with_op("output")],
        )
        return result
