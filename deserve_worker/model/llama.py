import torch

from deserve_worker.model.args import ModelArgs
from deserve_worker.model.context.forward import PagedForwardCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.model.layer.attention import Attention
from deserve_worker.model.layer.linear import FeedForward
from deserve_worker.model.layer.norm import RMSNorm
from deserve_worker.model.utils import trace_op
from deserve_utils.trace import LayerId


class TransformerBlock(torch.nn.Module):
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
        ctx: PagedForwardCtx,
    ) -> torch.Tensor:
        if isinstance(ctx, TraceForwardCtx):
            init_input_op = ctx.last_op_id
        h = x + self.attention.forward(
            self.attention_norm(x, ctx),
            ctx,
        )
        if isinstance(ctx, TraceForwardCtx):
            later_input_op = ctx.last_op_id
            trace_op(
                ctx,
                self.layer_id.with_component("attention").with_op("res"),
                h,
                [
                    init_input_op,
                    later_input_op,
                ],
            )
        out = h + self.feed_forward.forward(self.ffn_norm(h, ctx), ctx)
        if isinstance(ctx, TraceForwardCtx):
            later_input_op = ctx.last_op_id
            trace_op(
                ctx,
                self.layer_id.with_component("feed_forward").with_op("res"),
                out,
                [
                    self.layer_id.with_component("attention").with_op("res"),
                    later_input_op,
                ],
            )
        return out
