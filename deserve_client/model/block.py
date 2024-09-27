from torch import nn

from deserve_client.model.attn import Attention
from deserve_client.model.context import TraceCtx
from deserve_client.model.ffn import FeedForward
from deserve_client.model.norm import RMSNorm
from deserve_utils.trace import LayerId, OpId
from deserve_worker.model.args import ModelArgs


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: LayerId, args: ModelArgs):
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

    def forward(self, x: OpId, freqs_cis: OpId, mask: OpId, ctx: TraceCtx) -> OpId:
        attn_norm = self.attention_norm.forward(x, ctx)
        attn = self.attention.forward(attn_norm, freqs_cis, mask, ctx)
        h = ctx.binary(
            self.layer_id.with_component("attention").with_op("res"),
            x,
            attn,
            "add",
        )
        ffn_norm = self.ffn_norm.forward(h, ctx)
        ffn = self.feed_forward.forward(ffn_norm, ctx)
        h = ctx.binary(
            self.layer_id.with_component("feed_forward").with_op("res"),
            h,
            ffn,
            "add",
        )
        return h
