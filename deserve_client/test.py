import os

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from deserve_client.model.block import TransformerBlock
from deserve_client.model.context import (
    TraceBaseOp,
    TraceCtx,
    TraceRegisterOp,
    VerifyResult,
)
from deserve_client.model.linear import TraceLinear
from deserve_client.model.norm import RMSNorm
from deserve_utils.hash import BatchMerkleTree
from deserve_utils.trace import ComponentId, LayerId, OpId
from deserve_worker.model.args import ModelArgs

cache_dir = os.path.expanduser(
    "~/.cache/fleece-worker/models/llama-3-8b-instruct-slice/"
)

llama_3_8b_args = ModelArgs(
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000.0,
)


def setup() -> None:
    seed = 42
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def test_rms_norm() -> None:
    norm = RMSNorm(ComponentId(layer="norm", component="main"), 4096, eps=1e-5)
    norm.load_state_dict(
        torch.load(cache_dir + "norm.pt", map_location="cpu", weights_only=True)
    )
    x = torch.randn(64, 128, 4096, device="cpu")
    traces: dict[OpId, TraceBaseOp] = {}
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    trace_ctx = TraceCtx(
        traces,
        private_key,
    )
    trace_ctx.register(
        OpId(layer="norm", component="main", op="input"), x, keep_last_dim=True
    )
    norm.forward(OpId(layer="norm", component="main", op="input"), trace_ctx)
    for op, trace in traces.items():
        print(op, trace.output.tensor.shape)
        assert trace.try_verify_output(trace_ctx) is None
        if isinstance(trace, TraceRegisterOp):
            continue
        tensor_backup = trace.output.tensor.clone()
        merkle_root_backup = trace.output.merkle_root_sig
        trace.output.tensor[..., -1] += 1.0
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.WRONG_MEMBERSHIP
        trace.output.merkle_root_sig = BatchMerkleTree(
            trace.output.tensor, trace.output.keep_last_dim
        ).get_root()
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.VERIFIED
        trace.output.tensor = tensor_backup
        trace.output.merkle_root_sig = merkle_root_backup


@torch.inference_mode()
def test_linear() -> None:
    output = torch.nn.utils.skip_init(  # type: ignore
        TraceLinear,
        ComponentId(layer="output", component="main"),
        llama_3_8b_args.dim,
        llama_3_8b_args.vocab_size,
    )
    output.load_state_dict(
        torch.load(cache_dir + "output.pt", map_location="cpu", weights_only=True)
    )
    x = torch.randn(1, 32, 4096, device="cpu")  # FIXME: performance bottleneck
    traces: dict[OpId, TraceBaseOp] = {}
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    trace_ctx = TraceCtx(
        traces,
        private_key,
    )
    trace_ctx.register(
        OpId(layer="output", component="main", op="input"), x, keep_last_dim=True
    )
    output.forward(OpId(layer="output", component="main", op="input"), trace_ctx)
    for op, trace in traces.items():
        print(op, trace.output.tensor.shape)
        assert trace.try_verify_output(trace_ctx) is None
        if isinstance(trace, TraceRegisterOp):
            continue
        tensor_backup = trace.output.tensor.clone()
        merkle_root_backup = trace.output.merkle_root_sig
        trace.output.tensor[..., -1] += 1.0
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.WRONG_MEMBERSHIP
        trace.output.merkle_root_sig = BatchMerkleTree(
            trace.output.tensor, trace.output.keep_last_dim
        ).get_root()
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.VERIFIED
        trace.output.tensor = tensor_backup
        trace.output.merkle_root_sig = merkle_root_backup


def test_ffn() -> None:
    layer = TransformerBlock(LayerId.from_str("00"), llama_3_8b_args)
    layer.load_state_dict(
        torch.load(cache_dir + "layers.0.pt", map_location="cpu", weights_only=True)
    )
    ffn = layer.feed_forward
    x = torch.randn(1, 32, 4096, device="cpu")
    traces: dict[OpId, TraceBaseOp] = {}
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    trace_ctx = TraceCtx(
        traces,
        private_key,
    )
    trace_ctx.register(
        OpId(layer="ffn", component="input", op="x"), x, keep_last_dim=True
    )
    ffn.forward(
        OpId(layer="ffn", component="input", op="x"),
        trace_ctx,
    )
    for op, trace in traces.items():
        print(op, trace.output.tensor.shape)
        assert trace.try_verify_output(trace_ctx) is None
        if isinstance(trace, TraceRegisterOp):
            continue
        tensor_backup = trace.output.tensor.clone()
        merkle_root_backup = trace.output.merkle_root_sig
        trace.output.tensor[..., -1] += 1.0
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.WRONG_MEMBERSHIP
        trace.output.merkle_root_sig = BatchMerkleTree(
            trace.output.tensor, trace.output.keep_last_dim
        ).get_root()
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.VERIFIED
        trace.output.tensor = tensor_backup
        trace.output.merkle_root_sig = merkle_root_backup


def test_attention() -> None:
    layer = TransformerBlock(LayerId.from_str("00"), llama_3_8b_args)
    layer.load_state_dict(
        torch.load(cache_dir + "layers.0.pt", map_location="cpu", weights_only=True)
    )
    attention = layer.attention
    x = torch.randn(1, 32, 4096, device="cpu")
    traces: dict[OpId, TraceBaseOp] = {}
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    trace_ctx = TraceCtx(traces, private_key)
    trace_ctx.register(
        OpId(layer="attention", component="input", op="x"), x, keep_last_dim=True
    )
    freqs_cis = precompute_freqs_cis(
        llama_3_8b_args.dim // llama_3_8b_args.n_heads,
        llama_3_8b_args.max_seq_len * 2,
        llama_3_8b_args.rope_theta,
    )
    freqs_cis = freqs_cis[0:32]
    shape0, shape1 = freqs_cis.shape
    freqs_cis = freqs_cis.reshape(1, shape0, 1, shape1)
    trace_ctx.register(
        OpId(layer="attention", component="main", op="freqs_cis"),
        freqs_cis,
        keep_last_dim=True,
    )
    mask = torch.randn(1, 32, 32, device="cpu")
    trace_ctx.register(
        OpId(layer="attention", component="main", op="mask"), mask, keep_last_dim=False
    )
    attention.forward(
        OpId(layer="attention", component="input", op="x"),
        OpId(layer="attention", component="main", op="freqs_cis"),
        OpId(layer="attention", component="main", op="mask"),
        trace_ctx,
    )
    for op, trace in traces.items():
        print(op, trace.output.tensor.shape)
        assert trace.try_verify_output(trace_ctx) is None
        if isinstance(trace, TraceRegisterOp):
            continue
        tensor_backup = trace.output.tensor.clone()
        merkle_root_backup = trace.output.merkle_root_sig
        trace.output.tensor[..., -1] += 1.0
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.WRONG_MEMBERSHIP
        trace.output.merkle_root_sig = BatchMerkleTree(
            trace.output.tensor, trace.output.keep_last_dim
        ).get_root()
        vop = trace.try_verify_output(trace_ctx)
        assert vop is not None
        assert vop.verify_output() == VerifyResult.VERIFIED
        trace.output.tensor = tensor_backup
        trace.output.merkle_root_sig = merkle_root_backup


if __name__ == "__main__":
    setup()
    test_rms_norm()
    test_linear()
    test_ffn()
    test_attention()
