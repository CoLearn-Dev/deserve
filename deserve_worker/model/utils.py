import pickle
from typing import Any, Optional

import torch
from safetensors.torch import load, save

from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_utils.trace import OpId


def dumps(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> bytes:
    """
    Dump tensors and metadata into bytes
    """

    metadata_bytes = pickle.dumps(metadata)
    sharp_tensors = {}
    for k, v in tensors.items():
        if v.numel() == 0:
            sharp_tensors[f"#{k}"] = torch.ones((1,), dtype=v.dtype)
        else:
            sharp_tensors[k] = v
    tensors_bytes = save(sharp_tensors)
    return (
        len(metadata_bytes).to_bytes(4, byteorder="big")
        + metadata_bytes
        + tensors_bytes
    )


def loads(b: bytes) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Load tensors and metadata from bytes
    """

    metadata_length = int.from_bytes(b[:4], byteorder="big")
    metadata = pickle.loads(b[4 : 4 + metadata_length])
    sharp_tensors = load(b[4 + metadata_length :])
    tensors = {}
    for k, v in sharp_tensors.items():
        if k.startswith("#"):
            tensors[k[1:]] = torch.empty((0,), dtype=v.dtype)
        else:
            tensors[k] = v
    return tensors, metadata


def trace_op(
    ctx: ForwardCtx,
    op_id: OpId,
    output: torch.Tensor,
    op_inputs: Optional[list[OpId]],
) -> None:
    if isinstance(ctx, TraceForwardCtx):
        if op_inputs is None:
            op_inputs = [ctx.last_op_id]
        ctx.traces[op_id] = output
        ctx.output2input[op_id] = op_inputs
        ctx.last_op_id = op_id
