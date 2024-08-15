import pickle
from typing import Any

import torch
from safetensors.torch import load, save

from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.model.context.trace import TraceForwardCtx
from deserve_worker.trace import OpId


def dumps(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> bytes:
    """
    Dump tensors and metadata into bytes
    """

    metadata_bytes = pickle.dumps(metadata)
    tensors_bytes = save(tensors)
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
    tensors = load(b[4 + metadata_length :])
    return tensors, metadata


def trace_op(ctx: ForwardCtx, op_id: OpId, op_value: torch.Tensor) -> None:
    if isinstance(ctx, TraceForwardCtx):
        ctx.traces[op_id] = op_value
