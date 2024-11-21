import sys
from collections import defaultdict
from typing import Any, Dict, List, Set

import torch

# torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
_float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
_float8_e5m2 = getattr(torch, "float8_e5m2", None)

_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
    _float8_e4m3fn: 1,
    _float8_e5m2: 1,
}

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": _float8_e4m3fn,
    "F8_E5M2": _float8_e5m2,
}


def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]


def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        if len(v["data"]) == 0:
            # Workaround because frombuffer doesn't accept zero-size tensors
            assert any(x == 0 for x in v["shape"])
            arr = torch.empty(v["shape"], dtype=dtype)
        else:
            arr = torch.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        if sys.byteorder == "big":
            arr = torch.from_numpy(arr.numpy().byteswap(inplace=False))
        result[k] = arr

    return result


def storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _SIZE[tensor.dtype]
    else:
        stop = tensor.data_ptr()
    return stop


def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _SIZE[tensor.dtype]


def _filter_shared_not_shared(
    tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]
) -> List[Set[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors


def _find_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> List[Set[str]]:
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if (
            v.device != torch.device("meta")
            and storage_ptr(v) != 0
            and storage_size(v) != 0
        ):
            # Need to add device as key because of multiple GPU.
            tensors[(v.device, storage_ptr(v), storage_size(v))].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors


def _tobytes(tensor: torch.Tensor, name: str) -> bytes:
    if tensor.layout != torch.strided:
        raise ValueError(
            f"You are trying to save a sparse tensor: `{name}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    if not tensor.is_contiguous():
        raise ValueError(
            f"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you"
            " are trying to save tensors which are reference of each other in which case it's recommended to save"
            " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to"
            " pack it before saving."
        )
    if tensor.device.type != "cpu":
        # Moving tensor to cpu before saving
        tensor = tensor.to("cpu")

    import ctypes

    import numpy as np

    # When shape is empty (scalar), np.prod returns a float
    # we need a int for the following calculations
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = _SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    if ptr == 0:
        return b""
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    if sys.byteorder == "big":
        NPDTYPES = {
            torch.int64: np.int64,
            torch.float32: np.float32,
            torch.int32: np.int32,
            # XXX: This is ok because both have the same width
            torch.bfloat16: np.float16,
            torch.float16: np.float16,
            torch.int16: np.int16,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.bool: bool,
            torch.float64: np.float64,
            # XXX: This is ok because both have the same width and byteswap is a no-op anyway
            _float8_e4m3fn: np.uint8,
            _float8_e5m2: np.uint8,
        }
        npdtype = NPDTYPES[tensor.dtype]
        # Not in place as that would potentially modify a live running model
        data = data.view(npdtype).byteswap(inplace=False)
    return data.tobytes()


def _flatten(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(tensors, dict):
        raise ValueError(
            f"Expected a dict of [str, torch.Tensor] but received {type(tensors)}"
        )

    invalid_tensors = []
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(
                f"Key `{k}` is invalid, expected torch.Tensor but received {type(v)}"
            )

        if v.layout != torch.strided:
            invalid_tensors.append(k)
    if invalid_tensors:
        raise ValueError(
            f"You are trying to save a sparse tensors: `{invalid_tensors}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    shared_pointers = _find_shared_tensors(tensors)
    failing = []
    for names in shared_pointers:
        if len(names) > 1:
            failing.append(names)

    if failing:
        raise RuntimeError(
            f"""
            Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: {failing}.
            A potential way to correctly save your model is to use `save_model`.
            More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
            """
        )

    return {
        k: {
            "dtype": str(v.dtype).split(".")[-1],
            "shape": v.shape,
            "data": _tobytes(v, k),
        }
        for k, v in tensors.items()
    }
