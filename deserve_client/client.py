import pickle
from typing import Any, Optional

import requests
import torch
import typer
from safetensors.torch import load
from transformers import AutoTokenizer  # type: ignore

from deserve_client.llama import CheckCtx, Transformer, VerifyCtx, llama_3_8b_args
from deserve_controller.controller_api import Generation
from deserve_utils.trace import OpId

cli = typer.Typer()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loads(b: bytes) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Load tensors and metadata from bytes
    """

    metadata_length = int.from_bytes(b[:4], byteorder="big")
    metadata = pickle.loads(b[4 : 4 + metadata_length])
    tensors = load(b[4 + metadata_length :])
    return tensors, metadata


@cli.command()
def complete(
    model: str,
    prompt: str,
    dump_probs_num: int = 0,
    entry_point: str = "http://localhost:19000",
) -> None:
    response = requests.post(
        f"{entry_point}/complete",
        json={
            "model": model,
            "prompt": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_seq_len": 2048,
                "dump_probs_num": dump_probs_num,
            },
        },
        stream=True,
    )
    if response.status_code != 200:
        typer.echo("Error")
        return

    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            generation: Generation = pickle.loads(chunk)
            if generation.probs is not None:
                for token, prob in generation.probs:
                    print(f"{token}: {prob}", flush=True)
                print()
            else:
                print(generation.token, end="", flush=True)


@cli.command()
def trace(
    model: str, prompt: str, dump_path: str, entry_point: str = "http://localhost:19000"
) -> None:
    response = requests.post(
        f"{entry_point}/trace_complete",
        json={
            "model": model,
            "prompt": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_seq_len": 2048,
                "dump_probs_num": -1,
            },
        },
        stream=True,
    )
    if response.status_code != 200:
        typer.echo("Error")
        return

    tensors = {}
    next_token = None
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            temp_tensors, metadata = loads(chunk)
            if "token" in metadata:
                next_token = metadata["token"]
            tensors.update(temp_tensors)
    print("Next token:", next_token)
    with open(dump_path, "wb") as f:
        pickle.dump((next_token, metadata["probs"], tensors), f)
    print("Dumped (next_token, probs, tensors) to", dump_path)


@cli.command()
def verify(
    model: str, prompt: str, entry_point: str = "http://localhost:19000"
) -> None:
    response = requests.post(
        f"{entry_point}/trace_complete",
        json={"model": model, "prompt": prompt},
        stream=True,
    )
    if response.status_code != 200:
        typer.echo("Error")
        return
    tensors: dict[str, torch.Tensor] = {}
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            temp_tensors, _ = loads(chunk)
            tensors.update(temp_tensors)

    traces = {OpId.from_str(k): v for k, v in tensors.items()}
    diffs: dict[OpId, float] = {}
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(main_device)
    dtype = torch.float16
    transformer = Transformer(llama_3_8b_args, main_device)
    transformer.check(tokens, CheckCtx(dtype, dtype, traces, diffs, main_device))
    for op_id, diff in diffs.items():
        if diff > 0.03:
            print("Difference found for", op_id)
            if not transformer.verify(
                tokens, VerifyCtx(dtype, dtype, op_id, 0.03, traces, main_device)
            ):
                print("Verification passed for", op_id)
            else:
                print("Verification failed for", op_id)
            return

    print("No difference found for", op_id)


@cli.command()
def dryrun(
    model: str,
    bsz: int,
    prefill_len: int,
    decode_len: int,
    entry_point: str = "http://localhost:19000",
) -> None:
    assert prefill_len > 0
    response = requests.post(
        f"{entry_point}/dryrun",
        json={
            "model": model,
            "bsz": bsz,
            "prefill_len": prefill_len,
            "decode_len": decode_len,
        },
    )
    if response.status_code != 200:
        typer.echo("Error")
        return
    latency = response.json()
    print("Time consumed:", latency * 1000, "ms")


@cli.command()
def test_range(
    model: str, prompt: str, len: int, entry_point: str = "http://localhost:19000"
) -> None:
    max_tensors: dict[str, float] = {}
    min_tensors: dict[str, float] = {}
    for _ in range(len):
        response = requests.post(
            f"{entry_point}/trace",
            json={"model": model, "prompt": prompt},
            stream=True,
        )
        if response.status_code != 200:
            typer.echo("Error")
            return

        tensors = {}
        next_token: Optional[str] = None
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                temp_tensors, metadata = loads(chunk)
                if "token" in metadata:
                    next_token = metadata["token"]
                tensors.update(temp_tensors)
        for k, v in tensors.items():
            max_f = v.flatten().abs().max().item()
            min_f = v.flatten().abs().min().item()
            if k not in max_tensors:
                max_tensors[k] = max_f
            else:
                max_tensors[k] = max(max_tensors[k], max_f)
            if k not in min_tensors:
                min_tensors[k] = min_f
            else:
                min_tensors[k] = min(min_tensors[k], min_f)
        if next_token is not None:
            prompt += next_token
        else:
            print("No more tokens")
            break

    print("Prompt:", prompt)
    for k, maximum in max_tensors.items():
        minimum = min_tensors[k]
        print(f"{k}: {maximum} {minimum}")


if __name__ == "__main__":
    cli()
