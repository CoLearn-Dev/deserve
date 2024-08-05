import pickle
from typing import Any

import requests
import safetensors.torch
import torch
import typer
from transformers import AutoTokenizer  # type: ignore

from deserve_client.model import (
    CheckCtx,
    Transformer,
    VerifyCtx,
    llama_3_8b_args,
    main_device,
)
from deserve_controller.controller_api import app
from deserve_worker.trace import OpId

cli = typer.Typer()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


def loads(b: bytes) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Load tensors and metadata from bytes
    """

    metadata_length = int.from_bytes(b[:4], byteorder="big")
    metadata = pickle.loads(b[4 : 4 + metadata_length])
    tensors = safetensors.torch.load(b[4 + metadata_length :])
    return tensors, metadata


@cli.command()
def complete(model: str, prompt: str, entry_point: str = "http://localhost:19000"):
    response = requests.post(
        f"{entry_point}/complete",
        json={"model": model, "prompt": prompt},
        stream=True,
    )
    if response.status_code != 200:
        typer.echo("Error")
        return

    for chunk in response.iter_content():
        if chunk:
            print(chunk.decode("utf-8"), end="", flush=True)


@cli.command()
def trace(model: str, prompt: str, entry_point: str = "http://localhost:19000"):
    response = requests.post(
        f"{entry_point}/trace",
        json={"model": model, "prompt": prompt},
        stream=True,
    )
    if response.status_code != 200:
        typer.echo("Error")
        return
    
    tensors = {}    
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            temp_tensors, _ = loads(chunk)
            tensors.update(temp_tensors)
    print(list(tensors.keys())) 

@cli.command()
def verify(model: str, prompt: str, entry_point: str = "http://localhost:19000"):
    response = requests.post(
        f"{entry_point}/trace",
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
    transformer = Transformer(llama_3_8b_args)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(main_device)
    result = transformer.forward(tokens, CheckCtx(0.03, traces))
    if isinstance(result, torch.Tensor):
        print("No difference found")
    else: 
        if not transformer.verify(tokens, VerifyCtx(result.op_id, 0.03, traces)): 
            print("Difference found for", result.op_id)
        else: 
            print("Difference found but verification failed")


if __name__ == "__main__":
    cli()
