import argparse
import threading
import traceback
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request

from deserve_worker.engine.processor import Processor
from deserve_worker.engine.scheduler import Scheduler
from deserve_worker.request import (
    DecodeRequest,
    JoinRequest,
    PrefillRequest,
    TraceRequest,
)
from deserve_worker.task import SamplingParams, main_device

from .model.utils import loads

llama3_70b_layers = [
    "llama-3-70b-instruct-slice/tok_embeddings",
    *[f"llama-3-70b-instruct-slice/layers.{i}" for i in range(0, 80)],
    "llama-3-70b-instruct-slice/norm",
    "llama-3-70b-instruct-slice/output",
]
llama3_8b_layers = [
    "llama-3-8b-instruct-slice/tok_embeddings",
    *[f"llama-3-8b-instruct-slice/layers.{i}" for i in range(0, 32)],
    "llama-3-8b-instruct-slice/norm",
    "llama-3-8b-instruct-slice/output",
]
app = FastAPI()
llm_engine: Processor


def convert_name_to_id(name: str, max_layer: int) -> int:
    if name == "emb":
        return 0
    elif name.isdigit():
        return int(name) + 1
    elif name == "norm":
        return max_layer - 1
    elif name == "output":
        return max_layer
    else:
        raise ValueError("Invalid layer name")


@app.post("/prefill")
async def prefill(request: Request) -> None:
    body = await request.body()
    tensors, metadata = loads(body)
    task_id = metadata["task_id"]
    sampling_params = SamplingParams.model_validate(metadata["sampling_params"])
    llm_engine.add_request(
        PrefillRequest(
            x=tensors["x"].to(main_device),
            task_id=task_id,
            sampling_params=sampling_params,
        )
    )


@app.post("/decode")
async def decode(request: Request) -> None:
    body = await request.body()
    tensors, metadata = loads(body)
    group_id = metadata["group_id"]
    exec_task_ids = metadata["exec_task_ids"]
    offload_task_ids = metadata["offload_task_ids"]
    reload_task_ids = metadata["reload_task_ids"]
    cancel_task_ids = metadata["cancel_task_ids"]
    r2s_task_ids = metadata["r2s_task_ids"]
    e2s_task_ids = metadata["e2s_task_ids"]
    s2e_task_ids = metadata["s2e_task_ids"]
    r2e_task_ids = metadata["r2e_task_ids"]
    llm_engine.add_request(
        DecodeRequest(
            group_id=group_id,
            xs=tensors["xs"].to(main_device),
            exec_task_ids=exec_task_ids,
            offload_task_ids=offload_task_ids,
            reload_task_ids=reload_task_ids,
            cancel_task_ids=cancel_task_ids,
            r2s_task_ids=r2s_task_ids,
            e2s_task_ids=e2s_task_ids,
            s2e_task_ids=s2e_task_ids,
            r2e_task_ids=r2e_task_ids,
        )
    )


@app.post("/join")
async def join(request: Request) -> None:
    body = await request.body()
    tensors, metadata = loads(body)
    task_id = metadata["task_id"]
    llm_engine.add_request(JoinRequest(x=tensors["x"].to(main_device), task_id=task_id))


@app.post("/trace")
async def trace(request: Request) -> None:
    body = await request.body()
    tensors, metadata = loads(body)
    task_id = metadata["task_id"]
    sampling_params = SamplingParams.model_validate(metadata["sampling_params"])
    sampling_params.dump_probs_num = (
        -1
    )  # override dump_probs_num to -1 for full tracing
    llm_engine.add_request(
        TraceRequest(
            x=tensors["x"].to(main_device),
            task_id=task_id,
            sampling_params=sampling_params,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument("--layer-begin", type=str)
    parser.add_argument("--layer-end", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str)
    parser.add_argument("--next-worker-url", type=str)
    args = parser.parse_args()

    if args.model == "llama-3-70b":
        layers = llama3_70b_layers
    elif args.model == "llama-3-8b":
        layers = llama3_8b_layers
    else:
        raise ValueError("Invalid model")
    layer_begin = convert_name_to_id(args.layer_begin, len(layers))
    layer_end = convert_name_to_id(args.layer_end, len(layers))
    print(f"Serve from {layers[layer_begin]} to {layers[layer_end - 1]}")
    worker_url = f"http://localhost:{args.port}"
    if layer_begin == 0:
        llm_engine = Scheduler(
            args.num_rounds,
            9000,
            8,
            args.batch_size,
            layers[layer_begin:layer_end],
            next_worker_url=args.next_worker_url,
            controller_url=args.controller_url,
            worker_url=worker_url,
        )
    else:
        llm_engine = Processor(
            args.num_rounds,
            9000,
            8,
            args.batch_size,
            layers[layer_begin:layer_end],
            next_worker_url=args.next_worker_url,
            controller_url=args.controller_url,
            worker_url=worker_url,
        )
    threading.Thread(target=llm_engine.run, daemon=True).start()
    threading.Thread(target=llm_engine.heartbeat, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
