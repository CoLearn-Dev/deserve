import argparse
import asyncio
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request

from deserve_network import Server
from deserve_worker.engine.pipeline.processor import PipelineProcessor
from deserve_worker.engine.pipeline.scheduler import PipelineScheduler
from deserve_worker.request import (
    DecodeRequest,
    InitRequest,
    LLMRequest,
    PrefillRequest,
    StepRequest,
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
llm_engine: PipelineProcessor
reorder_buffer: dict[int, StepRequest] = {}
current_round = 0
simulated_latency = 0.0
latency_simulator = ThreadPoolExecutor(max_workers=64)
lock = threading.Lock()
counter = 0


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


def prefill(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    task_id = metadata["task_id"]
    sampling_params = SamplingParams.model_validate(metadata["sampling_params"])
    llm_engine.add_request(
        InitRequest(
            x=tensors["x"].to(main_device),
            task_id=task_id,
            sampling_params=sampling_params,
        )
    )


def add_request(request: StepRequest) -> None:
    if simulated_latency > 0:
        time.sleep(simulated_latency)
    with lock:
        global current_round
        if request.microbatch_id == current_round:
            llm_engine.add_request(request)
            current_round = (current_round + 1) % llm_engine.num_rounds
            while current_round in reorder_buffer:
                llm_engine.add_request(reorder_buffer.pop(current_round))
                current_round = (current_round + 1) % llm_engine.num_rounds
        else:
            reorder_buffer[request.microbatch_id] = request


def step(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    # global counter
    # counter += 1
    # print(f"Step {counter}")
    # print(f"Received {time.time() * 1000}")
    group_id = metadata["group_id"]
    exec_task_ids = metadata["exec_task_ids"]
    exec_seqlens = metadata["exec_seqlens"]
    cancel_task_ids = metadata["cancel_task_ids"]
    offload_task_ids = metadata["offload_task_ids"]
    reload_task_ids = metadata["reload_task_ids"]
    init_tasks = metadata["init_tasks"]
    llm_request = StepRequest(
        microbatch_id=group_id,
        xs=tensors["xs"].to(main_device),
        exec_task_ids=exec_task_ids,
        exec_seqlens=exec_seqlens,
        cancel_task_ids=cancel_task_ids,
        offload_task_ids=offload_task_ids,
        reload_task_ids=reload_task_ids,
        init_tasks=init_tasks,
    )
    latency_simulator.submit(add_request, llm_request)


def trace(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
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
    parser.add_argument("--num-main-pages", type=int)
    parser.add_argument("--num-swap-pages", type=int)
    parser.add_argument("--simulated-latency", type=float, default=0.0)
    parser.add_argument("--prefill-first-aggregate", action="store_true")
    parser.add_argument("--decode-first-aggregate", action="store_true")
    parser.add_argument("--buddy-height", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
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

    simulated_latency = args.simulated_latency
    print(f"Simulated latency: {args.simulated_latency * 1000}ms")

    worker_url = f"http://localhost:{args.port}"
    server = Server(f"0.0.0.0:{args.port}", ["/prefill", "/step", "/trace"], 64)
    if layer_begin == 0:
        llm_engine = PipelineScheduler(
            args.num_rounds,
            args.num_main_pages,
            args.num_swap_pages,
            8,
            args.batch_size,
            layers[layer_begin:layer_end],
            next_worker_url=args.next_worker_url,
            controller_url=args.controller_url,
            worker_url=worker_url,
            prefill_first_aggregate=args.prefill_first_aggregate,
            decode_first_aggregate=args.decode_first_aggregate,
            buddy_height=args.buddy_height,
            ignore_eos=args.ignore_eos,
            server=server,
        )
    else:
        llm_engine = PipelineProcessor(
            args.num_rounds,
            args.num_main_pages,
            args.num_swap_pages,
            8,
            args.batch_size,
            layers[layer_begin:layer_end],
            next_worker_url=args.next_worker_url,
            controller_url=args.controller_url,
            worker_url=worker_url,
            buddy_height=args.buddy_height,
            ignore_eos=args.ignore_eos,
            server=server,
        )
    threading.Thread(target=llm_engine.run, daemon=True).start()
    threading.Thread(target=llm_engine.heartbeat, daemon=True).start()
    threading.Thread(target=llm_engine.log, daemon=True).start()
    while True:
        route, tensors, metadata = server.recv_tensors()
        if route == "/prefill":
            prefill(tensors, metadata)
        elif route == "/step":
            step(tensors, metadata)
        elif route == "/trace":
            trace(tensors, metadata)
