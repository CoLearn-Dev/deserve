import argparse
import logging
import pickle
import queue
import traceback
import uuid
from typing import Any, Generator, Optional

import requests
import safetensors.torch
import torch
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

controller_url: str
app = FastAPI()
logger = logging.getLogger("uvicorn")
workers: TTLCache[str, str] = TTLCache(maxsize=128, ttl=2)
model2layers = {
    "meta-llama/Meta-Llama-3-70B-Instruct": 80,
    "meta-llama/Meta-Llama-3-8B-Instruct": 32,
}
model2alias = {
    "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b-instruct-slice",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-instruct-slice",
}
token_channels: dict[str, queue.Queue[Optional[str]]] = {}
trace_channels: dict[
    str, queue.Queue[Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]]
] = {}
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

STOP_TOKEN_IDS = [128001, 128009]


def dumps(tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> bytes:
    """
    Dump tensors and metadata into bytes
    """

    metadata_bytes = pickle.dumps(metadata)
    tensors_bytes = safetensors.torch.save(tensors)
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
    tensors = safetensors.torch.load(b[4 + metadata_length :])
    return tensors, metadata


class RegisterRequest(BaseModel):
    worker_id: str
    worker_url: str


@app.post("/register")
def register(request: RegisterRequest) -> str:
    workers[request.worker_id] = request.worker_url
    return "ok"


class HeartbeatRequest(BaseModel):
    worker_id: str
    worker_url: str


@app.post("/heartbeat")
def heartbeat(request: HeartbeatRequest) -> str:
    workers[request.worker_id] = request.worker_url
    return "ok"


class CompleteRequest:
    pass  # discuss about implementation details (how to send, how to retrieve)


class PlanStep(BaseModel):
    worker_id: str
    worker_url: str
    layers: list[str]


def generate_plan(model: str, worker_ids: list[str]) -> list[PlanStep]:
    worker_ids.sort()
    alias = model2alias[model]
    num_layer_total = model2layers[model]
    num_layer_worker = num_layer_total // len(worker_ids)
    layers = [
        (i * num_layer_worker, (i + 1) * num_layer_worker)
        for i in range(len(worker_ids) - 1)
    ]
    if len(layers) == 0:
        layers.append((0, num_layer_total))
    else:
        layers.append((layers[-1][1], num_layer_total))
    plans: list[PlanStep] = []
    for worker_id, layer in zip(worker_ids, layers):
        plans.append(
            PlanStep(
                worker_id=worker_id,
                worker_url=workers[worker_id],
                layers=[f"{alias}/layers.{i}" for i in range(layer[0], layer[1])],
            )
        )
    plans[0].layers.insert(0, f"{alias}/tok_embeddings")
    plans[-1].layers.append(f"{alias}/norm")
    plans[-1].layers.append(f"{alias}/output")
    return plans


def relay_tokens(
    channel: queue.Queue[Optional[str]],
) -> Generator[bytes, None, None]:
    while True:
        value = channel.get()
        if value is None:
            break
        yield value.encode("utf-8")


class OnlineCompleteRequest(BaseModel):
    model: str
    prompt: str


@app.post("/complete")
def complete(request: OnlineCompleteRequest) -> StreamingResponse:
    model = request.model
    prompt = request.prompt

    if model not in model2layers:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = str(uuid.uuid4())

    # init channel for relay
    token_channel = queue.Queue[Optional[str]]()
    token_channels[task_id] = token_channel

    # generate request
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    plan = generate_plan(model, list(workers.keys()))
    tensors = {"x": tokens}
    metadata = {
        "task_id": task_id,
        "round": 0,
        "plan": plan,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_total_len": 2048},
    }
    first_worker_url = plan[0].worker_url
    response = requests.post(
        f"{first_worker_url}/forward", data=dumps(tensors, metadata)
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Worker error")

    return StreamingResponse(relay_tokens(token_channel))


class OfflineCompleteRequest(BaseModel):
    model: str
    prompts: list[str]


@app.post("/offline-complete")
def offline_complete(request: OfflineCompleteRequest) -> None:
    pass


def relay_traces(
    channel: queue.Queue[Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]],
    total: int,
) -> Generator[bytes, None, None]:
    cnt = 0
    while cnt < total:
        value = channel.get()
        cnt += 1
        if value is None:
            break
        tensors, metadata = value
        bytes = dumps(tensors, metadata)
        yield bytes


class TraceRequest(BaseModel):
    model: str
    prompt: str


@app.post("/trace")
def trace(request: TraceRequest) -> Response:
    model = request.model
    prompt = request.prompt

    if model not in model2layers:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = str(uuid.uuid4())

    # init channel for relay, but we don't handle it inside tracing
    token_channel = queue.Queue[Optional[str]]()
    token_channels[task_id] = token_channel

    # init traces
    trace_channel = queue.Queue[
        Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]
    ]()
    trace_channels[task_id] = trace_channel

    # generate request
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    online_workers = list(workers.keys())
    plan = generate_plan(model, online_workers)
    tensors = {"x": tokens}
    metadata = {
        "task_id": task_id,
        "round": 0,
        "plan": plan,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_total_len": 2048},
    }
    first_worker_url = plan[0].worker_url
    response = requests.post(f"{first_worker_url}/trace", data=dumps(tensors, metadata))
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Worker error")
    return StreamingResponse(relay_traces(trace_channel, len(online_workers)))


class UpdateTaskRequest(BaseModel):
    task_id: str
    output_tokens: list[int]  # [bsz, seqlen], in normal case, bsz=1 and seqlen=1


@app.post("/update_tasks")
def update_tasks(requests: list[UpdateTaskRequest]) -> None:
    for request in requests:
        task_id = request.task_id
        for token_id in request.output_tokens:
            if token_id in STOP_TOKEN_IDS:
                token_channels[task_id].put(None)
            else:
                token = tokenizer.decode(token_id)
                if task_id in token_channels:
                    token_channels[task_id].put(token)
                else:
                    logger.warning(f"Task {task_id} not found")


@app.post("/update_traces")
async def update_traces(requests: Request) -> None:
    body = await requests.body()
    tensors, metadata = loads(body)
    task_id = metadata["task_id"]
    if task_id in trace_channels:
        if "token" in metadata:
            trace_channels[task_id].put(
                (tensors, {"token": tokenizer.decode(metadata["token"])})
            )
        else:
            trace_channels[task_id].put((tensors, {}))
    else:
        logger.warning(f"Task {task_id} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19000)
    args = parser.parse_args()

    controller_url = f"http://localhost:{args.port}"

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)
