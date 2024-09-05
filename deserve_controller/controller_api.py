import argparse
import logging
import pickle
import queue
import uuid
from typing import Any, Generator, Optional

import requests
import torch
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from safetensors.torch import load, save
from transformers import AutoTokenizer  # type: ignore

controller_url: str
app = FastAPI()
logger = logging.getLogger("uvicorn")
next_workers: TTLCache[str, str] = TTLCache(maxsize=128, ttl=2)
leaders: TTLCache[str, str] = TTLCache(maxsize=128, ttl=2)
model2layers = {
    "meta-llama/Meta-Llama-3-70B-Instruct": 80,
    "meta-llama/Meta-Llama-3-8B-Instruct": 32,
}
model2alias = {
    "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b-instruct-slice",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-instruct-slice",
}
trace_channels: dict[
    str, queue.Queue[Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]]
] = {}
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

STOP_TOKEN_IDS = [128001, 128009]


class Generation(BaseModel):
    token: str
    probs: Optional[list[tuple[str, float]]]


token_channels: dict[str, queue.Queue[Optional[Generation]]] = {}


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


class HeartbeatRequest(BaseModel):
    worker_url: str
    is_start: bool
    next_worker_url: str


@app.post("/heartbeat")
def heartbeat(request: HeartbeatRequest) -> str:
    if request.is_start:
        leaders[request.worker_url] = request.next_worker_url
    else:
        next_workers[request.worker_url] = request.next_worker_url
    return "ok"


class PlanStep(BaseModel):
    worker_id: str
    worker_url: str
    layers: list[str]


def relay_tokens(
    channel: queue.Queue[Optional[Generation]],
) -> Generator[bytes, None, None]:
    while True:
        value = channel.get()
        if value is None:
            break
        yield pickle.dumps(value)


class CompleteRequest(BaseModel):
    model: str
    prompt: str


@app.post("/complete")
def complete(request: CompleteRequest) -> StreamingResponse:
    model = request.model
    prompt = request.prompt

    if model not in model2layers:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = str(uuid.uuid4())

    # init channel for relay
    token_channel = queue.Queue[Optional[Generation]]()
    token_channels[task_id] = token_channel

    # generate request
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    tensors = {"x": tokens}
    metadata = {
        "task_id": task_id,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_seq_len": 2048,
            "dump_probs_num": 10,
        },
    }
    first_worker_url = list(leaders.keys())[0]
    response = requests.post(
        f"{first_worker_url}/prefill", data=dumps(tensors, metadata)
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Worker error")

    return StreamingResponse(relay_tokens(token_channel))


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    model = request.model
    messages = request.messages

    if model not in model2layers:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = str(uuid.uuid4())

    # init channel for relay
    token_channel = queue.Queue[Optional[Generation]]()
    token_channels[task_id] = token_channel

    # generate request
    tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )[0]
    tensors = {"x": tokens}
    metadata = {
        "task_id": task_id,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_seq_len": 2048},
    }
    first_worker_url = list(leaders.keys())[0]
    response = requests.post(
        f"{first_worker_url}/prefill", data=dumps(tensors, metadata)
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Worker error")

    return StreamingResponse(relay_tokens(token_channel))


def relay_traces(
    channel: queue.Queue[Optional[tuple[dict[str, torch.Tensor], dict[str, Any]]]],
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


@app.post("/trace_complete")
def trace_complete(request: CompleteRequest) -> Response:
    model = request.model
    prompt = request.prompt

    if model not in model2layers:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = str(uuid.uuid4())

    # init channel for relay, but we don't handle it inside tracing
    token_channel = queue.Queue[Optional[Generation]]()
    token_channels[task_id] = token_channel

    # init traces
    trace_channel = queue.Queue[
        Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]
    ]()
    trace_channels[task_id] = trace_channel

    # generate request
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    online_workers = list(leaders.keys())
    # plan = generate_plan(model, online_workers)
    tensors = {"x": tokens}
    metadata = {
        "task_id": task_id,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_seq_len": 2048},
    }
    first_worker_url = online_workers[0]
    response = requests.post(f"{first_worker_url}/trace", data=dumps(tensors, metadata))
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Worker error")
    return StreamingResponse(relay_traces(trace_channel, len(online_workers)))


@app.post("/trace_chat")
def trace_chat(request: ChatRequest) -> Response:
    model = request.model
    messages = request.messages

    if model not in model2layers:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = str(uuid.uuid4())

    # init channel for relay, but we don't handle it inside tracing
    token_channel = queue.Queue[Optional[Generation]]()
    token_channels[task_id] = token_channel

    # init traces
    trace_channel = queue.Queue[
        Optional[tuple[dict[str, torch.Tensor], dict[str, str]]]
    ]()
    trace_channels[task_id] = trace_channel

    # generate request
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
    tokens = tokens[: tokens.shape[0] - 1]
    online_workers = list(leaders.keys())
    # plan = generate_plan(model, online_workers)
    tensors = {"x": tokens}
    metadata = {
        "task_id": task_id,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_seq_len": 2048},
    }
    first_worker_url = online_workers[0]
    response = requests.post(f"{first_worker_url}/trace", data=dumps(tensors, metadata))
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Worker error")
    return StreamingResponse(relay_traces(trace_channel, len(online_workers)))


class UpdateTaskRequest(BaseModel):
    task_id: str
    output_token: int  # [seqlen], in normal case, seqlen=1
    needed_probs: Optional[list[tuple[int, float]]]


@app.post("/update_tasks")
def update_tasks(requests: list[UpdateTaskRequest]) -> None:
    for request in requests:
        task_id = request.task_id
        token_id = request.output_token
        token = tokenizer.decode(token_id)
        if request.needed_probs is not None:
            probs = [(tokenizer.decode(token), p) for token, p in request.needed_probs]
        else:
            probs = None
        if task_id in token_channels:
            token_channels[task_id].put(Generation(token=token, probs=probs))
        else:
            logger.warning(f"Task {task_id} not found")


@app.post("/update_traces")
async def update_traces(requests: Request) -> None:
    body = await requests.body()
    tensors, metadata = loads(body)
    task_id = metadata["task_id"]
    if task_id in trace_channels:
        trace_channels[task_id].put(
            (
                tensors,
                {
                    "output2input": metadata["output2input"],
                    "probs": metadata["probs"],
                },
            )
        )
    else:
        logger.warning(f"Task {task_id} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19000)
    args = parser.parse_args()

    controller_url = f"http://localhost:{args.port}"

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
