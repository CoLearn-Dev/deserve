import argparse
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from .model.utils import loads
from .task import PlanStep, SamplingParams, TaskInfo
from .worker import Worker

app = FastAPI()
worker: Worker
runtime_executor = ThreadPoolExecutor(max_workers=96)

pending_tasks = deque[tuple[dict[str, torch.Tensor], dict[str, Any]]]()
mutex = (
    threading.Lock()
)  # may not be necessary if the async worker has only one thread?
last_check_pending_time = 0.0


def check_pending() -> None:
    with mutex:
        print(
            "check_pending",
            len(pending_tasks),
            worker.page_pool.num_avails,
            worker.page_pool.num_blocks,
        )
        global last_check_pending_time
        if last_check_pending_time + 1.0 > time.time():
            return
        last_check_pending_time = time.time()
        num_avails = worker.page_pool.num_avails
        while (
            len(pending_tasks) > 0 and num_avails / worker.page_pool.num_blocks >= 0.2
        ):
            tensors, metadata = pending_tasks.popleft()
            num_avails -= tensors["x"].shape[0] // worker.page_pool.block_size + 1
            runtime_executor.submit(
                worker.forward,
                tensors["x"],
                metadata["task_id"],
                metadata["round"],
                [PlanStep.model_validate(step) for step in metadata["plan"]],
                SamplingParams.model_validate(metadata["sampling_params"]),
            )


@app.post("/batch_forward")
async def batch_forward(request: Request) -> str:
    try:
        body = await request.body()
        tensors, metadata = loads(body)
        if worker.worker_id == metadata["task_infos"][0]["plan"][0]["worker_id"]:
            check_pending()
        runtime_executor.submit(
            worker.batch_forward,
            tensors["x"],
            [
                TaskInfo.model_validate(task_info)
                for task_info in metadata["task_infos"]
            ],
        )
    except Exception as e:
        traceback.print_exc()
    return "ok"


@app.post("/forward")
async def forward(request: Request) -> str:
    try:
        body = await request.body()
        tensors, metadata = loads(body)
        if worker.worker_id == metadata["plan"][0]["worker_id"]:
            check_pending()
            if worker.page_pool.num_avails / worker.page_pool.num_blocks < 0.2:
                pending_tasks.append((tensors, metadata))
                return "ok"
        runtime_executor.submit(
            worker.forward,
            tensors["x"],
            metadata["task_id"],
            metadata["round"],
            [PlanStep.model_validate(step) for step in metadata["plan"]],
            SamplingParams.model_validate(metadata["sampling_params"]),
        )
    except Exception as e:
        traceback.print_exc()
    return "ok"


@app.post("/trace")
async def trace(request: Request) -> str:
    try:
        body = await request.body()
        tensors, metadata = loads(body)
        runtime_executor.submit(
            worker.trace,
            tensors["x"],
            metadata["task_id"],
            metadata["round"],
            [PlanStep.model_validate(step) for step in metadata["plan"]],
            SamplingParams.model_validate(metadata["sampling_params"]),
        )
    except Exception as e:
        traceback.print_exc()
    return "ok"


class CancelRequest(BaseModel):
    task_id: str
    start_index: int
    plan: list[PlanStep]


@app.post("/cancel")
async def cancel(request: CancelRequest) -> str:
    runtime_executor.submit(
        worker.cancel, request.task_id, request.start_index, request.plan
    )
    return "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str)
    args = parser.parse_args()

    worker = Worker(
        args.id,
        f"http://localhost:{args.port}",
        args.batch_size,
        args.controller_url,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
