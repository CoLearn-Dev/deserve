import argparse
import uvicorn
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from pydantic import BaseModel

from .model.llama import loads
from .task import PlanStep, SamplingParams, TaskInfo
from .worker import Worker

app = FastAPI()
worker: Worker
runtime_executor = ThreadPoolExecutor(max_workers=96)


@app.post("/batch_forward")
async def batch_forward(request: Request) -> str:
    body = await request.body()
    tensors, metadata = loads(body)
    runtime_executor.submit(
        worker.batch_forward,
        tensors["x"],
        [TaskInfo.model_validate(task_info) for task_info in metadata["task_infos"]],
    )
    return "ok"


@app.post("/forward")
async def forward(request: Request) -> str:
    try:
        body = await request.body()
        tensors, metadata = loads(body)
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
        args.id, f"http://localhost:{args.port}", args.batch_size, args.controller_url
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)
