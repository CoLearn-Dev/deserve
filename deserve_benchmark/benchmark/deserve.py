import argparse
import json
import logging
import multiprocessing
import pickle
import random
import threading
import time
import uuid
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from queue import Queue
from typing import Any, Optional

import requests
import torch
from flask import Flask, request
from safetensors.torch import load, save
from transformers import AutoTokenizer  # type: ignore

from deserve_benchmark.rater import Rater, RaterTimeLimitExceeded, Response
from deserve_benchmark.workload.oasst1 import Oasst1Dataset
from deserve_benchmark.workload.sharegpt import ShareGptDataset
from deserve_benchmark.workload.static import StaticWorkload
from deserve_benchmark.workload.utils import Workload

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

stop_tokens = {128001, 128009}


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
        len(tensors_bytes).to_bytes(4, byteorder="big") + tensors_bytes + metadata_bytes
    )


class DeServeClient:
    def __init__(
        self,
        workload: Workload,
        time_limit: int,
        first_worker_url: str,
        batch_size: int,
        max_tokens: int,
        trace: bool,
        warmup: int,
        variance: int,
    ):
        self.first_worker_url = first_worker_url
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.network_executor = ThreadPoolExecutor(max_workers=128)
        self.deserve_executor = ThreadPoolExecutor(max_workers=128)
        self.time_limit = time_limit
        self.rater = Rater(
            workload=workload, time_limit=time_limit, trace=trace, warmup=warmup
        )
        self.variance = variance

    def flask_service(self, events: Queue[int | None]) -> None:  # type: ignore
        app = Flask(__name__)
        app.logger.setLevel(logging.ERROR)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.route("/update_tasks", methods=["POST"])
        def update_tasks() -> str:
            request_json = request.json
            if request_json is None:
                return "No"
            data: list[dict[str, Any]] = request_json
            for task in data:
                request_id = int(task["task_id"].split("@")[0])
                token = task["output_token"]
                char = tokenizer.decode(token)
                try:
                    self.rater.post(
                        Response(
                            id=request_id, payload=char, finished=(token in stop_tokens)
                        )
                    )
                except RaterTimeLimitExceeded:
                    events.put(None)
                if token in stop_tokens:
                    events.put(request_id)
            return "OK"

        app.run(host="0.0.0.0", port=19000, debug=False)

    def polling(self, queue: Queue[int | None]) -> None:
        current = 0
        while True:
            if current >= self.batch_size:
                value = queue.get()
                if value is None:
                    break
            else:
                current += 1
            history = self.rater.get(1)
            if len(history) == 0:
                break
            id = history[0].id
            prompt = history[0].history
            tokens = tokenizer.encode(prompt, return_tensors="pt")[0]
            tensors = {"x": tokens}
            metadata = {
                "task_id": str(id) + "@" + str(uuid.uuid4()),
                "sampling_params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": self.max_tokens
                    + random.randint(-self.variance, self.variance),
                },
            }
            response = requests.post(
                f"{self.first_worker_url}/prefill",
                data=dumps(tensors, metadata),
            )

    def speedtest(self) -> dict[str, Any]:
        queue: Queue[int | None] = Queue()
        flask_thread = threading.Thread(
            target=self.flask_service, args=[queue], daemon=True
        )
        flask_thread.start()
        polling_thread = threading.Thread(
            target=self.polling, args=[queue], daemon=True
        )
        polling_thread.start()

        try:
            if self.time_limit > 0:
                for _ in range(self.time_limit):
                    time.sleep(1)
                    if not polling_thread.is_alive():
                        break
            else:
                while self.rater.requests_finished_total < self.rater.workload.size():
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        return self.rater.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=-1)
    parser.add_argument(
        "--batch-size", type=int, default=150
    )  # number of concurrent requests that the controller will send to the workers
    parser.add_argument(
        "--max-tokens", type=int, default=1024
    )  # max tokens per request
    parser.add_argument(
        "--workload", type=str, default="oasst1"
    )  # workload name, if it starts with "fixed", then the format is "fixed{size}:{length}:{variance}"
    parser.add_argument("--first-worker-url", type=str, default="http://localhost:8080")
    parser.add_argument("--warmup", type=int, default=0)  # warmup time in seconds
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--variance", type=int, default=0)  # variance of output tokens
    args = parser.parse_args()

    if args.workload == "oasst1":
        workload = Oasst1Dataset().into_workload()
    elif args.workload == "sharegpt":
        workload = ShareGptDataset().into_workload()
    elif args.workload.startswith("fixed"):
        raw = args.workload[len("fixed") :]
        size, length, variance = map(int, raw.split(":"))
        workload = StaticWorkload(size, length, variance)
    else:
        raise ValueError(f"Unknown workload: {args.workload}")
    client = DeServeClient(
        workload=workload,
        time_limit=args.time_limit,
        first_worker_url=args.first_worker_url,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        trace=args.trace,
        warmup=args.warmup,
        variance=args.variance,
    )
    print(json.dumps(client.speedtest()))
