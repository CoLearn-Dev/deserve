from typing import List, Tuple, Optional
from fastapi import FastAPI, HTTPException, Request
from fleece_network import Peer, loads
from pydantic import BaseModel
import anyio
import uvicorn
from .worker import Worker
from .__init__ import __version__
import argparse
import requests
import json
import torch
import concurrent.futures
from anyio.from_thread import BlockingPortal

app = FastAPI()
worker = Worker()


class LayersRequest(BaseModel):
    layer_names: List[str]


def preload_layers(req: LayersRequest):
    try:
        worker.preload_layers(req.layer_names)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


def unload_layers(req: LayersRequest):
    try:
        worker.unload_layers(req.layer_names)
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class ForwardRequest(BaseModel):
    task_id: str
    plan: List[Tuple[str, List[str]]]
    step: int
    round: int = -1
    payload: Optional[List] = None
    max_total_len: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    task_manager_url: Optional[str] = None
    signature: Optional[str] = None
    timestamp: Optional[int] = None


executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)


def forward(req: bytes):
    try:
        tensors, metadata = loads(req)
        if isinstance(metadata, dict):
            executor.submit(
                worker.forward,
                **tensors,
                **metadata,
            )
        elif isinstance(metadata, list):
            executor.submit(
                worker.forward_merged,
                tensors,
                metadata,
            )
        else:
            raise
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def app_forward(request: Request):
    buffer = await request.body()
    try:
        tensors, metadata = loads(buffer)
        if isinstance(metadata, dict):
            executor.submit(
                worker.forward,
                **tensors,
                **metadata,
            )
        elif isinstance(metadata, list):
            executor.submit(
                worker.forward_merged,
                tensors,
                metadata,
            )
        else:
            raise
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


class GetInfoRequest(BaseModel):
    node_list: List[str] = []
    timeout: int = 30


class GetInfoResponse(BaseModel):
    worker_nickname: Optional[str]
    gpu_mem_info: Tuple[int, int] = [0, 0]
    latency_list: List[Optional[float]] = []


def get_info(req: GetInfoRequest) -> GetInfoResponse:
    try:
        worker_nickname, gpu_mem_info, latency_list = worker.get_info(
            req.node_list, req.timeout
        )
        return GetInfoResponse(
            worker_nickname=worker_nickname,
            gpu_mem_info=gpu_mem_info,
            latency_list=latency_list,
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--controller-url")
    parser.add_argument("-w", "--worker-url")
    parser.add_argument("-t", "--api-token")
    parser.add_argument("--port")
    parser.add_argument("--worker-nickname")
    parser.add_argument("--heartbeat-interval")
    args = parser.parse_args()
    if args.worker_url is not None:
        worker_url = args.worker_url
        parsed = worker_url.split(':')
        if len(parsed) >= 3:
            port = int(parsed[2])
        else:
            port = 8080
    else:
        worker_url = "none"
        port = 8080
    if args.port is not None:
        port = int(args.port)
    worker.port = port
    if args.api_token is not None:
        worker.api_token = args.api_token
    if args.worker_nickname is not None:
        worker.worker_nickname = args.worker_nickname
    if args.heartbeat_interval is not None:
        worker.heartbeat_interval = int(args.heartbeat_interval)
    if args.controller_url is not None:
        worker.controller_url = args.controller_url
        data = {"url": worker_url, "version": __version__}
        if worker.worker_nickname is not None:
            data["nickname"] = worker.worker_nickname
        if torch.cuda.is_available():
            model = torch.cuda.get_device_name()
            memory = torch.cuda.mem_get_info()
            data["gpu_model"] = model
            data["gpu_total_memory"] = memory[1]
            data["gpu_remaining_memory"] = memory[0]
        else:
            data["gpu_model"] = "CPU"
            data["gpu_total_memory"] = 0
            data["gpu_remaining_memory"] = 0
        r = requests.post(f"{args.controller_url}/register_worker",
                          json=data,
                          headers={"api-token": worker.api_token})
        res = json.loads(r.content)
        worker.worker_id = res["id"]
        worker.pull_worker_url()
        worker.start_heartbeat_daemon()
        worker.start_layer_forward_engine()

        print("Worker ID: ", worker.worker_id)

        r = requests.get(
            f"{args.controller_url}/get_network_servers",
            headers={"api-token": worker.api_token}
        )

        servers = json.loads(r.content)
        signaling = servers["signaling"]["url"]
        turns = servers["turn"]
        async with BlockingPortal() as portal:
            worker.async_portal = portal
            async with anyio.create_task_group() as tg:
                worker.peer = Peer(
                    worker.worker_id,
                    signaling,
                    [(turn["url"], turn["username"], turn["password"]) for turn in turns],
                    {
                        "preload_layers": preload_layers,
                        "unload_layers": unload_layers,
                        "forward": forward,
                        "get_info": get_info,
                    },
                    tg,
                )

                # start the FastAPI server when public IP is available
                if worker_url != "none":
                    app.add_api_route("/preload_layers", preload_layers, methods=["POST"])
                    app.add_api_route("/unload_layers", unload_layers, methods=["POST"])
                    app.add_api_route("/forward", app_forward, methods=["POST"])
                    app.add_api_route("/get_info", get_info, methods=["POST"])

                    uviconfig = uvicorn.Config(app, host="0.0.0.0", port=port, access_log=False)
                    uviserver = uvicorn.Server(uviconfig)
                    tg.start_soon(uviserver.serve)
            await portal.sleep_until_stopped()
    else:
        worker.worker_id = "local"
        worker.start_layer_forward_engine()
        async with anyio.create_task_group() as tg:
            if worker_url != "none":
                app.add_api_route("/preload_layers", preload_layers, methods=["POST"])
                app.add_api_route("/unload_layers", unload_layers, methods=["POST"])
                app.add_api_route("/forward", app_forward, methods=["POST"])
                app.add_api_route("/get_info", get_info, methods=["POST"])

                uviconfig = uvicorn.Config(app, host="0.0.0.0", port=port, access_log=True)
                uviserver = uvicorn.Server(uviconfig)
                tg.start_soon(uviserver.serve)


if __name__ == '__main__':
    anyio.run(main)
