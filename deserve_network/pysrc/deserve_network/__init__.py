import pickle
from typing import Any, Optional

import torch

from ._deserve_network_rust import PyClient, PyServer
from .sede import _flatten, _view2torch


class Server:
    def __init__(
        self,
        address: str,
        routes: list[str],
        worker_threads: int = 64,
        enable_logging: bool = False,
    ):
        self.server = PyServer(address, routes, worker_threads, enable_logging)

    def send_tensors(
        self,
        address: str,
        tensor_dict: dict[str, torch.Tensor],
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if metadata is None:
            metadata = {}
        pickled_metadata = pickle.dumps(metadata)
        return self.server.send_tensors(
            address, _flatten(tensor_dict), pickled_metadata
        )

    def recv_tensors(self) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
        route, tensors, metadata = self.server.recv_tensors()
        return route, _view2torch(tensors), pickle.loads(metadata)


class Client:
    def __init__(self, worker_threads: int = 64):
        self.client = PyClient(worker_threads)

    def send_tensors(
        self,
        address: str,
        tensor_dict: dict[str, torch.Tensor],
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if metadata is None:
            metadata = {}
        pickled_metadata = pickle.dumps(metadata)
        return self.client.send_tensors(
            address, _flatten(tensor_dict), pickled_metadata
        )
