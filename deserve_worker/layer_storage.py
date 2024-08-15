import os
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import torch
from tqdm import tqdm

from deserve_worker.model.args import (
    ModelArgs,
    llama_2_7b_args,
    llama_2_13b_args,
    llama_2_70b_args,
    llama_3_8b_args,
    llama_3_70b_args,
)
from deserve_worker.model.context.forward import ForwardCtx
from deserve_worker.model.layer.linear import TraceEmbedding, TraceLinear
from deserve_worker.model.layer.norm import RMSNorm
from deserve_worker.model.llama import TransformerBlock
from deserve_worker.task import TaskData
from deserve_worker.trace import ComponentId, LayerId, OpId

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LayerManager:
    def __init__(self, main_device: torch.device):
        self.main_device = main_device
        self.network_executor = ThreadPoolExecutor(max_workers=80)
        self.cache_dir = os.path.expanduser("~/.cache/fleece-worker/models")
        self.layer_storages: dict[frozenset[str], LayerStorage] = {}
        self.layers: dict[str, torch.nn.Module] = {}
        self.mutex = threading.Lock()

    def get_model_args(self, model_name: str) -> ModelArgs:
        if model_name.startswith("llama-2-7b"):
            return llama_2_7b_args
        elif model_name.startswith("llama-2-13b"):
            return llama_2_13b_args
        elif model_name.startswith("llama-2-70b"):
            return llama_2_70b_args
        elif model_name.startswith("llama-3-8b"):
            return llama_3_8b_args
        elif model_name.startswith("llama-3-70b"):
            return llama_3_70b_args
        else:
            raise NotImplementedError("Unknown model")

    def get_layer_storage(self, layer_names: list[str]) -> "LayerStorage":
        model_name, _ = layer_names[0].split("/")
        model_args = self.get_model_args(model_name)

        frozen_layer_names = frozenset(layer_names)
        if frozen_layer_names not in self.layer_storages:
            with self.mutex:
                self.layer_storages[frozen_layer_names] = LayerStorage(
                    self.preload_layers(layer_names, model_args),
                    model_args,
                    self.main_device,
                )
        return self.layer_storages[frozen_layer_names]

    def fetch_layer(self, full_layer_name: str, model_args: ModelArgs) -> None:
        model_name, layer_name = full_layer_name.split("/")

        path = os.path.join(self.cache_dir, model_name, f"{layer_name}.pt")
        if not os.path.exists(path):  # TODO lock
            os.makedirs(os.path.join(self.cache_dir, model_name), exist_ok=True)
            with requests.get(
                f"https://huggingface.co/colearn/{model_name}/resolve/main/{layer_name}.pt",
                stream=True,
            ) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        l: torch.nn.Module
        if layer_name == "tok_embeddings":
            l = torch.nn.utils.skip_init(
                TraceEmbedding,
                ComponentId("tok_embeddings", "main"),
                model_args.vocab_size,
                model_args.dim,
            )  # type: ignore
        elif layer_name.startswith("layer"):
            l = TransformerBlock(LayerId(f"{int(layer_name[7:]):02}"), model_args)
        elif layer_name == "norm":
            l = RMSNorm(
                ComponentId("norm", "main"),
                model_args.dim,
                eps=model_args.norm_eps,
            )
        elif layer_name == "output":
            l = torch.nn.utils.skip_init(
                TraceLinear,
                ComponentId("output", "main"),
                model_args.dim,
                model_args.vocab_size,
            )  # type: ignore
        else:
            raise NotImplementedError("Unknown layers")

        l.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        l.to(self.main_device)
        self.layers[full_layer_name] = l

    def preload_layers(
        self, full_layer_names: list[str], model_args: ModelArgs
    ) -> dict[str, torch.nn.Module]:
        threads: list[Future[None]] = []
        result = {}
        for full_layer_name in full_layer_names:
            if full_layer_name not in self.layers:
                thread = self.network_executor.submit(
                    self.fetch_layer, full_layer_name, model_args
                )
                threads.append(thread)
        with tqdm(total=len(threads)) as pbar:
            for thread in as_completed(threads):
                thread.result()
                pbar.update(1)
        for full_layer_name in full_layer_names:
            result[full_layer_name] = self.layers[full_layer_name]
        return result


class LayerStorage:
    def __init__(
        self,
        layers: dict[str, torch.nn.Module],
        model_args: ModelArgs,
        main_device: torch.device,
    ):
        self.main_device = main_device
        self.layers = layers
        self.model_args = model_args
        self.need_sample = any(
            [full_layer_name.split("/")[1] == "output" for full_layer_name in layers]
        )

    def clear(self) -> None:
        self.layers.clear()

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, ctx: ForwardCtx) -> torch.Tensor:
        for full_layer_name in self.layers:
            _, layer_name = full_layer_name.split("/")
            if layer_name == "tok_embeddings":
                h = self.layers[full_layer_name](h, ctx)
            elif layer_name.startswith("layers."):
                h = self.layers[full_layer_name](h, ctx)
                ctx.layer_id += 1
            elif layer_name == "norm":
                h = self.layers[full_layer_name](h, ctx)
            elif layer_name == "output":
                h = self.layers[full_layer_name](h, ctx)
            else:
                raise NotImplementedError("Unknown layers")
        return h

    @torch.inference_mode()
    def sample(self, merged_h: torch.Tensor, task_datas: list[TaskData]) -> tuple[
        list[torch.Tensor],
        list[TaskData],
        list[torch.Tensor],
        list[TaskData],
        list[TaskData],
    ]:
        ongoing_tokens = []
        ongoing_datas = []
        all_tokens = []
        all_datas = []
        done_datas = []
        ptr = 0
        for task_data in task_datas:
            seqlen = task_data.seqlen
            h = merged_h[ptr : ptr + seqlen]
            ptr += seqlen
            task_data.start_pos += seqlen
            task_data.round += 1
            task_data.seqlen = (
                1  # TODO: make more sense for updating task data values here
            )
            sampling_params = task_data.sampling_params
            sampling_params.max_new_tokens -= 1
            if sampling_params.temperature > 0:
                probs = torch.softmax(h[-1] / sampling_params.temperature, dim=-1)
                next_token = sample_top_p(probs, sampling_params.top_p).reshape(1)
            else:
                next_token = torch.argmax(h[-1], dim=-1).reshape(1)
            next_token = next_token.to("cpu")
            if next_token[0] not in STOP_TOKEN_IDS and (
                sampling_params.max_new_tokens <= 0
                or task_data.start_pos >= sampling_params.max_seq_len
            ):
                next_token = torch.cat([next_token, torch.tensor([EOS_TOKEN_ID])])
            all_datas.append(task_data)
            all_tokens.append(next_token)
            if next_token[-1] in STOP_TOKEN_IDS or sampling_params.max_new_tokens <= 0:
                done_datas.append(task_data)
            else:
                ongoing_datas.append(task_data)
                ongoing_tokens.append(next_token)
        return ongoing_tokens, ongoing_datas, all_tokens, all_datas, done_datas
