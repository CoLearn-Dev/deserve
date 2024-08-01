import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import requests
import torch

from deserve_worker.task import TaskData

from .kvcache.kvcache import KVCache, KVCacheManager
from .model.llama import ModelArgs, RMSNorm, TransformerBlock

EOS_TOKEN_ID = 128001  # for llama 3 only
STOP_TOKEN_IDS = [128001, 128009]

llama_2_7b_args = {
    "dim": 4096,
    "multiple_of": 256,
    "n_heads": 32,
    "n_layers": 32,
    "norm_eps": 1e-06,
    "vocab_size": 32000,
}

llama_2_13b_args = {
    "dim": 5120,
    "multiple_of": 256,
    "n_heads": 40,
    "n_layers": 40,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
}

llama_2_70b_args = {
    "dim": 8192,
    "multiple_of": 4096,
    "ffn_dim_multiplier": 1.3,
    "n_heads": 64,
    "n_kv_heads": 8,
    "n_layers": 80,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
}

llama_3_8b_args = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "multiple_of": 1024,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
}

llama_3_70b_args = {
    "dim": 8192,
    "ffn_dim_multiplier": 1.3,
    "multiple_of": 1024,
    "n_heads": 64,
    "n_kv_heads": 8,
    "n_layers": 80,
    "norm_eps": 1e-05,
    "vocab_size": 128256,
    "rope_theta": 500000.0,
}


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

    def get_layer_storage(self, layer_names: list[str]) -> "LayerStorage":
        frozen_layer_names = frozenset(layer_names)
        if frozen_layer_names not in self.layer_storages:
            with self.mutex:
                self.layer_storages[frozen_layer_names] = LayerStorage(
                    self.preload_layers(layer_names), self.main_device
                )
        return self.layer_storages[frozen_layer_names]

    def fetch_layer(self, full_layer_name: str) -> str:
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
        return path

    def preload_layers(self, full_layer_names: list[str]) -> dict[str, torch.nn.Module]:
        threads: list[tuple[str, Future[str]]] = []
        result = {}
        for full_layer_name in full_layer_names:
            thread = self.network_executor.submit(self.fetch_layer, full_layer_name)
            threads.append((full_layer_name, thread))
        for full_layer_name, thread in threads:
            path = thread.result()
            model_name, layer_name = full_layer_name.split("/")
            if model_name.startswith("llama-2-7b"):
                model_args = ModelArgs(**llama_2_7b_args)  # type: ignore
            elif model_name.startswith("llama-2-13b"):
                model_args = ModelArgs(**llama_2_13b_args)  # type: ignore
            elif model_name.startswith("llama-2-70b"):
                model_args = ModelArgs(**llama_2_70b_args)  # type: ignore
            elif model_name.startswith("llama-3-8b"):
                model_args = ModelArgs(**llama_3_8b_args)  # type: ignore
            elif model_name.startswith("llama-3-70b"):
                model_args = ModelArgs(**llama_3_70b_args)  # type: ignore
            else:
                raise NotImplementedError("Unknown model")
            if layer_name == "tok_embeddings":
                l = torch.nn.utils.skip_init(  # type: ignore
                    torch.nn.Embedding, model_args.vocab_size, model_args.dim
                )
            elif layer_name.startswith("layer"):
                l = TransformerBlock(model_args)
            elif layer_name == "norm":
                l = RMSNorm(model_args.dim, eps=model_args.norm_eps)
            elif layer_name == "output":
                l = torch.nn.utils.skip_init(  # type: ignore
                    torch.nn.Linear,
                    model_args.dim,
                    model_args.vocab_size,
                    bias=False,
                )
            else:
                raise NotImplementedError("Unknown layers")
            l.load_state_dict(torch.load(path, map_location="cpu"))
            l.to(self.main_device)
            print("Loaded", full_layer_name)
            self.layers[full_layer_name] = l
        for full_layer_name in full_layer_names:
            result[full_layer_name] = self.layers[full_layer_name]
        return result


class LayerStorage:
    def __init__(self, layers: dict[str, torch.nn.Module], main_device: torch.device):
        self.main_device = main_device
        self.layers = layers

    def clear(self) -> None:
        self.layers.clear()

    @torch.inference_mode()
    def forward(
        self,
        h: torch.Tensor,
        bsz_list: list[int],
        start_pos_list: list[int],
        global_freqs_cis: torch.Tensor,
        kvcache_list: list[dict[int, KVCache]],
        kvcache_manager: KVCacheManager,
    ) -> torch.Tensor:
        _, seqlen = h.shape[:2]
        for full_layer_name in self.layers:
            _, layer_name = full_layer_name.split("/")
            if layer_name == "tok_embeddings":
                h = self.layers[full_layer_name](h)
            elif layer_name.startswith("layers."):
                layer_id = int(layer_name.split(".")[1])
                cur_kvcache_list = []
                for i, kv_cache in enumerate(kvcache_list):
                    kv_cache[layer_id].renew(1, seqlen, start_pos_list[i])
                    cur_kvcache_list.append(kv_cache[layer_id])
                h = self.layers[full_layer_name](
                    h,
                    bsz_list,
                    start_pos_list,
                    global_freqs_cis,
                    cur_kvcache_list,
                    kvcache_manager,
                )
            elif layer_name == "norm":
                h = self.layers[full_layer_name](h)
            elif layer_name == "output":
                h = self.layers[full_layer_name](h)
            else:
                raise NotImplementedError("Unknown layers")
        return h

    @torch.inference_mode()
    def sample(
        self, merged_h: torch.Tensor, task_datas: list[TaskData]
    ) -> tuple[list[torch.Tensor], list[str], list[torch.Tensor], list[str], list[str]]:
        ongoing_tokens = []
        ongoing_ids = []
        all_tokens = []
        all_ids = []
        done_ids = []
        for ptr, task_data in enumerate(task_datas):
            h = merged_h[ptr : ptr + 1]
            _, seqlen = h.shape[:2]
            task_data.start_pos += seqlen
            task_data.round += 1
            sampling_params = task_data.sampling_params
            if task_data.start_pos >= sampling_params.max_total_len:
                next_token = torch.tensor([[EOS_TOKEN_ID]])
            elif sampling_params.temperature > 0:
                probs = torch.softmax(h[:, -1] / sampling_params.temperature, dim=-1)
                next_token = sample_top_p(probs, sampling_params.top_p)
                next_token = next_token.reshape(1, -1)
            else:
                next_token = torch.argmax(h[:, -1], dim=-1)
                next_token = next_token.reshape(1, -1)
            next_token = next_token.to("cpu")
            all_ids.append(task_data.task_id)
            all_tokens.append(next_token)
            if next_token[0][0] in STOP_TOKEN_IDS:
                done_ids.append(task_data.task_id)
            else:
                ongoing_ids.append(task_data.task_id)
                ongoing_tokens.append(next_token)
        return ongoing_tokens, ongoing_ids, all_tokens, all_ids, done_ids
