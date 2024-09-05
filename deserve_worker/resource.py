import os
import sys

import torch

from deserve_worker.model.args import ModelArgs
from deserve_worker.task import TaskData, main_dtype


class ResourceCollector:
    def __init__(self, model_args: ModelArgs) -> None:
        self.model_args = model_args
        self.python_version = sys.version
        self.pytorch_version = torch.__version__
        self.cuda_version = torch.version.cuda
        self.cpu_arch = os.uname().machine
        self.cpu_count = os.cpu_count()
        self.precision = main_dtype
        _, self.gpu_mem = torch.cuda.mem_get_info()

    def calc_inner_dim(self) -> int:
        model_args = self.model_args
        hidden_dim = 4 * model_args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if model_args.ffn_dim_multiplier is not None:
            hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = model_args.multiple_of * (
            (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
        )
        return hidden_dim

    def model_layer_count(self) -> int:
        model_args = self.model_args
        head_dim = model_args.dim // model_args.n_heads
        assert model_args.n_kv_heads is not None
        attention_size = (
            2 * model_args.dim * model_args.n_heads * head_dim
            + 2 * model_args.dim * model_args.n_kv_heads * head_dim
        )
        inner_dim = self.calc_inner_dim()
        ffn_size = 3 * model_args.dim * inner_dim
        norm_size = model_args.dim
        return attention_size + ffn_size + 2 * norm_size

    def model_extra_count(self) -> int:
        model_args = self.model_args
        return (
            model_args.dim * model_args.vocab_size * 2 + model_args.dim
        )  # include tok_emb, norm, and output

    def kvcache_layer_count(self) -> int:
        return 2 * self.model_args.page_size * 8 * 128 * self.model_args.num_pages

    def temporary_count(self) -> int:
        attention_size = 128 * 1024 * 1024
        ffn_size = (self.model_args.dim + self.calc_inner_dim()) * max(
            self.model_args.max_batch_size, self.model_args.max_seq_len
        )  # for results for w1, w2, and w3
        return attention_size + ffn_size

    def get_num_layer(self) -> int:
        return (
            self.gpu_mem - self.temporary_count() * 2 - self.model_extra_count() * 2
        ) // ((self.kvcache_layer_count() + self.model_layer_count()) * 2)

    def print_resources(self) -> None:
        print("Python version:", self.python_version)
        print("PyTorch version:", self.pytorch_version)
        print("CUDA version:", self.cuda_version)
        print("CPU architecture:", self.cpu_arch)
        print("CPU count:", self.cpu_count)
        print(f"GPU memory: {self.gpu_mem / 1024 / 1024 / 1024:.2f} GB")
        print("Precision:", self.precision)
        print("Maximum number of layers served:", self.get_num_layer())
