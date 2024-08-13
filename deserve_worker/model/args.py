from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    page_size: int = 256


llama_2_7b_args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=None,
    vocab_size=32000,
    multiple_of=256,
    ffn_dim_multiplier=None,
    norm_eps=1e-06,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)

llama_2_13b_args = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    n_kv_heads=None,
    vocab_size=32000,
    multiple_of=256,
    ffn_dim_multiplier=None,
    norm_eps=1e-05,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)

llama_2_70b_args = ModelArgs(
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=32000,
    multiple_of=4096,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-05,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)

llama_3_8b_args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-05,
    rope_theta=500000.0,
    max_batch_size=32,
    max_seq_len=2048,
)

llama_3_70b_args = ModelArgs(
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-05,
    rope_theta=500000.0,
    max_batch_size=32,
    max_seq_len=2048,
)
