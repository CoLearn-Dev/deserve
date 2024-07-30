from typing import List, Optional, Tuple, Dict, Any, Set
import os
import torch
from torch import Tensor, nn
from .model import ModelArgs, TransformerBlock, RMSNorm, precompute_freqs_cis
from fleece_network import Peer, dumps
import requests
import threading
import concurrent.futures
import time
import socket
from urllib.parse import urlparse
import json
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
import queue
import traceback

torch.set_default_device("cpu")

llama_2_7b_args = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000}
llama_2_13b_args = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000}
llama_2_70b_args = {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000}
llama_3_8b_args = {"dim": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "vocab_size": 128256, "multiple_of": 1024, "ffn_dim_multiplier": 1.3, "norm_eps": 1e-05, "rope_theta": 500000.0}
llama_3_70b_args = {"dim": 8192, "ffn_dim_multiplier": 1.3, "multiple_of": 1024, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 128256, "rope_theta": 500000.0}

if torch.cuda.is_available():
    main_device = "cuda"
    # if torch.cuda.is_bf16_supported():
    #     main_dtype = torch.bfloat16
    #     torch.set_default_dtype(torch.bfloat16)
    # else:
    main_dtype = torch.float16
    torch.set_default_dtype(torch.float16)
else:
    main_device = "cpu"
    main_dtype = torch.float32
    torch.set_default_dtype(torch.float32)

# llama 2
# global_freqs_cis = precompute_freqs_cis(128, 4096).to(main_device)
# EOS_ID = 2
# STOP_TOKEN_IDS = [2]
# VOCAL_SIZE = 32000
# tokenizer = Tokenizer(model_path="/home/ubuntu/llama/tokenizer.model")
# print(tokenizer.bos_id) 1
# print(tokenizer.eos_id) 2
# print(tokenizer.pad_id) -1
# print(tokenizer.n_words) 32000

# llama 3
global_freqs_cis = precompute_freqs_cis(128, 8192, 500000.0).to(main_device)
EOS_ID = 128001
STOP_TOKEN_IDS = [128001, 128009]
VOCAL_SIZE = 128256
# tokenizer = Tokenizer(model_path="./Meta-Llama-3-8B-Instruct/tokenizer.model")
# print(tokenizer.bos_id) 128000
# print(tokenizer.eos_id) 128001
# print(tokenizer.pad_id) -1
# print(tokenizer.n_words) 128256

stop_tokens = torch.tensor(STOP_TOKEN_IDS, device=main_device)
stop_tokens_cpu = torch.tensor(STOP_TOKEN_IDS)


def parse_layer_name(layer_name: str):
    s = layer_name.split('/')
    return s[0], s[1]


KV_CACHE_BLOCK = 256


def get_kv_cache_length(cur, seqlen):
    while cur < seqlen:
        cur += KV_CACHE_BLOCK
    return cur


ENABLE_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_with_kvcache
    ENABLE_FLASH_ATTN = True
except ImportError as e:
    print("Package flash-attn is not found. Please install it for better performance. https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features")

NUM_BLOCKS = 12096
PAGE_BLOCK_SIZE = 256
MAX_TOTAL_BSZ = 64
if ENABLE_FLASH_ATTN:
    k_cache_paged = torch.randn(
        NUM_BLOCKS, PAGE_BLOCK_SIZE, 8, 128, device=main_device
    )
    v_cache_paged = torch.randn(
        NUM_BLOCKS, PAGE_BLOCK_SIZE, 8, 128, device=main_device
    )
    page_queue = queue.Queue()
    _ = [page_queue.put(x) for x in range(1, NUM_BLOCKS)]


def get_kv_cache_paged(x, start_pos, block_table, model):
    bsz, seqlen = x.shape[0], x.shape[1]
    if block_table is None:
        length = get_kv_cache_length(0, start_pos + seqlen)//PAGE_BLOCK_SIZE
        block_table = torch.zeros(
            (
                bsz,
                length,
            ),
            device=main_device,
            dtype=torch.int32
        )
        for i in range(length):
            for j in range(bsz):
                block_table[j, i] = page_queue.get()
        return block_table
    old_block_table = block_table
    if start_pos + seqlen > block_table.shape[1]*PAGE_BLOCK_SIZE:
        length = get_kv_cache_length(block_table.shape[1]*PAGE_BLOCK_SIZE, start_pos + seqlen)//PAGE_BLOCK_SIZE
        block_table = torch.zeros(
            (
                bsz,
                length,
            ),
            device=main_device,
            dtype=torch.int32
        )
        block_table[:, :old_block_table.shape[1]] = old_block_table[:, :]
        for i in range(old_block_table.shape[1], length):
            for j in range(bsz):
                block_table[j, i] = page_queue.get()
        return block_table
    else:
        return block_table


def get_kv_cache(x, start_pos, kv_cache, model):
    bsz, seqlen = x.shape[0], x.shape[1]
    if kv_cache is None:
        length = get_kv_cache_length(0, start_pos + seqlen)
        cache_k = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device=main_device
        )
        cache_v = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device=main_device
        )
        return (cache_k, cache_v)
    old_cache_k, old_cache_v = kv_cache
    if start_pos + seqlen > old_cache_k.shape[1]:
        length = get_kv_cache_length(old_cache_k.shape[1], start_pos + seqlen)
        cache_k = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device=main_device
        )
        cache_v = torch.zeros(
            (
                bsz,
                length,
                model.attention.n_local_kv_heads,
                model.attention.head_dim,
            ),
            device=main_device
        )
        cache_k[:, :start_pos, :, :], cache_v[:, :start_pos, :, :] = old_cache_k[:, :start_pos, :, :], old_cache_v[:, :start_pos, :, :]
        del_tensor(old_cache_k)
        del_tensor(old_cache_v)
        del kv_cache
        return (cache_k, cache_v)
    else:
        return kv_cache


def del_tensor(t):
    t.detach()
    t.grad = None
    t.untyped_storage().resize_(0)


executor = concurrent.futures.ThreadPoolExecutor(max_workers=400)
executor_forward = concurrent.futures.ThreadPoolExecutor(max_workers=40)


def requests_post(url, headers=None, data=None, json=None, worker=None, to_worker_id=None):
    try:
        if to_worker_id is not None:
            st = time.monotonic()
        # time.sleep(0.01)
        r = requests.post(url, headers=headers, data=data, json=json)
        assert r.status_code == 200
        if to_worker_id is not None:
            en = time.monotonic()
            latency = (en-st)*1000
            worker.perf_network.append((to_worker_id, latency))
    except Exception:
        if worker is not None:
            worker.cancel_task(json["task_id"])


def send_request(url, headers=None, data=None, exec=None, worker=None, to_worker_id=None):
    if exec is None:
        executor.submit(requests_post, url=url, headers=headers, data=data, worker=worker, to_worker_id=to_worker_id)
    else:
        exec.submit(requests_post, url=url, headers=headers, data=data, worker=worker, to_worker_id=to_worker_id)


executor_latency_test = concurrent.futures.ThreadPoolExecutor(max_workers=40)


def latency_test(host: str, port: int, timeout=60):
    st = time.monotonic()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.shutdown(socket.SHUT_RD)
    except socket.timeout:
        return None
    except OSError:
        return None
    en = time.monotonic()
    return (en-st)*1000


def measure_latency(node_list: List[str], timeout):
    # executor_latency_test
    jobs = []
    for node in node_list:
        parsed_url = urlparse(node)
        host = parsed_url.hostname
        if parsed_url.port is not None:
            port = parsed_url.port
        elif parsed_url.scheme == "http":
            port = 80
        elif parsed_url.scheme == "https":
            port = 443
        else:
            port = 22
        jobs.append(executor_latency_test.submit(latency_test, host, port, timeout))
    ans = []
    for job in jobs:
        ans.append(job.result())
    return ans


class LayerForward:
    def __init__(
        self,
        h: torch.Tensor,
        layer_names: List,
        bsz: int,
        is_new_task: bool,
        round: int,
        start_pos: int,
        seqlen: int,
        kv_cache_dict: Dict,
        metadata: Dict,
    ):
        self.h = h
        self.layer_names = layer_names
        self.bsz = bsz
        self.is_new_task = is_new_task
        self.round = round
        self.start_pos = start_pos
        self.seqlen = seqlen
        self.kv_cache_dict = kv_cache_dict
        self.metadata = metadata


class Worker:
    def __init__(
            self,
            worker_id: str = None,
            # mirror_url: str = "TODO",
            cache_dir: str = "~/.cache/fleece-worker/models",
    ):
        self.worker_id = worker_id
        # self.mirror_url = mirror_url
        self.controller_url = None
        self.api_token = None
        self.worker_nickname = worker_id
        self.heartbeat_interval = 300
        self.tm_pubkeys = {}
        self.worker_urls = {}
        self.perf_computation = []
        self.perf_network = []
        self.peer: Optional[Peer] = None
        self.async_portal = None

        self.cache_dir = os.path.expanduser(cache_dir)
        self.layers = dict()
        self.task_info: Dict[(str, int), Tuple[int, Dict[str, Any]]] = dict()
        self.mutex = threading.Lock()
        self.task_prompt_tokens: Dict[str, torch.Tensor] = dict()
        self.task_eos_reached: Dict[str, torch.Tensor] = dict()
        self.task_local_steps: Dict[str, List[int]] = dict()
        self.task_update_queue: Dict[str, queue.Queue[Tuple[int, List[int]]]] = dict()
        self.layer_forward_engine_queue: queue.Queue[LayerForward] = queue.Queue()
        self.canceled_task: Set[str] = set()

    def fetch_layer(self, full_layer_name):
        model_name, layer_name = parse_layer_name(full_layer_name)
        if model_name.startswith("dummy"):
            return None
        path = os.path.join(self.cache_dir, model_name, f"{layer_name}.pt")
        if not os.path.exists(path):  # TODO lock
            os.makedirs(os.path.join(self.cache_dir, model_name), exist_ok=True)
            with requests.get(f"https://huggingface.co/colearn/{model_name}/resolve/main/{layer_name}.pt", stream=True) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        return path

    def preload_layers(self, layer_names: List[str]):
        with self.mutex:  # TODO
            ths = []
            for full_layer_name in layer_names:
                if full_layer_name in self.layers:
                    continue
                th = executor.submit(self.fetch_layer, full_layer_name)
                ths.append((full_layer_name, th))
            for full_layer_name, th in ths:
                path = th.result()
                model_name, layer_name = parse_layer_name(full_layer_name)
                if model_name.startswith("dummy"):
                    continue
                if model_name.startswith("llama-2-7b"):
                    model_args = ModelArgs(**llama_2_7b_args)
                elif model_name.startswith("llama-2-13b"):
                    model_args = ModelArgs(**llama_2_13b_args)
                elif model_name.startswith("llama-2-70b"):
                    model_args = ModelArgs(**llama_2_70b_args)
                elif model_name.startswith("llama-3-8b"):
                    model_args = ModelArgs(**llama_3_8b_args)
                elif model_name.startswith("llama-3-70b"):
                    model_args = ModelArgs(**llama_3_70b_args)
                else:
                    raise NotImplementedError("Unknown model")
                if layer_name == "tok_embeddings":
                    l = torch.nn.utils.skip_init(nn.Embedding, model_args.vocab_size, model_args.dim)
                elif layer_name.startswith("layer"):
                    l = TransformerBlock(model_args)
                elif layer_name == "norm":
                    l = RMSNorm(model_args.dim, eps=model_args.norm_eps)
                elif layer_name == "output":
                    l = torch.nn.utils.skip_init(nn.Linear, model_args.dim, model_args.vocab_size, bias=False)
                else:
                    raise NotImplementedError("Unknown layers")
                l.load_state_dict(torch.load(path, map_location="cpu"))
                l.to(main_device)
                self.layers[full_layer_name] = l

    def unload_layers(self, layer_names: List[str]):
        for full_layer_name in layer_names:
            if full_layer_name not in self.layers:
                continue  # TODO continue or warning?
            del self.layers[full_layer_name]
            torch.cuda.empty_cache()

    def cancel_task(self, task_id: str, del_task=False):
        if del_task:
            self.del_task(task_id)
        self.canceled_task.add(task_id)

    def del_task(self, task_id: str):
        steps = self.task_local_steps.pop(task_id, None)
        if steps is None:
            return
        if task_id in self.task_prompt_tokens:
            del self.task_prompt_tokens[task_id]
        if task_id in self.task_eos_reached:
            del self.task_eos_reached[task_id]
        if task_id in self.task_update_queue:
            self.task_update_queue[task_id].put((None, None))
        for step in steps:
            _, kv_cache_dict = self.task_info[(task_id, step)]
            for _, kv_cache in kv_cache_dict.items():
                if ENABLE_FLASH_ATTN:
                    block_table = kv_cache.tolist()
                    for x in block_table:
                        for y in x:
                            page_queue.put(y)
                else:
                    k_cache, v_cache = kv_cache
                    del_tensor(k_cache)
                    del_tensor(v_cache)
            del self.task_info[(task_id, step)]
        if not ENABLE_FLASH_ATTN:
            torch.cuda.empty_cache()

    def pull_worker_url(self):
        r = requests.get(f"{self.controller_url}/get_worker_list",
                         headers={"api-token": self.api_token})
        res = json.loads(r.content)
        for worker in res["workers"]:
            self.worker_urls[worker["worker_id"]] = worker["url"]

    def get_worker_url(self, worker_id):
        if worker_id not in self.worker_urls:
            self.pull_worker_url()
        return self.worker_urls.get(worker_id)

    def verify(self, tm_url, task_id, plan, timestamp, signature_hex):
        public_key_bytes = bytes.fromhex(self.tm_pubkeys[tm_url])
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), public_key_bytes
        )
        signed_bytes = task_id.encode()+str(timestamp).encode()
        for x in plan:
            signed_bytes += x[0].encode()
            for y in x[1]:
                signed_bytes += y.encode()
        try:
            public_key.verify(
                bytes.fromhex(signature_hex),
                signed_bytes,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except:
            return False

    def send_forward(self, to_worker_id, tensors: dict[str, Tensor], metadata: dict[str, Any]):
        if to_worker_id == self.worker_id:
            if isinstance(metadata, dict):
                executor.submit(
                    self.forward,
                    **tensors,
                    **metadata,
                )
            elif isinstance(metadata, list):
                executor.submit(
                    self.forward_merged,
                    tensors,
                    metadata,
                )
            else:
                raise
            return
        url = self.get_worker_url(to_worker_id)
        buffer = dumps(tensors, metadata)
        if (url is not None and url != "none") and to_worker_id != self.worker_id:
            if to_worker_id == self.worker_id:
                # self.forward(**data)
                send_request(
                    f"http://127.0.0.1:{self.port}/forward",
                    data=buffer,
                    exec=executor_forward,
                    worker=self,
                    to_worker_id=to_worker_id)
            else:
                send_request(
                    f"{self.get_worker_url(to_worker_id)}/forward",
                    data=buffer,
                    exec=executor_forward,
                    worker=self,
                    to_worker_id=to_worker_id)
        else:
            async def send():
                connection = await self.peer.connect(to_worker_id)
                reply = await connection.send("forward", buffer)
                if reply.status_code != 200:
                    self.cancel_task(metadata["task_id"])
            self.async_portal.call(self.peer.tg.start_soon, send)

    def layer_forward_engine_step(self, task_list: List[LayerForward]):
        task = task_list[0]
        with torch.inference_mode():
            input_shapes = [list(t.h.shape) for t in task_list]
            st = time.monotonic()
            h = torch.cat([t.h for t in task_list])
            bsz_list, start_pos_list = [t.bsz for t in task_list], [t.start_pos for t in task_list]
            for full_layer_name in task.layer_names:
                model_name, layer_name = parse_layer_name(full_layer_name)
                if model_name.startswith("dummy"):
                    if layer_name == "output":
                        for t in task_list:
                            t.h = torch.zeros((t.bsz, 1, VOCAL_SIZE), dtype=main_dtype, device=main_device)
                            t.h[:, :, t.round+10] = 1.0
                            if t.round >= 320:
                                t.h = torch.zeros((t.bsz, 1, VOCAL_SIZE), dtype=main_dtype, device=main_device)
                                t.h[:, :, EOS_ID] = 1.0
                            # time.sleep(0.01)
                        h = torch.cat([t.h for t in task_list])
                    continue
                if layer_name == "tok_embeddings":
                    h = self.layers[full_layer_name](h)
                elif layer_name.startswith("layers."):
                    kv_cache_list = []
                    for t in task_list:
                        if t.is_new_task:
                            # if torch.cuda.is_available():
                            #     gpu_mem_info = torch.cuda.mem_get_info()
                            #     if gpu_mem_info[0]/gpu_mem_info[1] < 0.05 and gpu_mem_info[0] < 2e9:
                            #         return None, None  # TODO need fix
                            if ENABLE_FLASH_ATTN:
                                kv_cache_list.append(get_kv_cache_paged(t.h, t.start_pos, None, self.layers[full_layer_name]))
                            else:
                                kv_cache_list.append(get_kv_cache(t.h, t.start_pos, None, self.layers[full_layer_name]))
                        else:
                            if ENABLE_FLASH_ATTN:
                                kv_cache_list.append(get_kv_cache_paged(t.h, t.start_pos, t.kv_cache_dict[full_layer_name], self.layers[full_layer_name]))
                            else:
                                kv_cache_list.append(get_kv_cache(t.h, t.start_pos, t.kv_cache_dict[full_layer_name], self.layers[full_layer_name]))
                    if ENABLE_FLASH_ATTN:
                        h = self.layers[full_layer_name](h, bsz_list, start_pos_list, global_freqs_cis, (k_cache_paged, v_cache_paged), kv_cache_list)
                    else:
                        h = self.layers[full_layer_name](h, bsz_list, start_pos_list, global_freqs_cis, None, kv_cache_list)
                    for i, t in enumerate(task_list):
                        t.kv_cache_dict[full_layer_name] = kv_cache_list[i]
                elif layer_name == "norm":
                    h = self.layers[full_layer_name](h)
                elif layer_name == "output":
                    h = self.layers[full_layer_name](h)
                else:
                    raise NotImplementedError("Unknown layers")
            # start = 0
            # for t in task_list:
            #     bsz = t.bsz
            #     t.h = h[start:start+bsz]
            #     start += bsz
            en = time.monotonic()
            latency = (en-st)*1000
            self.perf_computation.append(((str(task.layer_names), str(input_shapes)), latency))
        return h
        # for task in task_list:
        #     task.call_back_queue.put((task.h, task.kv_cache_dict))

    def post_layer_forward_engine_step(self, task_list: List[LayerForward], merged_h):
        start = 0
        next_token_list = []
        task_update_list = []
        metadata_list = [{"plan": task_list[0].metadata["plan"]}]
        for task in task_list:
            self.task_info[(task.metadata["task_id"], task.metadata["step"])] = (task.start_pos+task.seqlen, task.kv_cache_dict)

            bsz = task.bsz
            h = merged_h[start:start+bsz]
            start += bsz

            # last node
            if task.metadata["step"] == len(task.metadata["plan"])-1:
                if task.metadata["temperature"] > 0:
                    probs = torch.softmax(h[:, -1] / task.metadata["temperature"], dim=-1)
                    next_token = sample_top_p(probs, task.metadata["top_p"])
                else:
                    next_token = torch.argmax(h[:, -1], dim=-1)
                next_token = next_token.reshape(-1)
                if task.start_pos > task.metadata["max_total_len"]:
                    next_token = torch.tensor([EOS_ID] * task.bsz)  # FIXME fake max length limit
                # print(next_token)
                next_token = next_token.to("cpu")

                # eos_reached
                self.task_eos_reached[task.metadata["task_id"]] |= torch.isin(next_token, stop_tokens_cpu)  # eos_id
                if not all(self.task_eos_reached[task.metadata["task_id"]]):
                    # next node
                    # tensors[str(len(metadata_list))] = next_token
                    next_token_list.append(next_token)
                    metadata_list.append({
                        "task_id": task.metadata["task_id"],
                        # "plan": task.metadata["plan"],
                        "step": 0,
                        "round": task.metadata["round"]+1,
                        "max_total_len": task.metadata["max_total_len"],
                        "temperature": task.metadata["temperature"],
                        "top_p": task.metadata["top_p"],
                        "task_manager_url": task.metadata["task_manager_url"],
                        "signature": task.metadata["signature"],
                        "timestamp": task.metadata["timestamp"],
                        "bsz": task.bsz,
                    })
                else:
                    self.send_forward(
                        task.metadata["plan"][0][0],
                        tensors={},
                        metadata={
                            "task_id": task.metadata["task_id"],
                            "plan": task.metadata["plan"],
                            "step": 0,
                            "task_manager_url": task.metadata["task_manager_url"],
                            "signature": task.metadata["signature"],
                            "timestamp": task.metadata["timestamp"],
                        })
                # update
                if task.metadata["task_manager_url"] is not None:
                    # self.new_task_update(task.metadata["task_manager_url"], task.metadata["task_id"], task.metadata["step"], task.metadata["round"], next_token.tolist())
                    task_update_list.append((task.metadata["task_manager_url"], task.metadata["task_id"], task.metadata["step"], task.metadata["round"], next_token.tolist()))
                if all(self.task_eos_reached[task.metadata["task_id"]]):
                    self.cancel_task(task.metadata["task_id"], True)
            else:
                # next node
                metadata_list.append({
                    "task_id": task.metadata["task_id"],
                    # "plan": task.metadata["plan"],
                    "step": task.metadata["step"]+1,
                    "round": task.metadata["round"],
                    "max_total_len": task.metadata["max_total_len"],
                    "temperature": task.metadata["temperature"],
                    "top_p": task.metadata["top_p"],
                    "task_manager_url": task.metadata["task_manager_url"],
                    "signature": task.metadata["signature"],
                    "timestamp": task.metadata["timestamp"],
                    "bsz": task.bsz,
                })
                # self.send_forward(
                #     task.metadata["plan"][task.metadata["step"]+1][0],
                #     tensors={
                #         "payload": h,
                #     },
                #     metadata=)
        # torch.cuda.synchronize()
        # print("1.1", time.monotonic())
        if task.metadata["step"] == len(task.metadata["plan"])-1:
            if len(next_token_list) > 0:
                merged_next_token = torch.cat(next_token_list)
                self.send_forward(task.metadata["plan"][0][0], tensors={"payload": merged_next_token}, metadata=metadata_list)
        else:
            self.send_forward(task.metadata["plan"][task.metadata["step"]+1][0], tensors={"payload": merged_h}, metadata=metadata_list)
        return task_update_list

    def layer_forward_engine(self):
        q = self.layer_forward_engine_queue
        while True: 
            q_buffered: list[list[LayerForward]] = [q.get()]
            while True: 
                try: 
                    tasks = q.get(block=False)
                    q_buffered.append(tasks)
                except queue.Empty:
                    break
            prefill_tasks_list = [tasks for tasks in q_buffered if tasks[0].seqlen > 1]
            decode_tasks_list = [tasks for tasks in q_buffered if tasks[0].seqlen == 1]
            
            for tasks in prefill_tasks_list:
                h = self.layer_forward_engine_step(tasks)
                task_update_list = self.post_layer_forward_engine_step(tasks, h)
                tmp_len = sum([len(task[4]) for task in task_update_list])
                print(time.monotonic(), len(tasks), sum([task.bsz for task in tasks]), tmp_len)
                executor_forward.submit(
                    self.tmptmp,
                    task_update_list
                )
            
            decode_tasks_list.sort(key=lambda x: x[0].bsz, reverse=False)
            while len(decode_tasks_list) > 0: 
                total_bsz = 0
                task_list = []
                for i in reversed(range(len(decode_tasks_list))): 
                    print(i)
                    cur_bsz = sum([task.bsz for task in decode_tasks_list[i]])
                    if total_bsz + cur_bsz > MAX_TOTAL_BSZ: 
                        continue
                    total_bsz += cur_bsz
                    task_list.extend(decode_tasks_list.pop(i))
                h = self.layer_forward_engine_step(task_list)
                task_update_list = self.post_layer_forward_engine_step(task_list, h)
                tmp_len = sum([len(task[4]) for task in task_update_list])
                print(time.monotonic(), len(task_list), sum([task.bsz for task in task_list]), tmp_len)
                executor_forward.submit(
                    self.tmptmp,
                    task_update_list
                )
                    
                    
    def layer_forward_engine_old(self):
        q = self.layer_forward_engine_queue
        while True:
            task_list = []
            total_bsz = 0
            tasks = q.get()
            for task in tasks:
                total_bsz += task.bsz
                task_list.append(task)
            while True:
                if total_bsz > MAX_TOTAL_BSZ:
                    break
                try:
                    tasks = q.get(block=False)
                    if tasks[0].seqlen == task_list[0].seqlen and tasks[0].layer_names == task_list[0].layer_names:  # FIXME
                        bsz = sum([task.bsz for task in tasks])
                        if total_bsz+bsz > MAX_TOTAL_BSZ:
                            if bsz > total_bsz:
                                q.put(task_list)
                                task_list = tasks
                                total_bsz = bsz
                            else:
                                q.put(tasks)
                            break
                        for task in tasks:
                            task_list.append(task)
                            total_bsz += task.bsz
                        if total_bsz > MAX_TOTAL_BSZ:
                            break
                    else:
                        q.put(tasks)
                        break
                except queue.Empty:
                    break
            # print("layer_forward_engine_step: ", len(task_list), total_bsz)
            # torch.cuda.synchronize()
            # print("0", time.monotonic())
            h = self.layer_forward_engine_step(task_list)
            # torch.cuda.synchronize()
            # print("1", time.monotonic())
            task_update_list = self.post_layer_forward_engine_step(task_list, h)
            # torch.cuda.synchronize()
            # print("2", time.monotonic())
            tmp_len = sum([len(task[4]) for task in task_update_list])
            print(time.monotonic(), len(task_list), total_bsz, tmp_len)
            executor_forward.submit(
                self.tmptmp,
                task_update_list
            )

    def tmptmp(self, task_update_list):
        req_list = []
        for task_update in task_update_list:
            # self.new_task_update(task_update[0], task_update[1], task_update[2], task_update[3], task_update[4])
            task_manager_url = task_update[0]
            req_list.append({
                "task_id": task_update[1],
                "plan_current_step": task_update[2],
                "plan_current_round": task_update[3],
                "output_tokens": [task_update[4]],
            })
        requests_post(
            f"{task_manager_url}/update_tasks",
            headers={"worker-id": self.worker_id, "api-token": self.api_token},
            json=req_list)

    def start_layer_forward_engine(self):
        heartbeat_thread = threading.Thread(target=self.layer_forward_engine)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

    def layers_forward(self, h, layer_names, bsz, is_new_task, round, start_pos, seqlen, kv_cache_dict, metadata):
        # q = queue.Queue()
        self.layer_forward_engine_queue.put([LayerForward(h, layer_names, bsz, is_new_task, round, start_pos, seqlen, kv_cache_dict, metadata)])
        # h, kv_cache_dict = q.get()
        # del q
        # return h, kv_cache_dict

    def send_update_task(self, task_manager_url, task_id, step):
        q = self.task_update_queue[task_id]
        while True:
            output_tokens_list = []
            round, output_tokens = q.get()
            if output_tokens is None:
                break
            output_tokens_list.append(output_tokens)
            ret_flag = False
            while True:
                try:
                    _, output_tokens = q.get(block=False)
                    if output_tokens is None:
                        ret_flag = True
                        break
                    else:
                        output_tokens_list.append(output_tokens)
                except queue.Empty:
                    break
            # print("requests_post", round, output_tokens_list)
            requests_post(
                f"{task_manager_url}/update_task",
                headers={"worker-id": self.worker_id, "api-token": self.api_token},
                json={
                    "task_id": task_id,
                    "plan_current_step": step,
                    "plan_current_round": round,
                    "output_tokens": output_tokens_list,
                },
                worker=self)
            if ret_flag:
                break
        # TODO del self.task_update_queue[task_id]?

    def new_task_update(self, task_manager_url, task_id, _step, round, output_tokens):
        if task_manager_url is not None:
            self.task_update_queue[task_id].put((round, output_tokens))

    # def forward_same_node(self, delta_round, h, layer_names, bsz, is_new_task, round, start_pos, seqlen, kv_cache_dict, temperature, top_p, max_total_len, eos_reached, prompt_tokens, task_manager_url, task_id, step):
    #     ans_tokens = []
    #     try:
    #         for i in range(delta_round):
    #             h, kv_cache_dict = self.layers_forward(h, layer_names, bsz, is_new_task, round+i, start_pos, seqlen, kv_cache_dict)
    #             # last node
    #             if temperature > 0:
    #                 probs = torch.softmax(h[:, -1] / temperature, dim=-1)
    #                 next_token = sample_top_p(probs, top_p)
    #             else:
    #                 next_token = torch.argmax(h[:, -1], dim=-1)
    #             next_token = next_token.reshape(-1)
    #             if start_pos > max_total_len:
    #                 next_token = torch.tensor([EOS_ID] * bsz, device=main_device)  # FIXME fake max length limit
    #             # print(next_token)
    #             # eos_reached
    #             if all(eos_reached | torch.isin(next_token, stop_tokens)) or i == delta_round-1:
    #                 return h, kv_cache_dict, ans_tokens, eos_reached

    #             # loop
    #             eos_reached |= torch.isin(next_token, stop_tokens)  # eos_id
    #             ans_tokens.append(next_token)
    #             start_pos = start_pos+seqlen
    #             seqlen = 1
    #             is_new_task = False

    #             # first node
    #             tokens = torch.zeros((bsz, 1), dtype=torch.long, device=main_device)
    #             for k, t in enumerate(prompt_tokens):
    #                 if len(t) > start_pos:
    #                     tokens[k, :] = torch.tensor([t[start_pos]], dtype=torch.long, device=main_device)
    #                 else:
    #                     tokens[k, :] = next_token[k]
    #             h = tokens
    #     finally:
    #         # update_task
    #         for i, output_tokens in enumerate(ans_tokens):
    #             self.new_task_update(task_manager_url, task_id, step, round+i, output_tokens.tolist())

    def forward_merged(self,
                       tensors: Dict[str, torch.Tensor],
                       metadata_list: List[Dict],
                       ):
        try:
            start = 0
            layers_forward_list = []
            plan = metadata_list[0]["plan"]
            metadata_list.pop(0)
            for i, task in enumerate(metadata_list):
                index = task["step"]
                is_new_task = task["round"] == 0
                task_id = task["task_id"]
                round = task["round"]
                step = task["step"]
                task["plan"] = plan
                if is_new_task:
                    if task_id in self.task_local_steps:
                        self.task_local_steps[task_id].append(step)
                    else:
                        self.task_local_steps[task_id] = [step]
                    self.task_info[(task_id, step)] = (0, dict())
                else:
                    if not task_id in self.task_local_steps:
                        return
                start_pos, kv_cache_dict = self.task_info[(task_id, step)]

                # first node
                if index == 0:
                    # payload = tensors[str(i)]
                    bsz = task["bsz"]
                    payload = tensors["payload"][start:start+bsz]
                    start += bsz
                    if is_new_task:
                        min_prompt_len = min(len(t) for t in payload)
                        self.task_prompt_tokens[task_id] = payload
                        tokens = torch.zeros((bsz, min_prompt_len), dtype=torch.long)
                        for k, t in enumerate(payload):
                            tokens[k, :] = torch.tensor(t[:min_prompt_len], dtype=torch.long)
                        h = tokens.to(main_device)
                    else:
                        prompt_tokens = self.task_prompt_tokens[task_id]
                        tokens = torch.zeros((bsz, 1), dtype=torch.long)
                        for k, t in enumerate(prompt_tokens):
                            if len(t) > start_pos:
                                tokens[k, :] = torch.tensor([t[start_pos]], dtype=torch.long)
                            else:
                                tokens[k, :] = torch.tensor([payload[k]], dtype=torch.long)
                        h = tokens.to(main_device)
                    # print(h)
                    bsz, seqlen = h.shape
                else:
                    bsz = task["bsz"]
                    payload = tensors["payload"][start:start+bsz]
                    start += bsz
                    h = payload.to(main_dtype).to(main_device)  # h = torch.tensor(payload, dtype=main_dtype, device=main_device)
                    if len(h.shape) > 2:
                        bsz, seqlen, _ = h.shape
                    else:
                        bsz, seqlen = h.shape

                # last node init
                if index == len(plan)-1 and is_new_task:
                    self.task_eos_reached[task_id] = torch.tensor([False] * bsz)
                    self.task_update_queue[task_id] = queue.Queue()
                    executor.submit(self.send_update_task, task["task_manager_url"], task_id, step)

                # forward
                _, layer_names = plan[index]
                self.preload_layers(layer_names)
                layers_forward_list.append(LayerForward(h, layer_names, bsz, is_new_task, round, start_pos, seqlen, kv_cache_dict, task))
            self.layer_forward_engine_queue.put(layers_forward_list)
        except Exception:
            print(traceback.format_exc())
        # print(tensors, metadata_list)

    def forward(self,
                task_id: str,
                plan: List[Tuple[str, List[str]]],
                step: int,
                round: int = -1,
                payload: Optional[List] = None,
                max_total_len: int = 2048,
                temperature: float = 0.0,
                top_p: float = 0.9,
                task_manager_url: Optional[str] = None,
                signature: Optional[str] = None,
                timestamp: Optional[int] = None,
                ):
        try:
            # self.verify(task_manager_url, task_id, plan, timestamp, signature)

            index = step
            is_new_task = round == 0
            if payload is None or task_id in self.canceled_task:
                self.del_task(task_id)
                if index < len(plan)-1:
                    # next node
                    self.send_forward(
                        plan[index+1][0],
                        tensors={},
                        metadata={
                            "task_id": task_id,
                            "plan": plan,
                            "step": step+1,
                            "task_manager_url": task_manager_url,
                            "signature": signature,
                            "timestamp": timestamp,
                        })
                return

            if is_new_task:
                if task_id in self.task_local_steps:
                    self.task_local_steps[task_id].append(step)
                else:
                    self.task_local_steps[task_id] = [step]
                self.task_info[(task_id, step)] = (0, dict())
            else:
                if not task_id in self.task_local_steps:
                    return
            start_pos, kv_cache_dict = self.task_info[(task_id, step)]

            # first node
            if index == 0:
                bsz = len(payload)
                if is_new_task:
                    min_prompt_len = min(len(t) for t in payload)
                    self.task_prompt_tokens[task_id] = payload
                    tokens = torch.zeros((bsz, min_prompt_len), dtype=torch.long)
                    for k, t in enumerate(payload):
                        tokens[k, :] = torch.tensor(t[:min_prompt_len], dtype=torch.long)
                    h = tokens.to(main_device)
                else:
                    prompt_tokens = self.task_prompt_tokens[task_id]
                    tokens = torch.zeros((bsz, 1), dtype=torch.long)
                    for k, t in enumerate(prompt_tokens):
                        if len(t) > start_pos:
                            tokens[k, :] = torch.tensor([t[start_pos]], dtype=torch.long)
                        else:
                            tokens[k, :] = torch.tensor([payload[k]], dtype=torch.long)
                    h = tokens.to(main_device)
                # print(h)
                bsz, seqlen = h.shape
            else:
                h = payload.to(main_dtype).to(main_device)  # h = torch.tensor(payload, dtype=main_dtype, device=main_device)
                if len(h.shape) > 2:
                    bsz, seqlen, _ = h.shape
                else:
                    bsz, seqlen = h.shape

            # last node init
            if index == len(plan)-1 and is_new_task:
                self.task_eos_reached[task_id] = torch.tensor([False] * bsz)
                self.task_update_queue[task_id] = queue.Queue()
                executor.submit(self.send_update_task, task_manager_url, task_id, step)

            # forward
            _, layer_names = plan[index]
            self.preload_layers(layer_names)  # preload
            # if len(plan) == 1:
            #     delta_round = 16
            #     eos_reached = self.task_eos_reached[task_id].to(main_device)
            #     prompt_tokens = self.task_prompt_tokens[task_id]
            #     h, kv_cache_dict, tokens, eos_reached = self.forward_same_node(delta_round, h, layer_names, bsz, is_new_task, round, start_pos, seqlen,
            #                                                                    kv_cache_dict, temperature, top_p, max_total_len, eos_reached, prompt_tokens, task_manager_url, task_id, step)
            #     self.task_eos_reached[task_id] = eos_reached.to("cpu")
            #     delta_round = len(tokens)+1
            #     round = round+delta_round-1
            #     start_pos = start_pos+delta_round-1
            # else:
            metadata = {
                "task_id": task_id,
                "plan": plan,
                "step": step,
                "round": round,
                "max_total_len": max_total_len,
                "temperature": temperature,
                "top_p": top_p,
                "task_manager_url": task_manager_url,
                "signature": signature,
                "timestamp": timestamp,
            }
            self.layers_forward(h, layer_names, bsz, is_new_task, round, start_pos, seqlen, kv_cache_dict, metadata)
            return
            h, kv_cache_dict = self.layers_forward(h, layer_names, bsz, is_new_task, round, start_pos, seqlen, kv_cache_dict, metadata)
            # if h is None:
            #     return
            # else:
            self.task_info[(task_id, step)] = (start_pos+seqlen, kv_cache_dict)

            # last node
            if index == len(plan)-1:
                if temperature > 0:
                    probs = torch.softmax(h[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(h[:, -1], dim=-1)
                next_token = next_token.reshape(-1)
                if start_pos > max_total_len:
                    next_token = torch.tensor([EOS_ID] * bsz)  # FIXME fake max length limit
                print(next_token)
                next_token = next_token.to("cpu")

                # eos_reached
                self.task_eos_reached[task_id] |= torch.isin(next_token, stop_tokens_cpu)  # eos_id
                if not all(self.task_eos_reached[task_id]):
                    # next node
                    self.send_forward(
                        plan[0][0],
                        tensors={
                            "payload": next_token,
                        },
                        metadata={
                            "task_id": task_id,
                            "plan": plan,
                            "step": 0,
                            "round": round+1,
                            "max_total_len": max_total_len,
                            "temperature": temperature,
                            "top_p": top_p,
                            "task_manager_url": task_manager_url,
                            "signature": signature,
                            "timestamp": timestamp,
                        })
                else:
                    self.send_forward(
                        plan[0][0],
                        tensors={},
                        metadata={
                            "task_id": task_id,
                            "plan": plan,
                            "step": 0,
                            "task_manager_url": task_manager_url,
                            "signature": signature,
                            "timestamp": timestamp,
                        })
                # update
                if task_manager_url is not None:
                    self.new_task_update(task_manager_url, task_id, step, round, next_token.tolist())
                if all(self.task_eos_reached[task_id]):
                    self.cancel_task(task_id, True)
            else:
                # next node
                self.send_forward(
                    plan[index+1][0],
                    tensors={
                        "payload": h,
                    },
                    metadata={
                        "task_id": task_id,
                        "plan": plan,
                        "step": step+1,
                        "round": round,
                        "max_total_len": max_total_len,
                        "temperature": temperature,
                        "top_p": top_p,
                        "task_manager_url": task_manager_url,
                        "signature": signature,
                        "timestamp": timestamp,
                    })
                # update
                # if task_manager_url is not None:
                #     send_request(
                #         f"{task_manager_url}/update_task",
                #         headers={"worker-id": self.worker_id, "api-token": self.api_token},
                #         json={
                #             "task_id": task_id,
                #             "plan_current_step": step,
                #             "plan_current_round": round,
                #         },
                #         worker=self)
        except Exception:
            print(traceback.format_exc())

    def get_info(self, node_list, timeout):
        gpu_mem_info = torch.cuda.mem_get_info()
        latency_list = measure_latency(node_list, timeout)
        return self.worker_nickname, gpu_mem_info, latency_list

    def send_heartbeat(self):
        info_data = {
            "loaded_layers": json.dumps(list(self.layers.keys())),
            "perf_computation": [],
            "perf_network": []
        }

        s = {}
        for k, v in self.perf_computation:
            if k not in s:
                s[k] = [v, 1]
            else:
                s[k][0] += v
                s[k][1] += 1
        for k, v in s.items():
            layers, input_shape = k
            avg_latency = v[0]/v[1]
            info_data["perf_computation"].append({"layers": layers, "input_shape": input_shape, "latency": avg_latency})
        s = {}
        for k, v in self.perf_network:
            if k not in s:
                s[k] = [v, 1]
            else:
                s[k][0] += v
                s[k][1] += 1
        for k, v in s.items():
            avg_latency = v[0]/v[1]
            info_data["perf_network"].append({"to_worker_id": k, "latency": avg_latency})

        if torch.cuda.is_available():
            memory = torch.cuda.mem_get_info()
            info_data["gpu_remaining_memory"] = memory[0]
        data = {"info_update": json.dumps(info_data)}
        try:
            r = requests.post(f"{self.controller_url}/worker_heartbeat",
                              json=data,
                              headers={"worker-id": self.worker_id, "api-token": self.api_token})
            res = json.loads(r.content)
            self.tm_pubkeys = res["pubkeys"]
        except:
            pass

    def start_heartbeat_daemon(self):
        def heartbeat_thread():
            while True:
                self.send_heartbeat()
                time.sleep(self.heartbeat_interval)
        heartbeat_thread = threading.Thread(target=heartbeat_thread)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()


def sample_top_p(probs, p):
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
