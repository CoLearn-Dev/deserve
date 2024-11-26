import argparse
import sys
import threading
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from deserve_worker.benchmark.utils import convert_name_to_id, layers
from deserve_worker.execution.exec import BatchDecode, BatchPrefill
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import GpuPagePool
from deserve_worker.kvcache.paged.page_table import PageTableAllocator
from deserve_worker.layer_storage import LayerManager, LayerStorage
from deserve_worker.model.args import llama_3_70b_args
from deserve_worker.task import SamplingParams, TaskData


def profile_decode(
    layer_storage: LayerStorage,
    gpu_page_pool: GpuPagePool,
    page_table_allocator: PageTableAllocator,
    begin: int,
    bsz: int,
    prefix: int,
) -> float:
    if begin == 0:
        prefill_input = torch.randint(
            1,
            llama_3_70b_args.vocab_size,
            (prefix,),
            dtype=torch.int,
            device=torch.device("cuda"),
        )
        decode_input = torch.randint(
            1,
            llama_3_70b_args.vocab_size,
            (bsz,),
            dtype=torch.int,
            device=torch.device("cuda"),
        )
    else:
        prefill_input = torch.rand(
            prefix,
            llama_3_70b_args.dim,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        decode_input = torch.rand(
            bsz,
            llama_3_70b_args.dim,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
    task_datas: list[TaskData] = []
    kvcaches: list[PagedKVCache[GpuPagePool]] = []
    for i in range(bsz):
        task_data = TaskData.empty(
            task_id=f"{prefix}-{bsz}-{i}",
            initial_seqlen=prefix,
            sampling_params=sparam,
        )
        task_data.init(prefix)
        task_datas.append(task_data)
        kvcache = PagedKVCache.empty(page_table_allocator, gpu_page_pool)
        kvcaches.append(kvcache)
        prefill = BatchPrefill(prefill_input, layer_storage, [task_data], [kvcache])
        prefill.prepare()
        prefill.step(ignore_eos=False)
        task_data.step()
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        time_begin = time.time()
        decode = BatchDecode(decode_input, layer_storage, task_datas, kvcaches)
        decode.prepare()
        decode.step(ignore_eos=False)
        for task_data in task_datas:
            task_data.step()
        torch.cuda.synchronize()
        time_end = time.time()
        if i >= 20:
            times.append((time_end - time_begin) * 1000)
    for kvcache in kvcaches:
        kvcache.free()
    return sum(times) / len(times)


if __name__ == "__main__":
    sparam = SamplingParams(temperature=0.0, top_p=1.0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--bsz", type=int)
    args = parser.parse_args()
    begin = convert_name_to_id(args.begin)
    end = convert_name_to_id(args.end)
    prefix = args.prefix
    bsz = args.bsz

    layer_manager = LayerManager(torch.device("cuda"))
    layer_storage = layer_manager.get_layer_storage(layers[begin:end])
    num_layers = sum(layer.count(".") for layer in layers[begin:end])
    gpu_page_pool = GpuPagePool(
        num_layers, 9000, 8, torch.device("cuda"), torch.float16
    )
    page_table_allocator = PageTableAllocator(256, 4096, 8, torch.device("cuda"))
    print(
        f"Decode time: {profile_decode(layer_storage, gpu_page_pool, page_table_allocator, begin, bsz, prefix)} ms",
    )
