import argparse
import time

import torch

from deserve_worker.benchmark.utils import convert_name_to_id, layers
from deserve_worker.execution.exec import BatchPrefill
from deserve_worker.kvcache.paged.kvcache import PagedKVCache
from deserve_worker.kvcache.paged.page_pool import GpuPagePool
from deserve_worker.kvcache.paged.page_table import PageTableAllocator
from deserve_worker.layer_storage import LayerManager, LayerStorage
from deserve_worker.model.args import llama_3_70b_args
from deserve_worker.task import SamplingParams, TaskData


def profile_prefill(
    layer_storage: LayerStorage,
    gpu_page_pool: GpuPagePool,
    page_table_allocator: PageTableAllocator,
    begin: int,
    bsz: int,
    prefix: int,
) -> float:
    times = []
    for i in range(20):
        if begin == 0:
            input = torch.randint(
                1,
                llama_3_70b_args.vocab_size,
                (prefix * bsz,),
                dtype=torch.int,
                device=torch.device("cuda"),
            )
        else:
            input = torch.rand(
                prefix * bsz,
                llama_3_70b_args.dim,
                dtype=torch.float16,
                device=torch.device("cuda"),
            )
        task_datas: list[TaskData] = []
        kvcaches: list[PagedKVCache[GpuPagePool]] = []
        for j in range(bsz):
            task_data = TaskData(
                task_id=f"{prefix}-{bsz}-{i}-{j}",
                start_pos=0,
                round=0,
                seqlen=prefix,
                initial_seqlen=prefix,
                sampling_params=sparam,
            )
            task_datas.append(task_data)
            kvcaches.append(PagedKVCache.empty(page_table_allocator, gpu_page_pool))
        torch.cuda.synchronize()
        begin_time = time.time()
        prefill = BatchPrefill(
            input,
            layer_storage,
            task_datas,
            kvcaches,
        )
        prefill.prepare()
        prefill.step(ignore_eos=False)
        torch.cuda.synchronize()
        end_time = time.time()
        for kvcache in kvcaches:
            kvcache.free()
        if i >= 5:
            times.append((end_time - begin_time) * 1000)
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
    page_table_allocator = PageTableAllocator(num_layers, 4096, 8, torch.device("cuda"))
    gpu_page_pool = GpuPagePool(
        num_layers, 9000, 8, torch.device("cuda"), torch.float16
    )
    print(
        f"Prefill time: {profile_prefill(layer_storage, gpu_page_pool, page_table_allocator, begin, bsz, prefix)} ms"
    )
