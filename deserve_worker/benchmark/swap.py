import argparse
import time

import torch

from deserve_worker.benchmark.utils import convert_name_to_id, layers
from deserve_worker.engine.microbatch.processor import MicroBatchProcessor
from deserve_worker.kvcache.manager import KVCacheManager
from deserve_worker.kvcache.paged.page_pool import CpuPagePool
from deserve_worker.kvcache.virtual import VirtualPagePool
from deserve_worker.layer_storage import LayerManager
from deserve_worker.model.args import llama_3_70b_args
from deserve_worker.task import SamplingParams, TaskData, TaskManager

if __name__ == "__main__":
    sparam = SamplingParams(temperature=0.0, top_p=1.0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--bsz", type=int)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--main-size", type=int)
    parser.add_argument("--swap-size", type=int)
    parser.add_argument("--adjust", action="store_true")
    args = parser.parse_args()
    begin = convert_name_to_id(args.begin)
    end = convert_name_to_id(args.end)
    bsz = args.bsz
    prefix = args.prefix
    main = args.main_size
    swap = args.swap_size
    adjust = args.adjust
    layer_manager = LayerManager(torch.device("cuda"))
    layer_storage = layer_manager.get_layer_storage(layers[begin:end])
    num_layers = sum(layer.count(".") for layer in layers[begin:end])

    virtual_page_pool = VirtualPagePool(
        num_layers, main, swap, 8, torch.device("cuda"), torch.float16
    )
    cpu_page_pool = CpuPagePool(num_layers, main * 3, 8, torch.float16)
    kvcache_manager = KVCacheManager(virtual_page_pool, cpu_page_pool)
    task_manager = TaskManager(main, 8)
    processor0 = MicroBatchProcessor(
        kvcache_manager,
        task_manager,
        layer_storage,
    )
    processor1 = MicroBatchProcessor(
        kvcache_manager,
        task_manager,
        layer_storage,
    )
    processor2 = MicroBatchProcessor(
        kvcache_manager,
        task_manager,
        layer_storage,
    )

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

    for i in range(bsz):
        task_data = TaskData.empty(
            task_id=f"{prefix}-{bsz}-{i}",
            seqlen=prefix,
            sampling_params=sparam,
        )
        task_manager.add(task_data)
        processor0.join([task_data.task_id])
        processor0.step(
            [task_data.task_id], prefill_input, True, processor1, processor1
        )
        task_data.step()

    times = []
    for i in range(100):
        torch.cuda.synchronize()
        time_begin = time.time()
        if adjust:
            processor0.adjust()
        processor0.step(
            list(task_manager.task_datas.keys()),
            decode_input,
            False,
            processor1,
            processor2,
        )
        for task_data in task_manager.task_datas.values():
            task_data.step()
        torch.cuda.synchronize()
        time_end = time.time()
        if i >= 20:
            times.append((time_end - time_begin) * 1000)
    print(f"Swap time: {sum(times) / len(times)} ms")
