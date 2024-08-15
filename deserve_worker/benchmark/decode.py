import argparse
import sys
import time

import torch

from deserve_worker.benchmark.utils import convert_name_to_id, layers
from deserve_worker.execution.exec import BatchDecode, BatchPrefill
from deserve_worker.model.args import llama_3_70b_args
from deserve_worker.task import SamplingParams, TaskData, TaskInfo
from deserve_worker.worker import Worker

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

    worker = Worker("test", "test", 48, "test")
    layer_storage = worker.layer_manager.get_layer_storage(layers[begin:end])
    if prefix * bsz >= 2048 * 48:
        print("skip")
        exit(0)
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
    for i in range(bsz):
        data = worker.init_task_data(
            TaskInfo(
                task_id=f"{prefix}-{bsz}-{i}",
                plan=[],
                round=0,
                seqlen=prefix,
                sampling_params=sparam,
            ),
            False,
        )
        assert isinstance(data, TaskData)
        task_datas.append(data)
        prefill = BatchPrefill(prefill_input, layer_storage, worker.page_pool, [data])
        prefill.step()
        data.seqlen = 1
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        time_begin = time.time()
        decode = BatchDecode(decode_input, layer_storage, worker.page_pool, task_datas)
        decode.step()
        torch.cuda.synchronize()
        time_end = time.time()
        if i >= 20:
            times.append((time_end - time_begin) * 1000)
    print(f"Decode time: {sum(times) / len(times)} ms")
    for data in task_datas:
        data.kvcache.clear()
