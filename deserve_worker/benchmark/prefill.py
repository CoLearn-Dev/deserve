import argparse
import sys
import time

import torch

from deserve_worker.benchmark.utils import convert_name_to_id, layers
from deserve_worker.execution.exec import BatchPrefill
from deserve_worker.model.args import llama_3_70b_args
from deserve_worker.task import SamplingParams, TaskData, TaskInfo
from deserve_worker.worker import Worker


def profile_prefill(
    worker: Worker, begin: int, end: int, bsz: int, prefix: int
) -> float:
    layer_storage = worker.layer_manager.get_layer_storage(layers[begin:end])
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
        for j in range(bsz):
            data = worker.init_task_data(
                TaskInfo(
                    task_id=f"{prefix}-{bsz}-{i}-{j}",
                    plan=[],
                    round=0,
                    seqlen=prefix,
                    sampling_params=sparam,
                ),
                False,
            )
            assert isinstance(data, TaskData)
            task_datas.append(data)
        torch.cuda.synchronize()
        begin_time = time.time()
        prefill = BatchPrefill(input, layer_storage, worker.page_pool, task_datas)
        prefill.step()
        torch.cuda.synchronize()
        end_time = time.time()
        for data in task_datas:
            data.kvcache.clear()
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

    worker = Worker("test", "test", 48, "test")
    print(f"Prefill time: {profile_prefill(worker, begin, end, bsz, prefix)} ms")
