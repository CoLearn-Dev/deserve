# Experiments

## Overview

The setup of experiments will consist of two parts: 
1. Workers. 
2. Benchmark clients. 

Overall, the experiments will be run in the following way:
1. Start workers on different GPUs. 
2. Run benchmark clients.

We will start from installation, then run workers, then run benchmark clients.

## Installation

### Worker

In installation, we will assume that the current directory is the root of the repository (`deserve/`).

It's recommended to use a new environment for the experiments. 

```bash
conda create -n deserve python=3.12
```

Then install the dependencies for deserve worker:

```bash
pip install -e deserve_worker --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/
```

You should change the `extra-index-url` to the correct one for your environment. Take a look at your `torch` version and `cuda` version.
If you cannot use `torch2.4`, you can try to install the correct version of `torch` and `flashinfer` by both modifying `deserve_worker/pyproject.toml` and `--extra-index-url` in the installation command.

### Benchmark Client

Install the dependencies for deserve benchmark:

```bash
pip install -e deserve_benchmark
```

For using the auto tokenizer provided by huggingface, you need to go through its verification process, or you can use local tokenizer by modifying code in `deserve_benchmark/rater.py` and `deserve_benchmark/benchmark/*.py`.

## Start Workers

A worker is a process that runs the specific part of the model. 
It will process the activations/tokens from the previous worker by forwarding through the layers and send the results to the next worker.
I will use an example to demonstrate how to start workers and the specific meaning of the arguments.

```bash
python3 -m deserve_worker.worker_api --model=llama-3-70b --num-rounds=2 --layer-begin=40 --layer-end=output --batch-size=128 --port=8081 --num-main-pages=9000 --num-swap-pages=0 --controller-url=http://localhost:19000 --next-worker-url=http://localhost:8080 --simulated-latency=0.0 --prefill-first-aggregate --buddy-height=16
```

Following is the meaning of the arguments:

- `--model`: the name of the model, currently support `llama-3-70b` and `llama-3-8b`.
- `--num-rounds`: the number of machines inside the pipeline. 
- `--layer-begin`: the layer to start from, set to `emb` for embedding layer, `output` for output layer, number for the specific layer.
- `--layer-end`: the layer to end at, set to `emb` for embedding layer, `output` for output layer, number for the specific layer.
- `--batch-size`: the micro-batch size.
- `--port`: the port to run the worker on.
- `--num-main-pages`: the number of main pages, which will be used by all microbatches.
- `--num-swap-pages`: the number of swap pages, which will be used by a specific microbatch.
- `--controller-url`: the url of the controller, which is the benchmark client in the context of experiments.
- `--next-worker-url`: the url of the next worker for sending activations/tokens to.
- `--simulated-latency`: the simulated network latency inside the pipeline.

To simplify the process of starting workers, we provide a script `start_expr.py` to simultaneously start workers on different GPUs. Please look at the argparser for more details.

## Run Experiments

After all workers are started, you can run experiments using `deserve_benchmark`. Here is an example command:

```bash
python3 -m src.benchmark.deserve --batch-size=512 --workload=fixed512:256:0 --max-tokens=256 --trace
```

Following is the meaning of the arguments:  

- `--batch-size`: the number of requests that will be simultaneously sent to the pipeline.
- `--workload`: the workload to run, in the format of `total_num_requests:input_length:variation`, also you can use `oasst1` or `sharegpt` for real-world workloads.
- `--max-tokens`: the maximum number of tokens to generate.
- `--trace`: whether to trace the execution.

You should look at `deserve_benchmark/benchmark/deserve.py` for more details about the arguments.
