## Example

### Setup

Firstly, install DeServe: 

```
git clone git@github.com:CoLearn-Dev/deserve.git
cd deserve 
pip install -e deserve_worker --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
pip install -e deserve_controller
pip install -e deserve_client
```

After that, setup workers and controller. For example, for a pipeline with 2 workers: 

```bash 
python3 -m deserve_worker.worker_api --model=llama-3-70b --num-rounds=2 --layer-begin=emb --layer-end=40 --batch-size=48 --port=8080 --controller-url=http://localhost:19000 --next-worker-url=http://localhost:8081
```

```bash 
python3 -m deserve_worker.worker_api --model=llama-3-70b --num-rounds=2 --layer-begin=40 --layer-end=output --batch-size=48 --port=8081 --controller-url=http://localhost:19000 --next-worker-url=http://localhost:8080
```

You could observe the basic resources information after the worker is started.

Then, setup the controller: 
```
python3 -m deserve_controller.controller_api
```

### Run

After the controller is started, you could use client to send requests to the controller. For example, if you want to run an inference of llama-3-70b, you could use the following command: 

```bash 
python3 -m deserve_client.client complete "meta-llama/Meta-Llama-3-70B-Instruct" "What's the capital of France?"
```

You could also use `trace` command to get more detailed information about the inference, such as the values of intermediate tensors: 

```bash 
python3 -m deserve_client.client trace "meta-llama/Meta-Llama-3-70B-Instruct" "What's the capital of France?" trace.pkl
```

Next token, probabilities, and traces would be dumped to `trace.pkl`. You could use `pickle` to load them in your script.

### Benchmark 

If you want to benchmark the performance of DeServe worker on different machines, here are some scripts for you to use. 

For benchmark prefilling speed, you could use the following command: 

```bash
python3 -m deserve_worker.benchmark.prefill --begin=emb --end=40 --prefix=256 --bsz=16
```

For benchmark decoding speed, you could use the following command: 

```bash
python3 -m deserve_worker.benchmark.decode --begin=emb --end=40 --prefix=128 --bsz=64
```

The `begin` and `end` arguments control the range of layers that the benchmark script will run on. For example, `begin=emb` and `end=40` means the script will run on layers from `emb` to `layers.40`. 