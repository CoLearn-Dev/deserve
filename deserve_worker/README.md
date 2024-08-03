# deserve worker 

## How to run

```bash 
python3 -m deserve_worker.worker_api <port> <worker_id>
```

For example,

```bash 
python3 -m deserve_worker.worker_api 8080 worker0
```

## API

### Inference

To inference, you need to pass a plan and other metadata in the request body. You have to send it to the first worker. The plan is a list of workers with their layers. The first worker will send the request to the next worker in the plan. The last worker will return the token to the controller. Here is an example: 

```python
plan = [
    {
        "worker_id": worker_id0,
        "worker_url": "http://localhost:8080",
        "layers": [
            "llama-3-8b-instruct-slice/tok_embeddings",
            *[f"llama-3-8b-instruct-slice/layers.{i}" for i in range(0, 16)],
        ],
    },
    {
        "worker_id": worker_id1,
        "worker_url": "http://localhost:8081",
        "layers": [
            *[f"llama-3-8b-instruct-slice/layers.{i}" for i in range(16, 32)],
            "llama-3-8b-instruct-slice/norm",
            "llama-3-8b-instruct-slice/output",
        ],
    },
]

metadata = {
    "task_id": task_id,
    "round": 0,
    "plan": plan,
    "sampling_params": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_total_len": 2048,
    },
}

tensors = {"x": tokens}

requests.post(
    "http://localhost:8080/forward", data=dumps(tensors, metadata)
)
```

### Trace 

To trace, the plan is also required. It is worth noting that trace use different kernel for computation and dumping.


### Cancel

You should not cancel a task. It's used for freeing resources like KV caches. 