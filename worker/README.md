## Installation

### Install From PyPI
```
pip install fleece-worker
```

### Install From Source
```
pip install -e .
```

### (Optional) Install FlashAttention
https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features

## Connect to a controller

```
python -m fleece-worker -c <controller_url>  -t <api_token>
```
Optional: `--worker-nickname abc`, `--heartbeat-interval 10`, `-w <worker_url>`

For example:

```
python -m fleece-worker -c https://serving-api.colearn.cloud:8443 -t <api_token>
```

## Try it out (deprecated)

```
export CUDA_VISIBLE_DEVICES=0
python -m fleece-worker -w http://127.0.0.1:8080
```

```
python send_forward.py
```

```
curl localhost:8080/forward -H 'Content-Type: application/json' -d '{"task_id":"123","step":0,"round":0,"plan":[["local",["llama-3-8b-instruct-slice/tok_embeddings", "llama-3-8b-instruct-slice/layers.0", "llama-3-8b-instruct-slice/layers.1", "llama-3-8b-instruct-slice/layers.2", "llama-3-8b-instruct-slice/layers.3", "llama-3-8b-instruct-slice/layers.4", "llama-3-8b-instruct-slice/layers.5", "llama-3-8b-instruct-slice/layers.6", "llama-3-8b-instruct-slice/layers.7", "llama-3-8b-instruct-slice/layers.8", "llama-3-8b-instruct-slice/layers.9", "llama-3-8b-instruct-slice/layers.10", "llama-3-8b-instruct-slice/layers.11", "llama-3-8b-instruct-slice/layers.12", "llama-3-8b-instruct-slice/layers.13", "llama-3-8b-instruct-slice/layers.14", "llama-3-8b-instruct-slice/layers.15", "llama-3-8b-instruct-slice/layers.16", "llama-3-8b-instruct-slice/layers.17", "llama-3-8b-instruct-slice/layers.18", "llama-3-8b-instruct-slice/layers.19", "llama-3-8b-instruct-slice/layers.20", "llama-3-8b-instruct-slice/layers.21", "llama-3-8b-instruct-slice/layers.22", "llama-3-8b-instruct-slice/layers.23", "llama-3-8b-instruct-slice/layers.24", "llama-3-8b-instruct-slice/layers.25", "llama-3-8b-instruct-slice/layers.26", "llama-3-8b-instruct-slice/layers.27", "llama-3-8b-instruct-slice/layers.28", "llama-3-8b-instruct-slice/layers.29", "llama-3-8b-instruct-slice/layers.30", "llama-3-8b-instruct-slice/layers.31", "llama-3-8b-instruct-slice/norm", "llama-3-8b-instruct-slice/output"]]],"payload":[[128000, 128006, 882, 128007, 271, 12840, 374, 279, 11363, 315, 1253, 13767, 1082, 30, 128009, 128006, 78191, 128007, 271]]}'
```
```
curl localhost:8080/forward -H 'Content-Type: application/json' -d '{"task_id":"123","step":0,"round":0,"plan":[["local",["llama-3-8b-instruct-slice/tok_embeddings", "llama-3-8b-instruct-slice/layers.0", "llama-3-8b-instruct-slice/layers.1", "llama-3-8b-instruct-slice/layers.2", "llama-3-8b-instruct-slice/layers.3", "llama-3-8b-instruct-slice/layers.4", "llama-3-8b-instruct-slice/layers.5", "llama-3-8b-instruct-slice/layers.6", "llama-3-8b-instruct-slice/layers.7", "llama-3-8b-instruct-slice/layers.8", "llama-3-8b-instruct-slice/layers.9", "llama-3-8b-instruct-slice/layers.10", "llama-3-8b-instruct-slice/layers.11", "llama-3-8b-instruct-slice/layers.12", "llama-3-8b-instruct-slice/layers.13", "llama-3-8b-instruct-slice/layers.14", "llama-3-8b-instruct-slice/layers.15", "llama-3-8b-instruct-slice/layers.16", "llama-3-8b-instruct-slice/layers.17", "llama-3-8b-instruct-slice/layers.18", "llama-3-8b-instruct-slice/layers.19", "llama-3-8b-instruct-slice/layers.20", "llama-3-8b-instruct-slice/layers.21", "llama-3-8b-instruct-slice/layers.22", "llama-3-8b-instruct-slice/layers.23", "llama-3-8b-instruct-slice/layers.24", "llama-3-8b-instruct-slice/layers.25", "llama-3-8b-instruct-slice/layers.26", "llama-3-8b-instruct-slice/layers.27", "llama-3-8b-instruct-slice/layers.28", "llama-3-8b-instruct-slice/layers.29", "llama-3-8b-instruct-slice/layers.30", "llama-3-8b-instruct-slice/layers.31", "llama-3-8b-instruct-slice/norm", "llama-3-8b-instruct-slice/output"]]],"payload":[[128000, 128006, 882, 128007, 271, 12840, 374, 279, 11363, 315, 1253, 13767, 1082, 30, 128009, 128006, 78191, 128007, 271], [128000, 128006, 9125, 128007, 271, 38195, 4320, 449, 14433, 39342, 128009, 128006, 882, 128007, 271, 40, 1097, 2133, 311, 12366, 11, 1148, 1288, 358, 1518, 30, 128009, 128006, 78191, 128007, 271], [128000, 128006, 9125, 128007, 271, 38195, 4320, 449, 100166, 128009, 128006, 882, 128007, 271, 4438, 311, 733, 505, 27647, 311, 12551, 30, 128009, 128006, 78191, 128007, 271]]}'
```
> note that the model will be automatically downloaded to `~/.cache`
