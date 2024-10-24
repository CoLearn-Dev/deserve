# DeServe Worker 

## Install
```
pip install -e deserve_worker --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/
```

## How to run

For example, for a pipeline with 2 workers:

```bash 
python3 -m deserve_worker.worker_api --model=llama-3-70b --num-rounds=2 --layer-begin=emb --layer-end=40 --batch-size=48 --port=8080 --num-rounds=2 --num-main-pages=7000 --num-swap-pages=1000 --controller-url=http://localhost:19000 --next-worker-url=http://localhost:8081
```

```bash 
python3 -m deserve_worker.worker_api --model=llama-3-70b --num-rounds=2 --layer-begin=40 --layer-end=output --batch-size=48 --port=8081 --num-rounds=2 --num-main-pages=7000 --num-swap-pages=1000 --controller-url=http://localhost:19000 --next-worker-url=http://localhost:8080
```