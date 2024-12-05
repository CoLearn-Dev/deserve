# Reproducibility

## How to Start Workers

Use `start_expr.py` to start workers. Please look at the argparser for more details.

## How to Run Experiments

Firstly you need to start workers using `start_expr.py`.

Then run experiments using `deserve_benchmark`. Here is an example command:

```bash
python3 -m src.benchmark.deserve --batch-size=512 --workload=fixed512:256:0 --max-tokens=256 --trace
```

You should look at `deserve_benchmark/benchmark/deserve.py` for more details about the arguments.