# DeServe

DeServe is a offline-serving framework for decentralized inference of large language models. Benefiting from following technologies, DeServe can achieve up to **12.6x** throughput improvement in high-latency network compared to the vLLM with pipeline parallelism. Following features are key to the performance:

- **KV Cache Swapping**: Maximizes GPU computation utilization by enlarging the KV cache size through swapping microbatch memory between CPU and GPU. 
- **Microbatch Scheduling**: Allocates microbatches inside the pipeline for different network latency to maximize the throughput.

| Latency (ms)  | real-world centralized | real-world east-west | sim 16 | sim 32 | sim 64 | sim 256 |
|----------------|-----------------------|----------------------|--------|--------|--------|---------|
| vLLM (tp)      | 253.0                 | failed               | /      | /      | /      | /       |
| vLLM (pp)      | 89.1                  | 37.3                 | 68.8   | 55.3   | 36.1   | /       |
| DeServe (pp)   | 194.6                 | 138.4                | 182.3  | 163.7  | 133.7  | /       |
| DeServe (opt)  | 445.2                 | 434.1                | 458.5  | 457.3  | 456.8  | 442.9   |

To start the experiments, please refer to [deserve_exp/readme.md](deserve_exp/readme.md).

## Citation

If you find this useful in your research, please consider citing:

```

```
