# DeServe

DeServe is a offline-serving framework for decentralized inference of large language models. Benefiting from following technologies, DeServe can achieve up to **12.6x** throughput improvement in high-latency network compared to the vLLM with pipeline parallelism. Following features are key to the performance:

- **KV Cache Swapping**: Maximizes GPU computation utilization by enlarging the KV cache size through swapping microbatch memory between CPU and GPU. 
- **Microbatch Scheduling**: Allocates microbatches inside the pipeline for different network latency to maximize the throughput.

To start the experiments, please refer to [deserve_exp/readme.md](deserve_exp/readme.md).

## Citation

If you find this useful in your research, please consider citing:

```

```
