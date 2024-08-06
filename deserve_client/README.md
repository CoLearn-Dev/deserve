# DeServe Client

## How To Run

For completion: 
```bash
python3 -m deserve_client.client complete meta-llama/Meta-Llama-3-8B-Instruct "Here is a text prompt."
```

For dumping traces of prefill: 
```bash 
python3 -m deserve_client.client trace meta-llama/Meta-Llama-3-8B-Instruct "Here is a text prompt."
```

For verifying the correctness of the trace: 
```bash
python3 -m deserve_client.client verify meta-llama/Meta-Llama-3-8B-Instruct "Here is a text prompt."
```
