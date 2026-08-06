# Performance Config Generator

Generates `server.yml` and `client.yml` for guidellm benchmarking in
[model-validation-configs](https://github.com/neuralmagic/model-validation-configs).

```bash
python generate_perf_config.py <model_id> [--gpu GPU] [--quantization QUANT]
```

## How it works

Every parameter is derived from one number: **how many concurrent 
benchmark requests can the KV cache hold**.

```
GPU memory
    │
    ├── Model weights (estimated from architecture + quantization)
    ├── Overhead (~1.5 GB for CUDA context, activations)
    └── KV cache (everything left over)
            │
            ├── KV per token = layers × 2 × kv_heads_per_gpu × head_dim × 2 bytes
            │
            └── Total KV token capacity = KV cache memory / KV per token
                    │
                    ├─→ max_concurrent_seqs = capacity / (prompt_tok + output_tok)
                    │       │
                    │       ├─→ concurrency rates (from [1,10,25,50,100], filtered)
                    │       └─→ prompt/output tokens (1000 default, reduced if tight)
                    │
                    └─→ max-model-len (capacity-driven, capped for large dense models)
```

## Parameter decisions

### Tensor Parallel Size

Picks the **smallest TP** where KV cache can sustain at least **25 concurrent
requests** at 2000 tokens each (1000 prompt + 1000 output):

```
for tp in [1, 2, 4, 8]:
    weight_per_gpu = model_weight / tp
    kv_headroom = gpu_memory × utilization − weight_per_gpu − overhead
    kv_capacity = kv_headroom / kv_per_token
    if kv_capacity / 2000 ≥ 25:
        use this tp
```

### max-model-len

Capacity-driven with a cap for large dense models:

| Concurrent workload seqs | max-model-len |
|--------------------------|---------------|
| ≥ 25                     | 10000         |
| ≥ 12                     | 8192          |
| ≥ 6                      | 4096          |
| < 6                      | 2048          |

Caps:
- Dense models > 40B active params, unquantized → 4096
- Dense models > 40B active params, quantized → 8192
- MoE models use **active params** (not total) for this check
- Multimodal models → 8192

### Prompt and output tokens

Target: **1000 prompt + 1000 output** (standard benchmark workload).

If KV cache is tight, steps down through [1000, 750, 512, 256] until
at least 25 concurrent sequences fit. This keeps the concurrency sweep
meaningful even on constrained hardware.

### Concurrency rates

Picked from `[1, 10, 25, 50, 100]`:
- Always includes **1** (baseline latency measurement)
- Only includes rates ≤ max concurrent sequences
- At most **4** rates

### gpu-memory-utilization

| Weight fraction per GPU | gpu-memory-utilization |
|-------------------------|------------------------|
| > 70%                   | 0.97                   |
| > 55%                   | 0.95                   |
| ≤ 55%                   | omitted (vLLM default 0.90) |

### max-num-batched-tokens

Only set for multimodal models (default 32768), to satisfy vLLM's
`max_num_batched_tokens ≥ max_tokens_per_mm_item` constraint.

## Weight estimation

### Dense models

```
params_per_layer = Q + K + V + O projections + MLP (gate + up + down) + layer norms
total = layers × params_per_layer + embeddings + lm_head
weight_bytes = total_params × dtype_bytes
```

### MoE models

Uses `moe_intermediate_size` (per-expert MLP width) when available,
not `intermediate_size` (shared MLP). A 80B MoE with 512 experts at
`moe_intermediate_size=512` is ~150 GB, not ~1400 GB.

```
expert_mlp = moe_intermediate_size × hidden_size × 3 × num_experts
shared_mlp = shared_expert_intermediate_size × hidden_size × 3
router = hidden_size × num_experts
```

### dtype_bytes

| Quantization | Bytes per param |
|--------------|-----------------|
| none (BF16)  | 2               |
| fp8          | 1               |
| w4a16        | 0.5             |
| w8a8         | 1               |

## Supported GPUs

| GPU | Memory |
|-----|--------|
| A100-80GB | 80 GB |
| A100-40GB | 40 GB |
| H100-80GB | 80 GB |
| H200-141GB | 141 GB |
| L40S-48GB | 48 GB |
| L4-24GB | 24 GB |
| MI300X-192GB | 192 GB |
| MI250X-128GB | 128 GB |
| MI210-64GB | 64 GB |

## Examples

```bash
# Auto-detect from HuggingFace
python generate_perf_config.py Qwen/Qwen3-Next-80B-A3B-Instruct

# Specific GPU
python generate_perf_config.py openai/gpt-oss-120b --gpu MI300X-192GB

# Quantized model
python generate_perf_config.py RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --quantization fp8

# Manual overrides
python generate_perf_config.py openai/gpt-oss-20b --tp 2 --max-model-len 8192

# Internal model (not on HuggingFace)
python generate_perf_config.py my-org/custom-model \
    --num-layers 44 --num-attention-heads 64 --num-kv-heads 8 \
    --hidden-size 8192 --intermediate-size 22016 --vocab-size 100000

# Preview without writing files
python generate_perf_config.py openai/gpt-oss-20b --dry-run
```
