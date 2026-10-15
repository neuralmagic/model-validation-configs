#!/usr/bin/env python3
"""Generate vLLM performance test configs (server.yml + client.yml).

Fetches model architecture from HuggingFace and computes optimal serving
parameters for benchmarking. For models not on HuggingFace, pass architecture
specs via CLI flags.

Usage:
    # Auto-detect from HuggingFace
    python generate_perf_config.py meta-llama/Llama-3.3-70B-Instruct

    # With quantization
    python generate_perf_config.py RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \\
        --quantization fp8

    # Internal model with manual specs
    python generate_perf_config.py openai/gpt-oss-20b \\
        --num-params 20 --num-layers 44 --num-kv-heads 8 \\
        --num-attention-heads 64 --hidden-size 8192 \\
        --intermediate-size 22016 --vocab-size 100000

    # Gated HF models (e.g. Llama)
    python generate_perf_config.py meta-llama/Llama-3.3-70B-Instruct \\
        --hf-token hf_xxx...

Algorithm doc: https://gist.github.com/cmiyai/db483579b4a69e4d91eea804bd1552d6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TP = [1, 2, 4, 8]
OVERHEAD_GB = 1.5
STANDARD_LENGTHS = [2048, 4096, 8192, 10000, 16384, 32768]

GPU_MEMORY_GB = {
    # CUDA
    "A100-80GB": 80,
    "A100-40GB": 40,
    "H100-80GB": 80,
    "H200-141GB": 141,
    "L40S-48GB": 48,
    "L4-24GB": 24,
    # ROCm
    "MI300X-192GB": 192,
    "MI250X-128GB": 128,
    "MI210-64GB": 64,
    # ADD MORE GPUS Here
}

DTYPE_BYTES = {
    "none": 2,       # BF16: Default
    "fp8": 1,
    "w4a16": 0.5,
    "w8a8": 1,
}

# Where huggingface-cli stores tokens
HF_TOKEN_PATHS = [
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
]

# ---------------------------------------------------------------------------
# Model architecture spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    model_id: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    head_dim: int = 0
    max_position_embeddings: int = 32768
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    is_multimodal: bool = False
    mm_tokens_per_item: int = 0
    tie_word_embeddings: bool = False
    tokenizer_class: str = ""

    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# HuggingFace config fetcher
# ---------------------------------------------------------------------------

# you will need your own huggingface token and permission for specific models
def _resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_var)
        if val:
            return val
    for path in HF_TOKEN_PATHS:
        try:
            return path.read_text().strip()
        except (FileNotFoundError, PermissionError):
            continue
    if HF_TOKEN:
        return HF_TOKEN
    return None


def fetch_hf_config(model_id: str, hf_token: str | None = None) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    headers = {"User-Agent": "vllm-perf-config-gen/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit(
                f"401 Unauthorized for '{model_id}' — this is a gated model.\n"
                f"Pass --hf-token <token> or set HF_TOKEN env var.\n"
                f"Get a token at https://huggingface.co/settings/tokens"
            )
        if e.code == 403:
            raise SystemExit(
                f"403 Forbidden for '{model_id}' — your token is valid but you\n"
                f"haven't accepted the model's license agreement.\n"
                f"Visit https://huggingface.co/{model_id} and accept the license,\n"
                f"then retry."
            )
        if e.code == 404:
            raise SystemExit(
                f"Could not fetch config.json for '{model_id}' from HuggingFace.\n"
                f"If this is an internal model, pass architecture flags manually:\n"
                f"  --num-params 20 --num-layers 44 --num-kv-heads 8 ..."
            )
        raise


MULTIMODAL_KEYS = {"image_size", "vision_config", "audio_config",
                   "mm_vision_tower", "visual", "vision_tower",
                   "image_token_id", "video_token_id"}


def parse_hf_config(model_id: str, config: dict[str, Any]) -> ModelSpec:
    text_cfg = config
    if "text_config" in config:
        text_cfg = config["text_config"]

    num_kv_heads = (
        text_cfg.get("num_key_value_heads")
        or text_cfg.get("num_kv_heads")
        or text_cfg.get("num_attention_heads", 32)
    )

    is_mm = bool(MULTIMODAL_KEYS & set(config.keys()))

    mm_tokens = 0
    if is_mm:
        vision = config.get("vision_config", {})
        img_size = vision.get("image_size", 0)
        patch_size = vision.get("patch_size", 1)
        if img_size and patch_size:
            mm_tokens = (img_size // patch_size) ** 2

    return ModelSpec(
        model_id=model_id,
        num_layers=text_cfg.get("num_hidden_layers", text_cfg.get("num_layers", 32)),
        num_attention_heads=text_cfg.get("num_attention_heads", 32),
        num_kv_heads=num_kv_heads,
        hidden_size=text_cfg.get("hidden_size", 4096),
        head_dim=text_cfg.get("head_dim", 0),
        intermediate_size=text_cfg.get("intermediate_size", 11008),
        vocab_size=text_cfg.get("vocab_size", 32000),
        max_position_embeddings=text_cfg.get("max_position_embeddings", 32768),
        num_experts=text_cfg.get("num_local_experts") or text_cfg.get("num_experts") or 0,
        num_experts_per_tok=text_cfg.get("num_experts_per_tok") or text_cfg.get("num_selected_experts") or 0,
        moe_intermediate_size=text_cfg.get("moe_intermediate_size") or 0,
        shared_expert_intermediate_size=text_cfg.get("shared_expert_intermediate_size") or 0,
        architectures=config.get("architectures", []),
        model_type=config.get("model_type", ""),
        is_multimodal=is_mm,
        mm_tokens_per_item=mm_tokens,
        tie_word_embeddings=text_cfg.get("tie_word_embeddings", False),
        tokenizer_class=config.get("tokenizer_class", ""),
    )


# ---------------------------------------------------------------------------
# Algorithm 1: Model weight size
# ---------------------------------------------------------------------------

def estimate_weight_bytes(spec: ModelSpec, quantization: str) -> int:
    # Attention params (same for dense and MoE)
    attn_params = (
        spec.hidden_size * spec.hidden_size                          # Q
        + spec.num_kv_heads * spec.head_dim * spec.hidden_size       # K
        + spec.num_kv_heads * spec.head_dim * spec.hidden_size       # V
        + spec.hidden_size * spec.hidden_size                        # O
    )

    # MLP params — MoE models have per-expert MLP (often much smaller)
    # plus an optional shared expert
    if spec.num_experts > 0 and spec.moe_intermediate_size > 0:
        per_expert_mlp = spec.moe_intermediate_size * spec.hidden_size * 3
        expert_mlp_total = per_expert_mlp * spec.num_experts
        shared_mlp = 0
        if spec.shared_expert_intermediate_size > 0:
            shared_mlp = spec.shared_expert_intermediate_size * spec.hidden_size * 3
        # Router gate
        router = spec.hidden_size * spec.num_experts
        mlp_params = expert_mlp_total + shared_mlp + router
    elif spec.num_experts > 0:
        # MoE but no separate moe_intermediate_size — use intermediate_size per expert
        mlp_params = spec.intermediate_size * spec.hidden_size * 3 * spec.num_experts
        mlp_params += spec.hidden_size * spec.num_experts  # router
    else:
        # Dense model
        mlp_params = spec.intermediate_size * spec.hidden_size * 3

    params_per_layer = (
        attn_params + mlp_params
        + spec.hidden_size * 4  # layer norms
    )

    embedding_params = spec.vocab_size * spec.hidden_size
    lm_head_params = 0 if spec.tie_word_embeddings else spec.vocab_size * spec.hidden_size

    total_params = spec.num_layers * params_per_layer + embedding_params + lm_head_params
    return int(total_params * DTYPE_BYTES[quantization])


# ---------------------------------------------------------------------------
# Algorithm 2: Tensor parallel size
#
# Pick the smallest TP where:
#   1. Weights fit on each GPU
#   2. KV cache can hold at least 25 concurrent workload-sized sequences
#      (enough for a meaningful guidellm concurrency sweep)
# ---------------------------------------------------------------------------

TARGET_CONCURRENT_SEQS = 25

def _kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int,
                        tp: int, kv_dtype_bytes: int = 2) -> int:
    kv_heads_per_gpu = max(1, num_kv_heads // tp)
    per_layer = 2 * kv_heads_per_gpu * head_dim * kv_dtype_bytes
    return num_layers * per_layer


def _kv_capacity(model_gb: float, tp: int, gpu_memory_gb: float,
                 gpu_util: float, kv_per_token: int) -> int:
    weight_per_gpu = model_gb / tp
    headroom_bytes = max(0, (gpu_memory_gb * gpu_util - weight_per_gpu - OVERHEAD_GB) * (1024**3))
    if kv_per_token <= 0:
        return 0
    return int(headroom_bytes / kv_per_token)


def pick_tp(model_weight_bytes: int, gpu_memory_gb: float, num_gpus: int,
            spec: ModelSpec) -> int:
    model_gb = model_weight_bytes / (1024**3)

    best_tp = min(num_gpus, VALID_TP[-1])
    for tp in VALID_TP:
        if tp > num_gpus:
            continue
        weight_per_gpu = model_gb / tp
        usable = gpu_memory_gb * 0.90 - OVERHEAD_GB
        if weight_per_gpu > usable:
            continue

        kv_pt = _kv_bytes_per_token(spec.num_layers, spec.num_kv_heads,
                                     spec.head_dim, tp)
        gpu_util = 0.97 if (weight_per_gpu / gpu_memory_gb) > 0.70 else \
                   0.95 if (weight_per_gpu / gpu_memory_gb) > 0.55 else 0.90
        capacity = _kv_capacity(model_gb, tp, gpu_memory_gb, gpu_util, kv_pt)
        max_concurrent = capacity / WORKLOAD_TOKENS

        if max_concurrent >= TARGET_CONCURRENT_SEQS:
            return tp
        best_tp = tp

    return best_tp


# ---------------------------------------------------------------------------
# Algorithm 3: KV cache per token
# ---------------------------------------------------------------------------

def kv_bytes_per_token(spec: ModelSpec, tp: int, kv_dtype_bytes: int = 2) -> int:
    return _kv_bytes_per_token(spec.num_layers, spec.num_kv_heads,
                               spec.head_dim, tp, kv_dtype_bytes)


# ---------------------------------------------------------------------------
# Algorithm 4: max-model-len
#
# Primary driver: KV cache capacity — how many concurrent workload-sized
# sequences (2000 tokens = 1000 prompt + 1000 output) can we serve?
#
# Secondary: large dense models cap at shorter lengths (benchmarking
# convention — teams prefer testing throughput/concurrency). For MoE,
# use active params since total params is misleading (120B MoE with
# 5B active behaves like a 5B model for inference).
# ---------------------------------------------------------------------------

WORKLOAD_TOKENS = 2000  # prompt + output per benchmarking request

def pick_max_model_len(kv_cache_memory_bytes: int, kv_per_token: int,
                       native_ctx: int, active_params_b: float,
                       total_params_b: float, quantization: str,
                       is_multimodal: bool = False,
                       num_experts: int = 0) -> int:
    total_capacity = kv_cache_memory_bytes / max(kv_per_token, 1)

    # How many concurrent workload-sized sequences fit in KV cache?
    workload_seqs = total_capacity / WORKLOAD_TOKENS

    # Capacity-driven target
    if workload_seqs >= 25:
        target = 10000
    elif workload_seqs >= 12:
        target = 8192
    elif workload_seqs >= 6:
        target = 4096
    else:
        target = 2048

    # Large dense models: cap context length — benchmarking convention
    # favors throughput/concurrency over long-context testing.
    # For MoE, use active params (total is misleading).
    effective_size = active_params_b if num_experts > 0 else total_params_b
    if effective_size > 40 and quantization == "none":
        target = min(target, 4096)
    elif effective_size > 40:
        target = min(target, 8192)

    # Multimodal: extra memory from vision/audio encoders
    if is_multimodal:
        target = min(target, 8192)

    # Hard cap: at least 4 concurrent sequences at max-model-len
    max_fits = int(total_capacity / 4) if total_capacity > 0 else 0

    cap = min(target, max_fits, native_ctx)

    # Snap down to nearest standard length
    chosen = STANDARD_LENGTHS[0]
    for length in STANDARD_LENGTHS:
        if length <= cap:
            chosen = length
    return chosen


# ---------------------------------------------------------------------------
# Algorithm 5: max-num-batched-tokens
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def pick_max_num_batched_tokens(spec: ModelSpec) -> int | None:
    if spec.is_multimodal:
        if spec.mm_tokens_per_item:
            return max(32768, _next_power_of_2(spec.mm_tokens_per_item))
        return 32768
    return None


# ---------------------------------------------------------------------------
# Algorithm 6: gpu-memory-utilization
# ---------------------------------------------------------------------------

def pick_gpu_memory_util(model_weight_bytes: int, gpu_memory_gb: float,
                         tp: int) -> float | None:
    model_gb = model_weight_bytes / (1024**3)
    weight_per_gpu = model_gb / tp
    fraction = weight_per_gpu / gpu_memory_gb

    if fraction > 0.70:
        return 0.97
    elif fraction > 0.55:
        return 0.95
    return None


# ---------------------------------------------------------------------------
# Algorithm 7: Concurrency rates
# ---------------------------------------------------------------------------

def compute_active_params_b(spec: ModelSpec) -> float:
    if spec.num_experts > 0 and spec.num_experts_per_tok > 0:
        expert_mlp_size = spec.moe_intermediate_size or spec.intermediate_size
        mlp_params = expert_mlp_size * spec.hidden_size * 3
        active_mlp = mlp_params * spec.num_experts_per_tok
        attn_params = spec.hidden_size * spec.hidden_size * 4
        active_per_layer = attn_params + active_mlp
        embedding = spec.vocab_size * spec.hidden_size
        return (spec.num_layers * active_per_layer + embedding) / 1e9

    expert_mult = max(spec.num_experts, 1)
    params_per_layer = (
        spec.hidden_size * spec.hidden_size * 4
        + spec.intermediate_size * spec.hidden_size * 3 * expert_mult
        + spec.hidden_size * 4
    )
    total = spec.num_layers * params_per_layer + spec.vocab_size * spec.hidden_size * 2
    return total / 1e9


RATE_CANDIDATES = [1, 10, 25, 50, 100]

def pick_concurrency_rates(max_concurrent_seqs: int) -> list[int]:
    """Pick up to 4 rates from candidates, always starting with 1.
    Only include rates the KV cache can actually sustain."""
    rates = [r for r in RATE_CANDIDATES if r <= max_concurrent_seqs]
    if not rates or rates[0] != 1:
        rates = [1] + rates
    if len(rates) > 4:
        # Keep 1, then spread evenly across remaining candidates
        rest = [r for r in rates if r != 1]
        step = max(1, len(rest) // 3)
        rates = [1] + rest[::step][:3]
    return rates[:4]


# ---------------------------------------------------------------------------
# Algorithm 8: Special flags
# ---------------------------------------------------------------------------

def detect_special_flags(spec: ModelSpec) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    flags["enable-chunked-prefill"] = True

    arch = (spec.architectures[0] if spec.architectures else "").lower()
    mtype = spec.model_type.lower()

    if "mamba" in arch or "mamba" in mtype:
        flags["max-num-seqs"] = 256
        del flags["enable-chunked-prefill"]

    is_mistral = (
        spec.tokenizer_class == "MistralTokenizer"
        or mtype.startswith("mistral")
    )
    if "mistral" in spec.model_id.lower() and is_mistral:
        flags["tokenizer_mode"] = "mistral"
        flags["config_format"] = "mistral"
        flags["load_format"] = "mistral"

    if spec.is_multimodal:
        flags["no-enable-prefix-caching"] = True

    if "phi-3" in spec.model_id.lower():
        flags["no-enable-prefix-caching"] = True

    return flags


# ---------------------------------------------------------------------------
# Algorithm 9: Client config
#
# Token sizes and concurrency rates are co-dependent:
#   max_concurrent = kv_capacity / (prompt_tokens + output_tokens)
#
# Target: 1000 prompt + 1000 output (standard benchmark workload).
# If KV cache is tight, reduce tokens so the rate sweep can still
# reach meaningful concurrency (at least 25 concurrent).
# ---------------------------------------------------------------------------

PREFERRED_TOKENS = 1000  # ideal prompt = output = 1000
TOKEN_STEPS = [1000, 750, 512, 256]  # step down if KV is tight

def pick_workload_tokens(kv_total_tokens: int, max_model_len: int) -> int:
    """Pick the largest token size where we can sustain TARGET_CONCURRENT_SEQS."""
    for tok in TOKEN_STEPS:
        if tok * 2 > max_model_len:
            continue
        workload = tok * 2  # prompt + output
        concurrent = kv_total_tokens / workload if workload > 0 else 0
        if concurrent >= TARGET_CONCURRENT_SEQS:
            return tok
    return max(max_model_len // 8, 128)


def generate_client_config(concurrency_rates: list[int],
                           prompt_tokens: int, output_tokens: int) -> dict:
    return {
        "data": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        },
        "rate-type": "concurrent",
        "max-seconds": 500,
        "max-requests": 300,
        "rate": ",".join(str(r) for r in concurrency_rates),
        "warmup": {"percent": 0.05, "mode": "requests"},
        "cooldown": {"percent": 0.05, "mode": "requests"},
        "backend-kwargs": json.dumps({"extras": {"body": {
            "temperature": 0,
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
        }}}),
    }


# ---------------------------------------------------------------------------
# YAML writer (no dependency on PyYAML)
# ---------------------------------------------------------------------------

def _yaml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if any(c in v for c in "{}[],:\"'") or v.startswith("{"):
            return f"'{v}'"
        return v
    return str(v)


def dict_to_yaml(d: dict, indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(dict_to_yaml(v, indent + 1))
        else:
            lines.append(f"{prefix}{k}: {_yaml_value(v)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class GeneratedConfig:
    server: dict[str, Any]
    client: dict[str, Any]
    spec: ModelSpec
    tp: int
    max_model_len: int
    model_weight_gb: float
    kv_per_token: int
    kv_cache_gb: float
    active_params_b: float
    max_concurrent_seqs: int
    gpu_memory_util: float | None


def generate(spec: ModelSpec, gpu_type: str, num_gpus: int,
             quantization: str, tp_override: int | None = None,
             mml_override: int | None = None) -> GeneratedConfig:
    gpu_mem = GPU_MEMORY_GB[gpu_type]

    weight_bytes = estimate_weight_bytes(spec, quantization)
    model_gb = weight_bytes / (1024**3)

    tp = tp_override if tp_override else pick_tp(weight_bytes, gpu_mem, num_gpus, spec)

    kv_per_tok = kv_bytes_per_token(spec, tp)
    weight_per_gpu = model_gb / tp
    gpu_util_override = pick_gpu_memory_util(weight_bytes, gpu_mem, tp)
    effective_util = gpu_util_override or 0.90
    kv_cache_bytes = int(
        (gpu_mem * effective_util - weight_per_gpu - OVERHEAD_GB) * (1024**3)
    )
    kv_cache_bytes = max(kv_cache_bytes, 0)

    active_b = compute_active_params_b(spec)
    total_params_b = weight_bytes / DTYPE_BYTES[quantization] / 1e9

    if mml_override:
        max_ml = mml_override
    else:
        max_ml = pick_max_model_len(kv_cache_bytes, kv_per_tok,
                                    spec.max_position_embeddings,
                                    active_b, total_params_b, quantization,
                                    is_multimodal=spec.is_multimodal,
                                    num_experts=spec.num_experts)

    batched_tokens = pick_max_num_batched_tokens(spec)
    special = detect_special_flags(spec)

    # Token sizes and concurrency are co-dependent
    kv_total_tokens = int(kv_cache_bytes / max(kv_per_tok, 1))
    tok = pick_workload_tokens(kv_total_tokens, max_ml)
    workload = tok * 2
    max_concurrent = kv_total_tokens // workload if workload > 0 else 0
    rates = pick_concurrency_rates(max_concurrent)

    server: dict[str, Any] = {}
    for k, v in special.items():
        server[k] = v
    server["max-model-len"] = max_ml
    if batched_tokens is not None:
        server["max-num-batched-tokens"] = batched_tokens
    server["tensor-parallel-size"] = tp
    server["trust-remote-code"] = True
    if gpu_util_override is not None:
        server["gpu-memory-utilization"] = gpu_util_override

    client = generate_client_config(rates, tok, tok)

    return GeneratedConfig(
        server=server,
        client=client,
        spec=spec,
        tp=tp,
        max_model_len=max_ml,
        model_weight_gb=model_gb,
        kv_per_token=kv_per_tok,
        kv_cache_gb=kv_cache_bytes / (1024**3),
        active_params_b=active_b,
        max_concurrent_seqs=max_concurrent,
        gpu_memory_util=gpu_util_override,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate vLLM performance test configs (server.yml + client.yml).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("model_id", help="HuggingFace model ID or internal model path")

    g = p.add_argument_group("deployment")
    g.add_argument("--gpu", default="A100-80GB", choices=list(GPU_MEMORY_GB),
                   help="GPU type (default: A100-80GB)")
    g.add_argument("--num-gpus", type=int, default=8,
                   help="GPUs available (default: 8)")
    g.add_argument("--quantization", default="none", choices=list(DTYPE_BYTES),
                   help="Quantization method (default: none / BF16)")
    g.add_argument("--tp", type=int, default=None,
                   help="Override tensor-parallel-size instead of auto-computing")
    g.add_argument("--max-model-len", type=int, default=None,
                   help="Override max-model-len instead of auto-computing")

    m = p.add_argument_group("manual architecture (when model is not on HuggingFace)")
    m.add_argument("--num-params", type=float, default=None,
                   help="Total params in billions (rough estimate, used if no HF config)")
    m.add_argument("--num-layers", type=int, default=None)
    m.add_argument("--num-attention-heads", type=int, default=None)
    m.add_argument("--num-kv-heads", type=int, default=None)
    m.add_argument("--hidden-size", type=int, default=None)
    m.add_argument("--head-dim", type=int, default=None)
    m.add_argument("--intermediate-size", type=int, default=None)
    m.add_argument("--vocab-size", type=int, default=None)
    m.add_argument("--max-position-embeddings", type=int, default=32768)
    m.add_argument("--num-experts", type=int, default=0)
    m.add_argument("--num-experts-per-tok", type=int, default=0)
    m.add_argument("--multimodal", action="store_true", default=False)

    a = p.add_argument_group("authentication")
    a.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token for gated models (or set HF_TOKEN env var)")

    o = p.add_argument_group("output")
    o.add_argument("--output-dir", type=str, default=None,
                   help="Base directory (default: ./<model_id>/performance/)")
    o.add_argument("--dry-run", action="store_true",
                   help="Print configs to stdout without writing files")
    return p


def spec_from_manual_args(args: argparse.Namespace) -> ModelSpec:
    required = ["num_layers", "num_attention_heads", "num_kv_heads",
                "hidden_size", "intermediate_size", "vocab_size"]
    missing = [f for f in required if getattr(args, f) is None]
    if missing:
        flags = [f"--{f.replace('_', '-')}" for f in missing]
        raise SystemExit(
            f"Model not found on HuggingFace. Provide manual architecture flags:\n"
            f"  Missing: {', '.join(flags)}"
        )
    return ModelSpec(
        model_id=args.model_id,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim or 0,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        is_multimodal=args.multimodal,
    )


def print_summary(cfg: GeneratedConfig, gpu_type: str, quantization: str):
    s = cfg.spec
    print(f"\n{'=' * 60}")
    print(f"  Model:          {s.model_id}")
    print(f"  Architecture:   {s.model_type or 'unknown'} "
          f"({'MoE ' + str(s.num_experts) + ' experts' if s.num_experts else 'dense'})")
    print(f"  Layers:         {s.num_layers}")
    print(f"  Heads (Q/KV):   {s.num_attention_heads} / {s.num_kv_heads}")
    print(f"  Hidden size:    {s.hidden_size}  (head_dim={s.head_dim})")
    print(f"  MLP size:       {s.intermediate_size}")
    print(f"  Vocab:          {s.vocab_size}")
    print(f"  Native ctx:     {s.max_position_embeddings}")
    if s.is_multimodal:
        print(f"  Multimodal:     yes (mm_tokens={s.mm_tokens_per_item})")
    print(f"  Quantization:   {quantization}")
    print(f"{'=' * 60}")
    print(f"  Estimated weights:    {cfg.model_weight_gb:.1f} GB")
    print(f"  Active params:        {cfg.active_params_b:.1f} B")
    print(f"  GPU:                  {gpu_type}")
    print(f"  Tensor parallel:      {cfg.tp}")
    print(f"  KV per token per GPU: {cfg.kv_per_token:,} bytes")
    print(f"  KV cache headroom:    {cfg.kv_cache_gb:.1f} GB per GPU")
    # Recover actual workload tokens from client config
    tok = cfg.client["data"]["prompt_tokens"]
    print(f"  Max concurrent seqs:  {cfg.max_concurrent_seqs} (at {tok}+{tok} tok/seq)")
    print(f"  max-model-len:        {cfg.max_model_len}")
    if cfg.gpu_memory_util:
        print(f"  gpu-memory-util:      {cfg.gpu_memory_util}")
    print(f"{'=' * 60}\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    hf_token = _resolve_hf_token(args.hf_token)

    spec = None
    has_manual = any(getattr(args, f) is not None for f in
                     ["num_layers", "num_attention_heads", "num_kv_heads",
                      "hidden_size", "intermediate_size", "vocab_size"])

    if not has_manual:
        try:
            print(f"Fetching config from HuggingFace for {args.model_id}...")
            hf_config = fetch_hf_config(args.model_id, hf_token)
            spec = parse_hf_config(args.model_id, hf_config)
            print("  OK — architecture detected from config.json")
        except SystemExit:
            raise
        except Exception as e:
            print(f"  Warning: HF fetch failed ({e}), trying manual flags...")

    if spec is None:
        spec = spec_from_manual_args(args)

    cfg = generate(spec, args.gpu, args.num_gpus, args.quantization,
                   tp_override=args.tp, mml_override=args.max_model_len)

    print_summary(cfg, args.gpu, args.quantization)

    server_yaml = dict_to_yaml(cfg.server)
    client_yaml = dict_to_yaml(cfg.client)

    if args.dry_run:
        print("--- server.yml ---")
        print(server_yaml)
        print()
        print("--- client.yml ---")
        print(client_yaml)
        return

    if args.output_dir:
        out = Path(args.output_dir) / args.model_id / "performance"
    else:
        out = Path(args.model_id) / "performance"

    out.mkdir(parents=True, exist_ok=True)
    (out / "server.yml").write_text(server_yaml + "\n")
    (out / "client.yml").write_text(client_yaml + "\n")

    print(f"Written to {out}/")
    print(f"  server.yml")
    print(f"  client.yml")


if __name__ == "__main__":
    main()
