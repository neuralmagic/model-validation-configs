#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "pyyaml"]
# ///
"""
Check PR-changed server.yml configs against upstream vLLM community serving
guidance from vllm-project/recipes (served as JSON at recipes.vllm.ai).

This is the CI counterpart to a one-time manual review done in August 2026
(see neuralmagic/nm-cicd's .github/scripts/compare_configs_to_recipes.py for
the full, repo-wide sweep). That review found a real bug -- a copy-pasted
`reasoning-parser: deepseek_r1` on a Qwen model -- but also produced two false
alarms (`tool-call-parser`/`enable-auto-tool-choice` "missing", `kv-cache-dtype`
"missing") that only looked like bugs until checked against actual vLLM/harness
behavior. This script is deliberately conservative to avoid reintroducing that
noise on every PR:

  * Only PR-changed model directories are checked (not a full repo sweep).
  * `tool-call-parser`/`enable-auto-tool-choice` are never flagged as
    "missing" or "extra" -- the OCP test harness in nm-cicd
    (ocp_model_deployment/fixtures.py) independently injects these at deploy
    time via toolparser/rules.py, and this repo has no visibility into that
    harness to know whether a given model's rule exists.
  * `kv-cache-dtype` is always a performance-tier note, never a correctness
    finding -- vLLM's own FP8-KV-cache docs frame the default (`auto`, ~bf16)
    as the higher-fidelity choice for standard attention backends; it's only
    correctness-critical for kernels with no bf16 KV-cache path at all (e.g.
    DeepSeek's fp8_ds_mla), which is too narrow to special-case generically.
  * A `reasoning-parser` or `tool-call-parser` conflict (both sides set,
    values disagree) is the highest-confidence signal available -- nothing
    downstream overrides an explicit, hardcoded wrong value in either -- and
    is surfaced most prominently, alongside tokenizer-mode/config-format/
    load-format conflicts. `enable-auto-tool-choice` conflicts (a boolean,
    not a parser selection) stay advisory-only.
  * "Expected divergence" flags (max-model-len, tensor-parallel-size,
    uvicorn-log-level, no-enable-prefix-caching, chat-template,
    trust-remote-code, model) are filtered out entirely -- not just
    deprioritized. This repo has no model_registry.yml, so it can't even
    attempt the TP-cosmetic cross-check nm-cicd's full sweep does; excluding
    TP outright avoids false noise from that gap.
  * Fuzzy-matched models (no exact HF-id match) are confidence-gated: only
    "exact" and "fuzzy-stem-quant-matched" (a precision-matched variant, e.g.
    our FP8 model matched to upstream's FP8 variant) drive the main findings.
    Anything less certain is shown separately, clearly labeled as unverified.
  * Network failures, rate limits, and "no upstream recipe for this model"
    are all expected, common outcomes -- never treated as errors.
  * Exits 0 by default (`--fail-on none`), for standalone/local use. The
    repo's own workflow opts into `--fail-on correctness`, so a high-confidence
    conflict (`reasoning-parser`/`tool-call-parser`/`tokenizer-mode`/
    `config-format`/`load-format`) fails the check and blocks merge; everything
    else (missing/extra, advisory notes, fuzzy matches) never fails the build.

Usage:
  uv run check_recipes.py --base <sha> --head <sha> --summary-out summary.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

DEFAULT_API_BASE = "https://recipes.vllm.ai"
WORKLOADS = ("performance", "accuracy")
FLAVOR_TO_HARDWARE = {
    "default": "h100",
    "cpu": "xeon6",
    "rocm": "mi300x",
    "tpu": "trillium",
}

_SENTINEL = object()

# ---------------------------------------------------------------------------
# Vendored from vllm-project/vllm tools/recipes/recipe_json_to_vllm_config.py
# (argv_to_config and its helpers), with the same local fix applied in
# nm-cicd's compare_configs_to_recipes.py: upstream's own SHORT_ALIASES don't
# include `-cc` (--compilation-config), and real recipes emit dotted
# `-cc.<field>=<value>` tokens (e.g. openai/gpt-oss-120b's AMD hardware
# override) that upstream's converter silently mis-parses as a swallowed
# value of the *preceding* flag. Confirmed this session by running upstream's
# script directly against that recipe.
# ---------------------------------------------------------------------------

SHORT_ALIASES = {
    "-tp": "tensor-parallel-size",
    "-pp": "pipeline-parallel-size",
    "-dp": "data-parallel-size",
    "-cc": "compilation-config",
}


def coerce(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def is_option_token(token: str) -> bool:
    if token.startswith("--"):
        return True
    if token in SHORT_ALIASES or token == "-O":
        return True
    if any(token.startswith(a + ".") for a in SHORT_ALIASES):
        return True
    return token.startswith("-O") and len(token) > 2


def merge_value(dst: dict[str, Any], path: list[str], value: Any) -> None:
    cur = dst
    for part in path[:-1]:
        existing = cur.get(part)
        if existing is None:
            cur[part] = {}
        elif not isinstance(existing, dict):
            raise ValueError(
                f"Cannot merge nested option {'.'.join(path)!r}: {part!r} is already a scalar"
            )
        cur = cur[part]

    leaf = path[-1]
    if leaf not in cur:
        cur[leaf] = value
        return

    old = cur[leaf]
    if not isinstance(old, list):
        old = [old]
    if isinstance(value, list):
        old.extend(value)
    else:
        old.append(value)
    cur[leaf] = old


def normalize_key(raw_key: str) -> list[str]:
    parts = raw_key.split(".")
    parts[0] = parts[0].replace("_", "-")
    return parts


def argv_to_config(argv: list[Any]) -> dict[str, Any]:
    argv = [str(x) for x in argv]

    if len(argv) < 3 or argv[0:2] != ["vllm", "serve"]:
        raise ValueError(
            f"Expected recipe argv to start with: ['vllm', 'serve', MODEL, ...]. Got: {argv[:4]!r}"
        )

    model = argv[2]
    if model.startswith("-"):
        raise ValueError(f"Expected model after 'vllm serve', got {model!r}")

    cfg: dict[str, Any] = {"model": model}

    i = 3
    while i < len(argv):
        token = argv[i]

        if token.startswith("-O") and token != "-O":
            value = token[3:] if token.startswith("-O=") else token[2:]
            merge_value(cfg, ["optimization-level"], coerce(value))
            i += 1
            continue

        if token == "-O":
            if i + 1 >= len(argv):
                raise ValueError("-O is missing its value")
            merge_value(cfg, ["optimization-level"], coerce(argv[i + 1]))
            i += 2
            continue

        if token in SHORT_ALIASES:
            if i + 1 >= len(argv):
                raise ValueError(f"{token} is missing its value")
            merge_value(cfg, [SHORT_ALIASES[token]], coerce(argv[i + 1]))
            i += 2
            continue

        matched_alias = next((a for a in SHORT_ALIASES if token.startswith(a + ".")), None)
        if matched_alias:
            rest = token[len(matched_alias) :]
            if "=" not in rest:
                raise ValueError(f"{token!r} is missing an inline value")
            dotted, raw_value = rest.lstrip(".").split("=", 1)
            path = normalize_key(SHORT_ALIASES[matched_alias]) + dotted.split(".")
            merge_value(cfg, path, coerce(raw_value))
            i += 1
            continue

        if not token.startswith("--"):
            raise ValueError(
                f"Unexpected positional/short argument {token!r}. "
                "The converter expects Recipes to emit long-form vLLM serve options."
            )

        if "=" in token:
            key, raw_value = token[2:].split("=", 1)
            merge_value(cfg, normalize_key(key), coerce(raw_value))
            i += 1
            continue

        key = token[2:]
        i += 1

        raw_values: list[str] = []
        while i < len(argv) and not is_option_token(argv[i]):
            raw_values.append(argv[i])
            i += 1

        if not raw_values:
            value: Any = True
        elif len(raw_values) == 1:
            value = coerce(raw_values[0])
        else:
            value = [coerce(v) for v in raw_values]

        merge_value(cfg, normalize_key(key), value)

    return cfg


# ---------------------------------------------------------------------------
# PR diff scoping
# ---------------------------------------------------------------------------

_MODEL_FILE_RE = re.compile(r"^([^/]+)/([^/]+)/(accuracy|performance)/server.*\.ya?ml$")


def changed_model_dirs(base: str, head: str) -> tuple[list[str], bool]:
    """Return (sorted unique 'org/model' ids touched by the diff, common_only)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        capture_output=True,
        text=True,
        check=True,
    )
    changed = result.stdout.splitlines()

    models: set[str] = set()
    common_touched = False
    for path in changed:
        if path.startswith("common/"):
            common_touched = True
            continue
        m = _MODEL_FILE_RE.match(path)
        if m:
            models.add(f"{m.group(1)}/{m.group(2)}")

    return sorted(models), common_touched and not models


# ---------------------------------------------------------------------------
# Local config loading (mirrors nm-cicd's ValidationConfig fallback chain)
# ---------------------------------------------------------------------------


def load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def normalize_local_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("_", "-"): v for k, v in raw.items()}


def detect_flavors(model_dir: Path) -> list[str]:
    flavors = ["default"]
    for flavor in ("cpu", "rocm", "tpu"):
        if any((model_dir / w / f"server-{flavor}.yml").exists() for w in WORKLOADS):
            flavors.append(flavor)
    return flavors


def effective_server_config(
    configs_root: Path, model_dir: Path, workload: str, flavor: str
) -> tuple[dict[str, Any], list[str]]:
    suffix = f"-{flavor}" if flavor != "default" else ""
    candidates = [f"server{suffix}.yml", "server.yml"] if suffix else ["server.yml"]
    candidates = list(dict.fromkeys(candidates))

    for name in candidates:
        p = model_dir / workload / name
        if p.exists():
            return normalize_local_config(load_yaml_dict(p)), [str(p)]

    if workload == "accuracy":
        perf_cfg, perf_prov = effective_server_config(configs_root, model_dir, "performance", flavor)
        if perf_cfg:
            return perf_cfg, [*perf_prov, "(accuracy fallback to performance)"]

    common_dir = configs_root / "common" / workload
    for name in candidates:
        p = common_dir / name
        if p.exists():
            return normalize_local_config(load_yaml_dict(p)), [str(p)]

    return {}, []


# ---------------------------------------------------------------------------
# Recipes API client (in-memory cache only -- single CI run, no disk state)
# ---------------------------------------------------------------------------


class RecipesClient:
    def __init__(self, api_base: str, client: httpx.Client):
        self.api_base = api_base.rstrip("/")
        self.client = client
        self._cache: dict[str, Any] = {}
        self.unreachable = False

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return f"{self.api_base}/{path_or_url.lstrip('/')}"

    def get_json(self, path_or_url: str) -> Any | None:
        url = self._url(path_or_url)
        if url in self._cache:
            return self._cache[url]
        if self.unreachable:
            return None
        try:
            resp = self.client.get(url, timeout=15)
        except httpx.HTTPError:
            self.unreachable = True
            return None
        if resp.status_code == 404:
            self._cache[url] = None
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            self.unreachable = True
            return None
        try:
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            self.unreachable = True
            return None
        self._cache[url] = data
        return data


# ---------------------------------------------------------------------------
# Matching our model IDs to upstream recipe IDs
# ---------------------------------------------------------------------------

_QUANT_TOKEN_RE = re.compile(
    r"^(fp8(-dynamic)?|nvfp4|mxfp4|mxfp8|int4|int8|w4a16|w8a8|w4a4|gptq|awq|marlin|"
    r"quantized|dynamic|block|sym|asym|qad|g\d+)$",
    re.IGNORECASE,
)

HIGH_CONFIDENCE = {"exact", "fuzzy-stem-quant-matched"}


def repo_stem(repo_name: str) -> str:
    parts = re.split(r"([-.])", repo_name)
    while len(parts) >= 3 and _QUANT_TOKEN_RE.match(parts[-1]):
        parts = parts[:-2]
    return "".join(parts).lower()


def quant_flavor(repo_name: str) -> str | None:
    name = repo_name.lower()
    if "nvfp4" in name:
        return "nvfp4"
    if "mxfp4" in name or "mxfp8" in name:
        return "mxfp"
    if "fp8" in name:
        return "fp8"
    if "w4a16" in name or "int4" in name:
        return "w4a16"
    if "w8a8" in name or "int8" in name:
        return "w8a8"
    if "gptq" in name:
        return "gptq"
    if "awq" in name:
        return "awq"
    return None


def match_recipe(our_hf_id: str, index: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Returns (recipe_hf_id, confidence) or None if no candidate at all."""
    by_id_lower = {m["hf_id"].lower(): m["hf_id"] for m in index if m.get("hf_id")}
    exact = by_id_lower.get(our_hf_id.lower())
    if exact:
        return exact, "exact"

    our_repo = our_hf_id.split("/")[-1]
    our_stem = repo_stem(our_repo)
    candidates = [
        m["hf_id"] for m in index if m.get("hf_id") and repo_stem(m["hf_id"].split("/")[-1]) == our_stem
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0], "fuzzy-stem"

    our_flavor = quant_flavor(our_repo)
    if our_flavor:
        flavor_matches = [c for c in candidates if quant_flavor(c.split("/")[-1]) == our_flavor]
        if len(flavor_matches) == 1:
            return flavor_matches[0], "fuzzy-stem-quant-matched"
        if flavor_matches:
            candidates = flavor_matches

    our_org = our_hf_id.split("/")[0].lower()
    same_org = [c for c in candidates if c.split("/")[0].lower() == our_org]
    pick = same_org[0] if same_org else sorted(candidates, key=len)[0]
    return pick, "fuzzy-stem-ambiguous"


# ---------------------------------------------------------------------------
# Diffing and tiering
# ---------------------------------------------------------------------------

# These never appear in output at all: intentional/structural, not findings.
EXPECTED_DIVERGENCE_FLAGS = {
    "model",
    "max-model-len",
    "tensor-parallel-size",
    "uvicorn-log-level",
    "no-enable-prefix-caching",
    "chat-template",
    "trust-remote-code",
}

# High-confidence correctness signal: shown ONLY when both sides set it and
# disagree (a "diff"), never for merely-missing/extra. See module docstring.
# `tool-call-parser` is included here *and* in DIFF_ONLY_ADVISORY_FLAGS below:
# a hardcoded wrong parser string is the same class of bug as a wrong
# `reasoning-parser` (both mis-select an output-parsing implementation for
# the model), so a "diff" gets top billing -- but "missing"/"extra" is still
# suppressed for it via DIFF_ONLY_ADVISORY_FLAGS, since the harness may
# legitimately inject it.
TOP_CORRECTNESS_FLAGS = {
    "reasoning-parser",
    "tool-call-parser",
    "tokenizer-mode",
    "config-format",
    "load-format",
}

# Compensated elsewhere (nm-cicd's OCP harness): "missing"/"extra" is never
# actionable for these, only an explicit value conflict. `tool-call-parser`
# is also in TOP_CORRECTNESS_FLAGS, so its conflicts render top-tier;
# `enable-auto-tool-choice` (a boolean toggle, not a parser selection) stays
# advisory-only.
DIFF_ONLY_ADVISORY_FLAGS = {
    "tool-call-parser",
    "enable-auto-tool-choice",
}


def diff_configs(local: dict[str, Any], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(local) | set(recipe)):
        if key in EXPECTED_DIVERGENCE_FLAGS:
            continue
        lv = local.get(key, _SENTINEL)
        rv = recipe.get(key, _SENTINEL)
        if lv is _SENTINEL:
            kind = "missing_local"
        elif rv is _SENTINEL:
            kind = "extra_local"
        elif lv == rv:
            continue  # matches; not a finding
        else:
            kind = "diff"

        if key in DIFF_ONLY_ADVISORY_FLAGS and kind != "diff":
            continue  # never actionable as missing/extra -- see docstring

        rows.append(
            {
                "key": key,
                "local": None if lv is _SENTINEL else lv,
                "recipe": None if rv is _SENTINEL else rv,
                "kind": kind,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-model review
# ---------------------------------------------------------------------------


def review_model(hf_id: str, model_dir: Path, configs_root: Path, index: list[dict[str, Any]], client: RecipesClient) -> dict[str, Any]:
    result: dict[str, Any] = {"model": hf_id}

    match = match_recipe(hf_id, index)
    if match is None:
        result["status"] = "no_recipe"
        return result
    recipe_hf_id, confidence = match
    result["recipe_hf_id"] = recipe_hf_id
    result["confidence"] = confidence

    # Link to the actual recipe page as evidence, not just a bare hf_id --
    # for derived variants (e.g. our FP8 model matched to an upstream FP8
    # entry) the index's "url" may point at a shared base-model page rather
    # than a page for recipe_hf_id itself, so look it up rather than assume.
    index_entry = next((m for m in index if m.get("hf_id") == recipe_hf_id), None)
    if index_entry and index_entry.get("url"):
        result["recipe_url"] = client._url(index_entry["url"])

    detail = client.get_json(f"/{recipe_hf_id}.json")
    if detail is None:
        result["status"] = "unreachable" if client.unreachable else "no_recipe"
        return result

    result["min_vllm_version"] = (detail.get("model") or {}).get("min_vllm_version")
    by_hardware = ((detail.get("recommended_command") or {}).get("by_hardware")) or {}

    flavors_out = []
    for flavor in detect_flavors(model_dir):
        hardware = FLAVOR_TO_HARDWARE[flavor]
        hw_path = by_hardware.get(hardware)
        if not hw_path:
            continue
        hw_json = client.get_json(hw_path)
        if not hw_json:
            continue
        argv = hw_json.get("argv")
        if not isinstance(argv, list):
            continue  # multi-process/non-single-node recipe; not comparable
        try:
            recipe_cfg = argv_to_config(argv)
        except ValueError:
            continue
        recipe_env = hw_json.get("env") or {}

        for workload in WORKLOADS:
            local_cfg, provenance = effective_server_config(configs_root, model_dir, workload, flavor)
            if not provenance:
                continue
            rows = diff_configs(local_cfg, recipe_cfg)
            if rows or recipe_env:
                flavors_out.append(
                    {
                        "flavor": flavor,
                        "hardware": hardware,
                        "workload": workload,
                        "rows": rows,
                        "recipe_env": recipe_env,
                    }
                )

    result["status"] = "ok"
    result["flavors"] = flavors_out
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_summary(results: list[dict[str, Any]], common_only: bool) -> str:
    lines: list[str] = []
    lines.append("## vllm-project/recipes check")
    lines.append("")

    if common_only:
        lines.append(
            "This PR only touches `common/` fallback configs. This check compares "
            "directly-changed model directories against upstream recipes and doesn't "
            "attempt a repo-wide sweep for `common/` changes -- run nm-cicd's "
            "`compare_configs_to_recipes.py` manually if you want to check "
            "downstream impact across all models that fall back to `common/`."
        )
        return "\n".join(lines) + "\n"

    if not results:
        lines.append("No changed model server.yml files detected in this PR.")
        return "\n".join(lines) + "\n"

    top_findings: list[str] = []
    advisory_lines: list[str] = []
    low_confidence_lines: list[str] = []
    no_recipe: list[str] = []
    unreachable: list[str] = []

    for r in results:
        model = r["model"]
        status = r.get("status")
        if status == "no_recipe":
            no_recipe.append(model)
            continue
        if status == "unreachable":
            unreachable.append(model)
            continue

        confidence = r["confidence"]
        recipe_id = r["recipe_hf_id"]
        recipe_ref = f"[`{recipe_id}`]({r['recipe_url']})" if r.get("recipe_url") else f"`{recipe_id}`"
        header = f"`{model}` -> {recipe_ref} (match: {confidence})"

        if confidence not in HIGH_CONFIDENCE:
            low_confidence_lines.append(
                f"- {header} -- unverified match, spot-check before trusting any finding below it"
            )
            continue

        model_top: list[str] = []
        model_advisory: list[str] = []
        env_notes: list[str] = []

        for fl in r.get("flavors", []):
            ctx = f"{fl['workload']}/{fl['flavor']} (hw={fl['hardware']})"
            for row in fl["rows"]:
                line = f"  - `{row['key']}` [{ctx}]: ours=`{row['local']!r}` vs recipe=`{row['recipe']!r}` ({row['kind']})"
                if row["key"] in TOP_CORRECTNESS_FLAGS and row["kind"] == "diff":
                    model_top.append(line)
                elif row["key"] in DIFF_ONLY_ADVISORY_FLAGS:
                    model_advisory.append(line)
                else:
                    model_advisory.append(line)
            if fl["recipe_env"]:
                env_notes.append(f"  - recipe requires env (not set anywhere here): `{fl['recipe_env']}` [{ctx}]")

        if r.get("min_vllm_version"):
            env_notes.append(f"  - upstream recipe requires vLLM >= `{r['min_vllm_version']}`")

        if model_top:
            top_findings.append(f"- {header}\n" + "\n".join(model_top))
        if model_advisory or env_notes:
            advisory_lines.append(f"- {header}\n" + "\n".join(model_advisory + env_notes))

    if top_findings:
        lines.append("### Worth a look (high-confidence conflicting values) -- blocks merge")
        lines.append("")
        lines.append(
            "Both our config and the upstream recipe explicitly set these, to different "
            "values -- this is the strongest signal this tool can produce (it's how we "
            "caught a real `reasoning-parser` bug during the manual review this check is "
            "based on; a hardcoded wrong `tool-call-parser` is the same class of bug). "
            "Fix the value (or if it's genuinely intentional, say why in the PR)."
        )
        lines.append("")
        lines.extend(top_findings)
        lines.append("")
    else:
        lines.append("No high-confidence conflicting values found on changed models.")
        lines.append("")

    if advisory_lines:
        lines.append("<details>")
        lines.append("<summary>Other observations (performance tuning, env vars, non-blocking)</summary>")
        lines.append("")
        lines.extend(advisory_lines)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if low_confidence_lines:
        lines.append("<details>")
        lines.append("<summary>Low-confidence recipe matches (not verified)</summary>")
        lines.append("")
        lines.extend(low_confidence_lines)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if no_recipe:
        lines.append(f"No upstream recipe found for: {', '.join(f'`{m}`' for m in no_recipe)}")
        lines.append("")

    if unreachable:
        lines.append(
            f"Could not verify (recipes.vllm.ai unreachable/rate-limited): "
            f"{', '.join(f'`{m}`' for m in unreachable)}"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base git ref/sha to diff against")
    ap.add_argument("--head", required=True, help="head git ref/sha")
    ap.add_argument("--configs-path", type=Path, default=Path("."))
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--summary-out", type=Path, default=Path("summary.md"))
    ap.add_argument(
        "--fail-on",
        choices=("correctness", "none"),
        default="none",
        help="exit 1 if a high-confidence correctness conflict exists (default: never fail)",
    )
    args = ap.parse_args()

    models, common_only = changed_model_dirs(args.base, args.head)

    if not models:
        args.summary_out.write_text(render_summary([], common_only), encoding="utf-8")
        return 0

    results: list[dict[str, Any]] = []
    with httpx.Client(headers={"User-Agent": "model-validation-configs-recipes-check/1.0"}) as http_client:
        client = RecipesClient(args.api_base, http_client)
        index = client.get_json("/models.json")
        if not isinstance(index, list):
            # API down/unreachable entirely: report every changed model as
            # unreachable rather than silently producing an empty summary.
            for hf_id in models:
                results.append({"model": hf_id, "status": "unreachable"})
        else:
            index = [m for m in index if isinstance(m, dict)]
            for hf_id in models:
                model_dir = args.configs_path / hf_id
                if not model_dir.is_dir():
                    continue
                results.append(review_model(hf_id, model_dir, args.configs_path, index, client))

    summary = render_summary(results, common_only=False)
    args.summary_out.write_text(summary, encoding="utf-8")

    if args.fail_on == "correctness":
        has_top_finding = any(
            row["key"] in TOP_CORRECTNESS_FLAGS and row["kind"] == "diff"
            for r in results
            if r.get("confidence") in HIGH_CONFIDENCE
            for fl in r.get("flavors", [])
            for row in fl["rows"]
        )
        if has_top_finding:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
