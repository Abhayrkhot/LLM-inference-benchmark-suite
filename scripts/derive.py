#!/usr/bin/env python3
"""Statistics that need the per-request records, which are too large to commit.

Run where the records live (the cluster's results directory); writes a small JSON that is
committed and read by build_readme.py:

    python scripts/derive.py /scratch/$USER/llmbench/results results/h200/derived.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from llmbench.metrics import per_request
from llmbench.report import bootstrap_diff
from llmbench.results import load_records, load_result


def word_vs_token(sharegpt: Path, tokenizer: str, n: int = 5000) -> dict[str, float]:
    """How far v0's whitespace word counts are from real token counts, on ShareGPT's recorded
    assistant replies (real LLM output) tokenized with the served model's tokenizer."""
    from llmbench.tokenizer import load_tokenizer

    tok = load_tokenizer(tokenizer)
    data = json.loads(sharegpt.read_text())
    replies = [
        c["conversations"][1]["value"]
        for c in data
        if len(c.get("conversations", [])) >= 2 and c["conversations"][1]["value"].strip()
    ]
    rng = np.random.default_rng(0)
    sample = [
        replies[i] for i in rng.choice(len(replies), size=min(n, len(replies)), replace=False)
    ]
    counts = tok.encode_batch(sample)
    words = np.array([len(t.split()) for t in sample], dtype=float)
    tokens = np.array([len(c) for c in counts], dtype=float)
    keep = tokens > 0
    ratio = words[keep] / tokens[keep]
    return {
        "replies": int(keep.sum()),
        "tokenizer": tokenizer,
        "words_per_token_mean": float(ratio.mean()),
        "words_per_token_p5": float(np.percentile(ratio, 5)),
        "words_per_token_p95": float(np.percentile(ratio, 95)),
        "aggregate_words_per_token": float(words[keep].sum() / tokens[keep].sum()),
    }


def sglang_burst(log: Path) -> dict[str, Any]:
    """How SGLang's scheduler handled the saturation burst, from its own log: the longest run of
    consecutive prefill-only batches (decode steps cannot run between them) once a large queue
    formed, and the scheduler settings that govern it."""
    lines = log.read_text(errors="replace").splitlines()
    start = next((i for i, ln in enumerate(lines) if (m := re.search(r"#queue-req: (\d+)", ln))
                  and int(m.group(1)) >= 500), None)  # fmt: skip
    out: dict[str, Any] = {}
    for key in ("chunked_prefill_size", "max_running_requests", "max_prefill_tokens"):
        m = re.search(rf"{key}[=:] ?'?(\d+)", log.read_text(errors="replace"))
        if m:
            out[key] = int(m.group(1))
    m = re.search(r"'enable_mixed_chunk': (True|False)", log.read_text(errors="replace"))
    if m:
        out["enable_mixed_chunk"] = m.group(1) == "True"
    if start is None:
        return out
    seq = "".join("P" if "Prefill batch" in ln else "D" for ln in lines[start : start + 5000]
                  if "Prefill batch" in ln or "Decode batch" in ln)  # fmt: skip
    runs = [len(r) for r in re.findall(r"P+", seq)]
    peak_queue = max(int(m.group(1)) for ln in lines[start : start + 5000]
                     if (m := re.search(r"#queue-req: (\d+)", ln)))  # fmt: skip
    out.update({"longest_prefill_run": max(runs), "peak_queue": peak_queue})
    return out


def compare(a: Path, b: Path) -> dict[str, Any]:
    ma, mb = per_request(load_records(a)), per_request(load_records(b))
    out: dict[str, Any] = {"baseline": load_result(a)["name"], "candidate": load_result(b)["name"]}
    for metric in ("ttft", "tpot", "e2e"):
        x, y = ma[metric], mb[metric]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        for q in (0.5, 0.99):
            pt, lo, hi = bootstrap_diff(x, y, q, n_boot=2000, seed=0)
            out[f"{metric}_p{int(q * 100)}"] = {
                "baseline": float(np.quantile(x, q)),
                "candidate": float(np.quantile(y, q)),
                "change": pt,
                "ci95": [lo, hi],
            }
    return out


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out: dict[str, Any] = {}
    sharegpt = src.parent / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"
    if sharegpt.exists():
        out["word_vs_token"] = word_vs_token(sharegpt, "Qwen/Qwen2.5-7B-Instruct")
    comps = {}
    for rate in ("16", "32", "48", "64"):
        for quant in ("bf16", "fp8"):
            a = src / f"vllm_qwen2.5-7b_{quant}/sweep/vllm_qwen2.5-7b_{quant}_rate{rate}.json"
            b = src / f"sglang_qwen2.5-7b_{quant}/sweep/sglang_qwen2.5-7b_{quant}_rate{rate}.json"
            if a.exists() and b.exists():
                comps[f"{quant}_rate{rate}"] = compare(a, b)
        for engine in ("vllm", "sglang"):
            a = src / f"{engine}_qwen2.5-7b_bf16/sweep/{engine}_qwen2.5-7b_bf16_rate{rate}.json"
            b = src / f"{engine}_qwen2.5-7b_fp8/sweep/{engine}_qwen2.5-7b_fp8_rate{rate}.json"
            if a.exists() and b.exists():
                comps[f"{engine}_fp8_vs_bf16_rate{rate}"] = compare(a, b)
    out["comparisons"] = comps
    trace = src.parent / "data" / "mooncake_toolagent_trace.jsonl"
    if trace.exists():
        from llmbench.datasets import TokenSpace, ideal_prefix_hit_rate, mooncake_workload

        wl = mooncake_workload(trace, TokenSpace(151_643), 1500, max_input_len=16384)
        out["mooncake_ideal_hit_rate"] = {
            str(bs): ideal_prefix_hit_rate(wl.requests, bs) for bs in (1, 16, 64)
        }
    for log in sorted((src / "sglang_qwen2.5-7b_bf16").glob("engine-[0-9]*.log")):
        burst = sglang_burst(log)
        if "longest_prefill_run" in burst:
            out["sglang_burst"] = {"log": log.name, **burst}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out.get("word_vs_token", {})), len(comps), "comparisons")


if __name__ == "__main__":
    main()
