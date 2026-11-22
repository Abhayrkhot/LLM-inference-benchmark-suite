"""Load the committed result files into one structure used by the README and the figures.

Layout (copied from the cluster without the large ``*.records.jsonl.gz`` files):

    results/h200/<engine>_<model>_<quant>/node.txt
    results/h200/<tag>/sweep/<tag>_sweep.json            latency-throughput sweep
    results/h200/<tag>/cap_<tier>/<tag>_cap_<tier>_capacity.json
    results/h200/<tag>/repeat/*.json                     five runs of one load point
    results/h200/<tag>/xval/{llmbench,vllm}_rate*.json   cross-validation with vllm bench
    results/h200/<tag>/prefix/*.json                     Mooncake replay, cache on/off
    results/client_overhead/*.json
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ENGINE_LABEL = {"vllm": "vLLM", "sglang": "SGLang"}


def _load(p: Path) -> dict[str, Any]:
    return dict(json.loads(p.read_text()))


def node_info(tag_dir: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    p = tag_dir / "node.txt"
    if not p.exists():
        return info
    lines = p.read_text().splitlines()
    for line in lines:
        for m in re.finditer(r"(\w+)=(\S+)", line):
            info.setdefault(m.group(1), m.group(2))
        if "H200" in line or "H100" in line or "A100" in line:
            info["gpu_line"] = line.strip()
        if line.startswith("Model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
    return info


def label(tag: str) -> str:
    """``vllm_qwen2.5-7b_fp8`` -> ``vLLM FP8`` (model added when it is not the 7B default)."""
    engine, model, quant = tag.split("_", 2)
    name = f"{ENGINE_LABEL.get(engine, engine)} {quant.upper()}"
    return name if model == "qwen2.5-7b" else f"{name} ({model})"


def configs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    base = RESULTS / "h200"
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("smoke")):
        tag = d.name
        c: dict[str, Any] = {"tag": tag, "label": label(tag), "node": node_info(d)}
        sw = list((d / "sweep").glob("*_sweep.json"))
        if sw:
            c["sweep"] = _load(sw[0])
        for tier in ("interactive", "relaxed"):
            cap = list((d / f"cap_{tier}").glob("*_capacity.json"))
            if cap:
                c[f"cap_{tier}"] = _load(cap[0])
        reps = sorted((d / "repeat").glob("*_rep*.json"))
        if reps:
            c["repeat"] = [_load(p) for p in reps]
        xv = d / "xval"
        if xv.exists():
            c["xval"] = {p.stem: _load(p) for p in sorted(xv.glob("*.json"))}
        bu = sorted((d / "burst").glob("*_burst.json"))
        if bu:
            c["burst"] = _load(bu[0])
        pf = sorted((d / "prefix").glob("*.json"))
        if pf:
            c["prefix"] = {p.stem: _load(p) for p in pf}
        c["curve"] = curve(c)
        out[tag] = c
    return out


def curve(c: dict[str, Any]) -> list[dict[str, Any]]:
    """Every finite-rate Poisson run of a config, by offered rate: sweep points plus capacity
    probes (same workload, 60 s each), which fill in the region around the knee."""
    pts = [p for p in c.get("sweep", {}).get("points", []) if p["load"]["rate"] != float("inf")]
    for tier in ("interactive", "relaxed"):
        pts += [p for p in c.get(f"cap_{tier}", {}).get("probes", []) if p.get("requests_per_s")]
    pts = [p for p in pts if isinstance(p["load"]["rate"], (int, float))]
    return sorted(pts, key=lambda p: p["load"]["rate"])


def derived() -> dict[str, Any]:
    """Statistics computed on the cluster from the per-request records (scripts/derive.py)."""
    p = RESULTS / "h200" / "derived.json"
    return _load(p) if p.exists() else {}


def client_overhead() -> dict[str, Any] | None:
    files = sorted((RESULTS / "client_overhead").glob("*.json"))
    return _load(files[-1]) if files else None


def repeat_stats(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and coefficient of variation across repeated runs for headline metrics."""
    metrics = {
        "output tok/s": lambda s: s["throughput"]["output_tokens_per_s"],
        "TTFT p50": lambda s: s["ttft_ms"]["p50"],
        "TTFT p99": lambda s: s["ttft_ms"]["p99"],
        "TPOT p50": lambda s: s["tpot_ms"]["p50"],
        "TPOT p99": lambda s: s["tpot_ms"]["p99"],
    }
    out = {}
    for name, f in metrics.items():
        xs = [float(f(r["summary"])) for r in runs]
        mean = statistics.fmean(xs)
        out[name] = {
            "mean": mean,
            "cv": statistics.stdev(xs) / mean if len(xs) > 1 and mean else 0.0,
            "min": min(xs),
            "max": max(xs),
        }
    return out
