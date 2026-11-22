"""Load sweeps and capacity search.

**Capacity at SLO** is the highest offered request rate at which aggregate latency constraints
(e.g. ``ttft_p99<=500ms,tpot_p99<=30ms``, as in MLPerf Inference's server scenario) still hold.
It is found by measuring the server's saturated throughput, which bounds any sustainable rate,
then bisecting the rate geometrically between a passing and a failing probe. Every probe uses a
fresh workload seed so prompts cached by an earlier probe cannot flatter a later one.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from llmbench.bench import BenchConfig, run_benchmark, wait_idle
from llmbench.metrics import Constraint
from llmbench.results import RunResult

__all__ = ["CapacityResult", "find_capacity", "reset_prefix_cache", "run_capacity", "run_sweep"]

log = logging.getLogger("llmbench")


async def reset_prefix_cache(base_url: str) -> str | None:
    """Empty the engine's prefix cache between runs. vLLM exposes ``POST /reset_prefix_cache``
    (with ``VLLM_SERVER_DEV_MODE=1``); SGLang exposes ``POST /flush_cache``."""
    base = base_url.rstrip("/")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        for path in ("/reset_prefix_cache", "/flush_cache"):
            try:
                async with s.post(base + path) as r:
                    if r.status == 200:
                        return path
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                continue
    return None


def _between_runs(cfg: BenchConfig, reset_cache: bool) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        idle = await wait_idle(cfg.base_url)
        reset = await reset_prefix_cache(cfg.base_url) if reset_cache else None
        return {"idle": idle, "cache_reset": reset}

    return asyncio.run(go())


def sweep_row(res: RunResult) -> dict[str, Any]:
    s = res.summary
    thr = s.get("throughput", {})
    row: dict[str, Any] = {
        "name": res.name,
        "load": res.config["load"],
        "requests_ok": s["requests"]["ok"],
        "requests_failed": s["requests"]["failed"],
        "duration_s": s.get("duration_s"),
        "requests_per_s": thr.get("requests_per_s"),
        "output_tokens_per_s": thr.get("output_tokens_per_s"),
        "total_tokens_per_s": thr.get("total_tokens_per_s"),
        "goodput_requests_per_s": thr.get("goodput_requests_per_s"),
        "goodput_fraction": thr.get("goodput_fraction"),
    }
    for m in ("ttft", "tpot", "itl", "e2e"):
        d = s.get(f"{m}_ms", {})
        for stat in ("p50", "p90", "p99", "mean"):
            row[f"{m}_{stat}_ms"] = d.get(stat)
    row["mean_in_flight"] = s.get("client", {}).get("mean_in_flight")
    row["dispatch_lag_p99_ms"] = s.get("client", {}).get("dispatch_lag_ms", {}).get("p99")
    for k, v in res.engine_metrics.get("stats", {}).items():
        row[f"engine_{k}"] = v
    if res.gpu:
        row["gpu_power_w_mean"] = res.gpu.get("power_w_mean")
        row["joules_per_output_token"] = res.gpu.get("joules_per_output_token")
    return row


def run_sweep(
    cfg: BenchConfig,
    out_dir: str | Path,
    *,
    rates: Sequence[float] = (),
    concurrencies: Sequence[int] = (),
    reset_cache: bool = True,
    save_records: bool = True,
    point_duration_s: float | None = None,
    min_requests: int = 200,
    max_requests: int = 20000,
) -> list[dict[str, Any]]:
    """One run per load point. With ``point_duration_s``, each finite-rate point sends
    ``rate * point_duration_s`` requests (clipped), so low and high rates get similar wall time and
    similar statistical weight; other points send ``cfg.workload.num_requests``."""
    if bool(rates) == bool(concurrencies):
        raise ValueError("give either rates or concurrencies")
    if rates and cfg.load.arrival == "trace":
        raise ValueError("trace replay ignores --rate; sweep --time-scale with separate runs")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    points: list[float] = [float(r) for r in rates] or [float(c) for c in concurrencies]
    for i, x in enumerate(points):
        c = copy.deepcopy(cfg)
        c.workload.seed = cfg.workload.seed + i
        if rates:
            c.load.rate, c.load.concurrency = x, None
            label = "inf" if math.isinf(x) else f"{x:g}"
            c.name = f"{cfg.name}_rate{label}"
            if point_duration_s and not math.isinf(x):
                n = int(x * point_duration_s)
                c.workload.num_requests = min(max(n, min_requests), max_requests)
        else:
            c.load.concurrency = int(x)
            c.name = f"{cfg.name}_conc{int(x)}"
        prep = _between_runs(c, reset_cache)
        log.info("sweep point %d/%d: %s (%s)", i + 1, len(points), c.load.describe(), prep)
        res = run_benchmark(c)
        res.save(out / c.name, records=save_records)
        rows.append(sweep_row(res))
        _log_row(rows[-1])
    (out / f"{cfg.name}_sweep.json").write_text(
        json.dumps({"name": cfg.name, "config": cfg.to_json(), "points": rows}, indent=1) + "\n"
    )
    return rows


def _log_row(r: dict[str, Any]) -> None:
    log.info(
        "  -> %.2f req/s, %.0f out tok/s, TTFT p50/p99 %.0f/%.0f ms, TPOT p50/p99 %.1f/%.1f ms, "
        "failed %d",
        r["requests_per_s"] or 0,
        r["output_tokens_per_s"] or 0,
        r["ttft_p50_ms"] or 0,
        r["ttft_p99_ms"] or 0,
        r["tpot_p50_ms"] or 0,
        r["tpot_p99_ms"] or 0,
        r["requests_failed"],
    )


@dataclass
class Probe:
    rate: float
    passed: bool
    observed: dict[str, float]
    row: dict[str, Any]


@dataclass
class CapacityResult:
    constraints: list[str]
    capacity_rps: float  # highest passing offered rate (0 if none passed)
    bracket: tuple[float, float]  # (highest passing, lowest failing) offered rate
    saturation_rps: float  # completed req/s with every request sent at once
    probes: list[Probe] = field(default_factory=list)

    @property
    def at_capacity(self) -> dict[str, Any] | None:
        best = [p for p in self.probes if p.passed and p.rate == self.capacity_rps]
        return best[0].row if best else None

    def to_json(self) -> dict[str, Any]:
        return {
            "constraints": self.constraints,
            "capacity_rps": self.capacity_rps,
            "bracket": list(self.bracket),
            "saturation_rps": self.saturation_rps,
            "at_capacity": self.at_capacity,
            "probes": [
                {"rate": p.rate, "passed": p.passed, "observed": p.observed, **p.row}
                for p in self.probes
            ],
        }


def find_capacity(
    probe: Callable[[float, int], dict[str, Any]],
    constraints: Sequence[Constraint],
    *,
    saturation: Callable[[], float],
    rel_tol: float = 0.05,
    max_probes: int = 10,
    min_rate: float = 0.05,
) -> CapacityResult:
    """``probe(rate, index)`` runs the server at ``rate`` and returns a sweep row plus the run
    summary under key ``"_summary"``; ``saturation()`` returns saturated throughput (req/s)."""
    sat = saturation()
    probes: list[Probe] = []
    # A server that sheds load fails fast, and the survivors look quick: latency limits alone
    # would pass. Unless the caller set one, failures above 1% fail the probe.
    if not any(c.metric == "error_rate" for c in constraints):
        constraints = [*constraints, Constraint("error_rate", None, "<=", 0.01)]

    def check(rate: float) -> bool:
        row = probe(rate, len(probes))
        summary = row.pop("_summary")
        ok = all(c.holds(summary) for c in constraints)
        probes.append(Probe(rate, ok, {str(c): c.observed(summary) for c in constraints}, row))
        log.info(
            "capacity probe %.3g req/s: %s %s", rate, "PASS" if ok else "FAIL", probes[-1].observed
        )
        return ok

    names = [str(c) for c in constraints]
    if sat <= 0:
        return CapacityResult(names, 0.0, (0.0, 0.0), sat, probes)
    lo, hi = 0.0, sat
    if check(sat):
        # A burst run includes its own ramp-up and drain, so it can understate what the server
        # sustains; a lenient SLO may hold above it. Grow until a probe fails.
        lo, r = sat, sat * 1.5
        while len(probes) < max_probes:
            if not check(r):
                hi = r
                break
            lo, r = r, r * 1.5
        else:
            return CapacityResult(names, lo, (lo, math.inf), sat, probes)
    else:
        r = sat / 2
        while len(probes) < max_probes and r >= min_rate:
            if check(r):
                lo = r
                break
            hi, r = r, r / 2
        if lo == 0.0:
            return CapacityResult(names, 0.0, (0.0, hi), sat, probes)
    while len(probes) < max_probes and hi / lo - 1 > rel_tol:
        mid = math.sqrt(lo * hi)
        if check(mid):
            lo = mid
        else:
            hi = mid
    return CapacityResult(names, lo, (lo, hi), sat, probes)


def run_capacity(
    cfg: BenchConfig,
    constraints: Sequence[Constraint],
    out_dir: str | Path,
    *,
    probe_duration_s: float = 60.0,
    min_requests: int = 200,
    max_requests: int = 20000,
    saturation_requests: int | None = None,
    rel_tol: float = 0.05,
    max_probes: int = 10,
    reset_cache: bool = True,
) -> CapacityResult:
    if cfg.load.arrival == "trace":
        raise ValueError("capacity search varies the request rate; trace replay ignores it")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def saturation() -> float:
        c = copy.deepcopy(cfg)
        c.load.rate, c.load.concurrency = math.inf, None
        c.workload.num_requests = saturation_requests or cfg.workload.num_requests
        c.workload.seed = cfg.workload.seed + 1000
        c.name = f"{cfg.name}_saturation"
        _between_runs(c, reset_cache)
        res = run_benchmark(c)
        res.save(out / c.name)
        rps = float(res.summary.get("throughput", {}).get("requests_per_s", 0.0))
        log.info("saturated throughput: %.3g req/s", rps)
        return rps

    def probe(rate: float, i: int) -> dict[str, Any]:
        c = copy.deepcopy(cfg)
        c.load.rate, c.load.concurrency = rate, None
        c.workload.num_requests = int(min(max(rate * probe_duration_s, min_requests), max_requests))
        c.workload.seed = cfg.workload.seed + 1 + i
        c.name = f"{cfg.name}_probe{i:02d}_rate{rate:.4g}"
        _between_runs(c, reset_cache)
        res = run_benchmark(c)
        res.save(out / c.name)
        row = sweep_row(res)
        _log_row(row)
        row["_summary"] = res.summary
        return row

    result = find_capacity(
        probe, constraints, saturation=saturation, rel_tol=rel_tol, max_probes=max_probes
    )
    (out / f"{cfg.name}_capacity.json").write_text(
        json.dumps({"name": cfg.name, "config": cfg.to_json(), **result.to_json()}, indent=1) + "\n"
    )
    return result
