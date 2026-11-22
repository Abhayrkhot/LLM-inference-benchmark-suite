"""Text reports: one-run summaries, multi-run tables and statistical comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from llmbench.metrics import per_request
from llmbench.results import load_records, load_result

__all__ = ["compare_runs", "format_summary", "summary_table", "sweep_table"]


def _f(x: Any, nd: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def format_summary(name: str, s: dict[str, Any], gpu: dict[str, Any]) -> str:
    req = s["requests"]
    lines = [
        f"== {name}",
        f"requests     {req['ok']} ok / {req['total']} sent"
        + (f"   errors: {req['errors']}" if req["failed"] else ""),
    ]
    if "throughput" not in s:
        return "\n".join(lines)
    t = s["throughput"]
    lines.append(f"duration     {s['duration_s']:.2f} s")
    lines.append(
        f"throughput   {t['requests_per_s']:.2f} req/s   {t['output_tokens_per_s']:,.0f} output "
        f"tok/s   {t['total_tokens_per_s']:,.0f} total tok/s"
    )
    if "goodput_requests_per_s" in t:
        lines.append(
            f"goodput      {t['goodput_requests_per_s']:.2f} req/s "
            f"({t['goodput_fraction']:.1%} of requests met {t['slo']})"
        )
    lines.append(f"{'':12} {'mean':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9}")
    for key, label in (
        ("ttft_ms", "TTFT ms"),
        ("tpot_ms", "TPOT ms"),
        ("itl_ms", "ITL ms"),
        ("e2e_ms", "E2E ms"),
    ):
        d = s.get(key, {})
        if d.get("n"):
            lines.append(
                f"{label:12} {d['mean']:9.1f} {d['p50']:9.1f} {d['p90']:9.1f} {d['p99']:9.1f} "
                f"{d['max']:9.1f}"
            )
    c = s.get("client", {})
    lag = c.get("dispatch_lag_ms", {})
    loop = c.get("loop_lag_ms", {})
    if lag.get("n"):
        lines.append(
            f"client       dispatch lag p99 {lag['p99']:.2f} ms, event-loop lag p99 "
            f"{loop.get('p99', 0):.2f} ms, mean in flight {c['mean_in_flight']:.1f}"
        )
    tok = s.get("tokens", {})
    if tok:
        lines.append(
            f"tokens       usage reported {tok['usage_reported_fraction']:.0%}, "
            f"{tok['tokens_per_chunk_mean']:.2f} tokens/chunk"
            + (
                f", client count == usage {tok['client_equals_usage_fraction']:.0%}"
                if "client_equals_usage_fraction" in tok
                else ""
            )
        )
    if gpu.get("energy_j"):
        lines.append(
            f"gpu          {gpu['power_w_mean']:.0f} W mean, {gpu['util_pct_mean']:.0f}% util, "
            f"{gpu.get('joules_per_output_token', 0):.3f} J/output token"
        )
    return "\n".join(lines)


def summary_table(results: Sequence[dict[str, Any]]) -> str:
    head = (
        "| run | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p99 | "
        "E2E p50 | failed |"
    )
    rows = [head, "|" + "---|" * 10]
    for r in results:
        s = r["summary"]
        t = s.get("throughput", {})
        rows.append(
            f"| {r['name']} | {_f(t.get('requests_per_s'), 2)} | "
            f"{_f(t.get('output_tokens_per_s'), 0)} | {_f(s.get('ttft_ms', {}).get('p50'))} | "
            f"{_f(s.get('ttft_ms', {}).get('p99'))} | {_f(s.get('tpot_ms', {}).get('p50'))} | "
            f"{_f(s.get('tpot_ms', {}).get('p99'))} | {_f(s.get('itl_ms', {}).get('p99'))} | "
            f"{_f(s.get('e2e_ms', {}).get('p50'))} | {s['requests']['failed']} |"
        )
    return "\n".join(rows)


def sweep_table(rows: Sequence[dict[str, Any]]) -> str:
    head = (
        "| load | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | "
        "goodput | failed |"
    )
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        ld = r["load"]
        load = f"{ld['concurrency']} users" if ld.get("concurrency") else f"{ld['rate']:g} req/s"
        lines.append(
            f"| {load} | {_f(r['requests_per_s'], 2)} | {_f(r['output_tokens_per_s'], 0)} | "
            f"{_f(r['ttft_p50_ms'])} | {_f(r['ttft_p99_ms'])} | {_f(r['tpot_p50_ms'])} | "
            f"{_f(r['tpot_p99_ms'])} | {_f(r.get('goodput_requests_per_s'), 2)} | "
            f"{r['requests_failed']} |"
        )
    return "\n".join(lines)


def bootstrap_diff(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    q: float,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Relative change of the ``q``-quantile from ``a`` to ``b`` with a 95% percentile-bootstrap
    CI: ``(point, lo, hi)`` as fractions (0.1 = +10%)."""
    rng = np.random.default_rng(seed)
    qa, qb = np.quantile(a, q), np.quantile(b, q)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    ra = np.quantile(a[ia], q, axis=1)
    rb = np.quantile(b[ib], q, axis=1)
    rel = rb / ra - 1
    return float(qb / qa - 1), float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))


def compare_runs(baseline: str, candidate: str, *, metrics: Sequence[str], seed: int = 0) -> str:
    ra, rb = load_result(baseline), load_result(candidate)
    ma, mb = per_request(load_records(baseline)), per_request(load_records(candidate))
    lines = [
        f"baseline:  {ra['name']}",
        f"candidate: {rb['name']}",
        "",
        "| metric | baseline | candidate | change | 95% CI | significant |",
        "|---|---|---|---|---|---|",
    ]
    for spec in metrics:
        name, stat = spec.split(":")
        a, b = ma[name], mb[name]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        q = 0.5 if stat == "p50" else float(stat[1:]) / 100
        pt, lo, hi = bootstrap_diff(a, b, q, seed=seed)
        sig = "yes" if lo > 0 or hi < 0 else "no"
        lines.append(
            f"| {name} {stat} | {np.quantile(a, q):.1f} ms | {np.quantile(b, q):.1f} ms | "
            f"{pt:+.1%} | [{lo:+.1%}, {hi:+.1%}] | {sig} |"
        )
    ta, tb = ra["summary"]["throughput"], rb["summary"]["throughput"]
    lines.append(
        f"| output tok/s | {ta['output_tokens_per_s']:,.0f} | {tb['output_tokens_per_s']:,.0f} | "
        f"{tb['output_tokens_per_s'] / ta['output_tokens_per_s'] - 1:+.1%} | - | - |"
    )
    return "\n".join(lines)
