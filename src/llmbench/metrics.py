"""Latency, throughput and goodput statistics from per-request records.

Definitions (all latencies in milliseconds):

* **TTFT**: arrival (``scheduled_ns``) to the first chunk carrying output text. A role-only first
  chunk, keep-alive comments and the usage chunk are not tokens.
* **TTFT (service)**: the same, but from when the request was actually sent.
* **TPOT**: ``(last token chunk - first token chunk) / (output_tokens - 1)``; the mean decode
  interval a user sees once streaming starts. Uses the server's token count, so it stays correct
  when one chunk carries several tokens (speculative decoding, ``stream_interval > 1``).
* **ITL**: every gap between consecutive token chunks, pooled over requests.
* **E2E**: arrival to the end of the response.
* **Goodput**: completed requests per second that met every per-request SLO (DistServe, OSDI'24).

Confidence intervals for percentiles are distribution-free (order statistics of the binomial),
so they need no resampling. They describe sampling error inside one run; they assume requests
are independent, which pooled ITL gaps are not, so no CI is reported for ITL.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from llmbench.types import RequestRecord

__all__ = ["SLO", "Constraint", "quantile_ci", "summarize"]

PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0, 99.9)
_Z95 = 1.959963984540054


def _pkey(p: float) -> str:
    return f"p{p:g}"


def quantile_ci(sorted_x: np.ndarray, q: float, conf_z: float = _Z95) -> tuple[float, float] | None:
    """Distribution-free CI for the ``q``-quantile from sorted samples (normal approximation to
    the binomial count of samples below the true quantile). ``None`` if n is too small."""
    n = sorted_x.size
    if n < 20:
        return None
    half = conf_z * math.sqrt(n * q * (1 - q))
    lo = math.floor(n * q - half)
    hi = math.ceil(n * q + half)
    if lo < 0 or hi > n - 1:
        return None
    return float(sorted_x[lo]), float(sorted_x[hi])


def describe(x: np.ndarray, *, ci: bool = True) -> dict[str, Any]:
    if x.size == 0:
        return {"n": 0}
    s = np.sort(x)
    out: dict[str, Any] = {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s[0]),
        "max": float(s[-1]),
    }
    for p in PERCENTILES:
        out[_pkey(p)] = float(np.percentile(s, p))
    if ci:
        cis = {}
        for p in (50.0, 90.0, 99.0):
            c = quantile_ci(s, p / 100)
            if c is not None:
                cis[_pkey(p)] = [c[0], c[1]]
        out["ci95"] = cis
    return out


@dataclass(frozen=True)
class SLO:
    """Per-request latency limits in ms; a request meets the SLO if it meets all of them."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None

    @classmethod
    def parse(cls, spec: str | None) -> SLO | None:
        """``"ttft<=500ms,tpot<=30ms"``; units ``ms`` (default) or ``s``."""
        if not spec:
            return None
        vals: dict[str, float] = {}
        for part in spec.split(","):
            m = re.fullmatch(r"\s*(ttft|tpot|e2e)\s*<=?\s*([0-9.]+)\s*(ms|s)?\s*", part)
            if not m:
                raise ValueError(f"bad SLO term {part!r}; expected e.g. 'ttft<=500ms'")
            v = float(m.group(2)) * (1000.0 if m.group(3) == "s" else 1.0)
            vals[f"{m.group(1)}_ms"] = v
        return cls(**vals)

    def __str__(self) -> str:
        terms = [
            f"{k[:-3]}<={v:g}ms"
            for k, v in (
                ("ttft_ms", self.ttft_ms),
                ("tpot_ms", self.tpot_ms),
                ("e2e_ms", self.e2e_ms),
            )
            if v is not None
        ]
        return ",".join(terms)


@dataclass(frozen=True)
class Constraint:
    """An aggregate condition on a run summary, e.g. ``ttft_p99<=500ms`` or ``error_rate<=0.01``.
    Capacity search looks for the highest load at which all constraints hold."""

    metric: str
    stat: str | None
    op: str
    value: float

    _RE = re.compile(
        r"\s*(ttft|ttft_service|tpot|itl|e2e)_(p[0-9.]+|mean)\s*(<=|>=|<|>)\s*([0-9.]+)\s*(ms|s)?\s*"
        r"|\s*(error_rate|goodput_fraction)\s*(<=|>=|<|>)\s*([0-9.]+)\s*"
    )

    @classmethod
    def parse_all(cls, spec: str) -> list[Constraint]:
        out = []
        for part in spec.split(","):
            m = cls._RE.fullmatch(part)
            if not m:
                raise ValueError(f"bad constraint {part!r}; expected e.g. 'ttft_p99<=500ms'")
            if m.group(1):
                v = float(m.group(4)) * (1000.0 if m.group(5) == "s" else 1.0)
                stat = m.group(2)
                if stat != "mean":
                    stat = _pkey(float(stat[1:]))  # p99.0 -> p99
                    if stat not in {_pkey(p) for p in PERCENTILES}:
                        raise ValueError(
                            f"constraint {part!r}: percentiles available are "
                            f"{', '.join(_pkey(p) for p in PERCENTILES)}"
                        )
                out.append(cls(m.group(1), stat, m.group(3), v))
            else:
                out.append(cls(m.group(6), None, m.group(7), float(m.group(8))))
        return out

    def observed(self, summary: dict[str, Any]) -> float:
        if self.stat is None:
            if self.metric == "error_rate":
                req = summary["requests"]
                return float(req["failed"] / req["total"]) if req["total"] else 1.0
            return float(summary["throughput"].get("goodput_fraction", 0.0))
        d = summary.get(f"{self.metric}_ms", {})
        return float(d.get(self.stat, math.inf))

    def holds(self, summary: dict[str, Any]) -> bool:
        x = self.observed(summary)
        return {
            "<=": x <= self.value,
            "<": x < self.value,
            ">=": x >= self.value,
            ">": x > self.value,
        }[self.op]

    def __str__(self) -> str:
        name = f"{self.metric}_{self.stat}" if self.stat else self.metric
        unit = "ms" if self.stat else ""
        return f"{name}{self.op}{self.value:g}{unit}"


def per_request(records: Sequence[RequestRecord]) -> dict[str, np.ndarray]:
    """Per-request metric arrays for successful requests (ms)."""
    ok = [r for r in records if r.ok and r.first_token_ns]
    ttft = np.array([(r.first_token_ns - r.scheduled_ns) / 1e6 for r in ok])
    ttft_svc = np.array([(r.first_token_ns - r.sent_ns) / 1e6 for r in ok])
    e2e = np.array([(r.end_ns - r.scheduled_ns) / 1e6 for r in ok])
    out_tok = np.array([r.output_tokens for r in ok], dtype=float)
    tpot = np.full(len(ok), np.nan)
    for i, r in enumerate(ok):
        if out_tok[i] > 1 and len(r.chunk_ns) > 1:
            tpot[i] = (r.chunk_ns[-1] - r.first_token_ns) / 1e6 / (out_tok[i] - 1)
    return {
        "ttft": ttft,
        "ttft_service": ttft_svc,
        "e2e": e2e,
        "tpot": tpot,
        "output_tokens": out_tok,
    }


def _itl(records: Iterable[RequestRecord]) -> np.ndarray:
    parts = [
        np.diff(np.frombuffer(r.chunk_ns, dtype=np.int64)) / 1e6
        for r in records
        if r.ok and len(r.chunk_ns) > 1
    ]
    return np.concatenate(parts) if parts else np.zeros(0)


def meets_slo(m: dict[str, np.ndarray], slo: SLO) -> np.ndarray:
    good = np.ones(m["ttft"].size, dtype=bool)
    if slo.ttft_ms is not None:
        good &= m["ttft"] <= slo.ttft_ms
    if slo.tpot_ms is not None:
        # One-token answers have no TPOT; they cannot violate it.
        good &= np.nan_to_num(m["tpot"], nan=0.0) <= slo.tpot_ms
    if slo.e2e_ms is not None:
        good &= m["e2e"] <= slo.e2e_ms
    return good


def summarize(
    records: Sequence[RequestRecord],
    *,
    slo: SLO | None = None,
    loop_lag_ns: Sequence[int] | None = None,
) -> dict[str, Any]:
    measured = [r for r in records if not r.warmup]
    ok = [r for r in measured if r.ok and r.first_token_ns]
    errors = Counter(r.error.value for r in measured if not r.ok)
    no_token = sum(1 for r in measured if r.ok and not r.first_token_ns)
    if no_token:
        errors["no_output"] = no_token
    out: dict[str, Any] = {
        "requests": {
            "total": len(measured),
            "ok": len(ok),
            "failed": len(measured) - len(ok),
            "errors": dict(errors),
        }
    }
    if not measured:
        return out
    t_start = min(r.scheduled_ns for r in measured)
    t_end = max(r.end_ns for r in measured)
    duration = max((t_end - t_start) / 1e9, 1e-9)
    out["duration_s"] = duration
    m = per_request(ok)
    out_tokens = float(m["output_tokens"].sum())
    in_tokens = float(
        sum(r.usage_prompt_tokens if r.usage_prompt_tokens >= 0 else r.prompt_tokens for r in ok)
    )
    thr: dict[str, Any] = {
        "requests_per_s": len(ok) / duration,
        "output_tokens_per_s": out_tokens / duration,
        "input_tokens_per_s": in_tokens / duration,
        "total_tokens_per_s": (in_tokens + out_tokens) / duration,
    }
    if slo is not None:
        good = int(meets_slo(m, slo).sum())
        thr["goodput_requests_per_s"] = good / duration
        thr["goodput_fraction"] = good / len(measured)
        thr["slo"] = str(slo)
    out["throughput"] = thr
    out["ttft_ms"] = describe(m["ttft"])
    out["ttft_service_ms"] = describe(m["ttft_service"])
    out["tpot_ms"] = describe(m["tpot"][~np.isnan(m["tpot"])])
    out["itl_ms"] = describe(_itl(ok), ci=False)
    out["e2e_ms"] = describe(m["e2e"])

    lag = np.array([(r.sent_ns - r.scheduled_ns) / 1e6 for r in measured])
    chunks = np.array([len(r.chunk_ns) for r in ok], dtype=float)
    usage = [r for r in ok if r.usage_completion_tokens >= 0]
    client: dict[str, Any] = {
        "dispatch_lag_ms": describe(lag, ci=False),
        "mean_in_flight": float(m["e2e"].sum() / 1e3 / duration),
    }
    if loop_lag_ns is not None and len(loop_lag_ns):
        client["loop_lag_ms"] = describe(np.asarray(loop_lag_ns, dtype=float) / 1e6, ci=False)
    out["client"] = client
    tok: dict[str, Any] = {
        "usage_reported_fraction": len(usage) / len(ok) if ok else 0.0,
        "output_tokens_mean": float(m["output_tokens"].mean()) if ok else 0.0,
        "tokens_per_chunk_mean": float(out_tokens / chunks.sum()) if chunks.sum() else 0.0,
        "hit_max_tokens_fraction": (
            sum(1 for r in ok if r.output_tokens == r.max_tokens) / len(ok) if ok else 0.0
        ),
    }
    both = [r for r in usage if r.client_completion_tokens >= 0]
    if both:
        diff = np.array([r.client_completion_tokens - r.usage_completion_tokens for r in both])
        tok["client_minus_usage_mean"] = float(diff.mean())
        tok["client_equals_usage_fraction"] = float((diff == 0).mean())
    out["tokens"] = tok
    return out
