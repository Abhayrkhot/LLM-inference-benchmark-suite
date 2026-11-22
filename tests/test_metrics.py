from __future__ import annotations

from array import array

import numpy as np
import pytest

from llmbench.metrics import SLO, Constraint, quantile_ci, summarize
from llmbench.types import ErrorKind, RequestRecord

MS = 1_000_000


def rec(
    i: int,
    sched_ms: float,
    first_ms: float,
    chunk_gaps_ms: list[float],
    *,
    tokens: int | None = None,
    sent_ms: float | None = None,
    end_extra_ms: float = 1.0,
) -> RequestRecord:
    chunks = [first_ms]
    for g in chunk_gaps_ms:
        chunks.append(chunks[-1] + g)
    r = RequestRecord(id=i, prompt_tokens=100, max_tokens=tokens or len(chunks))
    r.scheduled_ns = int(sched_ms * MS)
    r.sent_ns = int((sent_ms if sent_ms is not None else sched_ms) * MS)
    r.first_token_ns = int(first_ms * MS)
    r.chunk_ns = array("q", [int(c * MS) for c in chunks])
    r.chunk_chars = array("l", [3] * len(chunks))
    r.end_ns = int((chunks[-1] + end_extra_ms) * MS)
    r.usage_completion_tokens = tokens if tokens is not None else len(chunks)
    r.usage_prompt_tokens = 100
    return r


def test_latency_definitions() -> None:
    r = rec(0, 0, 50, [10, 10, 10], sent_ms=5)
    s = summarize([r])
    assert s["ttft_ms"]["p50"] == pytest.approx(50)
    assert s["ttft_service_ms"]["p50"] == pytest.approx(45)
    assert s["tpot_ms"]["p50"] == pytest.approx(10)
    assert s["itl_ms"]["n"] == 3 and s["itl_ms"]["p50"] == pytest.approx(10)
    assert s["e2e_ms"]["p50"] == pytest.approx(81)
    assert s["client"]["dispatch_lag_ms"]["p50"] == pytest.approx(5)


def test_tpot_uses_server_token_count_when_chunks_carry_several_tokens() -> None:
    # 3 chunks carrying 7 tokens (e.g. speculative decoding): 2 gaps of 30 ms over 6 intervals.
    r = rec(0, 0, 50, [30, 30], tokens=7)
    s = summarize([r])
    assert s["tpot_ms"]["p50"] == pytest.approx(10)
    assert s["itl_ms"]["p50"] == pytest.approx(30)
    assert s["tokens"]["tokens_per_chunk_mean"] == pytest.approx(7 / 3)


def test_throughput_and_duration() -> None:
    rs = [rec(i, i * 100, i * 100 + 20, [5] * 9) for i in range(10)]
    s = summarize(rs)
    duration = (rs[-1].end_ns - rs[0].scheduled_ns) / 1e9
    assert s["duration_s"] == pytest.approx(duration)
    assert s["throughput"]["output_tokens_per_s"] == pytest.approx(100 / duration)
    assert s["throughput"]["input_tokens_per_s"] == pytest.approx(1000 / duration)
    assert s["throughput"]["requests_per_s"] == pytest.approx(10 / duration)


def test_failures_and_warmup_are_excluded_from_latency() -> None:
    good = rec(0, 0, 50, [10])
    bad = rec(1, 0, 900, [10])
    bad.error = ErrorKind.TIMEOUT
    warm = rec(2, 0, 5000, [10])
    warm.warmup = True
    s = summarize([good, bad, warm])
    assert s["requests"] == {"total": 2, "ok": 1, "failed": 1, "errors": {"timeout": 1}}
    assert s["ttft_ms"]["max"] == pytest.approx(50)


def test_goodput_counts_requests_meeting_every_slo() -> None:
    rs = [
        rec(0, 0, 100, [10] * 4),  # meets both
        rec(1, 0, 900, [10] * 4),  # TTFT too slow
        rec(2, 0, 100, [80] * 4),  # TPOT too slow
        rec(3, 0, 100, []),  # one token: no TPOT, meets
    ]
    failed = rec(4, 0, 100, [10])
    failed.error = ErrorKind.HTTP
    s = summarize([*rs, failed], slo=SLO.parse("ttft<=500ms,tpot<=50ms"))
    assert s["throughput"]["goodput_fraction"] == pytest.approx(2 / 5)
    assert s["throughput"]["slo"] == "ttft<=500ms,tpot<=50ms"


def test_slo_parse() -> None:
    slo = SLO.parse("ttft<=0.5s, tpot<=30, e2e<=10s")
    assert slo == SLO(ttft_ms=500, tpot_ms=30, e2e_ms=10_000)
    assert SLO.parse(None) is None
    with pytest.raises(ValueError):
        SLO.parse("ttft>5")


def test_constraints() -> None:
    s = summarize([rec(i, 0, 100 + i, [10] * 3) for i in range(100)])
    cs = Constraint.parse_all("ttft_p99<=200ms,tpot_p50<0.011s,error_rate<=0.01")
    assert [str(c) for c in cs] == ["ttft_p99<=200ms", "tpot_p50<11ms", "error_rate<=0.01"]
    assert all(c.holds(s) for c in cs)
    assert not Constraint.parse_all("ttft_p50<=100ms")[0].holds(s)
    with pytest.raises(ValueError):
        Constraint.parse_all("latency<5")
    with pytest.raises(ValueError):  # not a reported percentile: would silently always fail
        Constraint.parse_all("ttft_p75<=1s")
    assert str(Constraint.parse_all("ttft_p99.0<=1s")[0]) == "ttft_p99<=1000ms"


def test_quantile_ci_coverage() -> None:
    """The distribution-free CI should cover the true p90 about 95% of the time."""
    rng = np.random.default_rng(0)
    true_p90 = -np.log(0.1)  # Exp(1)
    trials, covered = 1000, 0
    for _ in range(trials):
        c = quantile_ci(np.sort(rng.exponential(size=400)), 0.9)
        assert c is not None
        covered += c[0] <= true_p90 <= c[1]
    assert 0.93 <= covered / trials <= 0.975


def test_quantile_ci_small_samples() -> None:
    assert quantile_ci(np.arange(10.0), 0.5) is None
    assert quantile_ci(np.arange(100.0), 0.999) is None


def test_client_token_count_comparison() -> None:
    r = rec(0, 0, 10, [1, 1])
    r.client_completion_tokens = 3
    r2 = rec(1, 0, 10, [1, 1])
    r2.client_completion_tokens = 4
    tok = summarize([r, r2])["tokens"]
    assert tok["client_equals_usage_fraction"] == 0.5
    assert tok["client_minus_usage_mean"] == 0.5


def test_output_tokens_fallbacks() -> None:
    r = rec(0, 0, 10, [1, 1])
    assert r.output_tokens == 3
    r.usage_completion_tokens = -1
    r.client_completion_tokens = 5
    assert r.output_tokens == 5
    r.client_completion_tokens = -1
    assert r.output_tokens == 3  # chunk count


def test_empty() -> None:
    assert summarize([])["requests"]["total"] == 0
