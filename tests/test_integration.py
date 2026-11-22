"""End-to-end tests against the in-process fake server (real sockets, real SSE)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import ServerThread, free_port

from llmbench.arrivals import arrival_offsets
from llmbench.datasets import LengthDist, TokenSpace, random_workload
from llmbench.metrics import summarize
from llmbench.runner import ClientConfig, LoadSpec, run_load
from llmbench.types import ErrorKind, Request

Start = Callable[..., ServerThread]


def workload(n: int, out: int = 20, plen: int = 32) -> list[Request]:
    return random_workload(
        n, TokenSpace(1000), LengthDist.parse(str(plen)), LengthDist.parse(str(out))
    ).requests


def records(server: ServerThread, reqs: list[Request], load: LoadSpec, **kw: object):  # type: ignore[no-untyped-def]
    cfg = ClientConfig(base_url=server.url, model="fake-model", **kw)  # type: ignore[arg-type]
    outs = run_load(reqs, load, cfg)
    return [r for o in outs for r in o.records]


def test_fixed_timing_is_measured_accurately(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.05, itl_s=0.01)
    recs = records(srv, workload(20), LoadSpec.closed(5))
    assert len(recs) == 20 and all(r.ok for r in recs)
    s = summarize(recs)
    assert s["ttft_ms"]["p50"] == pytest.approx(50, abs=5)
    assert s["tpot_ms"]["p50"] == pytest.approx(10, abs=1)
    assert s["itl_ms"]["p50"] == pytest.approx(10, abs=1.5)
    assert all(r.usage_completion_tokens == 20 == len(r.chunk_ns) for r in recs)
    assert all(r.usage_prompt_tokens == 32 for r in recs)
    assert all(r.text == " tok" * 20 for r in recs)


def test_chat_role_chunk_does_not_count_as_first_token(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.08, itl_s=0.005)
    reqs = [Request(id=i, prompt="hello there", prompt_tokens=2, max_tokens=5) for i in range(4)]
    recs = records(srv, reqs, LoadSpec.closed(4), backend="chat")
    s = summarize(recs)
    assert s["ttft_ms"]["min"] > 70  # the role chunk arrives at ~0 ms
    assert all(len(r.chunk_ns) == 5 for r in recs)


def test_http_errors_are_classified(fake_server: Start) -> None:
    srv = fake_server(error_rate=1.0)
    recs = records(srv, workload(5), LoadSpec.closed(2))
    assert all(r.error is ErrorKind.HTTP and r.status == 500 for r in recs)
    assert summarize(recs)["requests"]["errors"] == {"http_status": 5}


def test_dropped_streams_are_incomplete(fake_server: Start) -> None:
    srv = fake_server(drop_rate=1.0, itl_s=0.001)
    recs = records(srv, workload(6), LoadSpec.closed(3))
    assert all(r.error is ErrorKind.INCOMPLETE for r in recs)


def test_error_events_are_server_errors(fake_server: Start) -> None:
    srv = fake_server(error_event_rate=1.0, itl_s=0.001)
    recs = records(srv, workload(4), LoadSpec.closed(2))
    assert all(r.error is ErrorKind.SERVER for r in recs)


def test_timeouts(fake_server: Start) -> None:
    srv = fake_server(ttft_s=2.0)
    recs = records(srv, workload(2), LoadSpec.closed(2), timeout_s=0.3)
    assert all(r.error is ErrorKind.TIMEOUT for r in recs)


def test_connection_refused() -> None:
    cfg = ClientConfig(base_url=f"http://127.0.0.1:{free_port()}", model="m")
    outs = run_load(workload(2), LoadSpec.closed(1), cfg)
    assert all(r.error is ErrorKind.CONNECTION for o in outs for r in o.records)


def test_open_loop_follows_schedule(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.01, itl_s=0.002)
    offsets = arrival_offsets("poisson", 40, rate=40.0, seed=3)
    recs = records(srv, workload(40, out=5), LoadSpec.open(offsets))
    recs.sort(key=lambda r: r.id)
    sched = np.array([r.scheduled_ns for r in recs], dtype=float)
    assert np.allclose((sched - sched.min()) / 1e9, offsets - offsets.min(), atol=1e-6)
    lag = summarize(recs)["client"]["dispatch_lag_ms"]
    assert lag["p99"] < 20


def test_max_duration_stops_dispatch(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.001, itl_s=0.001)
    offsets = np.arange(50) * 0.02  # 1 s of arrivals
    recs = records(srv, workload(50, out=2), LoadSpec.open(offsets, max_duration_s=0.3))
    assert 12 <= len(recs) <= 18


def test_closed_loop_respects_concurrency(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.02, itl_s=0.002)
    recs = records(srv, workload(30, out=10), LoadSpec.closed(3))
    events = sorted([(r.sent_ns, 1) for r in recs] + [(r.end_ns, -1) for r in recs])
    inflight = peak = 0
    for _, d in events:
        inflight += d
        peak = max(peak, inflight)
    assert peak <= 3


def test_open_loop_concurrency_cap_counts_queueing_in_ttft(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.05, itl_s=0.001)
    # 8 requests at once, at most 2 in flight: the last pair waits ~3 rounds before it is sent.
    recs = records(srv, workload(8, out=2), LoadSpec.open([0.0] * 8, max_concurrency=2))
    s = summarize(recs)
    assert s["ttft_ms"]["max"] > 150
    assert s["ttft_service_ms"]["max"] < 100


def test_multiprocess_workers_share_one_clock(fake_server: Start) -> None:
    srv = fake_server(ttft_s=0.05, itl_s=0.005)
    cfg = ClientConfig(base_url=srv.url, model="fake-model")
    offsets = arrival_offsets("poisson", 40, rate=50.0, seed=1)
    outs = run_load(workload(40, out=8), LoadSpec.open(offsets), cfg, workers=2)
    assert [o.worker for o in outs] == [0, 1]
    recs = [r for o in outs for r in o.records]
    assert sorted(r.id for r in recs) == list(range(40))
    assert {r.worker for r in recs} == {0, 1}
    s = summarize(recs)
    assert s["ttft_ms"]["p50"] == pytest.approx(50, abs=8)
    sched = np.sort([r.scheduled_ns for r in recs]).astype(float)
    assert np.allclose((sched - sched[0]) / 1e9, np.sort(offsets), atol=1e-6)


def test_closed_loop_hides_queueing_that_open_loop_reveals(fake_server: Start) -> None:
    """Coordinated omission: at the same overload, the closed-loop client slows its own sending
    and reports far lower TTFT than an open-loop client measuring from intended arrival."""
    srv = fake_server(
        mode="batch",
        max_num_seqs=4,
        step_base_s=0.005,
        step_per_seq_s=0.0,
        step_per_prefill_token_s=0.0,
    )
    reqs = workload(48, out=10)
    closed = summarize(records(srv, reqs, LoadSpec.closed(4)))
    # The engine finishes 4 requests every ~55 ms (~70 req/s); offer 150 req/s.
    offsets = arrival_offsets("constant", 48, rate=150.0)
    opened = summarize(records(srv, reqs, LoadSpec.open(offsets)))
    assert opened["ttft_ms"]["p99"] > 3 * closed["ttft_ms"]["p99"]
