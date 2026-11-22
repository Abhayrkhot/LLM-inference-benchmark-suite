"""Smaller paths: non-streaming servers, GPU telemetry, result round trips, sweep sizing."""

from __future__ import annotations

import asyncio
import math
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from aiohttp import web
from conftest import ServerThread, free_port

from llmbench.bench import BenchConfig, WorkloadConfig, run_benchmark
from llmbench.monitor import GPUMonitor, MetricsMonitor
from llmbench.results import load_records
from llmbench.runner import ClientConfig, LoadSpec, run_load
from llmbench.sweep import run_sweep
from llmbench.types import Request

Start = Callable[..., ServerThread]


def test_server_that_ignores_stream_true() -> None:
    """Some gateways buffer the response; the client then sees one chunk at the end."""

    async def handler(_: web.Request) -> web.Response:
        await asyncio.sleep(0.05)
        return web.json_response(
            {"choices": [{"text": "a b c"}], "usage": {"prompt_tokens": 3, "completion_tokens": 3}}
        )

    async def main() -> list[object]:
        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        port = free_port()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        try:
            cfg = ClientConfig(base_url=f"http://127.0.0.1:{port}", model="m")
            reqs = [Request(id=0, prompt="x y z", prompt_tokens=3, max_tokens=3)]
            outs = await asyncio.to_thread(run_load, reqs, LoadSpec.closed(1), cfg)
            return [r for o in outs for r in o.records]
        finally:
            await runner.cleanup()

    (rec,) = asyncio.run(main())
    assert rec.ok and rec.usage_completion_tokens == 3  # type: ignore[attr-defined]
    assert rec.first_token_ns == rec.chunk_ns[-1]  # type: ignore[attr-defined]


def test_gpu_monitor_with_fake_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in --query-gpu=index) echo 0; echo 1; exit 0;; esac\n'
        # every third tick GPU 1 reports [N/A]; those ticks must be dropped, not mixed
        "i=0; while true; do i=$((i+1)); echo '0, 100, 50, 1000';\n"
        "  if [ $((i % 3)) -eq 0 ]; then echo '1, [N/A], 100, 2000';\n"
        "  else echo '1, 300, 100, 2000'; fi; sleep 0.05; done\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert GPUMonitor.available()

    async def main() -> dict[str, object]:
        from llmbench.client import now_ns

        mon = GPUMonitor(interval_ms=50)
        t0 = now_ns()
        await mon.start()
        await asyncio.sleep(0.6)
        await mon.stop()
        return mon.summary(t0, now_ns())

    s = asyncio.run(main())
    assert s["power_w_mean"] == pytest.approx(400)  # two GPUs summed
    assert s["util_pct_mean"] == pytest.approx(75)
    assert s["mem_used_mib_max"] == 3000
    assert s["energy_j"] > 0


def test_metrics_monitor_counts_failures() -> None:
    async def main() -> MetricsMonitor:
        mon = MetricsMonitor(f"http://127.0.0.1:{free_port()}", interval_s=0.05)
        mon.start()
        await asyncio.sleep(0.2)
        await mon.stop()
        return mon

    mon = asyncio.run(main())
    assert mon.errors >= 1 and mon.summary()["samples"] == 0


def test_records_round_trip(fake_server: Start, tmp_path: Path) -> None:
    srv = fake_server(ttft_s=0.005, itl_s=0.001)
    cfg = BenchConfig(
        base_url=srv.url,
        model="fake-model",
        gpu_monitor=False,
        workload=WorkloadConfig(num_requests=6, vocab_size=500, input_len="8", output_len="4"),
    )
    res = run_benchmark(cfg)
    path = res.save(tmp_path / "rt")
    back = load_records(path)
    assert [r.to_json() for r in back] == [r.to_json() for r in res.records]


def test_point_duration_sizes_each_point(fake_server: Start, tmp_path: Path) -> None:
    srv = fake_server(ttft_s=0.001, itl_s=0.001)
    cfg = BenchConfig(
        base_url=srv.url,
        model="fake-model",
        name="pd",
        workload=WorkloadConfig(num_requests=7, vocab_size=500, input_len="8", output_len="2"),
    )
    rows = run_sweep(cfg, tmp_path, rates=[50.0, math.inf], point_duration_s=0.4, min_requests=5)
    assert [r["requests_ok"] for r in rows] == [20, 7]


def test_closed_loop_sharding_keeps_user_count() -> None:
    from llmbench.runner import _shard

    reqs = [Request(id=i, prompt=[1], prompt_tokens=1, max_tokens=1) for i in range(1000)]
    shards = _shard(reqs, LoadSpec.closed(8), 3)
    assert [s[1].concurrency for s in shards] == [3, 3, 2]
    # requests in proportion to users: 375 / 375 / 250
    assert [len(s[0]) for s in shards] == [375, 375, 250]
    assert sorted(r.id for s in shards for r in s[0]) == list(range(1000))


def test_bad_worker_splits_are_rejected() -> None:
    cfg = ClientConfig(base_url="http://127.0.0.1:1", model="m")
    reqs = [Request(id=i, prompt=[1], prompt_tokens=1, max_tokens=1) for i in range(4)]
    with pytest.raises(ValueError):
        run_load(reqs, LoadSpec.closed(2), cfg, workers=4)
    with pytest.raises(ValueError):
        run_load(reqs, LoadSpec.open([0.0] * 4, max_concurrency=2), cfg, workers=4)


def test_fake_batch_server_rejects_requests_that_can_never_fit(fake_server: Start) -> None:
    srv = fake_server(mode="batch", kv_tokens=100)
    reqs = [Request(id=0, prompt=[1] * 90, prompt_tokens=90, max_tokens=20)]
    outs = run_load(reqs, LoadSpec.closed(1), ClientConfig(base_url=srv.url, model="m"))
    assert outs[0].records[0].status == 400


def test_fake_batch_server_injects_error_events(fake_server: Start) -> None:
    from llmbench.types import ErrorKind

    srv = fake_server(mode="batch", error_event_rate=1.0)
    reqs = [Request(id=i, prompt=[1] * 4, prompt_tokens=4, max_tokens=8) for i in range(3)]
    outs = run_load(reqs, LoadSpec.closed(3), ClientConfig(base_url=srv.url, model="m"))
    assert all(r.error is ErrorKind.SERVER for r in outs[0].records)


def test_ratio_gauges_are_averaged_across_engines() -> None:
    from llmbench.monitor import parse_prometheus

    text = (
        'vllm:kv_cache_usage_perc{engine="0"} 0.8\nvllm:kv_cache_usage_perc{engine="1"} 0.6\n'
        'vllm:num_requests_running{engine="0"} 3\nvllm:num_requests_running{engine="1"} 4\n'
    )
    p = parse_prometheus(text)
    assert p["vllm:kv_cache_usage_perc"] == pytest.approx(0.7)
    assert p["vllm:num_requests_running"] == 7
