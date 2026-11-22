"""CLI, result files, sweeps, capacity search and telemetry against the fake server."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import ServerThread

from llmbench.bench import BenchConfig, LoadConfig, WorkloadConfig, run_benchmark
from llmbench.cli import main
from llmbench.metrics import Constraint
from llmbench.monitor import canonical_gauges, parse_prometheus
from llmbench.results import load_records, load_result
from llmbench.sweep import find_capacity, run_capacity, run_sweep

Start = Callable[..., ServerThread]


def base_args(srv: ServerThread, *extra: str) -> list[str]:
    return [
        "--base-url",
        srv.url,
        "--model",
        "fake-model",
        "--vocab-size",
        "1000",
        "--input-len",
        "64",
        "--output-len",
        "8",
        *extra,
    ]


def test_run_writes_results_and_report(
    fake_server: Start, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    srv = fake_server(ttft_s=0.02, itl_s=0.002)
    rc = main(
        [
            "run",
            *base_args(
                srv,
                "-n",
                "30",
                "--rate",
                "100",
                "--warmup",
                "3",
                "--slo",
                "ttft<=1s,tpot<=50ms",
                "--tag",
                "engine=fake",
            ),
            "--out",
            str(tmp_path),
            "--name",
            "smoke",
        ]
    )
    assert rc == 0
    res = load_result(tmp_path / "smoke.json")
    s = res["summary"]
    assert s["requests"] == {"total": 30, "ok": 30, "failed": 0, "errors": {}}
    assert s["throughput"]["goodput_fraction"] == 1.0
    assert res["server"]["engine"] == "vllm"  # the fake mimics vLLM's /version and metrics
    assert res["config"]["tags"] == {"engine": "fake"}
    assert res["workload"]["stats"]["prompt_tokens_mean"] == 64
    assert res["engine_metrics"]["samples"] >= 1
    assert res["engine_metrics"]["stats"]["generation_tokens_delta"] >= 0
    recs = load_records(tmp_path / "smoke.json")
    assert len(recs) == 33 and sum(r.warmup for r in recs) == 3
    out = capsys.readouterr().out
    assert "TTFT ms" in out and "goodput" in out

    assert main(["report", str(tmp_path / "smoke.json")]) == 0
    assert main(["report", "--table", str(tmp_path / "smoke.json")]) == 0
    assert "| smoke |" in capsys.readouterr().out


def test_compare_detects_a_real_difference(
    fake_server: Start, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fast = fake_server(ttft_s=0.02, itl_s=0.002)
    slow = fake_server(ttft_s=0.06, itl_s=0.002)
    for srv, name in ((fast, "fast"), (slow, "slow")):
        assert (
            main(
                [
                    "run",
                    *base_args(srv, "-n", "40", "--concurrency", "8"),
                    "--out",
                    str(tmp_path),
                    "--name",
                    name,
                ]
            )
            == 0
        )
    capsys.readouterr()
    assert (
        main(
            [
                "compare",
                str(tmp_path / "fast.json"),
                str(tmp_path / "slow.json"),
                "--metrics",
                "ttft:p50,tpot:p50",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    ttft_row = next(line for line in out.splitlines() if line.startswith("| ttft p50"))
    assert ttft_row.endswith("| yes |")


def test_sweep_and_plot(fake_server: Start, tmp_path: Path) -> None:
    srv = fake_server(mode="batch", max_num_seqs=8)
    cfg = BenchConfig(
        base_url=srv.url,
        model="fake-model",
        name="sw",
        workload=WorkloadConfig(num_requests=24, vocab_size=1000, input_len="32", output_len="8"),
    )
    rows = run_sweep(cfg, tmp_path, rates=[20.0, math.inf])
    assert len(rows) == 2 and all(r["requests_ok"] == 24 for r in rows)
    assert rows[1]["ttft_p99_ms"] > rows[0]["ttft_p99_ms"]
    sweep = json.loads((tmp_path / "sw_sweep.json").read_text())
    assert [p["name"] for p in sweep["points"]] == ["sw_rate20", "sw_rateinf"]
    out = tmp_path / "sweep.png"
    assert main(["plot", str(tmp_path / "sw_sweep.json"), "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000


def test_concurrency_sweep_via_cli(fake_server: Start, tmp_path: Path) -> None:
    srv = fake_server(ttft_s=0.01, itl_s=0.001)
    assert (
        main(
            [
                "sweep",
                *base_args(srv, "-n", "12", "--concurrencies", "1,4"),
                "--out",
                str(tmp_path),
                "--name",
                "cc",
            ]
        )
        == 0
    )
    assert (tmp_path / "cc_conc4.json").exists()


def test_find_capacity_brackets_the_true_limit() -> None:
    true_cap = 7.3

    def probe(rate: float, i: int) -> dict[str, object]:
        p99 = 100.0 if rate <= true_cap else 5000.0
        ok = {"total": 100, "failed": 0}
        return {"rate": rate, "_summary": {"ttft_ms": {"p99": p99}, "requests": ok}}

    cs = Constraint.parse_all("ttft_p99<=500ms")
    res = find_capacity(probe, cs, saturation=lambda: 20.0, rel_tol=0.02, max_probes=20)
    lo, hi = res.bracket
    assert lo <= true_cap < hi and hi / lo - 1 <= 0.02
    assert res.capacity_rps == lo and res.saturation_rps == 20.0
    assert res.at_capacity is not None

    # Saturation below the true limit: the search grows past it, then bisects.
    low_sat = find_capacity(probe, cs, saturation=lambda: 5.0, rel_tol=0.02, max_probes=20)
    assert low_sat.bracket[0] <= true_cap < low_sat.bracket[1]
    lenient = find_capacity(probe, cs, saturation=lambda: 1.0, max_probes=3)
    assert lenient.bracket == (2.25, math.inf)

    # Fast failures must not pass a latency-only SLO.
    def shedding(rate: float, i: int) -> dict[str, object]:
        failed = 0 if rate <= true_cap else 50
        return {
            "_summary": {"ttft_ms": {"p99": 100.0}, "requests": {"total": 100, "failed": failed}}
        }

    shed = find_capacity(shedding, cs, saturation=lambda: 20.0, rel_tol=0.02, max_probes=20)
    assert shed.bracket[0] <= true_cap < shed.bracket[1]
    assert "error_rate<=0.01" in shed.constraints
    assert find_capacity(probe, cs, saturation=lambda: 0.0).capacity_rps == 0.0
    never = find_capacity(
        lambda r, i: {"_summary": {"ttft_ms": {"p99": 1e9}, "requests": {"total": 1, "failed": 0}}},
        cs,
        saturation=lambda: 1.0,
        max_probes=4,
    )
    assert never.capacity_rps == 0.0


def test_capacity_search_on_batch_engine(fake_server: Start, tmp_path: Path) -> None:
    # 8 slots, 10 tokens each at ~6 ms/step: at most ~120 req/s.
    srv = fake_server(
        mode="batch",
        max_num_seqs=8,
        step_base_s=0.006,
        step_per_seq_s=0.0,
        step_per_prefill_token_s=0.0,
    )
    cfg = BenchConfig(
        base_url=srv.url,
        model="fake-model",
        name="cap",
        workload=WorkloadConfig(num_requests=40, vocab_size=1000, input_len="16", output_len="10"),
        load=LoadConfig(arrival="constant"),
    )
    res = run_capacity(
        cfg,
        Constraint.parse_all("ttft_p99<=60ms"),
        tmp_path,
        probe_duration_s=2,
        min_requests=30,
        rel_tol=0.15,
        max_probes=6,
    )
    lo, hi = res.bracket
    assert 30 < res.capacity_rps == lo < hi < 200
    assert (tmp_path / "cap_capacity.json").exists()


def test_bench_api_counts_tokens_client_side(fake_server: Start, tmp_path: Path) -> None:
    from test_datasets import make_tokenizer

    make_tokenizer(tmp_path)
    srv = fake_server(ttft_s=0.005, itl_s=0.001)
    cfg = BenchConfig(
        base_url=srv.url,
        model="fake-model",
        workload=WorkloadConfig(
            num_requests=5, tokenizer=str(tmp_path), input_len="8", output_len="6"
        ),
    )
    res = run_benchmark(cfg)
    # " tok" is not in the toy vocabulary, so each chunk becomes one [UNK] token.
    assert res.summary["tokens"]["client_equals_usage_fraction"] == 1.0


def test_prometheus_parsing() -> None:
    text = """# HELP vllm:num_requests_running x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="a",engine="0"} 3
vllm:num_requests_running{model_name="a",engine="1"} 2
vllm:kv_cache_usage_perc{model_name="a"} 0.25
vllm:prefix_cache_hits_total{model_name="a"} 10
sglang:num_queue_reqs 4
bad line here
nan_metric NaN
"""
    p = parse_prometheus(text)
    assert p["vllm:num_requests_running"] == 5
    assert "nan_metric" not in p
    g = canonical_gauges(p)
    assert g == {"running": 5, "waiting": 4, "kv_usage": 0.25, "prefix_hits_total": 10}
