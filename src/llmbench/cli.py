"""Command-line interface: ``llmbench {run,sweep,capacity,report,compare,plot,fake-server}``."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmbench import __version__

log = logging.getLogger("llmbench")


def _float_or_inf(s: str) -> float:
    return math.inf if s.lower() in ("inf", "infinity", "max") else float(s)


def _floats(s: str) -> list[float]:
    return [_float_or_inf(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _add_target(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("server")
    g.add_argument("--base-url", default="http://127.0.0.1:8000", help="server root URL")
    g.add_argument("--model", required=True, help="model name the server expects")
    g.add_argument("--backend", default="completions", choices=["completions", "chat"])
    g.add_argument("--api-key", default=None)
    g.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="let the model stop early (output lengths then depend on the model)",
    )
    g.add_argument("--extra-body", default=None, help="JSON merged into every request body")

    w = p.add_argument_group("workload")
    w.add_argument(
        "--dataset", default="random", choices=["random", "prefix", "sharegpt", "azure", "mooncake"]
    )
    w.add_argument("--dataset-path", default=None)
    w.add_argument("--num-requests", "-n", type=int, default=1000)
    w.add_argument(
        "--input-len", default="1024", help="random: '512', 'uniform:128:1024', 'lognormal:512:0.6'"
    )
    w.add_argument("--output-len", default="128", help="same syntax as --input-len")
    w.add_argument("--num-prefixes", type=int, default=8)
    w.add_argument("--prefix-len", type=int, default=2048)
    w.add_argument("--suffix-len", default="256")
    w.add_argument("--tokenizer", default=None, help="HF model id or path with tokenizer.json")
    w.add_argument("--vocab-size", type=int, default=None, help="token-id prompts w/o tokenizer")
    w.add_argument("--max-input-len", type=int, default=None)
    w.add_argument("--max-output-len", type=int, default=None)
    w.add_argument("--sharegpt-output-len", type=int, default=None)
    w.add_argument("--trace-start", type=float, default=0.0, help="azure: skip into trace (s)")
    w.add_argument("--seed", type=int, default=0)

    load = p.add_argument_group("load")
    load.add_argument(
        "--arrival", default="poisson", choices=["poisson", "gamma", "constant", "trace"]
    )
    load.add_argument("--rate", type=_float_or_inf, default=math.inf, help="req/s (open loop)")
    load.add_argument("--cv", type=float, default=1.0, help="gamma arrivals: gap CV (burstiness)")
    load.add_argument("--concurrency", type=int, default=None, help="closed loop with N users")
    load.add_argument("--max-concurrency", type=int, default=None, help="open-loop in-flight cap")
    load.add_argument("--time-scale", type=float, default=1.0, help="trace replay speed-up")
    load.add_argument("--duration", type=float, default=None, help="stop sending after (s)")
    load.add_argument("--workers", type=int, default=1, help="client processes")
    load.add_argument("--warmup", type=int, default=0, help="warm-up requests (not measured)")

    o = p.add_argument_group("output")
    o.add_argument(
        "--slo", default=None, help="per-request goodput SLO, e.g. 'ttft<=500ms,tpot<=30ms'"
    )
    o.add_argument("--no-engine-metrics", action="store_true", help="don't scrape /metrics")
    o.add_argument("--gpu-monitor", action="store_true", help="sample nvidia-smi power/util")
    o.add_argument("--no-text", action="store_true", help="don't keep generated text")
    o.add_argument("--name", default="run")
    o.add_argument("--tag", action="append", default=[], help="key=value label (repeatable)")


def _config(a: argparse.Namespace) -> Any:
    from llmbench.backends import SamplingOptions
    from llmbench.bench import BenchConfig, LoadConfig, WorkloadConfig

    return BenchConfig(
        base_url=a.base_url,
        model=a.model,
        backend=a.backend,
        api_key=a.api_key,
        timeout_s=a.timeout,
        workload=WorkloadConfig(
            dataset=a.dataset,
            path=a.dataset_path,
            num_requests=a.num_requests,
            input_len=a.input_len,
            output_len=a.output_len,
            num_prefixes=a.num_prefixes,
            prefix_len=a.prefix_len,
            suffix_len=a.suffix_len,
            tokenizer=a.tokenizer,
            vocab_size=a.vocab_size,
            max_input_len=a.max_input_len,
            max_output_len=a.max_output_len,
            sharegpt_output_len=a.sharegpt_output_len,
            trace_start_s=a.trace_start,
            seed=a.seed,
        ),
        load=LoadConfig(
            arrival=a.arrival,
            rate=a.rate,
            cv=a.cv,
            concurrency=a.concurrency,
            max_concurrency=a.max_concurrency,
            time_scale=a.time_scale,
            duration_s=a.duration,
            workers=a.workers,
            warmup_requests=a.warmup,
        ),
        sampling=SamplingOptions(
            temperature=a.temperature,
            ignore_eos=not a.no_ignore_eos,
            extra_body=json.loads(a.extra_body) if a.extra_body else {},
        ),
        slo=a.slo,
        engine_metrics=not a.no_engine_metrics,
        gpu_monitor=a.gpu_monitor,
        keep_text=not a.no_text,
        name=a.name,
        tags=dict(t.split("=", 1) for t in a.tag),
    )


def _print_summary(name: str, s: dict[str, Any], gpu: dict[str, Any] | None = None) -> None:
    from llmbench.report import format_summary

    print(format_summary(name, s, gpu or {}))


def cmd_run(a: argparse.Namespace) -> int:
    from llmbench.bench import run_benchmark

    cfg = _config(a)
    res = run_benchmark(cfg)
    out = res.save(Path(a.out) / cfg.name, records=not a.no_records)
    _print_summary(cfg.name, res.summary, res.gpu)
    print(f"\nwrote {out}")
    return 0 if res.summary["requests"]["ok"] else 1


def cmd_sweep(a: argparse.Namespace) -> int:
    from llmbench.report import sweep_table
    from llmbench.sweep import run_sweep

    cfg = _config(a)
    rows = run_sweep(
        cfg,
        a.out,
        rates=_floats(a.rates) if a.rates else (),
        concurrencies=_ints(a.concurrencies) if a.concurrencies else (),
        reset_cache=not a.no_cache_reset,
        save_records=not a.no_records,
        point_duration_s=a.point_duration,
        min_requests=a.min_requests,
    )
    print(sweep_table(rows))
    return 0


def cmd_capacity(a: argparse.Namespace) -> int:
    from llmbench.metrics import Constraint
    from llmbench.sweep import run_capacity

    cfg = _config(a)
    res = run_capacity(
        cfg,
        Constraint.parse_all(a.constraints),
        a.out,
        probe_duration_s=a.probe_duration,
        min_requests=a.min_requests,
        rel_tol=a.tolerance,
        max_probes=a.max_probes,
        reset_cache=not a.no_cache_reset,
    )
    _, hi = res.bracket
    print(
        f"capacity at {', '.join(res.constraints)}: {res.capacity_rps:.3g} req/s "
        f"(failing at {hi:.3g}; saturated throughput {res.saturation_rps:.3g} req/s)"
    )
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from llmbench.report import format_summary, summary_table
    from llmbench.results import load_result

    results = [load_result(p) for p in a.paths]
    if a.table or len(results) > 1:
        print(summary_table(results))
    else:
        r = results[0]
        print(format_summary(r["name"], r["summary"], r.get("gpu") or {}))
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from llmbench.report import compare_runs

    print(compare_runs(a.baseline, a.candidate, metrics=a.metrics.split(","), seed=a.seed))
    return 0


def cmd_plot(a: argparse.Namespace) -> int:
    from llmbench.plots import plot_sweeps

    plot_sweeps(a.sweeps, a.out, labels=a.label or None, title=a.title)
    print(f"wrote {a.out}")
    return 0


def cmd_fake_server(a: argparse.Namespace) -> int:
    from llmbench.fakeserver import FakeEngineConfig, serve

    cfg = FakeEngineConfig(
        mode=a.mode,
        ttft_s=a.ttft,
        itl_s=a.itl,
        prefill_per_token_s=a.prefill_per_token,
        max_num_seqs=a.max_num_seqs,
        max_batched_tokens=a.max_batched_tokens,
        kv_tokens=a.kv_tokens,
        step_base_s=a.step_base,
        step_per_seq_s=a.step_per_seq,
        step_per_prefill_token_s=a.step_per_prefill_token,
        error_rate=a.error_rate,
        drop_rate=a.drop_rate,
        trace_dir=a.trace_dir,
        model=a.model_name,
    )
    print(f"fake server ({a.mode}) on http://{a.host}:{a.port}", file=sys.stderr, flush=True)
    serve(cfg, a.host, a.port, processes=a.processes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmbench", description=__doc__)
    p.add_argument("--version", action="version", version=f"llmbench {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one benchmark run")
    _add_target(r)
    r.add_argument("--out", default="results")
    r.add_argument("--no-records", action="store_true", help="skip the per-request records file")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="run at several rates or concurrencies")
    _add_target(s)
    s.add_argument("--rates", default=None, help="comma list of req/s, 'inf' allowed")
    s.add_argument("--concurrencies", default=None, help="comma list (closed loop)")
    s.add_argument("--out", default="results")
    s.add_argument("--no-records", action="store_true")
    s.add_argument(
        "--no-cache-reset",
        action="store_true",
        help="don't flush the engine prefix cache between points",
    )
    s.add_argument(
        "--point-duration",
        type=float,
        default=None,
        help="size each finite-rate point to about this many seconds",
    )
    s.add_argument("--min-requests", type=int, default=200)
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("capacity", help="highest request rate that meets latency constraints")
    _add_target(c)
    c.add_argument(
        "--constraints", required=True, help="e.g. 'ttft_p99<=500ms,tpot_p99<=30ms' (MLPerf-style)"
    )
    c.add_argument("--probe-duration", type=float, default=60.0, help="seconds of load per probe")
    c.add_argument("--min-requests", type=int, default=200)
    c.add_argument("--tolerance", type=float, default=0.05, help="relative bracket width")
    c.add_argument("--max-probes", type=int, default=10)
    c.add_argument("--out", default="results")
    c.add_argument("--no-cache-reset", action="store_true")
    c.set_defaults(func=cmd_capacity)

    rp = sub.add_parser("report", help="print summaries of result files")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--table", action="store_true")
    rp.set_defaults(func=cmd_report)

    cp = sub.add_parser("compare", help="bootstrap comparison of two runs")
    cp.add_argument("baseline")
    cp.add_argument("candidate")
    cp.add_argument("--metrics", default="ttft:p50,ttft:p99,tpot:p50,tpot:p99,e2e:p50")
    cp.add_argument("--seed", type=int, default=0)
    cp.set_defaults(func=cmd_compare)

    pl = sub.add_parser("plot", help="latency-throughput curves from sweep files")
    pl.add_argument("sweeps", nargs="+", help="*_sweep.json files")
    pl.add_argument("--label", action="append")
    pl.add_argument("--title", default=None)
    pl.add_argument("--out", default="sweep.png")
    pl.set_defaults(func=cmd_plot)

    f = sub.add_parser("fake-server", help="OpenAI-compatible fake engine for tests and demos")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--port", type=int, default=8000)
    f.add_argument("--mode", default="fixed", choices=["fixed", "batch"])
    f.add_argument("--ttft", type=float, default=0.05, help="fixed: seconds to first token")
    f.add_argument("--itl", type=float, default=0.01, help="fixed: seconds between tokens")
    f.add_argument("--prefill-per-token", type=float, default=0.0)
    f.add_argument("--max-num-seqs", type=int, default=64)
    f.add_argument("--max-batched-tokens", type=int, default=2048)
    f.add_argument("--kv-tokens", type=int, default=1 << 20)
    f.add_argument("--step-base", type=float, default=0.008)
    f.add_argument("--step-per-seq", type=float, default=0.0002)
    f.add_argument("--step-per-prefill-token", type=float, default=0.00002)
    f.add_argument("--error-rate", type=float, default=0.0)
    f.add_argument("--drop-rate", type=float, default=0.0)
    f.add_argument("--trace-dir", default=None, help="dump server-side chunk send times here")
    f.add_argument("--processes", type=int, default=1, help="fixed mode: SO_REUSEPORT workers")
    f.add_argument("--model-name", default="fake-model")
    f.set_defaults(func=cmd_fake_server)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
