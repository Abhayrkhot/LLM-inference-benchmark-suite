#!/usr/bin/env python3
"""Benchmark the benchmark: how much latency does the load generator itself add?

A multi-process fake server streams tokens on an exact schedule (TTFT 50 ms, one token every
10 ms) and records when it wrote every chunk. Clients on the same machine record when each chunk
arrived; because ``perf_counter_ns`` is system-wide, the difference is the delay added by the
kernel, the HTTP stack and the client's event loop. Swept over concurrent streams and client
worker processes, it shows where a single Python event loop stops keeping up.

Optionally runs ``vllm bench serve`` against the same server and compares its reported TTFT and
ITL percentiles with the server's ground truth.

    python experiments/client_overhead.py --out results/client_overhead/box.json \\
        --concurrency 64,256,1024,2048 --workers 1,4 --server-procs 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from llmbench.datasets import LengthDist, TokenSpace, random_workload
from llmbench.metrics import summarize
from llmbench.runner import ClientConfig, LoadSpec, run_load


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeServer:
    def __init__(self, procs: int, ttft: float, itl: float, trace_dir: str) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.trace_dir = trace_dir
        cmd = [
            sys.executable,
            "-m",
            "llmbench",
            "fake-server",
            "--port",
            str(self.port),
            "--processes",
            str(procs),
            "--ttft",
            str(ttft),
            "--itl",
            str(itl),
            "--trace-dir",
            trace_dir,
        ]
        self.proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
        for _ in range(200):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=1).read()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("fake server did not start")

    def stop(self) -> dict[str, list[int]]:
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(timeout=60)
        trace: dict[str, list[int]] = {}
        for f in glob.glob(os.path.join(self.trace_dir, "trace-*.json")):
            trace.update(json.loads(Path(f).read_text()))
        return trace


def trace_rate(trace: dict[str, list[int]]) -> float:
    """Chunks per second the server actually sent, from first arrival to last send (excludes
    client start-up, e.g. the ~20 s it takes ``vllm bench serve`` to import vLLM)."""
    sends = [v for v in trace.values() if len(v) > 1]
    if not sends:
        return 0.0
    t0 = min(v[0] for v in sends)
    t1 = max(v[-1] for v in sends)
    return sum(len(v) - 1 for v in sends) / ((t1 - t0) / 1e9)


def truth(trace: dict[str, list[int]]) -> dict[str, np.ndarray]:
    ttft = np.array([(v[1] - v[0]) / 1e6 for v in trace.values() if len(v) > 1])
    itl = np.concatenate([np.diff(v[1:]) / 1e6 for v in trace.values() if len(v) > 2])
    return {"ttft": ttft, "itl": itl}


def pct(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {}
    return {
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "p99.9": float(np.percentile(x, 99.9)),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def run_llmbench(
    server: FakeServer, conc: int, workers: int, tokens: int, per_stream: int
) -> dict[str, Any]:
    wl = random_workload(
        conc * per_stream,
        TokenSpace(32000),
        LengthDist.parse("16"),
        LengthDist.parse(str(tokens)),
        seed=conc,
    )
    cfg = ClientConfig(base_url=server.url, model="fake-model", keep_text=False)
    outs = run_load(wl.requests, LoadSpec.closed(conc), cfg, workers=workers)
    recs = [r for o in outs for r in o.records]
    lag = [x for o in outs for x in o.loop_lag_ns]
    trace = server.stop()
    tr = truth(trace)
    delays, ttft_err, itl_err = [], [], []
    for r in recs:
        sends = trace.get(str(r.id))
        if not r.ok or not sends or len(sends) - 1 != len(r.chunk_ns):
            continue
        s = np.array(sends[1:], dtype=np.int64)
        c = np.frombuffer(r.chunk_ns, dtype=np.int64)
        delays.append((c - s) / 1e6)
        ttft_err.append((r.first_token_ns - r.sent_ns - (sends[1] - sends[0])) / 1e6)
        itl_err.append((np.diff(c) - np.diff(s)) / 1e6)
    summ = summarize(recs, loop_lag_ns=lag)
    return {
        "tool": "llmbench",
        "concurrency": conc,
        "workers": workers,
        "requests_ok": summ["requests"]["ok"],
        "requests_failed": summ["requests"]["failed"],
        "chunks_per_s": trace_rate(trace),
        "matched_requests": len(delays),
        "chunk_delay_ms": pct(np.concatenate(delays)) if delays else {},
        "ttft_error_ms": pct(np.array(ttft_err)),
        "itl_error_ms": pct(np.concatenate(itl_err)) if itl_err else {},
        "reported": {
            "ttft_ms": pct(np.array([(r.first_token_ns - r.sent_ns) / 1e6 for r in recs if r.ok])),
            "itl_ms": {k: summ["itl_ms"].get(k) for k in ("p50", "p99", "p99.9", "max", "mean")},
        },
        "truth": {"ttft_ms": pct(tr["ttft"]), "itl_ms": pct(tr["itl"])},
        "loop_lag_ms": {k: summ["client"]["loop_lag_ms"].get(k) for k in ("p50", "p99", "max")},
    }


def run_vllm_bench(
    server: FakeServer,
    conc: int,
    tokens: int,
    per_stream: int,
    cmd: str,
    tokenizer: str,
    workdir: str,
) -> dict[str, Any]:
    out = os.path.join(workdir, f"vllm_{conc}.json")
    args = [
        *shlex.split(cmd),
        "--backend",
        "vllm",
        "--base-url",
        server.url,
        "--model",
        "fake-model",
        "--tokenizer",
        tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        "16",
        "--random-output-len",
        str(tokens),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(conc * per_stream),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,itl",
        "--metric-percentiles",
        "50,99,99.9",
        "--save-result",
        "--result-dir",
        workdir,
        "--result-filename",
        os.path.basename(out),
    ]
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    trace = server.stop()
    tr = truth(trace)
    if p.returncode != 0 or not os.path.exists(out):
        return {"tool": "vllm-bench", "concurrency": conc, "error": p.stderr[-2000:]}
    r = json.loads(Path(out).read_text())
    return {
        "tool": "vllm-bench",
        "concurrency": conc,
        "workers": 1,
        "requests_ok": r.get("completed"),
        "chunks_per_s": trace_rate(trace),
        "reported": {
            "ttft_ms": {
                "p50": r.get("p50_ttft_ms"),
                "p99": r.get("p99_ttft_ms"),
                "p99.9": r.get("p99.9_ttft_ms"),
                "mean": r.get("mean_ttft_ms"),
            },
            "itl_ms": {
                "p50": r.get("p50_itl_ms"),
                "p99": r.get("p99_itl_ms"),
                "p99.9": r.get("p99.9_itl_ms"),
                "mean": r.get("mean_itl_ms"),
            },
        },
        "truth": {"ttft_ms": pct(tr["ttft"]), "itl_ms": pct(tr["itl"])},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", default="64,256,1024,2048")
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--server-procs", type=int, default=8)
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--itl", type=float, default=0.01)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--per-stream", type=int, default=3, help="requests per concurrent stream")
    ap.add_argument(
        "--vllm-bench", default=None, help="command prefix that runs 'vllm bench serve'"
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for conc in [int(x) for x in a.concurrency.split(",")]:
            for w in [int(x) for x in a.workers.split(",")]:
                tdir = os.path.join(tmp, f"llmbench_{conc}_{w}")
                srv = FakeServer(a.server_procs, a.ttft, a.itl, tdir)
                row = run_llmbench(srv, conc, w, a.tokens, a.per_stream)
                rows.append(row)
                print(
                    json.dumps(
                        {k: row[k] for k in ("tool", "concurrency", "workers", "chunks_per_s")}
                    ),
                    "delay",
                    row["chunk_delay_ms"],
                    "lag",
                    row["loop_lag_ms"],
                    flush=True,
                )
            if a.vllm_bench:
                tdir = os.path.join(tmp, f"vllm_{conc}")
                srv = FakeServer(a.server_procs, a.ttft, a.itl, tdir)
                row = run_vllm_bench(
                    srv, conc, a.tokens, a.per_stream, a.vllm_bench, a.tokenizer, tmp
                )
                rows.append(row)
                print(json.dumps(row)[:600], flush=True)
    cpu_model = (
        next(
            (
                ln.split(":", 1)[1].strip()
                for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")
            ),
            platform.processor(),
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    meta = {
        "cpu_model": cpu_model,
        "cpus_allocated": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "server_procs": a.server_procs,
        "ttft_s": a.ttft,
        "itl_s": a.itl,
        "tokens": a.tokens,
        "per_stream": a.per_stream,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
