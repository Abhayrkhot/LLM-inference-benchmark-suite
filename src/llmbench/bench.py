"""One benchmark run end to end: build the workload, probe the server, run the load with
telemetry, and summarize."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np

from llmbench.arrivals import arrival_offsets
from llmbench.backends import SamplingOptions
from llmbench.datasets import (
    LengthDist,
    TokenSpace,
    Workload,
    azure_workload,
    mooncake_workload,
    prefix_workload,
    random_workload,
    sharegpt_workload,
)
from llmbench.metrics import SLO, summarize
from llmbench.monitor import GPUMonitor, MetricsMonitor, canonical_gauges, parse_prometheus
from llmbench.results import RunResult, environment, probe_server
from llmbench.runner import ClientConfig, LoadSpec, run_load
from llmbench.tokenizer import Tokenizer, load_tokenizer
from llmbench.types import Request, RequestRecord

__all__ = ["BenchConfig", "LoadConfig", "WorkloadConfig", "build_workload", "run_benchmark"]

log = logging.getLogger("llmbench")


@dataclass
class WorkloadConfig:
    dataset: str = "random"  # random | prefix | sharegpt | azure | mooncake
    path: str | None = None
    num_requests: int = 1000
    input_len: str = "1024"
    output_len: str = "128"
    num_prefixes: int = 8
    prefix_len: int = 2048
    suffix_len: str = "256"
    tokenizer: str | None = None
    vocab_size: int | None = None
    max_input_len: int | None = None
    max_output_len: int | None = None
    sharegpt_output_len: int | None = None
    trace_start_s: float = 0.0
    seed: int = 0


@dataclass
class LoadConfig:
    arrival: str = "poisson"  # poisson | gamma | constant | trace
    rate: float = math.inf
    cv: float = 1.0
    concurrency: int | None = None  # set => closed loop
    max_concurrency: int | None = None
    time_scale: float = 1.0
    duration_s: float | None = None
    workers: int = 1
    warmup_requests: int = 0

    def describe(self) -> str:
        if self.concurrency:
            return f"closed loop, {self.concurrency} users"
        if self.arrival == "trace":
            return f"trace replay x{self.time_scale:g}"
        rate = "inf" if math.isinf(self.rate) else f"{self.rate:g}"
        return f"{self.arrival} {rate} req/s"


@dataclass
class BenchConfig:
    base_url: str
    model: str
    backend: str = "completions"
    api_key: str | None = None
    timeout_s: float | None = 600.0
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    slo: str | None = None
    engine_metrics: bool = True
    gpu_monitor: bool = False
    keep_text: bool = True
    count_tokens_client_side: bool = True
    name: str = "run"
    tags: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("api_key", None)
        return d


@functools.lru_cache(maxsize=4)
def _tokenizer(name: str) -> Tokenizer:
    return load_tokenizer(name)


def _space(w: WorkloadConfig) -> TokenSpace:
    if w.tokenizer:
        return TokenSpace.from_tokenizer(_tokenizer(w.tokenizer))
    if w.vocab_size:
        return TokenSpace(w.vocab_size)
    raise ValueError(f"dataset {w.dataset!r} needs --tokenizer (or --vocab-size)")


def build_workload(w: WorkloadConfig) -> Workload:
    if w.dataset == "random":
        return random_workload(
            w.num_requests,
            _space(w),
            LengthDist.parse(w.input_len),
            LengthDist.parse(w.output_len),
            seed=w.seed,
        )
    if w.dataset == "prefix":
        return prefix_workload(
            w.num_requests,
            _space(w),
            num_prefixes=w.num_prefixes,
            prefix_len=w.prefix_len,
            suffix_len=LengthDist.parse(w.suffix_len),
            output_len=LengthDist.parse(w.output_len),
            seed=w.seed,
        )
    if not w.path:
        raise ValueError(f"dataset {w.dataset!r} needs --dataset-path")
    if w.dataset == "sharegpt":
        if not w.tokenizer:
            raise ValueError("sharegpt needs --tokenizer to measure prompt and reply lengths")
        return sharegpt_workload(
            w.path,
            _tokenizer(w.tokenizer),
            w.num_requests,
            seed=w.seed,
            output_len=w.sharegpt_output_len,
        )
    if w.dataset == "azure":
        return azure_workload(
            w.path,
            _space(w),
            w.num_requests,
            start_s=w.trace_start_s,
            seed=w.seed,
            max_input_len=w.max_input_len,
            max_output_len=w.max_output_len,
        )
    if w.dataset == "mooncake":
        return mooncake_workload(
            w.path,
            _space(w),
            w.num_requests,
            seed=w.seed,
            max_input_len=w.max_input_len,
            max_output_len=w.max_output_len,
        )
    raise ValueError(f"unknown dataset {w.dataset!r}")


def load_spec(load: LoadConfig, wl: Workload, seed: int) -> LoadSpec:
    if load.concurrency:
        return LoadSpec.closed(load.concurrency, max_duration_s=load.duration_s)
    offsets = arrival_offsets(
        load.arrival,
        len(wl.requests),
        rate=load.rate,
        cv=load.cv,
        seed=seed,
        trace=wl.arrivals_s,
        time_scale=load.time_scale,
    )
    return LoadSpec.open(offsets, load.max_concurrency, load.duration_s)


def _warmup_requests(cfg: BenchConfig, wl: Workload) -> list[Request]:
    """Fresh prompts (never the measured ones, which would pre-fill the prefix cache)."""
    n = cfg.load.warmup_requests
    rng = np.random.default_rng(cfg.workload.seed + 10_000_019)
    plen = int(np.median([r.prompt_tokens for r in wl.requests]))
    olen = min(int(np.median([r.max_tokens for r in wl.requests])), 64)
    try:
        space = _space(cfg.workload)
    except ValueError:
        return [
            Request(
                id=-1 - i,
                prompt=f"Warm-up {i}: write a short poem about load testing.",
                prompt_tokens=16,
                max_tokens=olen,
            )
            for i in range(n)
        ]
    if cfg.backend == "chat":
        tok = _tokenizer(cfg.workload.tokenizer) if cfg.workload.tokenizer else None
        return [
            Request(
                id=-1 - i,
                prompt=tok.decode(space.sample(rng, plen)) if tok else "Hello",
                prompt_tokens=plen,
                max_tokens=olen,
            )
            for i in range(n)
        ]
    return [
        Request(id=-1 - i, prompt=space.sample(rng, plen), prompt_tokens=plen, max_tokens=olen)
        for i in range(n)
    ]


async def wait_idle(base_url: str, timeout_s: float = 120.0) -> bool:
    """Wait until the engine reports no running or waiting requests (requests the client gave up
    on may still be running server-side). Returns False if the server has no usable metrics."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
        while loop.time() < deadline:
            try:
                async with s.get(base_url.rstrip("/") + "/metrics") as r:
                    if r.status != 200:
                        return False
                    g = canonical_gauges(parse_prometheus(await r.text()))
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                return False
            if "running" not in g:
                return False
            if g.get("running", 0) == 0 and g.get("waiting", 0) == 0:
                return True
            await asyncio.sleep(1.0)
    return False


async def run_benchmark_async(cfg: BenchConfig, wl: Workload | None = None) -> RunResult:
    wl = wl if wl is not None else build_workload(cfg.workload)
    client = ClientConfig(
        base_url=cfg.base_url,
        model=cfg.model,
        backend=cfg.backend,
        sampling=cfg.sampling,
        timeout_s=cfg.timeout_s,
        api_key=cfg.api_key,
        keep_text=cfg.keep_text,
    )
    server = await probe_server(cfg.base_url)
    spec = load_spec(cfg.load, wl, cfg.workload.seed)

    records: list[RequestRecord] = []
    if cfg.load.warmup_requests:
        warm = _warmup_requests(cfg, wl)
        outs = await asyncio.to_thread(run_load, warm, LoadSpec.closed(min(len(warm), 8)), client)
        for o in outs:
            for r in o.records:
                r.warmup = True
            records.extend(o.records)
        failed = [r for o in outs for r in o.records if not r.ok]
        if failed:
            raise RuntimeError(f"warm-up request failed: {failed[0].error_detail}")

    mon = MetricsMonitor(cfg.base_url) if cfg.engine_metrics else None
    gpu = GPUMonitor() if cfg.gpu_monitor and GPUMonitor.available() else None
    if mon:
        mon.start()
    if gpu:
        await gpu.start()
    try:
        outputs = await asyncio.to_thread(
            run_load, wl.requests, spec, client, workers=cfg.load.workers
        )
    finally:
        if mon:
            await mon.stop()
        if gpu:
            await gpu.stop()

    lag: list[int] = []
    for o in outputs:
        records.extend(o.records)
        lag.extend(o.loop_lag_ns)
    records.sort(key=lambda r: (r.warmup is False, r.scheduled_ns))

    if cfg.count_tokens_client_side and cfg.workload.tokenizer and cfg.keep_text:
        tok = _tokenizer(cfg.workload.tokenizer)
        for r in records:
            if r.ok and r.text:
                r.client_completion_tokens = tok.count(r.text)

    summary = summarize(records, slo=SLO.parse(cfg.slo), loop_lag_ns=lag)
    measured = [r for r in records if not r.warmup]
    gpu_summary: dict[str, Any] = {}
    if gpu and measured:
        gpu_summary = gpu.summary(
            min(r.scheduled_ns for r in measured), max(r.end_ns for r in measured)
        )
        out_tok = summary.get("throughput", {}).get("output_tokens_per_s", 0) * summary.get(
            "duration_s", 0
        )
        if gpu_summary.get("energy_j") and out_tok:
            gpu_summary["joules_per_output_token"] = gpu_summary["energy_j"] / out_tok
    return RunResult(
        name=cfg.name,
        config=cfg.to_json(),
        summary=summary,
        environment=environment(),
        server=server,
        workload={"name": wl.name, **wl.description, "stats": wl.stats()},
        engine_metrics=mon.summary() if mon else {},
        gpu=gpu_summary,
        records=records,
    )


def run_benchmark(cfg: BenchConfig, wl: Workload | None = None) -> RunResult:
    return asyncio.run(run_benchmark_async(cfg, wl))
