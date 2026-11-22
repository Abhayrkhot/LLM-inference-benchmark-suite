"""Server-side telemetry sampled during a run: engine Prometheus metrics and GPU power/utilization.

Client latencies say *what* happened; the engine's queue depth, KV-cache occupancy and preemption
counters say *why* (for example, TTFT jumps when the KV cache fills and requests start queueing).
Both vLLM and SGLang (with ``--enable-metrics``) expose ``/metrics``.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import math
import re
import shutil
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from llmbench.client import now_ns

__all__ = ["GPUMonitor", "MetricsMonitor", "canonical_gauges", "parse_prometheus"]

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")

# Engine metric names mapped to one vocabulary so reports can compare engines.
CANONICAL: dict[str, tuple[str, ...]] = {
    "running": ("vllm:num_requests_running", "sglang:num_running_reqs"),
    "waiting": ("vllm:num_requests_waiting", "sglang:num_queue_reqs"),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"),
    "preemptions_total": ("vllm:num_preemptions_total", "sglang:num_retracted_reqs_total"),
    "prefix_queries_total": ("vllm:prefix_cache_queries_total",),
    "prefix_hits_total": ("vllm:prefix_cache_hits_total",),
    "cache_hit_rate": ("sglang:cache_hit_rate",),
    "prompt_tokens_total": ("vllm:prompt_tokens_total", "sglang:prompt_tokens_total"),
    "generation_tokens_total": ("vllm:generation_tokens_total", "sglang:generation_tokens_total"),
}


# Ratios (0..1) are averaged across label sets (e.g. vLLM data-parallel engines); everything
# else (counts, counters) is summed.
RATIO_METRICS = frozenset(CANONICAL["kv_usage"] + CANONICAL["cache_hit_rate"])


def parse_prometheus(text: str) -> dict[str, float]:
    """One value per metric name across label sets: ratios averaged, counts summed."""
    out: dict[str, float] = {}
    count: dict[str, int] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if math.isnan(v):
            continue
        name = m.group(1)
        out[name] = out.get(name, 0.0) + v
        count[name] = count.get(name, 0) + 1
    for name in RATIO_METRICS & out.keys():
        out[name] /= count[name]
    return out


def canonical_gauges(sample: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, names in CANONICAL.items():
        for n in names:
            if n in sample:
                out[key] = sample[n]
                break
    return out


@dataclass
class MetricsMonitor:
    """Polls ``{base_url}/metrics`` every ``interval_s`` until stopped."""

    base_url: str
    interval_s: float = 1.0
    samples: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    errors: int = 0
    _task: asyncio.Task[None] | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def _run(self) -> None:
        url = self.base_url.rstrip("/") + "/metrics"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            while not self._stop.is_set():
                t = now_ns()
                try:
                    async with s.get(url) as r:
                        if r.status == 200:
                            sample = canonical_gauges(parse_prometheus(await r.text()))
                            self.samples.append((t, sample))
                        else:
                            self.errors += 1
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    self.errors += 1
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), self.interval_s)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0, "errors": self.errors}
        keys = sorted({k for _, s in self.samples for k in s})
        series = {k: [s.get(k, math.nan) for _, s in self.samples] for k in keys}
        out: dict[str, Any] = {
            "samples": len(self.samples),
            "errors": self.errors,
            "t_ns": [t for t, _ in self.samples],
            "series": series,
        }
        stats: dict[str, float] = {}
        for k, v in series.items():
            vals = [x for x in v if not math.isnan(x)]
            if not vals:
                continue
            if k.endswith("_total"):
                stats[f"{k[:-6]}_delta"] = vals[-1] - vals[0]
            else:
                stats[f"{k}_mean"] = sum(vals) / len(vals)
                stats[f"{k}_max"] = max(vals)
        q, h = stats.get("prefix_queries_delta"), stats.get("prefix_hits_delta")
        if q:
            stats["prefix_hit_rate"] = (h or 0.0) / q
        out["stats"] = stats
        return out


@dataclass
class GPUMonitor:
    """Samples ``nvidia-smi`` (power, utilization, memory) every ``interval_ms``. Energy is the
    trapezoidal integral of power over the samples; only meaningful when the client runs on the
    same node as the GPU."""

    interval_ms: int = 250
    samples: list[tuple[int, list[float]]] = field(default_factory=list)
    _proc: asyncio.subprocess.Process | None = None
    _task: asyncio.Task[None] | None = None
    _ngpu: int = 1
    _first_index: int = 0

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    async def start(self) -> None:
        probe = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await probe.communicate()
        indices = [int(x) for x in out.split()] or [0]
        self._ngpu = len(indices)
        self._first_index = min(indices)
        self._proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            f"-lms={self.interval_ms}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        # nvidia-smi prints one line per GPU per tick, starting at the lowest index. A tick
        # counts only if every GPU reported a number (a GPU can report "[N/A]").
        tick: list[list[float]] = []
        async for raw in self._proc.stdout:
            parts = raw.decode().split(",")
            try:
                idx = int(parts[0])
                vals = [float(x) for x in parts[1:4]]
            except (ValueError, IndexError):
                tick = []  # drop the incomplete tick
                continue
            if idx == self._first_index and tick:
                self._commit(tick)
                tick = []
            tick.append(vals)
        if tick:
            self._commit(tick)

    def _commit(self, tick: list[list[float]]) -> None:
        if len(tick) != self._ngpu:
            return
        n = len(tick)
        power = sum(v[0] for v in tick)
        util = sum(v[1] for v in tick) / n
        mem = sum(v[2] for v in tick)
        self.samples.append((now_ns(), [power, util, mem]))

    async def stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await self._proc.wait()
        if self._task is not None:
            with contextlib.suppress(Exception):
                await self._task

    def summary(self, t_start_ns: int, t_end_ns: int) -> dict[str, Any]:
        s = [(t, v) for t, v in self.samples if t_start_ns <= t <= t_end_ns]
        if len(s) < 2:
            return {"samples": len(s)}
        sampled = 0.0
        for (t0, v0), (t1, v1) in itertools.pairwise(s):
            sampled += (v0[0] + v1[0]) / 2 * (t1 - t0) / 1e9
        span = (s[-1][0] - s[0][0]) / 1e9
        power = sampled / span if span else 0.0
        # Energy over the whole window (the window the token counts cover), at the
        # time-weighted mean power of the samples inside it.
        energy = power * (t_end_ns - t_start_ns) / 1e9
        return {
            "samples": len(s),
            "power_w_mean": power,
            "util_pct_mean": sum(v[1] for _, v in s) / len(s),
            "mem_used_mib_max": max(v[2] for _, v in s),
            "energy_j": energy,
            "t_ns": [t for t, _ in s],
            "power_w": [v[0] for _, v in s],
            "util_pct": [v[1] for _, v in s],
        }
