"""Drives a workload against a server: open- or closed-loop, in one process or several.

A Python client parsing thousands of SSE chunks per second can fall behind the server; when it
does, chunks sit in socket buffers and the client reports inflated latencies that belong to it,
not to the engine. Two defences are built in: every run records the client's own event-loop lag
and dispatch lag, so a saturated client is visible in the results, and ``workers > 1`` shards the
load over processes that share one start time.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import queue
import sys
from array import array
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from llmbench.backends import SamplingOptions, get_backend
from llmbench.client import dumps, make_session, now_ns, send_request
from llmbench.types import Request, RequestRecord

__all__ = ["ClientConfig", "LoadSpec", "WorkerOutput", "run_async", "run_load", "run_load_async"]

T = TypeVar("T")


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    model: str
    backend: str = "completions"
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    timeout_s: float | None = 600.0
    api_key: str | None = None
    keep_text: bool = True

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + get_backend(self.backend).path


@dataclass(frozen=True)
class LoadSpec:
    """How requests are released.

    ``open``: request *i* is scheduled at ``offsets_s[i]`` after the start, regardless of how the
    server is doing; ``max_concurrency`` optionally caps in-flight requests, and time spent waiting
    for a slot counts toward latency because the request had already arrived.

    ``closed``: ``concurrency`` users each send a request, wait for the full answer, pause for
    ``think_time_s``, and repeat until the workload is used up.
    """

    mode: Literal["open", "closed"]
    offsets_s: tuple[float, ...] = ()
    max_concurrency: int | None = None
    concurrency: int = 1
    think_time_s: float = 0.0
    max_duration_s: float | None = None

    @staticmethod
    def open(
        offsets_s: Sequence[float] | npt.NDArray[np.float64],
        max_concurrency: int | None = None,
        max_duration_s: float | None = None,
    ) -> LoadSpec:
        return LoadSpec(
            "open",
            offsets_s=tuple(float(x) for x in offsets_s),
            max_concurrency=max_concurrency,
            max_duration_s=max_duration_s,
        )

    @staticmethod
    def closed(
        concurrency: int, think_time_s: float = 0.0, max_duration_s: float | None = None
    ) -> LoadSpec:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        return LoadSpec(
            "closed",
            concurrency=concurrency,
            think_time_s=think_time_s,
            max_duration_s=max_duration_s,
        )


@dataclass
class WorkerOutput:
    worker: int
    records: list[RequestRecord]
    # Overshoot of a 5 ms sleep, sampled throughout the run: how late the event loop runs.
    loop_lag_ns: array[int]


async def _lag_monitor(stop: asyncio.Event, out: array[int], interval_s: float = 0.005) -> None:
    interval_ns = int(interval_s * 1e9)
    while not stop.is_set():
        t = now_ns()
        await asyncio.sleep(interval_s)
        out.append(now_ns() - t - interval_ns)


async def run_load_async(
    requests: Sequence[Request],
    load: LoadSpec,
    cfg: ClientConfig,
    *,
    start: Callable[[], Awaitable[int]] | None = None,
    start_delay_s: float = 0.25,
    worker: int = 0,
    warmup: bool = False,
) -> WorkerOutput:
    """Run ``requests`` in this process's event loop and return one record per sent request.

    Bodies are encoded and the HTTP session is opened first; only then is the start time fixed,
    by ``start()`` (which multi-process runs use as a barrier) or as ``now + start_delay_s``, so
    setup work never delays the first scheduled arrivals."""
    backend = get_backend(cfg.backend)
    url = cfg.url
    # Serialize every body before the clock starts so JSON encoding is off the hot path.
    payloads = [dumps(backend.body(r, cfg.model, cfg.sampling)) for r in requests]
    records: list[RequestRecord] = []
    lag: array[int] = array("q")
    stop = asyncio.Event()

    async with make_session(api_key=cfg.api_key) as session:

        async def fire(i: int, scheduled_ns: int, sem: asyncio.Semaphore | None) -> None:
            r = requests[i]
            rec = RequestRecord(
                id=r.id,
                prompt_tokens=r.prompt_tokens,
                max_tokens=r.max_tokens,
                scheduled_ns=scheduled_ns,
                worker=worker,
                warmup=warmup,
                tags=dict(r.tags),
            )
            records.append(rec)
            if sem is None:
                await send_request(
                    session,
                    url,
                    payloads[i],
                    backend,
                    rec,
                    timeout_s=cfg.timeout_s,
                    keep_text=cfg.keep_text,
                )
            else:
                async with sem:
                    await send_request(
                        session,
                        url,
                        payloads[i],
                        backend,
                        rec,
                        timeout_s=cfg.timeout_s,
                        keep_text=cfg.keep_text,
                    )

        t0 = await start() if start is not None else now_ns() + int(start_delay_s * 1e9)
        monitor = asyncio.create_task(_lag_monitor(stop, lag))
        deadline = t0 + int(load.max_duration_s * 1e9) if load.max_duration_s else None
        try:
            if load.mode == "open":
                await _open_loop(fire, load, t0, deadline, len(requests))
            else:
                await _closed_loop(fire, load, t0, deadline, len(requests))
        finally:
            stop.set()
            await monitor
    return WorkerOutput(worker=worker, records=records, loop_lag_ns=lag)


async def _open_loop(fire: Any, load: LoadSpec, start: int, deadline: int | None, n: int) -> None:
    if len(load.offsets_s) != n:
        raise ValueError(f"{len(load.offsets_s)} arrival offsets for {n} requests")
    sem = asyncio.Semaphore(load.max_concurrency) if load.max_concurrency else None
    order = sorted(range(n), key=lambda i: load.offsets_s[i])
    tasks: list[asyncio.Task[None]] = []
    for i in order:
        target = start + int(load.offsets_s[i] * 1e9)
        if deadline is not None and target > deadline:
            break
        delay = target - now_ns()
        if delay > 0:
            await asyncio.sleep(delay / 1e9)
        tasks.append(asyncio.create_task(fire(i, target, sem)))
    if tasks:
        await asyncio.gather(*tasks)


async def _closed_loop(fire: Any, load: LoadSpec, start: int, deadline: int | None, n: int) -> None:
    delay = start - now_ns()
    if delay > 0:
        await asyncio.sleep(delay / 1e9)
    pending = deque(range(n))

    async def user() -> None:
        while pending:
            if deadline is not None and now_ns() > deadline:
                return
            i = pending.popleft()
            await fire(i, now_ns(), None)
            if load.think_time_s > 0:
                await asyncio.sleep(load.think_time_s)

    await asyncio.gather(*(user() for _ in range(min(load.concurrency, n))))


def raise_fd_limit() -> None:
    """One socket per in-flight request: lift the soft open-file limit to the hard limit so a
    burst of thousands of requests does not fail with EMFILE (which would be recorded as a
    connection error and blamed on the server)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard if hard != resource.RLIM_INFINITY else 1 << 20
    if soft == resource.RLIM_INFINITY or soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):  # macOS caps below RLIM_INFINITY
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target, 10240), hard))


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` on uvloop when it is installed (2-4x less per-chunk overhead)."""
    try:
        import uvloop
    except ImportError:
        return asyncio.run(coro)
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            return runner.run(coro)
    return uvloop.run(coro)  # pragma: no cover


# ---------------------------------------------------------------- multi-process ----------------


def _shard(
    requests: Sequence[Request], load: LoadSpec, workers: int
) -> list[tuple[list[Request], LoadSpec]]:
    """Split the load over workers.

    Open loop: requests round-robin, each keeping its scheduled offset; a concurrency cap is split
    per worker. Closed loop: user ``u`` lives on worker ``u % W`` and request ``i`` belongs to
    user ``i % C``, so each worker gets work in proportion to its users and the total number of
    concurrent users stays ``C`` until the workload runs out."""
    shards: list[tuple[list[Request], LoadSpec]] = []
    for w in range(workers):
        if load.mode == "open":
            idx = list(range(w, len(requests), workers))
        else:
            c_all = load.concurrency
            idx = [i for i in range(len(requests)) if (i % c_all) % workers == w]
        reqs = [requests[i] for i in idx]
        if load.mode == "open":
            mc = load.max_concurrency
            wl = LoadSpec(
                "open",
                offsets_s=tuple(load.offsets_s[i] for i in idx),
                max_concurrency=(mc // workers + (w < mc % workers)) if mc else None,
                max_duration_s=load.max_duration_s,
            )
        else:
            c = sum(1 for u in range(load.concurrency) if u % workers == w)
            wl = LoadSpec(
                "closed",
                concurrency=c,
                think_time_s=load.think_time_s,
                max_duration_s=load.max_duration_s,
            )
        shards.append((reqs, wl))
    return shards


def _worker_main(
    w: int,
    reqs: list[Request],
    load: LoadSpec,
    cfg: ClientConfig,
    ready_q: Any,
    go: Any,
    t0: Any,
    result_q: Any,
) -> None:
    async def barrier() -> int:
        ready_q.put(w)
        await asyncio.to_thread(go.wait)
        return int(t0.value)

    async def main() -> WorkerOutput:
        return await run_load_async(reqs, load, cfg, start=barrier, worker=w)

    try:
        raise_fd_limit()
        out = run_async(main())
        result_q.put((w, [r.to_json() for r in out.records], list(out.loop_lag_ns), None))
    except BaseException as exc:  # report, never hang the parent
        result_q.put((w, [], [], repr(exc)))


def run_load(
    requests: Sequence[Request],
    load: LoadSpec,
    cfg: ClientConfig,
    *,
    workers: int = 1,
    start_delay_s: float = 0.25,
) -> list[WorkerOutput]:
    """Blocking entry point. ``workers > 1`` runs one event loop per process."""
    if workers <= 1:
        raise_fd_limit()
        return [run_async(run_load_async(requests, load, cfg, start_delay_s=start_delay_s))]
    if load.mode == "closed" and workers > load.concurrency:
        raise ValueError(f"{workers} workers for {load.concurrency} closed-loop users")
    if load.mode == "open" and load.max_concurrency and load.max_concurrency < workers:
        raise ValueError("--max-concurrency must be at least --workers (it is split per worker)")

    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    go = ctx.Event()
    t0 = ctx.Value("q", 0)
    procs = []
    for w, (reqs, wl) in enumerate(_shard(requests, load, workers)):
        p = ctx.Process(
            target=_worker_main,
            args=(w, reqs, wl, cfg, ready_q, go, t0, result_q),
            daemon=True,
        )
        p.start()
        procs.append(p)
    try:
        for _ in procs:
            ready_q.get(timeout=120)
        t0.value = now_ns() + int(start_delay_s * 1e9)
        go.set()
        outputs: list[WorkerOutput] = []
        errors: list[str] = []
        delivered: set[int] = set()
        while len(delivered) < len(procs):
            try:
                w, recs, lag, err = result_q.get(timeout=5)
            except queue.Empty:
                # A worker that exited without delivering will never deliver (give the queue's
                # feeder thread one more poll to flush before declaring it lost).
                gone = [i for i, p in enumerate(procs) if i not in delivered and not p.is_alive()]
                if gone:
                    try:
                        w, recs, lag, err = result_q.get(timeout=5)
                    except queue.Empty:
                        code = procs[gone[0]].exitcode
                        raise RuntimeError(
                            f"load worker {gone[0]} exited ({code}) without results"
                        ) from None
                else:
                    continue
            delivered.add(w)
            if err:
                errors.append(f"worker {w}: {err}")
            else:
                records = [RequestRecord.from_json(d) for d in recs]
                outputs.append(WorkerOutput(w, records, array("q", lag)))
        if errors:
            raise RuntimeError("; ".join(errors))
        return sorted(outputs, key=lambda o: o.worker)
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
