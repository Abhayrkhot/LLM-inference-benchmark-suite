"""An OpenAI-compatible fake inference server for tests, CI and validating the client itself.

Two engine models:

* ``fixed``: every request gets its first token after ``ttft`` (+ ``prefill_per_token`` per
  prompt token) and then one token every ``itl``, on an absolute schedule, independent of load.
  Because the true timing is known, comparing it with what the client measures gives the
  client's own measurement error.
* ``batch``: a continuous-batching engine (FCFS admission, at most ``max_num_seqs`` running, a
  per-step token budget with chunked prefill, a KV-token budget). Step time grows with the batch,
  so it saturates and queues the way a real engine does, which exercises capacity search and
  shows coordinated omission without a GPU.

With ``trace_dir`` set, the server records each request's arrival and the time it wrote each
token chunk, and dumps one JSON file per process on shutdown, so client-side receive times can
be compared with server-side send times on the same machine (``perf_counter_ns`` is
system-wide).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import signal
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from llmbench import __version__

__all__ = ["FakeEngineConfig", "make_app", "serve"]

now_ns = time.perf_counter_ns


@dataclass
class FakeEngineConfig:
    mode: str = "fixed"  # fixed | batch
    ttft_s: float = 0.05
    itl_s: float = 0.01
    prefill_per_token_s: float = 0.0
    # batch mode
    max_num_seqs: int = 64
    max_batched_tokens: int = 2048
    kv_tokens: int = 1 << 20
    step_base_s: float = 0.008
    step_per_seq_s: float = 0.0002
    step_per_prefill_token_s: float = 0.00002
    # faults
    error_rate: float = 0.0  # HTTP 500 before streaming
    drop_rate: float = 0.0  # connection closed mid-stream
    error_event_rate: float = 0.0  # {"error": ...} event mid-stream
    seed: int = 0
    trace_dir: str | None = None
    model: str = "fake-model"


@dataclass
class _Req:
    rid: str
    prompt_tokens: int
    max_tokens: int
    queue: asyncio.Queue[int | None] = field(default_factory=asyncio.Queue)
    prefilled: int = 0
    generated: int = 0


class BatchEngine:
    """Discrete-step continuous batching; one asyncio task drives all requests."""

    def __init__(self, cfg: FakeEngineConfig) -> None:
        self.cfg = cfg
        self.waiting: deque[_Req] = deque()
        self.running: list[_Req] = []
        self.kv_used = 0
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def submit(self, r: _Req) -> None:
        self.waiting.append(r)
        self._wake.set()

    async def _loop(self) -> None:
        c = self.cfg
        while True:
            if not self.waiting and not self.running:
                self._wake.clear()
                await self._wake.wait()
            while (
                self.waiting
                and len(self.running) < c.max_num_seqs
                and self.kv_used + self.waiting[0].prompt_tokens + self.waiting[0].max_tokens
                <= c.kv_tokens
            ):
                r = self.waiting.popleft()
                self.kv_used += r.prompt_tokens + r.max_tokens
                self.running.append(r)
            # Decode tokens first (one per decoding sequence), then fill the budget with prefill.
            decoding = [r for r in self.running if r.prefilled >= r.prompt_tokens]
            budget = c.max_batched_tokens - len(decoding)
            prefill_tokens = 0
            finishing_prefill: list[_Req] = []
            for r in self.running:
                if r.prefilled < r.prompt_tokens and budget > 0:
                    take = min(budget, r.prompt_tokens - r.prefilled)
                    r.prefilled += take
                    budget -= take
                    prefill_tokens += take
                    if r.prefilled >= r.prompt_tokens:
                        finishing_prefill.append(r)
            step = (
                c.step_base_s
                + c.step_per_seq_s * len(self.running)
                + c.step_per_prefill_token_s * prefill_tokens
            )
            await asyncio.sleep(step)
            self.prompt_tokens_total += prefill_tokens
            for r in decoding + finishing_prefill:
                r.generated += 1
                self.generation_tokens_total += 1
                r.queue.put_nowait(r.generated)
                if r.generated >= r.max_tokens:
                    r.queue.put_nowait(None)
            done = [r for r in self.running if r.generated >= r.max_tokens]
            for r in done:
                self.running.remove(r)
                self.kv_used -= r.prompt_tokens + r.max_tokens

    def abort(self, r: _Req) -> None:
        if r in self.running:
            self.running.remove(r)
            self.kv_used -= r.prompt_tokens + r.max_tokens
        elif r in self.waiting:
            self.waiting.remove(r)


def _dump_trace(trace_dir: str, trace: dict[str, list[int]]) -> None:
    """One file per process: ``{request_id: [arrival_ns, send_ns_0, send_ns_1, ...]}``."""
    os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.join(trace_dir, f"trace-{os.getpid()}.json"), "w") as f:
        json.dump(trace, f)


def _prompt_tokens(body: dict[str, Any]) -> int:
    p = body.get("prompt")
    if p is None:
        msgs = body.get("messages") or []
        p = " ".join(str(m.get("content", "")) for m in msgs)
    if isinstance(p, list):
        return len(p)
    return max(len(str(p).split()), 1)


def make_app(cfg: FakeEngineConfig) -> web.Application:
    app = web.Application()
    rng = random.Random(cfg.seed + os.getpid())
    engine = BatchEngine(cfg) if cfg.mode == "batch" else None
    trace: dict[str, list[int]] = {}
    state = {"running": 0, "prompt_tokens": 0, "generation_tokens": 0}

    async def on_startup(_: web.Application) -> None:
        if engine:
            engine.start()

    async def on_shutdown(_: web.Application) -> None:
        if engine:
            await engine.stop()
        if cfg.trace_dir and trace:
            await asyncio.to_thread(_dump_trace, cfg.trace_dir, trace)

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    async def generate(request: web.Request) -> web.StreamResponse:
        t_arrival = now_ns()
        chat = request.path.endswith("/chat/completions")
        body = await request.json()
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
        n_prompt = _prompt_tokens(body)
        rid = request.headers.get("X-Request-Id") or f"r{rng.getrandbits(48):x}"
        if cfg.error_rate and rng.random() < cfg.error_rate:
            return web.json_response({"error": {"message": "injected failure"}}, status=500)
        if engine is not None and n_prompt + max_tokens > cfg.kv_tokens:
            msg = f"prompt + max_tokens = {n_prompt + max_tokens} exceeds kv_tokens"
            return web.json_response({"error": {"message": msg}}, status=400)
        if not body.get("stream"):
            return web.json_response({"error": {"message": "fake server only streams"}}, status=400)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        resp = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await resp.prepare(request)
        times: list[int] | None = None
        if cfg.trace_dir:
            times = trace[rid] = [t_arrival]
        drop_at = rng.randint(1, max(max_tokens - 1, 1)) if rng.random() < cfg.drop_rate else -1
        err_at = (
            rng.randint(1, max(max_tokens - 1, 1)) if rng.random() < cfg.error_event_rate else -1
        )
        created = int(time.time())

        # Every token chunk is identical, so it is encoded once per request.
        if chat:
            delta: dict[str, Any] = {"content": " tok"}
            obj: dict[str, Any] = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": cfg.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        else:
            obj = {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": cfg.model,
                "choices": [{"index": 0, "text": " tok", "finish_reason": None}],
            }
        token_chunk = b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"

        state["running"] += 1
        state["prompt_tokens"] += n_prompt
        r: _Req | None = None
        try:
            if chat:  # real servers send the role first; it is not a token
                role = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": cfg.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                await resp.write(b"data: " + json.dumps(role).encode() + b"\n\n")
            if engine is None:
                ttft = cfg.ttft_s + cfg.prefill_per_token_s * n_prompt
                t_first = t_arrival + int(ttft * 1e9)
                itl = int(cfg.itl_s * 1e9)
                for i in range(max_tokens):
                    delay = (t_first + i * itl - now_ns()) / 1e9
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if i == drop_at:
                        assert request.transport is not None
                        request.transport.close()
                        return resp
                    if i == err_at:
                        await resp.write(b'data: {"error":{"message":"injected","code":500}}\n\n')
                        return resp
                    if times is not None:
                        times.append(now_ns())
                    await resp.write(token_chunk)
                    state["generation_tokens"] += 1
            else:
                r = _Req(rid, n_prompt, max_tokens)
                engine.submit(r)
                i = 0
                while (item := await r.queue.get()) is not None:
                    del item
                    if i == drop_at:
                        assert request.transport is not None
                        request.transport.close()
                        return resp
                    if i == err_at:
                        await resp.write(b'data: {"error":{"message":"injected","code":500}}\n\n')
                        return resp
                    if times is not None:
                        times.append(now_ns())
                    await resp.write(token_chunk)
                    i += 1
                r = None
            if include_usage:
                usage: dict[str, Any] = {
                    "id": rid,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": created,
                    "model": cfg.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": max_tokens,
                        "total_tokens": n_prompt + max_tokens,
                    },
                }
                await resp.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            state["running"] -= 1
            if r is not None and engine is not None:
                engine.abort(r)

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": cfg.model, "object": "model", "max_model_len": 1 << 20}],
            }
        )

    async def version(_: web.Request) -> web.Response:
        return web.json_response({"version": f"fake-{__version__}"})

    async def health(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def metrics(_: web.Request) -> web.Response:
        if engine:
            running, waiting = len(engine.running), len(engine.waiting)
            kv = engine.kv_used / cfg.kv_tokens
            ptok, gtok = engine.prompt_tokens_total, engine.generation_tokens_total
        else:
            running, waiting, kv = state["running"], 0, 0.0
            ptok, gtok = state["prompt_tokens"], state["generation_tokens"]
        m = cfg.model
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            f'vllm:num_requests_running{{model_name="{m}"}} {running}',
            f'vllm:num_requests_waiting{{model_name="{m}"}} {waiting}',
            f'vllm:kv_cache_usage_perc{{model_name="{m}"}} {kv}',
            f'vllm:num_preemptions_total{{model_name="{m}"}} 0',
            f'vllm:prompt_tokens_total{{model_name="{m}"}} {ptok}',
            f'vllm:generation_tokens_total{{model_name="{m}"}} {gtok}',
        ]
        return web.Response(text="\n".join(lines) + "\n")

    app.router.add_post("/v1/completions", generate)
    app.router.add_post("/v1/chat/completions", generate)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/version", version)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    return app


async def _serve_async(cfg: FakeEngineConfig, host: str, port: int, reuse_port: bool) -> None:
    app = make_app(cfg)
    runner = web.AppRunner(app, access_log=None, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, host, port, reuse_port=reuse_port, backlog=4096)
    await site.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await runner.cleanup()


def _serve_one(cfg: FakeEngineConfig, host: str, port: int, reuse_port: bool) -> None:
    from llmbench.runner import run_async

    run_async(_serve_async(cfg, host, port, reuse_port))


def serve(
    cfg: FakeEngineConfig, host: str = "127.0.0.1", port: int = 8000, processes: int = 1
) -> None:
    """Serve until SIGINT/SIGTERM. ``processes > 1`` (fixed mode, Linux/macOS) shares the port
    with SO_REUSEPORT so the server is never the bottleneck in client-overhead experiments."""
    if processes <= 1:
        _serve_one(cfg, host, port, False)
        return
    if cfg.mode == "batch":
        raise ValueError("batch mode simulates one engine; it cannot run in several processes")
    if not hasattr(socket, "SO_REUSEPORT"):
        raise RuntimeError("multi-process serving needs SO_REUSEPORT")
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_serve_one, args=(cfg, host, port, True)) for _ in range(processes)]
    for p in procs:
        p.start()

    def forward(signum: int, _frame: Any) -> None:
        for p in procs:
            if p.pid:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(p.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    for p in procs:
        p.join()
