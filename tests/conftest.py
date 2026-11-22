from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest
from aiohttp import web

from llmbench.fakeserver import FakeEngineConfig, make_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerThread:
    """Runs the fake server's aiohttp app on its own event loop in a background thread, so the
    code under test can call ``asyncio.run`` in the main thread."""

    def __init__(self, cfg: FakeEngineConfig) -> None:
        self.cfg = cfg
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._runner: web.AppRunner | None = None

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def start() -> None:
            self._runner = web.AppRunner(make_app(self.cfg), access_log=None)
            await self._runner.setup()
            await web.TCPSite(self._runner, "127.0.0.1", self.port).start()
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    def start(self) -> ServerThread:
        self._thread.start()
        assert self._ready.wait(10), "fake server did not start"
        return self

    def stop(self) -> None:
        async def cleanup() -> None:
            assert self._runner is not None
            await self._runner.cleanup()

        asyncio.run_coroutine_threadsafe(cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


@pytest.fixture
def fake_server() -> Iterator[Callable[..., ServerThread]]:
    servers: list[ServerThread] = []

    def start(**kw: object) -> ServerThread:
        s = ServerThread(replace(FakeEngineConfig(), **kw)).start()  # type: ignore[arg-type]
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.stop()
