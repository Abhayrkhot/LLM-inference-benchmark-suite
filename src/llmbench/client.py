"""Streaming HTTP client: sends one request and timestamps every chunk of its response."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from llmbench.backends import Backend, Delta, StreamError
from llmbench.sse import SSEDecoder
from llmbench.types import ErrorKind, RequestRecord

__all__ = ["dumps", "make_session", "now_ns", "send_request"]

# CLOCK_MONOTONIC on Linux, mach_absolute_time on macOS: both are system-wide, so timestamps
# taken in different worker processes can be compared directly.
now_ns = time.perf_counter_ns

try:
    import orjson

    def dumps(obj: Any) -> bytes:
        return orjson.dumps(obj)

except ImportError:  # pragma: no cover

    def dumps(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode()


def make_session(*, api_key: str | None = None, keepalive_s: float = 4.0) -> aiohttp.ClientSession:
    """One pooled session per worker.

    uvicorn (vLLM, SGLang) closes idle keep-alive connections after 5 s, and aiohttp does not
    retry a POST that lands on a connection the server just closed, so idle connections are
    dropped client-side before that. ``trust_env`` stays off: HPC nodes often export an
    ``http_proxy`` that must not see traffic to a local server.
    """
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=keepalive_s, ttl_dns_cache=300)
    # auto_decompress is off (it buffers), so ask proxies not to compress the stream.
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return aiohttp.ClientSession(
        connector=connector,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=None),
        auto_decompress=False,
        read_bufsize=2**18,
    )


def _apply(rec: RequestRecord, d: Delta, t: int, texts: list[str] | None) -> None:
    if d.text:
        if not rec.first_token_ns:
            rec.first_token_ns = t
        rec.chunk_ns.append(t)
        rec.chunk_chars.append(len(d.text))
        if texts is not None:
            texts.append(d.text)
    if d.usage_completion_tokens >= 0:
        rec.usage_completion_tokens = d.usage_completion_tokens
    if d.usage_prompt_tokens >= 0:
        rec.usage_prompt_tokens = d.usage_prompt_tokens


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: bytes,
    backend: Backend,
    rec: RequestRecord,
    *,
    timeout_s: float | None = None,
    keep_text: bool = True,
) -> None:
    """Send ``payload`` and fill ``rec`` in place. Never raises for request-level failures."""
    texts: list[str] | None = [] if keep_text else None
    done = False
    timeout = aiohttp.ClientTimeout(total=timeout_s) if timeout_s else None
    rec.sent_ns = now_ns()
    try:
        async with session.post(
            url, data=payload, headers={"X-Request-Id": str(rec.id)}, timeout=timeout
        ) as resp:
            rec.status = resp.status
            if resp.status != 200:
                body = await resp.read()
                rec.error = ErrorKind.HTTP
                rec.error_detail = f"HTTP {resp.status}: {body[:300].decode(errors='replace')}"
                return
            ctype = resp.headers.get("Content-Type", "")
            if not ctype.startswith("text/event-stream"):
                # Server ignored stream=true and returned one JSON body: one chunk at the end.
                body = await resp.read()
                t = now_ns()
                _apply(rec, _parse_unary(body), t, texts)
                done = True
                return
            decoder = SSEDecoder()
            async for chunk in resp.content.iter_any():
                t = now_ns()
                for data in decoder.feed(chunk):
                    if data == b"[DONE]":
                        done = True
                    else:
                        _apply(rec, backend.parse(data), t, texts)
            for data in decoder.close():
                if data == b"[DONE]":
                    done = True
                else:
                    _apply(rec, backend.parse(data), now_ns(), texts)
    except StreamError as exc:
        rec.error, rec.error_detail = ErrorKind.SERVER, str(exc)[:500]
    except asyncio.TimeoutError:
        rec.error, rec.error_detail = ErrorKind.TIMEOUT, f"no response within {timeout_s}s"
    except aiohttp.ClientPayloadError as exc:
        rec.error, rec.error_detail = ErrorKind.INCOMPLETE, repr(exc)[:500]
    except (aiohttp.ClientError, OSError) as exc:
        rec.error, rec.error_detail = ErrorKind.CONNECTION, repr(exc)[:500]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        rec.error, rec.error_detail = ErrorKind.PROTOCOL, repr(exc)[:500]
    finally:
        rec.end_ns = now_ns()
        if texts:
            rec.text = "".join(texts)
    if rec.ok and not done:
        rec.error = ErrorKind.INCOMPLETE
        rec.error_detail = "stream ended without [DONE]"


def _parse_unary(body: bytes) -> Delta:
    obj = json.loads(body)
    d = Delta()
    choices = obj.get("choices") or [{}]
    c = choices[0]
    d.text = c.get("text") or (c.get("message") or {}).get("content") or ""
    usage = obj.get("usage") or {}
    d.usage_prompt_tokens = int(usage.get("prompt_tokens", -1))
    d.usage_completion_tokens = int(usage.get("completion_tokens", -1))
    return d
