"""Request bodies and stream parsing for the OpenAI-compatible APIs that serving engines expose.

vLLM, SGLang, TensorRT-LLM (``trtllm-serve``), TGI, llama.cpp and Ollama all serve
``/v1/completions`` and ``/v1/chat/completions``. Engine-specific sampling options such as
``ignore_eos`` go in ``extra_body``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmbench.types import Request

__all__ = ["BACKENDS", "Backend", "Delta", "StreamError", "get_backend"]

try:  # orjson is ~3x faster on the many small chunks of a stream
    import orjson

    def _loads(data: bytes) -> Any:
        return orjson.loads(data)

except ImportError:  # pragma: no cover - exercised only without the 'fast' extra

    def _loads(data: bytes) -> Any:
        return json.loads(data)


class StreamError(Exception):
    """The server sent an error object inside an otherwise successful stream."""


@dataclass(slots=True)
class Delta:
    text: str = ""
    usage_prompt_tokens: int = -1
    usage_completion_tokens: int = -1


@dataclass(frozen=True)
class SamplingOptions:
    temperature: float = 0.0
    ignore_eos: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str
    path: str

    def body(self, req: Request, model: str, opts: SamplingOptions) -> dict[str, Any]: ...

    def parse(self, data: bytes) -> Delta: ...


def _usage(obj: dict[str, Any], d: Delta) -> None:
    usage = obj.get("usage")
    if usage:
        d.usage_prompt_tokens = int(usage.get("prompt_tokens") or -1)
        ct = usage.get("completion_tokens")
        d.usage_completion_tokens = int(ct) if ct is not None else -1


def _common(req: Request, model: str, opts: SamplingOptions) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": req.max_tokens,
        "temperature": opts.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if opts.ignore_eos:
        body["ignore_eos"] = True
    body.update(opts.extra_body)
    return body


def _check_error(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise StreamError(f"unexpected event payload: {obj!r:.200}")
    if "error" in obj and not obj.get("choices"):
        raise StreamError(json.dumps(obj["error"])[:500])
    return obj


class CompletionsBackend:
    name = "completions"
    path = "/v1/completions"

    def body(self, req: Request, model: str, opts: SamplingOptions) -> dict[str, Any]:
        body = _common(req, model, opts)
        body["prompt"] = req.prompt
        return body

    def parse(self, data: bytes) -> Delta:
        obj = _check_error(_loads(data))
        d = Delta()
        choices = obj.get("choices")
        if choices:
            d.text = choices[0].get("text") or ""
        _usage(obj, d)
        return d


class ChatBackend:
    name = "chat"
    path = "/v1/chat/completions"

    def body(self, req: Request, model: str, opts: SamplingOptions) -> dict[str, Any]:
        if not isinstance(req.prompt, str):
            raise ValueError("the chat backend needs text prompts; use --backend completions")
        body = _common(req, model, opts)
        body["messages"] = [{"role": "user", "content": req.prompt}]
        return body

    def parse(self, data: bytes) -> Delta:
        obj = _check_error(_loads(data))
        d = Delta()
        choices = obj.get("choices")
        if choices:
            delta = choices[0].get("delta") or {}
            # The first chunk usually carries only {"role": "assistant"}; it is not a token.
            # Reasoning models stream thinking tokens in a separate field; they are output too.
            # vLLM has used both names for the same text; never count it twice.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            d.text = (delta.get("content") or "") + reasoning
        _usage(obj, d)
        return d


BACKENDS: dict[str, type[CompletionsBackend] | type[ChatBackend]] = {
    "completions": CompletionsBackend,
    "chat": ChatBackend,
}


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
