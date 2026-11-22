"""Core data types shared by the load generator, metrics and result files."""

from __future__ import annotations

import enum
from array import array
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ErrorKind", "Request", "RequestRecord"]


@dataclass(frozen=True, slots=True)
class Request:
    """One request in a workload.

    ``prompt`` is either text or a list of token ids. Token-id prompts (sent to ``/v1/completions``)
    reach the engine with exactly ``prompt_tokens`` tokens, which text prompts cannot guarantee
    because re-tokenizing decoded text does not round-trip.
    """

    id: int
    prompt: str | list[int]
    prompt_tokens: int
    max_tokens: int
    # Offset of this request's arrival from the start of the run, for trace replay (seconds).
    arrival_s: float | None = None
    # Free-form labels carried into the result file (e.g. the prefix group of a request).
    tags: dict[str, Any] = field(default_factory=dict)


class ErrorKind(str, enum.Enum):
    NONE = ""
    HTTP = "http_status"
    CONNECTION = "connection_error"
    TIMEOUT = "timeout"
    INCOMPLETE = "stream_incomplete"  # connection closed before [DONE]
    SERVER = "server_error_event"  # an error object inside the stream
    PROTOCOL = "protocol_error"  # unparseable event, wrong content type


@dataclass(slots=True)
class RequestRecord:
    """Everything measured for one request. Timestamps are ``time.perf_counter_ns()`` values.

    ``scheduled_ns`` is when the workload says the request arrives. In an open-loop run it is
    fixed in advance, so latency measured from it includes any delay in sending (client backlog,
    concurrency cap) and cannot hide queueing (coordinated omission). ``sent_ns`` is when the
    request was handed to the HTTP client.
    """

    id: int
    prompt_tokens: int
    max_tokens: int
    scheduled_ns: int = 0
    sent_ns: int = 0
    first_token_ns: int = 0
    end_ns: int = 0
    # Arrival time of every chunk that carried output text, and the characters in each.
    chunk_ns: array[int] = field(default_factory=lambda: array("q"))
    chunk_chars: array[int] = field(default_factory=lambda: array("l"))
    # Token counts the server reported in its usage block (-1 when it sent none).
    usage_prompt_tokens: int = -1
    usage_completion_tokens: int = -1
    # Output tokens counted client-side by re-tokenizing the text (-1 when no tokenizer).
    client_completion_tokens: int = -1
    text: str = ""
    status: int = 0
    error: ErrorKind = ErrorKind.NONE
    error_detail: str = ""
    worker: int = 0
    warmup: bool = False
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is ErrorKind.NONE

    @property
    def output_tokens(self) -> int:
        """Best available output token count: server usage, then client tokenizer, then chunks."""
        if self.usage_completion_tokens >= 0:
            return self.usage_completion_tokens
        if self.client_completion_tokens >= 0:
            return self.client_completion_tokens
        return len(self.chunk_ns)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "scheduled_ns": self.scheduled_ns,
            "sent_ns": self.sent_ns,
            "first_token_ns": self.first_token_ns,
            "end_ns": self.end_ns,
            "chunk_ns": list(self.chunk_ns),
            "chunk_chars": list(self.chunk_chars),
            "usage_prompt_tokens": self.usage_prompt_tokens,
            "usage_completion_tokens": self.usage_completion_tokens,
            "client_completion_tokens": self.client_completion_tokens,
            "status": self.status,
            "error": self.error.value,
            "error_detail": self.error_detail,
            "worker": self.worker,
            "warmup": self.warmup,
            "tags": self.tags,
            "text": self.text,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RequestRecord:
        return cls(
            id=d["id"],
            prompt_tokens=d["prompt_tokens"],
            max_tokens=d["max_tokens"],
            scheduled_ns=d["scheduled_ns"],
            sent_ns=d["sent_ns"],
            first_token_ns=d["first_token_ns"],
            end_ns=d["end_ns"],
            chunk_ns=array("q", d["chunk_ns"]),
            chunk_chars=array("l", d["chunk_chars"]),
            usage_prompt_tokens=d["usage_prompt_tokens"],
            usage_completion_tokens=d["usage_completion_tokens"],
            client_completion_tokens=d.get("client_completion_tokens", -1),
            status=d["status"],
            error=ErrorKind(d["error"]),
            error_detail=d.get("error_detail", ""),
            worker=d.get("worker", 0),
            warmup=d.get("warmup", False),
            tags=d.get("tags", {}),
            text=d.get("text", ""),
        )
