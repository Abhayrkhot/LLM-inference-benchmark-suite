from __future__ import annotations

import json

import pytest

from llmbench.backends import SamplingOptions, StreamError, get_backend
from llmbench.types import Request


def test_completions_body_has_stream_usage_and_ignore_eos() -> None:
    b = get_backend("completions")
    req = Request(id=0, prompt=[1, 2, 3], prompt_tokens=3, max_tokens=7)
    body = b.body(req, "m", SamplingOptions(extra_body={"top_k": 5}))
    assert body["prompt"] == [1, 2, 3]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["ignore_eos"] is True
    assert body["max_tokens"] == 7
    assert body["top_k"] == 5


def test_ignore_eos_can_be_disabled() -> None:
    req = Request(id=0, prompt="hi", prompt_tokens=1, max_tokens=4)
    body = get_backend("chat").body(req, "m", SamplingOptions(ignore_eos=False))
    assert "ignore_eos" not in body
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_rejects_token_id_prompts() -> None:
    req = Request(id=0, prompt=[1, 2], prompt_tokens=2, max_tokens=4)
    with pytest.raises(ValueError):
        get_backend("chat").body(req, "m", SamplingOptions())


def test_chat_role_chunk_is_not_a_token() -> None:
    b = get_backend("chat")
    role = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
    assert b.parse(json.dumps(role).encode()).text == ""
    tok = {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}
    assert b.parse(json.dumps(tok).encode()).text == "Hi"


def test_chat_reasoning_tokens_count_as_output() -> None:
    b = get_backend("chat")
    obj = {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]}
    assert b.parse(json.dumps(obj).encode()).text == "hmm"


def test_usage_chunk() -> None:
    b = get_backend("completions")
    obj = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
    d = b.parse(json.dumps(obj).encode())
    assert (d.text, d.usage_prompt_tokens, d.usage_completion_tokens) == ("", 11, 5)


def test_error_event_raises() -> None:
    with pytest.raises(StreamError):
        get_backend("completions").parse(b'{"error": {"message": "boom"}}')


def test_unknown_backend() -> None:
    with pytest.raises(ValueError):
        get_backend("grpc")
