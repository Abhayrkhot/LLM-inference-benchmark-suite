from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from llmbench.sse import SSEDecoder


def decode_all(chunks: list[bytes]) -> list[bytes]:
    d = SSEDecoder()
    out: list[bytes] = []
    for c in chunks:
        out.extend(d.feed(c))
    out.extend(d.close())
    return out


def test_basic_events_and_done() -> None:
    stream = b'data: {"a":1}\n\ndata: {"a":2}\n\ndata: [DONE]\n\n'
    assert decode_all([stream]) == [b'{"a":1}', b'{"a":2}', b"[DONE]"]


def test_comments_and_other_fields_are_ignored() -> None:
    stream = b": keep-alive\n\nevent: message\nid: 7\nretry: 100\ndata: x\n\n"
    assert decode_all([stream]) == [b"x"]


def test_multiline_data_is_joined_with_newline() -> None:
    assert decode_all([b"data: a\ndata: b\ndata:c\n\n"]) == [b"a\nb\nc"]


def test_only_one_leading_space_is_stripped() -> None:
    assert decode_all([b"data:  two\n\n"]) == [b" two"]


def test_crlf_and_cr_line_endings() -> None:
    assert decode_all([b"data: a\r\n\r\ndata: b\r\rdata: c\n\n"]) == [b"a", b"b", b"c"]


def test_crlf_split_across_reads() -> None:
    # A CR at the end of one read must not end the line before the LF arrives.
    assert decode_all([b"data: a\r", b"\n\r", b"\n"]) == [b"a"]


def test_unterminated_event_is_flushed_on_close() -> None:
    d = SSEDecoder()
    assert d.feed(b"data: tail") == []
    assert d.close() == [b"tail"]


def test_blank_lines_without_data_dispatch_nothing() -> None:
    assert decode_all([b"\n\n\n: c\n\n"]) == []


EVENT = st.binary(min_size=0, max_size=40).filter(lambda b: b"\n" not in b and b"\r" not in b)


@settings(max_examples=300, deadline=None)
@given(
    payloads=st.lists(EVENT, min_size=1, max_size=12),
    eol=st.sampled_from([b"\n", b"\r\n", b"\r"]),
    cuts=st.lists(st.integers(min_value=0, max_value=2000), max_size=20),
)
def test_chunk_boundaries_do_not_change_events(
    payloads: list[bytes], eol: bytes, cuts: list[int]
) -> None:
    stream = b"".join(b"data: " + p + eol + eol for p in payloads)
    points = sorted({c % (len(stream) + 1) for c in cuts})
    chunks, prev = [], 0
    for c in points:
        chunks.append(stream[prev:c])
        prev = c
    chunks.append(stream[prev:])
    assert decode_all(chunks) == payloads
    assert decode_all([stream]) == payloads
