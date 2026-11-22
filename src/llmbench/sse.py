"""Incremental decoder for ``text/event-stream`` bodies.

Network reads split server-sent events at arbitrary byte offsets, so the decoder buffers partial
lines and only returns events whose terminating blank line has arrived. It follows the WHATWG
event-stream grammar: lines end in LF, CRLF or a lone CR; lines starting with ``:`` are comments;
``data`` fields of one event are joined with newlines; one leading space after the colon is
stripped. Only ``data`` matters for OpenAI-style streams, so other fields are ignored.
"""

from __future__ import annotations

__all__ = ["SSEDecoder"]


class SSEDecoder:
    """Feed raw bytes with :meth:`feed`; get back the ``data`` payload of each complete event."""

    __slots__ = ("_buf", "_cr_mode", "_data")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._data: list[bytes] = []
        # Streams from OpenAI-compatible servers use LF only. Until a CR is seen, lines are split
        # with bytes.split, which is several times faster than scanning for both terminators.
        self._cr_mode = False

    def feed(self, chunk: bytes) -> list[bytes]:
        events: list[bytes] = []
        if not chunk:
            return events
        if not self._cr_mode and b"\r" in chunk:
            self._cr_mode = True
        self._buf += chunk
        if self._cr_mode:
            self._feed_general(events)
        else:
            lines = self._buf.split(b"\n")
            tail = lines.pop()
            for line in lines:
                self._line(bytes(line), events)
            self._buf = bytearray(tail)
        return events

    def close(self) -> list[bytes]:
        """Flush at end of stream.

        The spec discards an event that was never terminated by a blank line. Some servers omit
        the final blank line, so a pending event is returned instead; callers that want strict
        behaviour can ignore the result.
        """
        events: list[bytes] = []
        if self._buf:
            line = bytes(self._buf).rstrip(b"\r")
            self._buf.clear()
            self._line(line, events)
        if self._data:
            events.append(b"\n".join(self._data))
            self._data.clear()
        return events

    def _feed_general(self, events: list[bytes]) -> None:
        buf = self._buf
        n = len(buf)
        start = 0
        while start < n:
            i_lf = buf.find(b"\n", start)
            i_cr = buf.find(b"\r", start)
            if i_lf == -1 and i_cr == -1:
                break
            if i_cr != -1 and (i_lf == -1 or i_cr < i_lf):
                if i_cr == n - 1:
                    break  # might be the first half of CRLF; wait for the next byte
                line = bytes(buf[start:i_cr])
                start = i_cr + 2 if buf[i_cr + 1] == 0x0A else i_cr + 1
            else:
                line = bytes(buf[start:i_lf])
                start = i_lf + 1
            self._line(line, events)
        del buf[:start]

    def _line(self, line: bytes, events: list[bytes]) -> None:
        if not line:
            if self._data:
                events.append(self._data[0] if len(self._data) == 1 else b"\n".join(self._data))
                self._data = []
            return
        if line[0] == 0x3A:  # ':' comment / keep-alive
            return
        colon = line.find(b":")
        if colon == -1:
            field, value = line, b""
        else:
            field = line[:colon]
            value = line[colon + 1 :]
            if value[:1] == b" ":
                value = value[1:]
        if field == b"data":
            self._data.append(value)
