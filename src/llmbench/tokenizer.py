"""Thin tokenizer wrapper over the ``tokenizers`` library (no transformers / torch needed)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property
from typing import Any

__all__ = ["Tokenizer", "load_tokenizer"]


class Tokenizer:
    def __init__(self, tok: Any, name: str) -> None:
        self._tok = tok
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [list(e.ids) for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=False))

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    @cached_property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size(with_added_tokens=False))

    @cached_property
    def special_ids(self) -> frozenset[int]:
        added = self._tok.get_added_tokens_decoder()
        return frozenset(int(i) for i, t in added.items() if getattr(t, "special", True))


def load_tokenizer(name_or_path: str) -> Tokenizer:
    """Load ``tokenizer.json`` from a local directory/file or a Hugging Face model id."""
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tokenizer support needs: pip install 'llmbench[hf]'") from exc
    path = name_or_path
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(name_or_path, "tokenizer.json")
    return Tokenizer(HFTokenizer.from_file(path), name_or_path)
