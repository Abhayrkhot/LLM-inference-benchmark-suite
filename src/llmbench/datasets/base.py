"""Shared pieces for workload generators: token spaces and length distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from llmbench.tokenizer import Tokenizer
from llmbench.types import Request

__all__ = ["LengthDist", "TokenSpace", "Workload", "ideal_prefix_hit_rate"]


@dataclass(frozen=True)
class TokenSpace:
    """The ids synthetic prompts are drawn from: ``[0, vocab_size)`` minus special tokens."""

    vocab_size: int
    banned: frozenset[int] = frozenset()

    @classmethod
    def from_tokenizer(cls, tok: Tokenizer) -> TokenSpace:
        return cls(tok.vocab_size, tok.special_ids)

    def sample(self, rng: np.random.Generator, n: int) -> list[int]:
        ids = rng.integers(0, self.vocab_size, size=n)
        if self.banned:
            banned = np.fromiter(self.banned, dtype=np.int64)
            bad = np.isin(ids, banned)
            while bad.any():
                ids[bad] = rng.integers(0, self.vocab_size, size=int(bad.sum()))
                bad = np.isin(ids, banned)
        return [int(i) for i in ids]


@dataclass(frozen=True)
class LengthDist:
    """A token-length distribution parsed from a short spec.

    ``"512"`` fixed; ``"uniform:128:1024"`` inclusive range; ``"lognormal:256:0.8"`` with that
    mean and log-space sigma; ``"normal:256:64"``. Samples are clipped to ``[lo, hi]``.
    """

    kind: str
    a: float
    b: float = 0.0
    lo: int = 1
    hi: int = 1 << 20

    @classmethod
    def parse(cls, spec: str | int, lo: int = 1, hi: int = 1 << 20) -> LengthDist:
        if isinstance(spec, int):
            return cls("fixed", spec, lo=lo, hi=hi)
        parts = spec.split(":")
        try:
            if len(parts) == 1:
                return cls("fixed", float(parts[0]), lo=lo, hi=hi)
            kind, a, b = parts[0], float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            raise ValueError(f"bad length spec {spec!r}") from None
        if kind not in ("uniform", "lognormal", "normal"):
            raise ValueError(f"unknown length distribution {kind!r}")
        return cls(kind, a, b, lo=lo, hi=hi)

    def sample(self, rng: np.random.Generator, n: int) -> npt.NDArray[np.int64]:
        x: npt.NDArray[np.float64]
        if self.kind == "fixed":
            x = np.full(n, self.a, dtype=np.float64)
        elif self.kind == "uniform":
            x = rng.integers(int(self.a), int(self.b) + 1, size=n).astype(float)
        elif self.kind == "lognormal":
            sigma = self.b
            mu = math.log(self.a) - sigma * sigma / 2  # so that E[x] = a
            x = rng.lognormal(mu, sigma, size=n)
        else:
            x = rng.normal(self.a, self.b, size=n)
        out: npt.NDArray[np.int64] = np.clip(np.rint(x), self.lo, self.hi).astype(np.int64)
        return out


@dataclass
class Workload:
    """Requests plus, for traces, their recorded arrival offsets."""

    name: str
    requests: list[Request]
    description: dict[str, object] = field(default_factory=dict)

    @property
    def arrivals_s(self) -> list[float] | None:
        a = [r.arrival_s for r in self.requests]
        if any(x is None for x in a):
            return None
        return [float(x) for x in a if x is not None]

    def stats(self) -> dict[str, float]:
        p = np.array([r.prompt_tokens for r in self.requests], dtype=float)
        o = np.array([r.max_tokens for r in self.requests], dtype=float)
        if p.size == 0:
            return {"requests": 0}
        return {
            "requests": float(p.size),
            "prompt_tokens_mean": float(p.mean()),
            "prompt_tokens_p50": float(np.median(p)),
            "prompt_tokens_p99": float(np.percentile(p, 99)),
            "output_tokens_mean": float(o.mean()),
            "output_tokens_p50": float(np.median(o)),
            "output_tokens_p99": float(np.percentile(o, 99)),
        }


def ideal_prefix_hit_rate(requests: list[Request], block_size: int = 16) -> float:
    """Fraction of prompt tokens an infinite, never-evicting prefix cache with ``block_size``-token
    blocks would serve, replaying ``requests`` in order. Block hashes are chained, so a block only
    hits when the whole prefix up to and including it was seen before (as in vLLM). Real caches
    evict, so this is an upper bound for the engine-reported hit rate."""
    seen: set[int] = set()
    hits = total = 0
    for r in requests:
        if not isinstance(r.prompt, list):
            raise ValueError("needs token-id prompts")
        p = r.prompt
        h = 0
        for k in range(len(p) // block_size):
            h = hash((h, *p[k * block_size : (k + 1) * block_size]))
            if h in seen:
                hits += block_size
            else:
                seen.add(h)
        total += len(p)
    return hits / total if total else 0.0
