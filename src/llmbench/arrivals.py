"""Arrival processes for open-loop load: when each request is sent, fixed before the run starts.

An open-loop generator keeps sending at the offered rate no matter how slowly the server answers,
which is how independent users behave. A closed-loop generator (N users, each waiting for its
previous answer) slows down with the server and so under-reports latency at overload; see
``runner.LoadSpec``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "arrival_offsets",
    "gamma_offsets",
    "poisson_offsets",
    "trace_offsets",
    "uniform_offsets",
]


def poisson_offsets(rate: float, n: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Poisson process: exponential gaps with mean ``1/rate``. The first request goes at t=0."""
    return gamma_offsets(rate, 1.0, n, rng)


def gamma_offsets(
    rate: float, cv: float, n: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Renewal process with gamma-distributed gaps: mean ``1/rate``, coefficient of variation
    ``cv``. ``cv=1`` is Poisson, ``cv>1`` is burstier, ``cv<1`` more regular, ``cv=0`` constant.
    """
    if n <= 0:
        return np.zeros(0)
    if math.isinf(rate):
        return np.zeros(n)
    if rate <= 0:
        raise ValueError("rate must be positive")
    if cv < 0:
        raise ValueError("cv must be non-negative")
    if cv == 0:
        return uniform_offsets(rate, n)
    shape = 1.0 / (cv * cv)
    gaps = rng.gamma(shape, 1.0 / (rate * shape), size=n)
    gaps[0] = 0.0
    out: npt.NDArray[np.float64] = np.cumsum(gaps)
    return out


def uniform_offsets(rate: float, n: int) -> npt.NDArray[np.float64]:
    if math.isinf(rate):
        return np.zeros(n)
    return np.arange(n, dtype=float) / rate


def trace_offsets(arrivals_s: Sequence[float], time_scale: float = 1.0) -> npt.NDArray[np.float64]:
    """Replay recorded arrival times, shifted to start at 0. ``time_scale=2`` replays 2x faster."""
    a = np.asarray(arrivals_s, dtype=float)
    if a.size == 0:
        return a
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    out: npt.NDArray[np.float64] = (a - a.min()) / time_scale
    return out


def arrival_offsets(
    kind: str,
    n: int,
    *,
    rate: float = math.inf,
    cv: float = 1.0,
    seed: int = 0,
    trace: Sequence[float] | None = None,
    time_scale: float = 1.0,
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    if kind == "poisson":
        return poisson_offsets(rate, n, rng)
    if kind == "gamma":
        return gamma_offsets(rate, cv, n, rng)
    if kind in ("constant", "uniform"):
        return uniform_offsets(rate, n)
    if kind == "trace":
        if trace is None:
            raise ValueError("arrival 'trace' needs a dataset with recorded arrival times")
        return trace_offsets(trace, time_scale)
    raise ValueError(f"unknown arrival process {kind!r}")
