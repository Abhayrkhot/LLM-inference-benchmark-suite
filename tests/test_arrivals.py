from __future__ import annotations

import math

import numpy as np
import pytest

from llmbench.arrivals import arrival_offsets, gamma_offsets, trace_offsets, uniform_offsets


def gaps(x: np.ndarray) -> np.ndarray:
    return np.diff(x)


def test_poisson_rate_and_cv() -> None:
    x = arrival_offsets("poisson", 200_000, rate=50.0, seed=1)
    g = gaps(x)
    assert x[0] == 0.0
    assert g.mean() == pytest.approx(1 / 50, rel=0.01)
    assert g.std() / g.mean() == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("cv", [0.5, 2.0, 4.0])
def test_gamma_cv(cv: float) -> None:
    g = gaps(gamma_offsets(10.0, cv, 300_000, np.random.default_rng(2)))
    assert g.mean() == pytest.approx(0.1, rel=0.02)
    assert g.std() / g.mean() == pytest.approx(cv, rel=0.05)


def test_cv_zero_is_constant() -> None:
    assert np.allclose(
        gamma_offsets(4.0, 0.0, 5, np.random.default_rng(0)), [0, 0.25, 0.5, 0.75, 1]
    )


def test_infinite_rate_sends_everything_at_once() -> None:
    assert not arrival_offsets("poisson", 10, rate=math.inf).any()
    assert not uniform_offsets(math.inf, 3).any()


def test_seed_determinism() -> None:
    a = arrival_offsets("gamma", 100, rate=3, cv=2, seed=7)
    b = arrival_offsets("gamma", 100, rate=3, cv=2, seed=7)
    c = arrival_offsets("gamma", 100, rate=3, cv=2, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_trace_offsets_shift_and_scale() -> None:
    assert np.allclose(trace_offsets([10.0, 12.0, 16.0], time_scale=2.0), [0, 1, 3])


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        arrival_offsets("poisson", 5, rate=-1)
    with pytest.raises(ValueError):
        arrival_offsets("trace", 5)
    with pytest.raises(ValueError):
        arrival_offsets("weibull", 5, rate=1)
