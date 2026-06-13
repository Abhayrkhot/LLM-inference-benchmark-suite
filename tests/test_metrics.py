import math
import os
import sys

import pytest

# Ensure the module under test is importable by adjusting sys.path.  The
# "metrics.py" module resides in the parent directory of this tests folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import (
    compute_latencies,
    summarise,
    tokens_per_second,
    requests_per_second,
)


def test_compute_latencies_basic():
    # Single record with two tokens
    record = {
        "start": 0.0,
        "first_token": 0.5,
        "end": 1.0,
        "num_tokens": 2,
    }
    e2e, ttft, tpot = compute_latencies([record])
    assert e2e.tolist() == [1000.0]
    assert ttft.tolist() == [500.0]
    assert tpot.tolist() == [500.0]


def test_tokens_per_second_system_and_user():
    records = [
        {"start": 0.0, "first_token": 0.5, "end": 1.0, "num_tokens": 4},
        {"start": 0.1, "first_token": 0.6, "end": 1.1, "num_tokens": 6},
    ]
    system_tps, per_user_tps = tokens_per_second(records)
    # Total tokens = 10, duration = 1.1 - 0.0 = 1.1s, so system TPS ≈ 9.09
    assert math.isclose(system_tps, 10.0 / 1.1, rel_tol=1e-5)
    # Each request: time per token = (end - first) / (tokens - 1)
    # req1: (1.0 - 0.5) / 3 = 0.1667s; req2: (1.1 - 0.6) / 5 = 0.1s
    # Average ≈ 0.1333s; per_user_tps ≈ 7.5 tokens/s
    assert math.isclose(per_user_tps, 1.0 / ((0.5 / 3 + 0.5 / 5) / 2), rel_tol=1e-2)


def test_requests_per_second():
    records = [
        {"start": 0.0, "first_token": 0.5, "end": 1.0, "num_tokens": 2},
        {"start": 0.2, "first_token": 0.6, "end": 1.2, "num_tokens": 2},
    ]
    rps = requests_per_second(records)
    # Number of requests = 2, duration = 1.2 - 0.0 = 1.2s => RPS ≈ 1.67
    assert math.isclose(rps, 2.0 / 1.2, rel_tol=1e-5)


def test_summarise_percentiles():
    records = [
        {"start": 0.0, "first_token": 0.25, "end": 0.5, "num_tokens": 2},  # 500 ms
        {"start": 0.0, "first_token": 0.5, "end": 1.0, "num_tokens": 2},   # 1000 ms
        {"start": 0.0, "first_token": 0.75, "end": 1.5, "num_tokens": 2},  # 1500 ms
    ]
    summary = summarise(records)
    assert summary["p50_latency"] == 1000.0  # median of [500,1000,1500]
    # With NumPy's default linear interpolation, the 95th percentile for
    # [500, 1000, 1500] is 1450 ms and the 99th percentile is 1490 ms.
    assert summary["p95_latency"] == 1450.0
    assert summary["p99_latency"] == 1490.0
    # Average TTFT = average of (250ms, 500ms, 750ms) = 500ms
    assert summary["avg_ttft"] == 500.0
    # Average TPOT = average of (250ms, 500ms, 750ms) = 500ms
    assert summary["avg_tpot"] == 500.0