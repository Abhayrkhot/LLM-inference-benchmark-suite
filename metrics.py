"""
Helper functions for computing latency and throughput metrics for LLM serving.

This module provides utilities to compute common performance metrics from a list of
per‑request timing records.  Each record should be a dictionary with the following
keys:

* `start` – float, timestamp when the request was sent (seconds)
* `first_token` – float, timestamp when the first token was received (seconds)
* `end` – float, timestamp when the response completed (seconds)
* `num_tokens` – int, number of output tokens produced

Based on these values, the functions compute end‑to‑end latency, time to first
token (TTFT), time per output token (TPOT/ITL), tokens per second (TPS),
requests per second (RPS) and latency percentiles.

The definitions of the metrics follow guidance from Anyscale's LLM benchmarking
documentation【333032983385440†L87-L95】【333032983385440†L119-L135】.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Tuple


def compute_latencies(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of end‑to‑end latencies, TTFTs and TPOTs in milliseconds.

    Parameters
    ----------
    records : list of dict
        Each dict must contain `start`, `first_token`, `end` and `num_tokens` keys.

    Returns
    -------
    tuple of np.ndarray
        Arrays (in milliseconds) for E2E latencies, TTFTs and TPOTs.
    """
    e2e = []
    ttft = []
    tpot = []
    for r in records:
        start = r["start"]
        first = r["first_token"]
        end = r["end"]
        n_tokens = max(r.get("num_tokens", 0), 1)
        e2e_ms = (end - start) * 1000.0
        ttft_ms = (first - start) * 1000.0
        # Avoid division by zero when num_tokens == 1
        if n_tokens > 1:
            tpot_ms = ((end - first) / (n_tokens - 1)) * 1000.0
        else:
            tpot_ms = 0.0
        e2e.append(e2e_ms)
        ttft.append(ttft_ms)
        tpot.append(tpot_ms)
    return np.array(e2e), np.array(ttft), np.array(tpot)


def percentile(values: np.ndarray, perc: float) -> float:
    """Compute the given percentile from an array of values.

    Parameters
    ----------
    values : np.ndarray
        The input data.
    perc : float
        The desired percentile (e.g. 50 for p50).

    Returns
    -------
    float
        The computed percentile value.
    """
    return float(np.percentile(values, perc))


def tokens_per_second(records: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Compute system and per‑user tokens per second.

    System TPS is the total number of tokens produced divided by the total time
    elapsed from the earliest start to the latest end across all requests
    【333032983385440†L196-L216】.  Per‑user TPS is computed as the inverse of the
    average time per output token (TPOT)
    【333032983385440†L139-L154】.

    Parameters
    ----------
    records : list of dict
        Timing records with `start`, `end` and `num_tokens` fields.

    Returns
    -------
    tuple of float
        (system TPS, per‑user TPS)
    """
    if not records:
        return 0.0, 0.0
    # System TPS: total tokens divided by test duration
    total_tokens = sum(max(r.get("num_tokens", 0), 0) for r in records)
    start_times = [r["start"] for r in records]
    end_times = [r["end"] for r in records]
    duration = max(end_times) - min(start_times)
    if duration <= 0:
        system_tps = 0.0
    else:
        system_tps = total_tokens / duration
    # Per‑user TPS: 1 / mean TPOT in seconds (for requests with >=2 tokens)
    # Skip records with zero TPOT to avoid division by zero
    tpot_seconds = []
    for r in records:
        n_tokens = max(r.get("num_tokens", 0), 1)
        if n_tokens > 1:
            tpot_s = (r["end"] - r["first_token"]) / (n_tokens - 1)
            tpot_seconds.append(tpot_s)
    if tpot_seconds:
        mean_tpot = float(np.mean(tpot_seconds))
        if mean_tpot > 0:
            per_user_tps = 1.0 / mean_tpot
        else:
            per_user_tps = 0.0
    else:
        per_user_tps = 0.0
    return system_tps, per_user_tps


def requests_per_second(records: List[Dict[str, Any]]) -> float:
    """Compute requests per second (RPS).

    RPS is the number of completed requests divided by the total test duration
    【333032983385440†L236-L249】.

    Parameters
    ----------
    records : list of dict
        Timing records with `start` and `end` fields.

    Returns
    -------
    float
        Requests per second.
    """
    n = len(records)
    if n == 0:
        return 0.0
    start_times = [r["start"] for r in records]
    end_times = [r["end"] for r in records]
    duration = max(end_times) - min(start_times)
    if duration <= 0:
        return 0.0
    return n / duration


def summarise(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Produce a summary of key metrics from timing records.

    Returns a dictionary with the following keys: `p50_latency`, `p95_latency`,
    `p99_latency`, `avg_ttft`, `avg_tpot`, `system_tps`, `per_user_tps`,
    `rps`.  All latencies are in milliseconds; TPS and RPS are in tokens/second
    and requests/second respectively.
    """
    if not records:
        return {
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "avg_ttft": 0.0,
            "avg_tpot": 0.0,
            "system_tps": 0.0,
            "per_user_tps": 0.0,
            "rps": 0.0,
        }
    e2e, ttft, tpot = compute_latencies(records)
    system_tps, per_user_tps = tokens_per_second(records)
    return {
        "p50_latency": percentile(e2e, 50.0),
        "p95_latency": percentile(e2e, 95.0),
        "p99_latency": percentile(e2e, 99.0),
        "avg_ttft": float(np.mean(ttft)),
        "avg_tpot": float(np.mean(tpot)),
        "system_tps": system_tps,
        "per_user_tps": per_user_tps,
        "rps": requests_per_second(records),
    }
