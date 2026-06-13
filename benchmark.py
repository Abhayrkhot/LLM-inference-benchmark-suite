#!/usr/bin/env python3
"""
Asynchronous benchmark driver for LLM inference endpoints.

This script sends a configurable number of concurrent requests to a specified
HTTP endpoint and measures the time to first token, end‑to‑end latency and
throughput.  Results are aggregated into summary statistics and written to
CSV for further analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import csv
import sys
import time
from typing import Any, Dict, List

import httpx
import numpy as np  # type: ignore

from metrics import summarise  # type: ignore


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a single HTTP POST request and record timing information.

    Parameters
    ----------
    client : httpx.AsyncClient
        The HTTP client to use.
    url : str
        The full endpoint URL.
    payload : dict
        The JSON payload to send in the request body.

    Returns
    -------
    dict
        A record containing `start`, `first_token`, `end`, `num_tokens` and
        optional `error` fields.
    """
    start = time.perf_counter()
    first_token_time: float | None = None
    num_tokens = 0
    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            # Detect server‑sent events (SSE) stream. Using client.stream is
            # important here because client.post buffers the whole response and
            # would make TTFT look the same as end-to-end latency.
            if content_type.startswith("text/event-stream"):
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    # Try to parse JSON data if the line contains a JSON message
                    line_data = line
                    if line.startswith("data:"):
                        line_data = line[len("data:"):].strip()
                    try:
                        data = json.loads(line_data)
                        # Extract content tokens from typical OpenAI streaming response
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if isinstance(content, str):
                                # Split on whitespace as an approximation to tokens
                                num_tokens += len(content.split())
                    except json.JSONDecodeError:
                        # Fallback: count words in the line
                        num_tokens += len(line_data.split())
            else:
                # Non‑streaming response
                body = await response.aread()
                now = time.perf_counter()
                first_token_time = now
                text = body.decode(response.encoding or "utf-8", errors="replace")
                num_tokens = len(text.split())
        end = time.perf_counter()
        # Ensure first_token_time is set
        if first_token_time is None:
            first_token_time = end
        return {
            "start": start,
            "first_token": first_token_time,
            "end": end,
            "num_tokens": num_tokens,
        }
    except Exception as exc:
        end = time.perf_counter()
        if first_token_time is None:
            first_token_time = end
        return {
            "start": start,
            "first_token": first_token_time,
            "end": end,
            "num_tokens": 0,
            "error": repr(exc),
        }


async def run_benchmark(
    url: str,
    payload: Dict[str, Any],
    num_requests: int,
    concurrency: int,
    output_file: str,
) -> None:
    """Run the benchmark and write results to CSV.

    Parameters
    ----------
    url : str
        The endpoint URL to benchmark.
    payload : dict
        The request payload to send.
    num_requests : int
        Total number of requests to send.
    concurrency : int
        Maximum number of concurrent requests.
    output_file : str
        Path to the CSV file where per‑request metrics will be saved.
    """
    records: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=None) as client:
        async def task_wrapper() -> None:
            async with semaphore:
                record = await send_request(client, url, payload)
                records.append(record)

        tasks = [asyncio.create_task(task_wrapper()) for _ in range(num_requests)]
        await asyncio.gather(*tasks)

    # Write per‑request details to CSV
    keys = ["start", "first_token", "end", "num_tokens", "error"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            # Fill missing keys with empty strings
            row = {k: r.get(k, "") for k in keys}
            writer.writerow(row)

    # Compute and display summary statistics
    # Filter out records with errors when computing latencies
    success_records = [r for r in records if "error" not in r]
    summary = summarise(success_records)
    if success_records:
        print("Summary (milliseconds unless noted):")
        print(f"  p50 latency : {summary['p50_latency']:.2f} ms")
        print(f"  p95 latency : {summary['p95_latency']:.2f} ms")
        print(f"  p99 latency : {summary['p99_latency']:.2f} ms")
        print(f"  avg TTFT    : {summary['avg_ttft']:.2f} ms")
        print(f"  avg TPOT    : {summary['avg_tpot']:.2f} ms")
        print(f"  system TPS  : {summary['system_tps']:.2f} tokens/s")
        print(f"  per-user TPS: {summary['per_user_tps']:.2f} tokens/s")
        print(f"  RPS         : {summary['rps']:.2f} requests/s")
    else:
        print("No successful records to summarise.  All requests failed.")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an inference latency benchmark.")
    parser.add_argument("--url", required=True, help="The inference endpoint URL")
    parser.add_argument(
        "--data",
        required=True,
        help="JSON payload to send with each request (string or @file.json)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Total number of requests to send (default 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests (default 1)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="Output CSV filename for per‑request metrics",
    )
    args = parser.parse_args(argv)
    # Parse payload JSON
    data_arg = args.data
    if data_arg.startswith("@"):
        # Read from file
        filename = data_arg[1:]
        try:
            with open(filename, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"Error reading JSON from {filename}: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            payload = json.loads(data_arg)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON payload: {exc}", file=sys.stderr)
            return 1
    # Run benchmark
    asyncio.run(
        run_benchmark(
            url=args.url,
            payload=payload,
            num_requests=args.requests,
            concurrency=args.concurrency,
            output_file=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
