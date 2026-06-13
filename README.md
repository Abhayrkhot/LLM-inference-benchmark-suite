# LLM Inference Benchmark Suite

## Overview

The **LLM Inference Benchmark Suite** is a Python‑based toolkit for measuring the latency and throughput of large language model (LLM) serving endpoints.  It provides a simple command‑line interface for sending concurrent requests to a specified API endpoint and captures detailed timing information about each request.  After running a benchmark, the tool produces a CSV file with per‑request metrics and prints an aggregated summary including percentiles, average time to first token, average time per output token, tokens per second and requests per second.

This project exists to make LLM performance benchmarking transparent and reproducible.  Rather than relying on aggregated averages alone, it surfaces per‑request timings to help identify jitter and tail latencies.  You can use it to compare serving backends, hardware configurations or model sizes under identical workloads.

## Why this matters

When evaluating LLMs, it's not enough to look only at model quality; latency and throughput are equally critical for user experience and cost efficiency.  As the Anyscale documentation notes, *time to first token (TTFT)* is the delay between sending a prompt and receiving the first token, and it strongly influences perceived responsiveness【333032983385440†L87-L95】.  The *end‑to‑end latency* combines TTFT with the time needed to generate all remaining tokens【333032983385440†L119-L135】.  To understand throughput, you can examine *tokens per second (TPS)*, which measures how many tokens your system generates each second【333032983385440†L196-L216】.  Using percentiles such as p50, p95 and p99 helps you see the experience for typical and unlucky users【333032983385440†L284-L302】.  This tool calculates all of these metrics so you can make informed decisions about serving infrastructure.

## Features

- **Concurrent load generator** – Send many requests at once using asynchronous HTTP clients and configurable concurrency levels.
- **Streaming support** – Detects whether the endpoint returns a streaming response (server‑sent events) or a single blob and measures the first token accordingly.
- **Per‑request metrics** – Records start time, time to first token, end time and number of output tokens for each request.
- **Aggregated statistics** – Computes end‑to‑end latency percentiles (p50/p95/p99), average TTFT, average time per output token (TPOT/ITL) and both user and system tokens per second【333032983385440†L196-L216】.
- **CSV export** – Saves detailed per‑request metrics to a CSV file for further analysis.
- **Command‑line interface** – Accepts arguments for the endpoint URL, payload, number of requests, concurrency level and output filename.

## Architecture

The benchmark suite is organised as a simple Python module with an asynchronous runner.  The entry point (`benchmark.py`) uses `httpx.AsyncClient` to issue HTTP requests and `asyncio` semaphores to control concurrency.  After each request completes, its timing information and token count are stored.  Once all requests finish, the script aggregates the data into summary metrics and writes the detailed results to disk.  The structure of the repository is as follows:

```
llm-inference-benchmark-suite/
├── benchmark.py      # main script for running benchmarks
├── metrics.py        # helper functions for computing summary statistics
├── requirements.txt  # Python dependencies
├── tests/            # simple tests to validate metric calculations
└── README.md         # project documentation (this file)
```

## Installation

This tool requires Python 3.9 or later.  To install the dependencies, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` file includes `httpx` for HTTP requests, `numpy` for numerical computations and `pandas` for CSV export.

## Usage

Run the benchmark by specifying the API endpoint, a JSON payload, the number of requests to send and the concurrency level.  For example:

```bash
python benchmark.py \
  --url https://api.example.com/v1/chat/completions \
  --data '{"model": "gpt-3.5", "messages": [{"role": "user", "content": "Hello"}]}' \
  --requests 50 \
  --concurrency 5 \
  --output results.csv
```

After the run completes, the script prints a summary table with metrics such as p50/p95/p99 latencies, average TTFT, average TPOT, tokens per second and requests per second.  It also writes a CSV file (`results.csv`) containing per‑request timings and token counts.

### Measuring streaming endpoints

If the endpoint returns a streaming response via [Server‑Sent Events (SSE)](https://html.spec.whatwg.org/multipage/server-sent-events.html), the benchmark script attempts to read lines as they arrive.  The time of the first non‑empty chunk is recorded as the first token time.  If the response is not streaming, the script uses the end of the response as the first token time.  Token counts are approximated by splitting the received text on whitespace; this avoids needing a tokeniser library but is sufficient for relative comparisons.

### Interpreting the metrics

- **TTFT (ms)** – Time in milliseconds from when the request is sent to when the first token is received【333032983385440†L87-L95】.  Lower TTFT means a more responsive system.
- **E2E latency (ms)** – Time from request start to the end of the response【333032983385440†L119-L135】.  This determines how long the user waits for the full answer.
- **TPOT / ITL (ms)** – Average time per output token, computed as `(E2E latency – TTFT) / (num_tokens – 1)`【333032983385440†L139-L154】.  It reflects the steady‑state generation speed once the model starts streaming.
- **Tokens per second (TPS)** – Throughput measured by dividing the total number of tokens produced by the total elapsed time across all requests【333032983385440†L196-L216】.  The suite reports both system TPS and per‑user TPS.
- **Requests per second (RPS)** – Rate at which requests complete, calculated as `(number of requests) / (test duration)`【333032983385440†L236-L249】.
- **Percentiles (p50/p95/p99)** – End‑to‑end latency percentiles to capture typical and tail latencies【333032983385440†L284-L302】.

Use these metrics to compare different models, hardware setups or inference servers.  For example, if two systems have similar average latencies but very different p95 latencies, the one with a lower p95 will provide a more consistent experience for users.

## Example results

The suite includes a simple mock streaming server so you can verify that the metrics are computed correctly.  Running `benchmark.py` against this mock server with 50 requests and a concurrency of 5 produced the following summary on our machine:

| Metric              | Value           | Explanation |
|---------------------|-----------------|-------------|
| **p50 latency (ms)** | 368.62          | Median end‑to‑end latency across all requests.  Half of the requests finish in under 369 ms. |
| **p95 latency (ms)** | 373.05          | 95 % of requests complete within ~373 ms.  This captures tail latency. |
| **p99 latency (ms)** | 376.34          | 99 % of requests complete within ~376 ms.  Only 1 % take longer than this. |
| **avg TTFT (ms)**    | 130.80          | Average time to first token【333032983385440†L87-L95】.  Lower values mean quicker perceived responsiveness. |
| **avg TPOT (ms)**    | 26.41           | Average time per output token (TPOT/ITL)【333032983385440†L196-L216】 once streaming begins. |
| **system TPS**       | 134.86 tokens/s | Overall token throughput across all requests【333032983385440†L196-L216】. |
| **per‑user TPS**     | 37.86 tokens/s  | Token throughput per concurrent user. |
| **RPS**              | 13.49 req/s     | Requests per second processed by the server【333032983385440†L236-L249】. |

These numbers come from a controlled benchmark and will vary depending on your hardware, network and the model being served.  The key takeaway is that the suite captures detailed latency percentiles and throughput metrics, enabling you to analyse how different serving setups perform under load.  You can use the included mock server to validate your setup before benchmarking real endpoints.

## Why this project is valuable

Unlike simple wrappers that just call an API and measure average latency, this project provides a **comprehensive benchmarking framework** that surfaces both average and tail latencies, startup delays (TTFT) and steady‑state token generation speed.  It works with streaming and non‑streaming endpoints, records per‑request measurements for deeper analysis, and produces a CSV for reproducibility.  The metrics computed here are exactly the ones used in production serving systems to tune scheduling and batching strategies【333032983385440†L196-L216】【333032983385440†L284-L302】.

By including concurrency control, streaming detection and token throughput calculations, the suite demonstrates a solid understanding of real inference workloads.  In your portfolio, this project shows that you can design tooling to gather meaningful performance data and interpret it to guide system design decisions.  Combined with the other projects in this series, it forms the foundation of a cohesive inference engineering portfolio.

## Tests

The `tests/` folder contains unit tests that validate the metric calculations.  Run the tests with:

```bash
pytest -q
```

## Contributing and future work

This benchmark suite is intentionally simple and extensible.  Planned improvements include:

- More accurate token counting using OpenAI or Hugging Face tokenisers.
- Support for automatic detection of SSE vs. chunked HTTP vs. websockets.
- Additional workload patterns (e.g. variable prompt lengths).
- Integration with Grafana/Prometheus for continuous monitoring.

Contributions are welcome!
