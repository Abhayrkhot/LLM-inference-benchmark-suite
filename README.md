# LLM Inference Benchmark Suite

A lightweight benchmarking tool for measuring how an LLM inference endpoint behaves under concurrent load.

This project focuses on the metrics that matter in real model serving systems: **time to first token (TTFT)**, **time per output token (TPOT/ITL)**, **end-to-end latency**, **p50/p95/p99 tail latency**, **tokens per second**, and **requests per second**.

The goal is not just to call an LLM API. The goal is to understand whether an inference service is fast, stable, and scalable when multiple users are sending requests at the same time.

---

## What this project accomplishes

This benchmark suite helps answer questions like:

- How long does the model take to start responding?
- How fast does it stream tokens after the first token?
- What is the p50, p95, and p99 latency under concurrency?
- How many requests per second can the endpoint complete?
- How many output tokens per second can the system produce?
- Does latency degrade when concurrent users increase?
- Are tail latencies much worse than median latency?

These are the kinds of questions inference engineers care about when evaluating LLM serving systems, because a model can have good average latency but still feel slow if TTFT or p95 latency is high.

---

## Why this matters

LLM inference performance is usually split into two major phases:

1. **Prefill phase**  
   The server processes the input prompt and prepares the model state before producing the first output token. This phase affects **TTFT**, or time to first token.

2. **Decode phase**  
   The model generates output tokens one at a time. This phase affects **TPOT**, **inter-token latency**, and perceived streaming speed.

A real benchmark should measure both. Looking only at total latency hides important details. For example, two systems can finish in the same total time, but one may start streaming much earlier and feel faster to the user.

This project measures both phases separately and also reports aggregate throughput metrics.

---

## Key metrics measured

| Metric | Meaning | Why it matters |
|---|---|---|
| **TTFT** | Time from request start to first output token | Measures responsiveness for chat and interactive apps |
| **E2E latency** | Time from request start to completed response | Measures total user wait time |
| **TPOT / ITL** | Average time between output tokens after generation starts | Measures streaming speed |
| **p50 latency** | Median end-to-end latency | Shows the typical user experience |
| **p95 latency** | 95th percentile end-to-end latency | Shows tail latency for slower requests |
| **p99 latency** | 99th percentile end-to-end latency | Shows near worst-case behavior |
| **System TPS** | Total output tokens produced per second across all requests | Measures total serving throughput |
| **Per-user TPS** | Approximate token streaming speed experienced by one request | Measures individual response speed |
| **RPS** | Completed requests per second | Measures request throughput |

---

## Features

- Asynchronous concurrent load generation using `asyncio` and `httpx`
- Configurable number of requests and concurrency level
- Streaming endpoint support using HTTP streaming
- TTFT measurement for first received streaming chunk
- End-to-end latency tracking per request
- Output token counting using a lightweight whitespace approximation
- p50, p95, and p99 latency calculation
- System tokens-per-second and per-user tokens-per-second calculation
- Requests-per-second calculation
- CSV export of per-request benchmark data
- Unit tests for metric calculations
- Mock streaming server for local testing without needing a real LLM API

---

## Architecture

```text
llm-inference-benchmark-suite/
├── benchmark.py
├── metrics.py
├── requirements.txt
├── results.csv
├── examples/
│   └── mock_streaming_server.py
├── tests/
│   └── test_metrics.py
└── README.md
```

### `benchmark.py`

The main benchmark runner. It accepts an endpoint URL, a JSON payload, total request count, concurrency level, and output CSV path.

It is responsible for:

- Creating asynchronous HTTP requests
- Limiting concurrency with an async semaphore
- Detecting streaming responses
- Recording start time, first-token time, end time, and output token count
- Writing per-request results to CSV
- Printing the benchmark summary

### `metrics.py`

The metrics module. It converts raw request timing records into useful inference performance statistics.

It computes:

- End-to-end latency
- Time to first token
- Time per output token
- Percentiles
- System tokens per second
- Per-user tokens per second
- Requests per second

### `examples/mock_streaming_server.py`

A local mock endpoint that simulates an LLM streaming response. This allows the benchmark suite to be tested without using a paid model API.

The mock server is useful for validating:

- Streaming measurement logic
- TTFT calculation
- Token counting
- Concurrency behavior
- CSV output

---

## How the benchmark works

At a high level, the benchmark performs this flow:

```text
User config
   ↓
Benchmark runner
   ↓
Concurrent async requests
   ↓
Streaming response reader
   ↓
Per-request timing records
   ↓
Metric aggregation
   ↓
CSV + terminal summary
```

For each request, the tool records:

```text
start_time       → when the request is sent
first_token_time → when the first response chunk arrives
end_time         → when the full response finishes
num_tokens       → approximate number of output tokens
```

Then it calculates:

```text
TTFT = first_token_time - start_time
E2E latency = end_time - start_time
TPOT = (end_time - first_token_time) / (num_tokens - 1)
System TPS = total output tokens / total benchmark duration
RPS = total completed requests / total benchmark duration
```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run the tests

```bash
pytest -q
```

Expected result:

```text
4 passed
```

---

## Run with a real endpoint

```bash
python benchmark.py \
  --url https://api.example.com/v1/chat/completions \
  --data '{"model": "example-model", "messages": [{"role": "user", "content": "Explain KV cache in simple terms"}], "stream": true}' \
  --requests 50 \
  --concurrency 5 \
  --output results.csv
```

---

## Run locally with the mock streaming server

Start the mock server:

```bash
python examples/mock_streaming_server.py
```

In another terminal, run the benchmark:

```bash
python benchmark.py \
  --url http://localhost:8000/v1/chat/completions \
  --data '{"model": "mock-llm", "messages": [{"role": "user", "content": "Hello"}], "stream": true}' \
  --requests 5 \
  --concurrency 5 \
  --output results.csv
```

---

## Benchmark result from local run

This project was tested against the included mock streaming LLM endpoint using concurrent requests.

### Terminal summary

```text
Summary (milliseconds unless noted):
  p50 latency : 368.62 ms
  p95 latency : 373.05 ms
  p99 latency : 376.34 ms
  avg TTFT    : 130.80 ms
  avg TPOT    : 26.41 ms
  system TPS  : 134.86 tokens/s
  per-user TPS: 37.86 tokens/s
  RPS         : 13.49 requests/s
```

### CSV sample

```csv
start,first_token,end,num_tokens,error
127.379929894,127.510703645,127.748072437,10,
127.379583364,127.510620425,127.749708137,10,
127.372853163,127.510384556,127.750014948,10,
127.378541247,127.510782334,127.750306597,10,
```

---

## How to interpret the result

The local benchmark shows that the mock endpoint starts responding in about **130.80 ms** on average. That is the measured TTFT.

The full response completes in about **368.62 ms** at p50, with p95 latency at **373.05 ms** and p99 latency at **376.34 ms**. The p95 and p99 numbers are close to the median, which means the mock endpoint had stable latency with low tail-latency variance in this run.

The average TPOT is **26.41 ms**, which means that after the first token arrives, the endpoint streams a new output token roughly every 26 ms. This corresponds to an approximate per-user streaming rate of **37.86 tokens per second**.

The system-level throughput was **134.86 tokens per second** across concurrent requests, and the request throughput was **13.49 requests per second**.

For a real LLM endpoint, these same numbers would help compare model servers, GPU configurations, batching strategies, or deployment settings.

---

## What this proves

This project demonstrates practical understanding of LLM inference serving metrics.

It shows the ability to:

- Build a concurrent benchmark runner
- Measure streaming response behavior
- Separate TTFT from decode latency
- Compute throughput and tail latency metrics
- Export reproducible benchmark data
- Validate metric calculations with tests
- Explain inference performance using production-style terminology

This is useful for roles involving LLM serving, runtime optimization, inference benchmarking, model deployment, and production AI infrastructure.

---

## Example resume bullet

```text
Built an LLM inference benchmarking suite measuring TTFT, TPOT, p50/p95/p99 latency, request throughput, and token throughput across concurrent streaming workloads
```

---

## Limitations

This is intentionally lightweight and designed for portfolio-quality benchmarking. It has a few known limitations:

- Token counting uses whitespace splitting instead of a tokenizer-specific token counter
- It currently targets HTTP and SSE-style streaming endpoints
- It does not yet include workload distributions with variable prompt lengths
- It does not yet generate charts from the CSV output
- It does not yet integrate with Prometheus or Grafana

---

## Future improvements

Planned improvements:

- Add tokenizer-aware token counting with `tiktoken` or Hugging Face tokenizers
- Add benchmark profiles for short prompts, long prompts, and mixed workloads
- Add chart generation for latency and throughput trends
- Add warmup requests before measurement
- Add support for multiple endpoints in one benchmark run
- Add Prometheus metrics export
- Add Dockerfile and GitHub Actions CI

---

## Tech stack

- Python
- asyncio
- httpx
- NumPy
- Pytest
- CSV-based result export

