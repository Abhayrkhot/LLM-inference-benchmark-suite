"""Real-world workloads: ShareGPT conversations and production arrival traces.

* ShareGPT (``ShareGPT_V3_unfiltered_cleaned_split.json``): real prompts; output lengths come
  from the recorded replies. Filtered as vLLM's benchmark does so results are comparable.
* Azure LLM inference trace 2023 (``AzureLLMInferenceTrace_{conv,code}.csv``): arrival time plus
  input/output token counts of production requests; prompts are synthesized at those lengths.
* Mooncake / Kimi traces (FAST'25, ``*_trace.jsonl``): arrival time, lengths and the hash of every
  512-token block of each prompt. Equal hashes become identical token blocks, so the trace's real
  prefix-sharing structure reaches the engine's prefix cache intact.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from llmbench.datasets.base import TokenSpace, Workload
from llmbench.tokenizer import Tokenizer
from llmbench.types import Request

__all__ = ["azure_workload", "mooncake_workload", "sharegpt_workload"]


def sharegpt_workload(
    path: str | Path,
    tok: Tokenizer,
    n: int,
    *,
    seed: int = 0,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    output_len: int | None = None,
) -> Workload:
    data = json.loads(Path(path).read_text())
    convs = [
        (c["conversations"][0]["value"], c["conversations"][1]["value"])
        for c in data
        if len(c.get("conversations", [])) >= 2
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(convs))
    reqs: list[Request] = []
    batch = 512
    for start in range(0, len(order), batch):
        idx = order[start : start + batch]
        prompts = [convs[i][0] for i in idx]
        replies = [convs[i][1] for i in idx]
        p_ids = tok.encode_batch(prompts)
        r_ids = tok.encode_batch(replies)
        for prompt, p, r in zip(prompts, p_ids, r_ids, strict=True):
            out = output_len if output_len is not None else len(r)
            if len(p) < 4 or out < 4:
                continue
            if len(p) > max_prompt_len or len(p) + out > max_total_len:
                continue
            reqs.append(Request(id=len(reqs), prompt=prompt, prompt_tokens=len(p), max_tokens=out))
            if len(reqs) == n:
                return Workload("sharegpt", reqs, {"path": str(path), "seed": seed})
    if len(reqs) < n:
        raise ValueError(f"only {len(reqs)} ShareGPT conversations pass the length filters")
    return Workload("sharegpt", reqs, {"path": str(path), "seed": seed})


def _parse_ts(s: str) -> float:
    # "2023-11-16 18:15:46.6805900" -> 7 fractional digits; datetime takes at most 6.
    s = s.strip()
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s).timestamp()


def azure_workload(
    path: str | Path,
    space: TokenSpace,
    n: int | None = None,
    *,
    start_s: float = 0.0,
    seed: int = 0,
    max_input_len: int | None = None,
    max_output_len: int | None = None,
) -> Workload:
    """Requests arriving in ``[start_s, start_s + ...)`` of the trace, in order."""
    with open(path, newline="") as f:
        rows = sorted(
            (_parse_ts(row["TIMESTAMP"]), int(row["ContextTokens"]), int(row["GeneratedTokens"]))
            for row in csv.DictReader(f)
        )
    t_first = rows[0][0]
    rng = np.random.default_rng(seed)
    reqs: list[Request] = []
    for ts, inp, out in rows:
        t = ts - t_first
        if t < start_s:
            continue
        if max_input_len:
            inp = min(inp, max_input_len)
        if max_output_len:
            out = min(out, max_output_len)
        inp, out = max(inp, 1), max(out, 1)
        reqs.append(
            Request(
                id=len(reqs),
                prompt=space.sample(rng, inp),
                prompt_tokens=inp,
                max_tokens=out,
                arrival_s=t - start_s,
            )
        )
        if n is not None and len(reqs) == n:
            break
    return Workload("azure", reqs, {"path": str(path), "start_s": start_s, "seed": seed})


def mooncake_workload(
    path: str | Path,
    space: TokenSpace,
    n: int | None = None,
    *,
    block_size: int = 512,
    seed: int = 0,
    max_input_len: int | None = None,
    max_output_len: int | None = None,
) -> Workload:
    """Rebuild prompts block by block: ``hash_ids[k]`` names the k-th ``block_size``-token block,
    and each distinct hash maps to one fixed random block, so requests that share hashes share
    exactly those prompt tokens. Requests longer than ``max_input_len`` are skipped rather than
    truncated, because truncation would change the sharing structure."""
    blocks: dict[int, list[int]] = {}
    reqs: list[Request] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            inp, out = int(e["input_length"]), int(e["output_length"])
            if max_input_len and inp > max_input_len:
                skipped += 1
                continue
            if max_output_len:
                out = min(out, max_output_len)
            prompt: list[int] = []
            for h in e["hash_ids"]:
                blk = blocks.get(h)
                if blk is None:
                    # Seed per hash so a block is identical whichever request mentions it first.
                    blk = space.sample(np.random.default_rng((seed, int(h))), block_size)
                    blocks[h] = blk
                prompt.extend(blk)
            prompt = prompt[:inp]
            if len(prompt) < inp:  # trace rounding: pad the last block deterministically
                extra = np.random.default_rng((seed, -1, len(reqs)))
                prompt.extend(space.sample(extra, inp - len(prompt)))
            reqs.append(
                Request(
                    id=len(reqs),
                    prompt=prompt,
                    prompt_tokens=inp,
                    max_tokens=max(out, 1),
                    arrival_s=float(e["timestamp"]) / 1000.0,
                )
            )
            if n is not None and len(reqs) == n:
                break
    return Workload(
        "mooncake",
        reqs,
        {"path": str(path), "block_size": block_size, "skipped_too_long": skipped, "seed": seed},
    )
