"""Synthetic workloads with exact token counts, sent as token-id prompts."""

from __future__ import annotations

import numpy as np

from llmbench.datasets.base import LengthDist, TokenSpace, Workload
from llmbench.types import Request

__all__ = ["prefix_workload", "random_workload"]


def random_workload(
    n: int,
    space: TokenSpace,
    input_len: LengthDist,
    output_len: LengthDist,
    seed: int = 0,
) -> Workload:
    """Independent random-token prompts. Different random first tokens mean two prompts share no
    prefix (beyond chance), so prefix caching does not flatter the result."""
    rng = np.random.default_rng(seed)
    ins = input_len.sample(rng, n)
    outs = output_len.sample(rng, n)
    reqs = [
        Request(
            id=i,
            prompt=space.sample(rng, int(ins[i])),
            prompt_tokens=int(ins[i]),
            max_tokens=int(outs[i]),
        )
        for i in range(n)
    ]
    return Workload(
        "random",
        reqs,
        {"input_len": input_len.__dict__, "output_len": output_len.__dict__, "seed": seed},
    )


def prefix_workload(
    n: int,
    space: TokenSpace,
    *,
    num_prefixes: int,
    prefix_len: int,
    suffix_len: LengthDist,
    output_len: LengthDist,
    seed: int = 0,
) -> Workload:
    """``num_prefixes`` shared system prompts of ``prefix_len`` tokens, each request picking one
    uniformly and appending a unique suffix: the shape of multi-tenant chat and few-shot traffic.
    The best possible prefix-cache hit rate is ``prefix_len / (prefix_len + mean suffix)`` once
    every prefix has been seen."""
    rng = np.random.default_rng(seed)
    prefixes = [space.sample(rng, prefix_len) for _ in range(num_prefixes)]
    groups = rng.integers(0, num_prefixes, size=n)
    sufs = suffix_len.sample(rng, n)
    outs = output_len.sample(rng, n)
    reqs = []
    for i in range(n):
        g = int(groups[i])
        prompt = prefixes[g] + space.sample(rng, int(sufs[i]))
        reqs.append(
            Request(
                id=i,
                prompt=prompt,
                prompt_tokens=len(prompt),
                max_tokens=int(outs[i]),
                tags={"prefix": g},
            )
        )
    return Workload(
        "prefix",
        reqs,
        {"num_prefixes": num_prefixes, "prefix_len": prefix_len, "seed": seed},
    )
