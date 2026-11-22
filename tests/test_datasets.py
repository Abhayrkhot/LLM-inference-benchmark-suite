from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llmbench.datasets import (
    LengthDist,
    TokenSpace,
    azure_workload,
    ideal_prefix_hit_rate,
    mooncake_workload,
    prefix_workload,
    random_workload,
    sharegpt_workload,
)
from llmbench.tokenizer import Tokenizer


def make_tokenizer(tmp_path: Path) -> Tokenizer:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import models, pre_tokenizers

    words = "hello world how are you today fine thanks the a of to and is it what why".split()
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    tok = HFTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<|eot|>"])
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    from llmbench.tokenizer import load_tokenizer

    return load_tokenizer(str(tmp_path))


def test_length_dist_parsing_and_sampling() -> None:
    rng = np.random.default_rng(0)
    assert (LengthDist.parse("64").sample(rng, 5) == 64).all()
    u = LengthDist.parse("uniform:10:20").sample(rng, 10_000)
    assert u.min() == 10 and u.max() == 20
    ln = LengthDist.parse("lognormal:300:0.7").sample(rng, 200_000)
    assert ln.mean() == pytest.approx(300, rel=0.02)
    clipped = LengthDist.parse("normal:5:50", lo=1, hi=8).sample(rng, 1000)
    assert clipped.min() >= 1 and clipped.max() <= 8
    for bad in ("x", "uniform:1", "poisson:1:2"):
        with pytest.raises(ValueError):
            LengthDist.parse(bad)


def test_token_space_never_samples_banned_ids() -> None:
    space = TokenSpace(10, frozenset({0, 3, 9}))
    ids = space.sample(np.random.default_rng(0), 10_000)
    assert not {0, 3, 9} & set(ids)
    assert set(ids) == {1, 2, 4, 5, 6, 7, 8}


def test_random_workload_exact_lengths_and_no_shared_prefixes() -> None:
    wl = random_workload(
        200, TokenSpace(50_000), LengthDist.parse("uniform:100:300"), LengthDist.parse("32")
    )
    assert all(len(r.prompt) == r.prompt_tokens for r in wl.requests)
    assert all(r.max_tokens == 32 for r in wl.requests)
    assert ideal_prefix_hit_rate(wl.requests, block_size=16) == 0.0
    again = random_workload(
        200, TokenSpace(50_000), LengthDist.parse("uniform:100:300"), LengthDist.parse("32")
    )
    assert [r.prompt for r in again.requests] == [r.prompt for r in wl.requests]


def test_prefix_workload_sharing_matches_theory() -> None:
    n, groups, plen, slen = 400, 4, 512, 128
    wl = prefix_workload(
        n,
        TokenSpace(50_000),
        num_prefixes=groups,
        prefix_len=plen,
        suffix_len=LengthDist.parse(str(slen)),
        output_len=LengthDist.parse("8"),
    )
    by_group: dict[int, list[list[int]]] = {}
    for r in wl.requests:
        assert isinstance(r.prompt, list)
        by_group.setdefault(r.tags["prefix"], []).append(r.prompt)
    for prompts in by_group.values():
        assert all(p[:plen] == prompts[0][:plen] for p in prompts)
    # Every request after the first in its group re-uses the whole prefix.
    expected = (n - groups) * plen / (n * (plen + slen))
    assert ideal_prefix_hit_rate(wl.requests, block_size=16) == pytest.approx(expected)


def test_mooncake_rebuilds_shared_blocks(tmp_path: Path) -> None:
    rows = [
        {"timestamp": 0, "input_length": 1024, "output_length": 10, "hash_ids": [1, 2]},
        {"timestamp": 500, "input_length": 1500, "output_length": 20, "hash_ids": [1, 2, 3]},
        {"timestamp": 900, "input_length": 600, "output_length": 5, "hash_ids": [4, 5]},
        {"timestamp": 1200, "input_length": 9999, "output_length": 5, "hash_ids": [9] * 20},
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wl = mooncake_workload(path, TokenSpace(30_000), block_size=512, max_input_len=4096)
    assert len(wl.requests) == 3 and wl.description["skipped_too_long"] == 1
    a, b, c = (r.prompt for r in wl.requests)
    assert isinstance(a, list) and isinstance(b, list) and isinstance(c, list)
    assert (len(a), len(b), len(c)) == (1024, 1500, 600)
    assert b[:1024] == a
    assert c[:512] != a[:512]
    assert [r.arrival_s for r in wl.requests] == [0.0, 0.5, 0.9]
    assert wl.arrivals_s == [0.0, 0.5, 0.9]
    # Request b re-uses a's two blocks (1024 tokens) out of 1024 + 1500 + 600 prompt tokens.
    assert ideal_prefix_hit_rate(wl.requests, block_size=512) == pytest.approx(1024 / 3124)


def test_azure_trace(tmp_path: Path) -> None:
    path = tmp_path / "azure.csv"
    path.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2023-11-16 18:15:46.6805900,374,44\n"
        "2023-11-16 18:15:50.9951690,396,109\n"
        "2023-11-16 18:15:45.6805900,10,5\n"
    )
    wl = azure_workload(path, TokenSpace(1000), max_input_len=380)
    assert [r.prompt_tokens for r in wl.requests] == [10, 374, 380]
    assert wl.arrivals_s is not None
    assert wl.arrivals_s[0] == 0.0
    assert wl.arrivals_s[1] == pytest.approx(1.0)
    assert wl.arrivals_s[2] == pytest.approx(5.314578, abs=1e-5)


def test_sharegpt_filters_like_vllm(tmp_path: Path) -> None:
    tok = make_tokenizer(tmp_path)
    data = [
        {
            "conversations": [
                {"value": "hello how are you today"},
                {"value": "fine thanks and you what is it"},
            ]
        },
        {"conversations": [{"value": "hi"}, {"value": "too short prompt"}]},  # prompt < 4
        {"conversations": [{"value": "only one turn here ok"}]},
        {"conversations": [{"value": "what is the a of it"}, {"value": "to and is it why"}]},
    ]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))
    wl = sharegpt_workload(path, tok, 2, seed=0)
    assert sorted((r.prompt_tokens, r.max_tokens) for r in wl.requests) == [(5, 7), (6, 5)]
    assert all(isinstance(r.prompt, str) for r in wl.requests)
    with pytest.raises(ValueError):
        sharegpt_workload(path, tok, 3)


def test_tokenizer_wrapper(tmp_path: Path) -> None:
    tok = make_tokenizer(tmp_path)
    assert tok.count("hello world") == 2
    assert tok.vocab_size == 18
    assert tok.special_ids == frozenset({18})
    assert TokenSpace.from_tokenizer(tok).banned == frozenset({18})
