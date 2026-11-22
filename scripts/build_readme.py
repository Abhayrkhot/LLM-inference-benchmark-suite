#!/usr/bin/env python3
"""Fill the generated blocks of README.md from the committed results.

    python scripts/build_readme.py          # rewrite README.md
    python scripts/build_readme.py --check  # exit 1 if README.md is stale (CI)

Every number in a ``<!-- BEGIN GENERATED: x -->`` block comes from ``results/``.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import ROOT, client_overhead, configs, derived, repeat_stats

README = ROOT / "README.md"


def cpu_name(s: str) -> str:
    """'INTEL(R) XEON(R) GOLD 6548Y+' -> 'Intel Xeon Gold 6548Y+'."""
    s = re.sub(r"\((R|TM)\)|\bCPU\b|@.*$", "", s, flags=re.I)
    words = [w if any(c.isdigit() for c in w) else w.capitalize() for w in s.split()]
    return " ".join(words)


def f0(x: float) -> str:
    return f"{x:,.0f}"


def f1(x: float) -> str:
    return f"{x:,.1f}"


def pct(x: float) -> str:
    return f"{x:+.0%}" if abs(x) >= 0.1 else f"{x:+.1%}"


def cap_cell(cap: dict[str, Any] | None) -> str:
    if not cap:
        return "-"
    lo, hi = cap["bracket"]
    if hi == math.inf or (isinstance(hi, str) and hi.lower() == "infinity"):
        return f"≥ {lo:.1f}"
    return f"**{lo:.1f}** ({lo:.1f}–{hi:.1f})"


def at_cap(cap: dict[str, Any] | None, key: str) -> float | None:
    if not cap or not cap.get("at_capacity"):
        return None
    v = cap["at_capacity"].get(key)
    return float(v) if v is not None else None


def block_setup(cfgs: dict[str, dict[str, Any]]) -> str:
    vers = {}
    gpu = cpu = ""
    for c in cfgs.values():
        n = c["node"]
        vers[c["tag"].split("_")[0]] = n.get("engine_version", "?")
        gpu = gpu or n.get("gpu_line", "")
        cpu = cpu or n.get("cpu", "")
    if not cfgs:
        return "_No GPU results committed yet._"
    gpu_name = gpu.split(",")[0] if gpu else "GPU"
    driver = gpu.split(",")[1].strip() if gpu.count(",") >= 1 else "?"
    parts = [
        f"{e} {v}" for e, v in (("vLLM", vers.get("vllm")), ("SGLang", vers.get("sglang"))) if v
    ]
    engines = " and ".join(parts)
    return (
        f"One NVIDIA {gpu_name.replace('NVIDIA ', '')} (141 GB, driver {driver}) per engine, "
        f"{cpu_name(cpu)} host. {engines} from their official container images, at default "
        "settings apart from an 8k context limit (SGLang also needs `--enable-metrics`). FP8 "
        "means each engine's `--quantization fp8`: weights and activations quantized on load, "
        "no calibration data. Model Qwen2.5-7B-Instruct. Client and engine share the node; the "
        "client runs 4 worker processes."
    )


def block_capacity(cfgs: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| engine | capacity, interactive SLO (req/s) | output tok/s there | capacity, relaxed "
        "SLO (req/s) | output tok/s there | 2,000-request burst (req/s) | KV cache (tokens) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cfgs.values():
        ci, cr = c.get("cap_interactive"), c.get("cap_relaxed")
        if not ci and not cr:
            continue
        sat = (ci or cr or {}).get("saturation_rps")
        ti, tr = at_cap(ci, "output_tokens_per_s"), at_cap(cr, "output_tokens_per_s")
        kv = c["node"].get("kv_tokens_main")
        rows.append(
            f"| {c['label']} | {cap_cell(ci)} | {f0(ti) if ti else '-'} | {cap_cell(cr)} | "
            f"{f0(tr) if tr else '-'} | {f1(sat) if sat else '-'} | "
            f"{f0(int(kv)) if kv and kv.isdigit() else '-'} |"
        )
    return "\n".join(rows)


def block_sweep(cfgs: dict[str, dict[str, Any]]) -> str:
    rates = sorted(
        {p["load"]["rate"] for c in cfgs.values() if "sweep" in c for p in c["sweep"]["points"]}
    )
    pick = [r for r in rates if r in (16, 32, 48, 64) or math.isinf(r)]
    rows = [
        "| engine | offered req/s | achieved req/s | output tok/s | TTFT p50 / p99 (ms) | "
        "TPOT p50 / p99 (ms) | ITL p99 (ms) | goodput (req/s) | J / output token |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cfgs.values():
        if "sweep" not in c:
            continue
        for p in c["sweep"]["points"]:
            r = p["load"]["rate"]
            if r not in pick:
                continue
            jpt = p.get("joules_per_output_token")
            rows.append(
                f"| {c['label']} | {'∞' if math.isinf(r) else f'{r:g}'} | "
                f"{p['requests_per_s']:.1f} | {f0(p['output_tokens_per_s'])} | "
                f"{f0(p['ttft_p50_ms'])} / {f0(p['ttft_p99_ms'])} | "
                f"{f1(p['tpot_p50_ms'])} / {f1(p['tpot_p99_ms'])} | {f1(p['itl_p99_ms'])} | "
                f"{p['goodput_requests_per_s']:.1f} | {f'{jpt:.3f}' if jpt else '-'} |"
            )
    return "\n".join(rows)


def block_prefix(cfgs: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| engine | replay speed | prefix cache | TTFT p50 / p99 (ms) | E2E p50 (ms) | "
        "input tok/s served | engine-reported hit rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cfgs.values():
        for name, r in sorted(
            c.get("prefix", {}).items(), key=lambda kv: kv[0].rsplit("_x", 1)[::-1]
        ):
            m = re.search(r"_cache(on|off)_x([0-9.]+)$", name)
            if not m:
                continue
            s = r["summary"]
            st = r.get("engine_metrics", {}).get("stats", {})
            hit = st.get("prefix_hit_rate", st.get("cache_hit_rate_mean"))
            rows.append(
                f"| {c['label']} | {m.group(2)}× | {m.group(1)} | {f0(s['ttft_ms']['p50'])} / "
                f"{f0(s['ttft_ms']['p99'])} | {f0(s['e2e_ms']['p50'])} | "
                f"{f0(s['throughput']['input_tokens_per_s'])} | "
                f"{f'{hit:.0%}' if hit is not None and m.group(1) == 'on' else '-'} |"
            )
    ideal = derived().get("mooncake_ideal_hit_rate", {})
    vllm_on = next(
        (
            r
            for c in cfgs.values()
            if c["tag"].startswith("vllm")
            for n, r in c.get("prefix", {}).items()
            if n.endswith("_cacheon_x1")
        ),
        None,
    )
    if ideal:
        best = ideal["16"]
        note = (
            f"_An infinite prefix cache that never evicts would serve {best:.0%} of these "
            f"prompt tokens (16-token blocks)."
        )
        hit = (vllm_on or {}).get("engine_metrics", {}).get("stats", {}).get("prefix_hit_rate")
        if hit is not None and abs(hit - best) < 0.01:
            note += (
                " vLLM reports exactly that: with over 2 million tokens of KV cache it "
                "never had to evict a reusable block."
            )
        note += (
            " SGLang's figure is a recent-window gauge averaged over the run, so it is not "
            "directly comparable._"
        )
        rows += ["", note]
    return "\n".join(rows)


def block_xval(cfgs: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| offered req/s | metric | llmbench | vllm bench serve | difference |",
        "|---|---|---|---|---|",
    ]
    for c in cfgs.values():
        xv = c.get("xval")
        if not xv:
            continue
        for rate in ("4", "16", "32", "inf"):
            ours, theirs = xv.get(f"llmbench_rate{rate}"), xv.get(f"vllm_rate{rate}")
            if not ours or not theirs:
                continue
            s = ours["summary"]
            pairs = [
                (
                    "output tok/s",
                    s["throughput"]["output_tokens_per_s"],
                    theirs.get("output_throughput"),
                ),
                ("TTFT p50 (ms)", s["ttft_ms"]["p50"], theirs.get("median_ttft_ms")),
                ("TPOT p50 (ms)", s["tpot_ms"]["p50"], theirs.get("median_tpot_ms")),
                ("ITL p50 (ms)", s["itl_ms"]["p50"], theirs.get("median_itl_ms")),
            ]
            for i, (name, a, b) in enumerate(pairs):
                if b is None:
                    continue
                rows.append(
                    f"| {('∞' if rate == 'inf' else rate) if i == 0 else ''} | {name} | "
                    f"{f1(a)} | {f1(b)} | {pct(a / b - 1)} |"
                )
    return "\n".join(rows)


def block_client(co: dict[str, Any] | None) -> str:
    if not co:
        return "_Not run yet._"
    rows = [
        "| streams | chunks/s sent | client | reported TTFT p50 (ms) | delay added to each chunk, "
        "p50 / p99 (ms) | reported vs. true ITL p99 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    llm = [r for r in co["rows"] if r["tool"] == "llmbench" and r.get("chunk_delay_ms")]
    top_w = max((r["workers"] for r in llm), default=1)
    for r in co["rows"]:
        if r["tool"] == "llmbench" and r.get("chunk_delay_ms") and r["workers"] in (1, top_w):
            d = r["chunk_delay_ms"]
            n = r["workers"]
            rows.append(
                f"| {r['concurrency']:,} | {f0(r['chunks_per_s'])} | llmbench, {n} process"
                f"{'es' if n > 1 else ''} | {r['reported']['ttft_ms']['p50']:.1f} | "
                f"{d['p50']:.2f} / {d['p99']:.2f} | {r['reported']['itl_ms']['p99']:.1f} vs. "
                f"{r['truth']['itl_ms']['p99']:.1f} |"
            )
        elif r["tool"] == "vllm-bench" and "reported" in r:
            rows.append(
                f"| {r['concurrency']:,} | {f0(r['chunks_per_s'])} | vllm bench serve | "
                f"{r['reported']['ttft_ms']['p50']:.1f} | not exposed | "
                f"{r['reported']['itl_ms']['p99']:.1f} vs. {r['truth']['itl_ms']['p99']:.1f} |"
            )
    m = co["meta"]
    truth = [r["truth"]["ttft_ms"]["p50"] for r in llm]
    model = cpu_name(m.get("cpu_model", "")) or "CPU"
    cpus = f"{m['cpus_allocated']} cores of one {model} node" if m.get("cpus_allocated") else model
    rows.append("")
    rows.append(
        f"_Server-side truth: TTFT p50 {min(truth):.1f}–{max(truth):.1f} ms in every run. Client "
        f"and a {m['server_procs']}-process fake server shared {cpus}; "
        f"{m['tokens']} tokens per request, every request on its own stream. Rows with 2 "
        f"and 4 llmbench processes are in the JSON and the figure._"
    )
    return "\n".join(rows)


def block_variance(cfgs: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| engine | load | metric | mean of 5 runs | run-to-run CV | min – max |",
        "|---|---|---|---|---|---|",
    ]
    for c in cfgs.values():
        runs = c.get("repeat")
        if not runs:
            continue
        rate = runs[0]["config"]["load"]["rate"]
        for i, (name, st) in enumerate(repeat_stats(runs).items()):
            rows.append(
                f"| {c['label'] if i == 0 else ''} | {f'{rate:g} req/s' if i == 0 else ''} | "
                f"{name} | {f1(st['mean'])} | {st['cv']:.1%} | {f1(st['min'])} – {f1(st['max'])} |"
            )
    return "\n".join(rows)


def _cap(c: dict[str, Any], tier: str) -> float | None:
    cap = c.get(f"cap_{tier}")
    return float(cap["capacity_rps"]) if cap else None


def _sat_point(c: dict[str, Any]) -> dict[str, Any] | None:
    """The 2,000-requests-at-once run: the dedicated burst run if there is one (its engine log
    is the one analysed in derived.json), else the sweep's infinite-rate point."""
    if "burst" in c:
        r = c["burst"]
        stats = r.get("engine_metrics", {}).get("stats", {})
        return {
            "tpot_p99_ms": r["summary"]["tpot_ms"]["p99"],
            "requests_per_s": r["summary"]["throughput"]["requests_per_s"],
            "engine_running_max": stats.get("running_max", 0),
        }
    pts = [p for p in c.get("sweep", {}).get("points", []) if math.isinf(p["load"]["rate"])]
    return pts[0] if pts else None


def block_headline(cfgs: dict[str, dict[str, Any]], co: dict[str, Any] | None) -> str:
    bullets: list[str] = []
    caps = sorted(
        ((c, v) for c in cfgs.values() if (v := _cap(c, "interactive"))), key=lambda t: -t[1]
    )
    if caps:
        parts = []
        for c, v in caps:
            tok = at_cap(c.get("cap_interactive"), "output_tokens_per_s")
            parts.append(f"{c['label']} {v:.0f} req/s ({f0(tok or 0)} output tok/s)")
        bullets.append(
            "**Capacity of one H200 serving Qwen2.5-7B within an interactive SLO** (TTFT p99 ≤ "
            "500 ms, TPOT p99 ≤ 30 ms, ShareGPT, Poisson arrivals): " + "; ".join(parts) + "."
        )
    fp8 = []
    for eng in ("vllm", "sglang"):
        b, q = cfgs.get(f"{eng}_qwen2.5-7b_bf16"), cfgs.get(f"{eng}_qwen2.5-7b_fp8")
        cb, cq = (_cap(b, "interactive") if b else None), (_cap(q, "interactive") if q else None)
        if b and cb and cq:
            fp8.append(f"{pct(cq / cb - 1)} on {b['label'].split()[0]}")
    if fp8:
        bullets.append(
            f"**FP8 (W8A8) raises interactive capacity {' and '.join(fp8)}**, and "
            "lowers TPOT at every load."
        )
    v, g = cfgs.get("vllm_qwen2.5-7b_bf16"), cfgs.get("sglang_qwen2.5-7b_bf16")
    sv, sg = (_sat_point(v) if v else None), (_sat_point(g) if g else None)
    burst = derived().get("sglang_burst", {})
    if sv and sg:
        why = ""
        if burst.get("longest_prefill_run"):
            why = (
                f": its scheduler ran up to {burst['longest_prefill_run']} prefill-only "
                f"batches of {f0(burst.get('chunked_prefill_size', 0))} tokens back to back "
                f"while {f0(burst['peak_queue'])} requests queued, and running streams got no "
                "tokens in between"
            )
        bullets.append(
            f"**A 2,000-request burst exposes a scheduling difference:** vLLM admits at most "
            f"{f0(sv.get('engine_running_max', 0))} requests at once and mixes decode tokens into "
            f"every chunked-prefill batch, so streams keep moving (TPOT p99 "
            f"{f0(sv['tpot_p99_ms'])} ms); SGLang admits {f0(sg.get('engine_running_max', 0))} "
            f"and its TPOT p99 reaches {f0(sg['tpot_p99_ms'])} ms{why}. Both at default "
            "settings."
        )
    pre = []
    for c in cfgs.values():
        pf = c.get("prefix", {})
        on = next((r for n, r in pf.items() if n.endswith("_cacheon_x2")), None)
        off = next((r for n, r in pf.items() if n.endswith("_cacheoff_x2")), None)
        if on and off:
            pre.append(
                f"{c['label'].split()[0]} {f0(off['summary']['ttft_ms']['p99'] / 1000)} s → "
                f"{on['summary']['ttft_ms']['p99'] / 1000:.1f} s"
            )
    if pre:
        bullets.append(
            "**Prefix caching is the difference between overload and comfort on real agent "
            "traffic:** replaying the Mooncake tool-agent trace at 2× speed, TTFT p99 goes "
            + ", ".join(pre)
            + " with caching on."
        )
    if co:
        rows = co["rows"]
        llm = {
            (r["concurrency"], r["workers"]): r
            for r in rows
            if r["tool"] == "llmbench" and r.get("chunk_delay_ms")
        }
        vb = {r["concurrency"]: r for r in rows if r["tool"] == "vllm-bench" and "reported" in r}
        c = 1024 if (1024, 1) in llm else max((k[0] for k in llm), default=None)
        if c is not None and (c, 1) in llm:
            w = max(k[1] for k in llm if k[0] == c)
            one, many = llm[(c, 1)], llm[(c, w)]
            truth = one["truth"]["ttft_ms"]["p50"]
            vtxt = ""
            if c in vb:
                vtxt = (
                    f"vLLM's own `vllm bench serve` reports "
                    f"{f0(vb[c]['reported']['ttft_ms']['p50'])} ms and "
                )
            bullets.append(
                f"**Benchmark clients can be the bottleneck:** against a server with a true "
                f"{truth:.0f} ms TTFT streaming to {c:,} clients, {vtxt}a single llmbench "
                f"process reports {f0(one['reported']['ttft_ms']['p50'])} ms median TTFT; with "
                f"{w} processes llmbench reports {many['reported']['ttft_ms']['p50']:.1f} ms and "
                f"its per-token timestamps stay within {many['chunk_delay_ms']['p99']:.1f} ms "
                f"(p99) of the server's."
            )
    return "\n".join(f"- {b}" for b in bullets) if bullets else "_Results pending._"


def block_v0() -> str:
    w = derived().get("word_vs_token")
    if not w or not w.get("replies"):
        return ""
    return (
        f"How wrong were v0's token counts? On {w['replies']:,} assistant replies from ShareGPT, "
        f"tokenized with Qwen2.5's tokenizer, whitespace splitting finds "
        f"{w['aggregate_words_per_token']:.2f} words per real token overall, and between "
        f"{w['words_per_token_p5']:.2f} and {w['words_per_token_p95']:.2f} for the middle 90% of "
        f"replies (code and non-English text are far off). v0's throughput and TPOT were "
        f"therefore off by about {abs(1 - w['aggregate_words_per_token']):.0%} on average, by a "
        "different amount for every response."
    )


def block_compare() -> str:
    comps = derived().get("comparisons", {})
    if not comps:
        return ""
    rows = [
        "| comparison | load | metric | baseline | candidate | change [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    names = {
        "bf16": "SGLang vs vLLM, BF16",
        "fp8": "SGLang vs vLLM, FP8",
        "vllm_fp8_vs_bf16": "vLLM FP8 vs BF16",
        "sglang_fp8_vs_bf16": "SGLang FP8 vs BF16",
    }
    for key, c in comps.items():
        m = re.match(r"(.+)_rate(\d+)$", key)
        if not m or m.group(2) not in ("32", "64"):
            continue
        for metric in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99"):
            d = c[metric]
            lo, hi = d["ci95"]
            sig = "" if lo <= 0 <= hi else " **"
            rows.append(
                f"| {names.get(m.group(1), m.group(1))} | {m.group(2)} req/s | "
                f"{metric.replace('_', ' ').upper().replace(' P', ' p')} | "
                f"{d['baseline']:.1f} ms | {d['candidate']:.1f} ms | "
                f"{pct(d['change'])} [{pct(lo)}, {pct(hi)}]{sig} |"
            )
    rows.append("")
    rows.append(
        "_** marks changes whose bootstrap 95% CI excludes zero (2,000 resamples of the "
        "requests in each run)._"
    )
    return "\n".join(rows)


def blocks() -> dict[str, str]:
    cfgs = configs()
    return {
        "v0": block_v0(),
        "compare": block_compare(),
        "headline": block_headline(cfgs, client_overhead()),
        "setup": block_setup(cfgs),
        "capacity": block_capacity(cfgs),
        "sweep": block_sweep(cfgs),
        "prefix": block_prefix(cfgs),
        "xval": block_xval(cfgs),
        "client": block_client(client_overhead()),
        "variance": block_variance(cfgs),
    }


def render(text: str, values: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            return m.group(0)
        return f"<!-- BEGIN GENERATED: {key} -->\n{values[key]}\n<!-- END GENERATED: {key} -->"

    return re.sub(
        r"<!-- BEGIN GENERATED: (\w+) -->.*?<!-- END GENERATED: \1 -->", repl, text, flags=re.S
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    old = README.read_text()
    new = render(old, blocks())
    if a.check:
        if new != old:
            print("README.md is out of date; run: python scripts/build_readme.py")
            return 1
        return 0
    README.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
