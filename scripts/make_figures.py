#!/usr/bin/env python3
"""Render the README figures from the committed results: ``python scripts/make_figures.py``."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import ROOT, client_overhead, configs

IMG = ROOT / "docs" / "img"
COLORS = {
    "vLLM BF16": "#1f77b4",
    "vLLM FP8": "#7fb8e6",
    "SGLang BF16": "#d62728",
    "SGLang FP8": "#f4a3a3",
}
SLO = {"ttft_p99_ms": 500.0, "tpot_p99_ms": 30.0}


def color(label: str) -> str:
    return COLORS.get(label, "#555555")


def latency_throughput(cfgs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    panels = (
        ("ttft_p99_ms", "TTFT p99 (ms)"),
        ("tpot_p99_ms", "TPOT p99 (ms)"),
        ("goodput_requests_per_s", "goodput (req/s within TTFT ≤ 500 ms and TPOT ≤ 30 ms)"),
    )
    for c in cfgs.values():
        pts = c.get("curve", [])
        if not pts:
            continue
        for ax, (key, _) in zip(axes, panels, strict=True):
            x = [p["load"]["rate"] for p in pts]
            y = [p.get(key) or 0 for p in pts]
            ax.plot(x, y, marker="o", ms=3.5, lw=1.4, label=c["label"], color=color(c["label"]))
    for ax, (key, ylabel) in zip(axes, panels, strict=True):
        ax.set_xlabel("offered load (requests/s, Poisson)")
        ax.set_ylabel(ylabel)
        if key in SLO:
            ax.set_yscale("log")
            ax.axhline(SLO[key], color="k", ls="--", lw=0.8)
            ax.annotate(
                "SLO",
                (0.01, SLO[key]),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        else:
            lim = max(ax.get_xlim()[1], 1)
            ax.plot([0, lim], [0, lim], color="0.7", lw=0.8, ls=":")
            ax.annotate("offered", (lim * 0.7, lim * 0.72), color="0.5", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        "Qwen2.5-7B-Instruct on one H200: ShareGPT prompts, Poisson arrivals, 60 s per point"
    )
    fig.savefig(IMG / "latency_throughput.png", dpi=140)
    plt.close(fig)


def capacity_bars(cfgs: dict[str, dict[str, Any]]) -> None:
    rows = [
        (c["label"], c.get("cap_interactive"), c.get("cap_relaxed"))
        for c in cfgs.values()
        if c.get("cap_interactive") or c.get("cap_relaxed")
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    width = 0.38
    for i, (lab, ci, cr) in enumerate(rows):
        for j, cap in enumerate((ci, cr)):
            if not cap:
                continue
            lo, hi = cap["bracket"]
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                lo,
                width,
                color=color(lab),
                alpha=1.0 if j == 0 else 0.5,
                edgecolor="k",
                lw=0.5,
                hatch="" if j == 0 else "//",
            )
            top = lo
            if hi != math.inf:
                ax.errorbar(x, lo, yerr=[[0], [hi - lo]], color="k", capsize=3, lw=0.8)
                top = hi
            ax.text(x, top + 1.5, f"{lo:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylabel("highest request rate meeting the SLO (req/s)")
    ax.set_title(
        "Capacity at SLO. Solid: interactive (TTFT p99 ≤ 0.5 s, TPOT p99 ≤ 30 ms); "
        "hatched: relaxed\n(TTFT p99 ≤ 2 s, TPOT p99 ≤ 100 ms). Whisker: up to the "
        "lowest failing rate.",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(IMG / "capacity.png", dpi=140)
    plt.close(fig)


def engine_telemetry(cfgs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    for c in cfgs.values():
        pts = c.get("curve", [])
        if not pts:
            continue
        x = [p["load"]["rate"] for p in pts]
        kw = {"marker": "o", "ms": 3.5, "lw": 1.4, "label": c["label"], "color": color(c["label"])}
        axes[0].plot(x, [p.get("engine_running_mean") or 0 for p in pts], **kw)
        axes[1].plot(x, [p.get("engine_waiting_mean") or 0 for p in pts], **kw)
        axes[2].plot(x, [1000 * (p.get("joules_per_output_token") or math.nan) for p in pts], **kw)
    axes[0].set_ylabel("requests running in the engine (mean)")
    axes[1].set_ylabel("requests waiting in the engine queue (mean)")
    axes[2].set_ylabel("GPU energy per output token (mJ)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("offered load (requests/s)")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.savefig(IMG / "engine_telemetry.png", dpi=140)
    plt.close(fig)


def prefix_bars(cfgs: dict[str, dict[str, Any]]) -> None:
    rows = []
    for c in cfgs.values():
        for name, r in sorted(c.get("prefix", {}).items()):
            m = re.search(r"_cache(on|off)_x([0-9.]+)$", name)
            if m:
                rows.append((c["label"].split()[0], m.group(1), m.group(2), r["summary"]))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
    for ax, speed in zip(axes, ("1", "2"), strict=True):
        sel = [r for r in rows if r[2] == speed]
        labels = [f"{e}\ncache {mode}" for e, mode, _, _ in sel]
        p50 = [s["ttft_ms"]["p50"] for *_, s in sel]
        p99 = [s["ttft_ms"]["p99"] for *_, s in sel]
        xs = range(len(sel))
        cols = [color(f"{e} BF16") for e, *_ in sel]
        ax.bar([x - 0.2 for x in xs], p50, 0.4, color=cols, edgecolor="k", lw=0.5, label="p50")
        ax.bar(
            [x + 0.2 for x in xs],
            p99,
            0.4,
            color=cols,
            alpha=0.45,
            edgecolor="k",
            lw=0.5,
            hatch="//",
            label="p99",
        )
        for x, v in zip(xs, p99, strict=True):
            ax.text(x + 0.2, v * 1.05, f"{v / 1000:.1f} s", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(list(xs), labels, fontsize=8)
        ax.set_title(f"Mooncake tool-agent trace, {speed}× recorded speed", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("TTFT (ms)")
    axes[0].legend()
    fig.savefig(IMG / "prefix_cache.png", dpi=140)
    plt.close(fig)


def client_overhead_fig(co: dict[str, Any]) -> None:
    rows = [r for r in co["rows"] if r["tool"] == "llmbench" and r.get("chunk_delay_ms")]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for w in sorted({r["workers"] for r in rows}):
        rs = sorted((r for r in rows if r["workers"] == w), key=lambda r: r["concurrency"])
        x = [r["concurrency"] for r in rs]
        lab = f"llmbench, {w} process{'es' if w > 1 else ''}"
        axes[0].plot(x, [r["reported"]["ttft_ms"]["p50"] for r in rs], marker="o", label=lab)
        axes[1].plot(x, [r["chunk_delay_ms"]["p99"] for r in rs], marker="o", label=lab)
    vb = sorted(
        (r for r in co["rows"] if r["tool"] == "vllm-bench" and "reported" in r),
        key=lambda r: r["concurrency"],
    )
    if vb:
        axes[0].plot(
            [r["concurrency"] for r in vb],
            [r["reported"]["ttft_ms"]["p50"] for r in vb],
            marker="s",
            color="k",
            ls="--",
            label="vllm bench serve",
        )
    truth = sorted(rows, key=lambda r: r["concurrency"])
    axes[0].plot(
        [r["concurrency"] for r in truth],
        [r["truth"]["ttft_ms"]["p50"] for r in truth],
        color="0.6",
        lw=3,
        alpha=0.6,
        label="server ground truth",
        zorder=0,
    )
    axes[0].set_ylabel("reported TTFT p50 (ms)")
    axes[1].set_ylabel("delay the client adds to each chunk, p99 (ms)")
    for ax in axes:
        ax.set_xlabel("concurrent streams (each 100 tokens/s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "A fake server streams on an exact schedule (TTFT 50 ms, then a token every "
        "10 ms); the clients report what they measured"
    )
    fig.savefig(IMG / "client_overhead.png", dpi=140)
    plt.close(fig)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    cfgs = configs()
    latency_throughput(cfgs)
    capacity_bars(cfgs)
    engine_telemetry(cfgs)
    prefix_bars(cfgs)
    co = client_overhead()
    if co:
        client_overhead_fig(co)
    print("figures in", IMG)


if __name__ == "__main__":
    main()
