"""Latency-throughput curves from sweep files (needs the ``plot`` extra)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["plot_sweeps"]


def plot_sweeps(
    paths: Sequence[str | Path],
    out: str | Path,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps: list[dict[str, Any]] = [json.loads(Path(p).read_text()) for p in paths]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    panels = (
        ("ttft_p99_ms", "TTFT p99 (ms)"),
        ("tpot_p99_ms", "TPOT p99 (ms)"),
        ("e2e_p50_ms", "E2E p50 (ms)"),
    )
    for i, sw in enumerate(sweeps):
        label = labels[i] if labels and i < len(labels) else sw["name"]
        pts = [p for p in sw["points"] if p.get("output_tokens_per_s")]
        x = [p["output_tokens_per_s"] for p in pts]
        for ax, (key, _) in zip(axes, panels, strict=True):
            ax.plot(x, [p[key] for p in pts], marker="o", label=label)
    for ax, (_, ylabel) in zip(axes, panels, strict=True):
        ax.set_xlabel("output tokens/s")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    if title:
        fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)
