"""Result files: one JSON summary per run plus gzipped JSON-lines of every request record."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from llmbench import __version__
from llmbench.types import RequestRecord

__all__ = ["RunResult", "environment", "load_records", "load_result", "probe_server"]

SCHEMA_VERSION = 1


@dataclass
class RunResult:
    name: str
    config: dict[str, Any]
    summary: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    engine_metrics: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    records: list[RequestRecord] = field(default_factory=list, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "config": self.config,
            "environment": self.environment,
            "server": self.server,
            "workload": self.workload,
            "summary": self.summary,
            "engine_metrics": self.engine_metrics,
            "gpu": self.gpu,
        }

    def save(self, path: str | Path, *, records: bool = True) -> Path:
        """Write ``<path>.json`` and, if asked, ``<path>.records.jsonl.gz``."""
        p = Path(path)
        if p.suffix == ".json":
            p = p.with_suffix("")
        p.parent.mkdir(parents=True, exist_ok=True)
        out = p.with_name(p.name + ".json")
        out.write_text(json.dumps(self.to_json(), indent=1, default=_default) + "\n")
        if records and self.records:
            with gzip.open(p.with_name(p.name + ".records.jsonl.gz"), "wt") as f:
                for r in self.records:
                    f.write(json.dumps(r.to_json(), separators=(",", ":")) + "\n")
        return out


def _default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def load_result(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def load_records(path: str | Path) -> list[RequestRecord]:
    p = Path(path)
    if p.suffix == ".json":
        p = p.with_name(p.stem + ".records.jsonl.gz")
    with gzip.open(p, "rt") as f:
        return [RequestRecord.from_json(json.loads(line)) for line in f if line.strip()]


def _git_sha() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def environment() -> dict[str, Any]:
    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except ImportError:
            return False

    return {
        "llmbench_version": __version__,
        "git_sha": _git_sha(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "uvloop": has("uvloop"),
        "orjson": has("orjson"),
        "gpus": _gpu_names(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }


async def probe_server(base_url: str) -> dict[str, Any]:
    """Collect what the server says about itself: models, engine version, server args."""
    base = base_url.rstrip("/")
    info: dict[str, Any] = {"base_url": base}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:

        async def get(path: str) -> Any:
            try:
                async with s.get(base + path) as r:
                    if r.status != 200:
                        return None
                    try:
                        return await r.json(content_type=None)
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        return (await r.text())[:2000]
            except (aiohttp.ClientError, OSError, TimeoutError):
                return None

        models = await get("/v1/models")
        if isinstance(models, dict):
            info["models"] = [m.get("id") for m in models.get("data", [])]
            data = models.get("data") or [{}]
            if data[0].get("max_model_len"):
                info["max_model_len"] = data[0]["max_model_len"]
        sgl = await get("/get_server_info") or await get("/server_info")
        if isinstance(sgl, dict) and ("version" in sgl or "model_path" in sgl):
            info["engine"] = "sglang"
            info["engine_version"] = sgl.get("version")
            keep = (
                "model_path",
                "tp_size",
                "mem_fraction_static",
                "max_running_requests",
                "chunked_prefill_size",
                "schedule_policy",
                "disable_radix_cache",
                "quantization",
                "kv_cache_dtype",
                "attention_backend",
                "cuda_graph_max_bs",
                "max_total_num_tokens",
                "dtype",
                "stream_interval",
                "page_size",
            )
            info["engine_args"] = {k: sgl[k] for k in keep if k in sgl}
        else:
            ver = await get("/version")
            if isinstance(ver, dict) and "version" in ver:
                info["engine"] = "vllm"
                info["engine_version"] = ver["version"]
    return info
