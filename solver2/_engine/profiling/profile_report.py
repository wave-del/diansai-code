"""profile-case 与 benchmark 共用的报告输出。"""

from __future__ import annotations

import csv
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any

import numpy
import scipy
import shapely


def portable_profile(runtime_config: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    blas = {
        name: os.environ.get(name)
        for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "shapely_version": shapely.__version__,
        "cpu_count": os.cpu_count(),
        "worker_count": runtime_config.get("worker_count", 1),
        "blas_threads": blas,
        "benchmark_percentiles": performance,
        "peak_memory_mb": performance.get("peak_memory_mb"),
    }


def write_profile_report(
    output: str | Path,
    runs: list[dict[str, Any]],
    memory: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    stage_names = sorted(
        {
            name
            for run in runs
            for name in run.get("stage_times", {})
        }
    )
    stage_rows = []
    for index, run in enumerate(runs):
        for name in stage_names:
            value = run.get("stage_times", {}).get(name, {})
            stage_rows.append({
                "repeat": index,
                "stage": name,
                "seconds": float(value.get("seconds", 0.0)),
                "calls": int(value.get("calls", 0)),
            })
    with (root / "stage_times.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repeat", "stage", "seconds", "calls"])
        writer.writeheader()
        writer.writerows(stage_rows)
    totals = [float(run.get("elapsed_seconds", 0.0)) for run in runs]
    summary = {
        "repeat": len(runs),
        "elapsed_seconds": totals,
        "mean_seconds": sum(totals) / len(totals) if totals else 0.0,
        "minimum_seconds": min(totals, default=0.0),
        "maximum_seconds": max(totals, default=0.0),
        "stage_mean_seconds": {
            name: (
                sum(
                    float(run.get("stage_times", {}).get(name, {}).get("seconds", 0.0))
                    for run in runs
                ) / len(runs)
                if runs else 0.0
            )
            for name in stage_names
        },
        "runtime_config": runtime_config,
    }
    (root / "profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "memory_summary.json").write_text(
        json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "portable_profile.json").write_text(
        json.dumps(
            portable_profile(runtime_config, {**summary, **memory}),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return summary
