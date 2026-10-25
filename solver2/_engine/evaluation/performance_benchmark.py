"""单案例重复 profiling 入口。"""

from __future__ import annotations

import cProfile
import json
from pathlib import Path
import pstats
from time import perf_counter
from typing import Any

from ..profiling import MemoryTracker, write_profile_report
from ..runtime_config import RuntimeConfig, resolve_runtime_config
from ..solver.solve import solve_observations


def run_profile_case(
    input_path: str | Path,
    output: str | Path,
    repeat: int = 10,
    runtime_config: RuntimeConfig | dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat 必须 >= 1")
    config = resolve_runtime_config(runtime_config)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    memory = MemoryTracker(enabled=True, track_python_allocations=True)
    memory.start()
    runs: list[dict[str, Any]] = []
    profiler = cProfile.Profile()
    for index in range(repeat):
        run_output = root / f"repeat_{index:03d}"
        started = perf_counter()
        if index == 0:
            profiler.enable()
        result = solve_observations(
            input_path,
            run_output,
            print_plan=False,
            runtime_config=config,
        )
        if index == 0:
            profiler.disable()
        runs.append({
            "elapsed_seconds": perf_counter() - started,
            "success": bool(result.get("success")),
            "search_nodes": int(result.get("search_nodes", 0)),
            "stage_times": result.get("performance_profile", {}).get(
                "stage_times", {}
            ),
        })
    profiler.dump_stats(str(root / "cprofile.prof"))
    with (root / "top_functions.txt").open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle)
        stats.strip_dirs().sort_stats("cumulative").print_stats(80)
    memory_summary = memory.stop()
    summary = write_profile_report(
        root, runs, memory_summary, config.as_dict()
    )
    summary["runs"] = runs
    (root / "profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
