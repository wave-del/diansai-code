"""Single atomic output boundary for official solver artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from ..planning.motion_plan import write_motion_csv


def atomic_write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_failure_artifacts(
    output: Path,
    result: dict[str, Any],
    search_diagnostics: dict[str, Any],
) -> None:
    """Write a non-executable failure and a header-only motion plan."""
    atomic_write_json(output / "solution.json", result)
    atomic_write_json(output / "candidate_scores.json", [])
    atomic_write_json(
        output / "interface_diagnostics.json",
        search_diagnostics,
    )
    write_motion_csv(output / "motion_plan.csv", [], 0.0, False)


def write_solution_artifacts(
    output: Path,
    result: dict[str, Any],
    *,
    moves: list[Any],
    confidence: float,
    executable: bool,
    diagnostics_level: str,
    candidate_rows: list[dict[str, Any]],
    search_diagnostics: dict[str, Any],
    block_reasons: list[str],
    timer: Any | None = None,
    solver_started: float | None = None,
) -> None:
    """Write diagnostics and exactly one final official solution snapshot."""
    output_started = perf_counter()
    if not executable and diagnostics_level != "off":
        atomic_write_json(
            output / "rejected_motion_diagnostics.json",
            {
                "executable": False,
                "candidate_id": search_diagnostics.get(
                    "selected_candidate_id"
                ),
                "variant_id": search_diagnostics.get(
                    "selected_variant_id"
                ),
                "block_reasons": sorted(set(block_reasons)),
                "moves": result["moves"],
            },
        )
    if diagnostics_level == "full":
        atomic_write_json(
            output / "candidate_scores.json", candidate_rows
        )
        atomic_write_json(
            output / "interface_diagnostics.json",
            search_diagnostics,
        )
    elif diagnostics_level == "summary":
        selected_rows = [
            row for row in candidate_rows
            if row.get("selected_after_optimization")
        ]
        summary_rows = candidate_rows[:10]
        for row in selected_rows:
            if row not in summary_rows:
                summary_rows.append(row)
        atomic_write_json(
            output / "candidate_scores.json", summary_rows
        )
        atomic_write_json(
            output / "interface_diagnostics.json",
            {
                key: value
                for key, value in search_diagnostics.items()
                if key != "interface_candidates"
            },
        )
    write_motion_csv(
        output / "motion_plan.csv",
        moves if executable else [],
        confidence,
        executable,
    )
    if timer is not None:
        timer.add("json_output", perf_counter() - output_started)
        if solver_started is not None:
            timer.add("total", perf_counter() - solver_started)
        result["performance_profile"]["stage_times"] = timer.snapshot()
    if solver_started is not None:
        result["elapsed_seconds"] = perf_counter() - solver_started
    atomic_write_json(output / "solution.json", result)
