"""Small deterministic smoke matrix for patterned-candidate plumbing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..appearance.ambiguity_detector import detect_geometry_ambiguity
from ..appearance.candidate_resolver import resolve_candidates_by_appearance
from ..appearance.models import CandidateGeometryDescriptor
from ..generator.export import generate_case
from ..runtime_config import RuntimeConfig
from ..solver.solve import solve_observations


def _descriptor(
    candidate_id: str,
    *,
    swapped: bool = False,
    rotated: bool = False,
    safe: bool = True,
) -> CandidateGeometryDescriptor:
    square = np.asarray(
        [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]
    )
    left, right = square.copy(), square + np.asarray([20.0, 0.0])
    return CandidateGeometryDescriptor(
        candidate_id=candidate_id,
        piece_vertices={"piece_0": square, "piece_1": square},
        target_vertices={
            "piece_0": right if swapped else left,
            "piece_1": left if swapped else right,
        },
        rotations_deg={
            "piece_0": 180.0 if swapped or rotated else 0.0,
            "piece_1": 180.0 if swapped or rotated else 0.0,
        },
        adjacency_pairs={frozenset(("piece_0", "piece_1"))},
        interface_ids=("interface_b" if swapped else "interface_a",),
        width_mm=40.0,
        height_mm=20.0,
        geometry_score=0.1,
        safe=safe,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _motion_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, len(list(csv.reader(handle))) - 1)


def run_appearance_smoke(output: str | Path) -> dict[str, Any]:
    """Run 12 light cases plus two fixed solver boundary checks."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        execution_profile="benchmark", diagnostics_level="summary"
    )
    unique = [_descriptor("candidate_001")]
    ambiguous = [
        _descriptor("candidate_001"),
        _descriptor("candidate_002", swapped=True),
    ]
    unsafe_alternative = [
        _descriptor("candidate_001"),
        _descriptor("candidate_002", swapped=True, safe=False),
    ]
    specifications = [
        ("unique_geometry", unique, None),
        ("similar_piece_swap", ambiguous, None),
        ("global_180_rotation", ambiguous, None),
        ("unsafe_alternative_filtered", unsafe_alternative, None),
        (
            "clear_appearance_winner",
            ambiguous,
            {"candidate_scores": {
                "candidate_001": 0.92, "candidate_002": 0.61,
            }},
        ),
        (
            "insufficient_appearance_margin",
            ambiguous,
            {"candidate_scores": {
                "candidate_001": 0.81, "candidate_002": 0.76,
            }},
        ),
    ]
    rows: list[dict[str, Any]] = []
    unresolved_names: list[str] = []
    for surface_mode in ("white", "patterned"):
        for name, candidates, scores in specifications:
            geometry = detect_geometry_ambiguity(
                candidates, config.geometry_ambiguity_threshold
            )
            if surface_mode == "white":
                appearance = None
                executable = True
                block_reason = None
            else:
                pool = (
                    candidates
                    if geometry.detected
                    else [candidate for candidate in candidates if candidate.safe][:1]
                )
                appearance = resolve_candidates_by_appearance(
                    pool, scores, config
                )
                executable = appearance.resolved
                block_reason = (
                    None if executable else "appearance_ambiguity"
                )
                if not executable:
                    unresolved_names.append(f"{surface_mode}_{name}")
            rows.append({
                "case": name,
                "surface_mode": surface_mode,
                "safe_candidate_count": sum(
                    candidate.safe for candidate in candidates
                ),
                "geometry_ambiguity_detected": geometry.detected,
                "ambiguity_type": geometry.ambiguity_type or "",
                "appearance_completed": (
                    appearance.completed if appearance else False
                ),
                "appearance_resolved": (
                    appearance.resolved if appearance else False
                ),
                "winner_candidate_id": (
                    appearance.winner_candidate_id if appearance else ""
                ),
                "executable": executable,
                "block_reason": block_reason or "",
            })

    # Exercise the official motion-output boundary twice.
    case = root / "fixed_patterned_boundary"
    generate_case(
        "fixed", 24002, 0.0, case, surface_mode="patterned"
    )
    observations_path = case / "public" / "observations.json"
    missing = solve_observations(
        observations_path,
        case / "unresolved_solution",
        runtime_config=config,
        print_plan=False,
    )
    observations = json.loads(observations_path.read_text("utf-8"))
    observations["appearance_observation"] = {
        "candidate_scores": {"fixed_candidate_001": 0.95}
    }
    scored_path = case / "public" / "observations_scored.json"
    _write_json(scored_path, observations)
    resolved = solve_observations(
        scored_path,
        case / "resolved_solution",
        runtime_config=config,
        print_plan=False,
    )
    random_case = root / "random_patterned_ambiguous"
    generate_case(
        "random",
        24001,
        0.0,
        random_case,
        piece_count=2,
        generation_mode="mixed",
        surface_mode="patterned",
    )
    random_observations_path = (
        random_case / "public" / "observations.json"
    )
    random_unresolved = solve_observations(
        random_observations_path,
        random_case / "unresolved_solution",
        runtime_config=config,
        print_plan=False,
    )
    ambiguity_groups = random_unresolved["geometry_ambiguity"][
        "candidate_groups"
    ]
    candidate_ids = (
        ambiguity_groups[0]
        if ambiguity_groups
        else [
            item["candidate_id"]
            for item in random_unresolved["recovery"].get(
                "safe_candidate_pool", []
            )
        ][:2]
    )
    random_observations = json.loads(
        random_observations_path.read_text("utf-8")
    )
    random_observations["appearance_observation"] = {
        "candidate_scores": {
            candidate_id: 0.95 - 0.30 * index
            for index, candidate_id in enumerate(candidate_ids)
        }
    }
    random_scored_path = (
        random_case / "public" / "observations_scored.json"
    )
    _write_json(random_scored_path, random_observations)
    random_resolved = solve_observations(
        random_scored_path,
        random_case / "resolved_solution",
        runtime_config=config,
        print_plan=False,
    )
    with (root / "cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved_lines = [
        "# Unresolved patterned smoke cases",
        "",
        *[f"- `{name}`" for name in unresolved_names],
        "",
        "The official unresolved motion CSV is header-only:",
        "",
        (
            "- [motion_plan.csv]"
            "(fixed_patterned_boundary/unresolved_solution/motion_plan.csv)"
        ),
    ]
    (root / "unresolved_cases.md").write_text(
        "\n".join(unresolved_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "case_rows": len(rows),
        "solver_invocations": 4,
        "geometry_ambiguity_cases": sum(
            row["geometry_ambiguity_detected"] for row in rows
        ),
        "unresolved_patterned_cases": len(unresolved_names),
        "fixed_missing_appearance_executable": missing["executable"],
        "fixed_missing_appearance_motion_rows": _motion_data_rows(
            case / "unresolved_solution" / "motion_plan.csv"
        ),
        "fixed_scored_appearance_resolved": resolved[
            "appearance_resolution"
        ]["resolved"],
        "fixed_scored_block_reasons": resolved[
            "execution_block_reasons"
        ],
        "random_geometry_ambiguity_detected": random_unresolved[
            "geometry_ambiguity"
        ]["detected"],
        "random_missing_appearance_motion_rows": _motion_data_rows(
            random_case / "unresolved_solution" / "motion_plan.csv"
        ),
        "random_scored_appearance_resolved": random_resolved[
            "appearance_resolution"
        ]["resolved"],
        "random_scored_winner_candidate_id": random_resolved[
            "appearance_resolution"
        ]["winner_candidate_id"],
        "random_selected_candidate_id": random_resolved[
            "recovery"
        ]["selected_attempt"],
        "random_scored_block_reasons": random_resolved[
            "execution_block_reasons"
        ],
    }
    _write_json(root / "summary.json", summary)
    return summary
