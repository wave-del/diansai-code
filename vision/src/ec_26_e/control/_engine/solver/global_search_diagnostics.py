"""全局接口搜索的结构化诊断和调试图输出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .placed_interface_discovery import discover_interfaces_in_placed_state
from .preprocess import ProcessedPiece
from .search import RandomCandidate
from .observation_uncertainty import (
    DEFAULT_HARD_ERROR_RADIUS_MM,
    ObservationUncertaintyProfile,
)


def _physical_edges(pieces: list[ProcessedPiece]) -> list[dict[str, Any]]:
    rows = []
    for piece in pieces:
        for edge_index, length in enumerate(piece.edge_lengths):
            rows.append(
                {
                    "piece_id": piece.piece_id,
                    "edge_index": edge_index,
                    "start_point_local": piece.vertices[edge_index].tolist(),
                    "end_point_local": piece.vertices[
                        (edge_index + 1) % len(piece.vertices)
                    ].tolist(),
                    "length_mm": float(length),
                }
            )
    return rows


def _unique_intervals(search: dict[str, Any]) -> list[dict[str, Any]]:
    unique: dict[tuple, dict[str, Any]] = {}
    for candidate in search.get("interface_candidates", []):
        for segment in candidate.get("segments_a", []) + candidate.get(
            "segments_b", []
        ):
            key = (
                segment["piece_id"],
                int(segment["edge_index"]),
                round(float(segment["t_start"]), 9),
                round(float(segment["t_end"]), 9),
            )
            unique[key] = segment
    return list(unique.values())


def write_global_search_diagnostics(
    output: Path,
    pieces: list[ProcessedPiece],
    search: dict[str, Any],
    best: RandomCandidate | None,
    noise_mm: float = 0.0,
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
) -> None:
    intervals = _unique_intervals(search)
    interfaces = search.get("interface_candidates", [])
    junction_sources: dict[str, set[tuple[str, int, float]]] = {}
    for interval in intervals:
        for side, parameter in (
            ("start_sources", interval["t_start"]),
            ("end_sources", interval["t_end"]),
        ):
            for source in interval.get(side, []):
                if source["source_type"] == "physical_vertex":
                    continue
                junction_sources.setdefault(source["source_id"], set()).add(
                    (
                        interval["piece_id"],
                        int(interval["edge_index"]),
                        float(parameter),
                    )
                )
    junctions = [
        {
            "junction_id": f"junction_{index:05d}",
            "junction_type": (
                "MULTI_T_JUNCTION" if len(entries) >= 3 else "T_JUNCTION"
            ),
            "source_id": source_id,
            "local_parameters": [list(item) for item in sorted(entries)],
        }
        for index, (source_id, entries) in enumerate(
            sorted(junction_sources.items()), 1
        )
    ]
    pose_hypotheses = [
        {
            "interface_id": item["interface_id"],
            "piece_a": item["piece_a"],
            "piece_b": item["piece_b"],
            "constraint_rank": item.get("constraint_rank", 2),
            "remaining_dof": item.get("remaining_dof", 0),
            "deterministic": item.get("determines_full_pose", True),
            "residual": item.get("endpoint_rms", 0.0),
            "source_constraints": item.get("junction_ids", []),
        }
        for item in interfaces
    ]
    placed_discovery: dict[str, Any] = {}
    if best is not None:
        profile = uncertainty_profile or ObservationUncertaintyProfile(
            DEFAULT_HARD_ERROR_RADIUS_MM,
            None,
            None,
            None,
            None,
            "fallback_from_candidate_statistics",
            "medium",
            False,
        )
        placed_discovery = discover_interfaces_in_placed_state(
            {
                pose.piece_id: pose.target_vertices_local
                for pose in best.poses
            },
            tolerance_mm=max(
                1.0,
                profile.hard_error_radius_mm
                + float(profile.typical_sigma_mm or 0.0),
            ),
        )
    result = {
        "physical_edges": _physical_edges(pieces),
        "atomic_split_points": [
            {
                "piece_id": interval["piece_id"],
                "edge_index": interval["edge_index"],
                "t_start": interval["t_start"],
                "t_end": interval["t_end"],
                "start_sources": interval.get("start_sources", []),
                "end_sources": interval.get("end_sources", []),
            }
            for interval in intervals
        ],
        "atomic_intervals": intervals,
        "junction_hypotheses": junctions,
        "pose_hypotheses": pose_hypotheses,
        "deterministic_interface_count": sum(
            bool(item.get("determines_full_pose", True))
            for item in interfaces
        ),
        "underconstrained_interface_count": sum(
            not bool(item.get("determines_full_pose", True))
            for item in interfaces
        ),
        "partial_to_partial_candidates": [
            item
            for item in interfaces
            if item.get("interface_type") == "partial_to_partial"
        ],
        "middle_interval_candidates": [
            interval
            for interval in intervals
            if float(interval["t_start"]) > 1e-9
            and float(interval["t_end"]) < 1.0 - 1e-9
        ],
        "search_statistics": {
            key: search.get(key)
            for key in (
                "generated_states",
                "expanded_states",
                "duplicate_pruned",
                "rectangle_pruned",
                "overlap_pruned",
                "cycle_pruned",
                "interface_conflict_pruned",
                "budget_exhausted",
                "search_nodes",
                "legal_complete_candidate_count",
            )
        },
        "search_termination_reason": (
            "complete_candidate_found"
            if best is not None
            else search.get("failure_type", "no_complete_search_candidate")
        ),
        "closest_complete_state": {
            "placed_piece_count": len(best.poses) if best is not None else 0,
            "rectangle_feasible": bool(
                best is not None and best.geometry_valid
            ),
            "missing_constraints": [] if best is not None else ["complete_pose"],
        },
        "placed_state_discovery": placed_discovery,
        "correct_center_interface_represented_without_truth": bool(
            any(
                item.get("interface_type") == "partial_to_partial"
                for item in interfaces
            )
            or any(
                item.get("junction_type") == "THREE_WAY_JUNCTION"
                for item in placed_discovery.get("junctions", [])
            )
            or any(
                len(intervals) > 1
                for intervals in placed_discovery.get(
                    "occupied_intervals", {}
                ).values()
            )
        ),
    }
    (output / "global_search_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _render_debug(output / "global_search_debug.png", pieces, best)


def _render_debug(
    path: Path,
    pieces: list[ProcessedPiece],
    best: RandomCandidate | None,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors
    for piece_index, piece in enumerate(pieces):
        vertices = piece.vertices
        closed = np.vstack((vertices, vertices[0]))
        axes[0].plot(
            closed[:, 0],
            closed[:, 1],
            color=colors[piece_index],
            linewidth=2,
        )
        for edge_index in range(len(vertices)):
            midpoint = (
                vertices[edge_index]
                + vertices[(edge_index + 1) % len(vertices)]
            ) / 2.0
            axes[0].text(
                midpoint[0],
                midpoint[1],
                f"{piece.piece_id}:e{edge_index}",
                fontsize=7,
            )
    axes[0].set_title("Observed physical edges")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    if best is not None:
        for piece_index, pose in enumerate(best.poses):
            vertices = pose.target_vertices_local
            closed = np.vstack((vertices, vertices[0]))
            axes[1].fill(
                closed[:, 0],
                closed[:, 1],
                color=colors[piece_index],
                alpha=0.25,
            )
            axes[1].plot(
                closed[:, 0],
                closed[:, 1],
                color=colors[piece_index],
                linewidth=2,
                label=pose.piece_id,
            )
        for interface in best.interfaces:
            first = next(
                pose
                for pose in best.poses
                if pose.piece_id == interface.first_piece
            )
            start = first.target_vertices_local[
                interface.first_segment.edge_index
            ]
            end = first.target_vertices_local[
                (interface.first_segment.edge_index + 1)
                % len(first.target_vertices_local)
            ]
            a = start + interface.first_segment.t_start * (end - start)
            b = start + interface.first_segment.t_end * (end - start)
            axes[1].plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="black",
                linewidth=4,
                alpha=0.55,
            )
            axes[1].text(
                (a[0] + b[0]) / 2,
                (a[1] + b[1]) / 2,
                (
                    f"t={interface.first_segment.t_start:.2f}"
                    f"..{interface.first_segment.t_end:.2f}"
                ),
                fontsize=7,
            )
        axes[1].legend(fontsize=8)
    axes[1].set_title("Best placed state and confirmed interfaces")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
