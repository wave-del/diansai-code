"""生成 public/truth 隔离目录与 JSON、检查图。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, COORDINATE_SYSTEM, DIVIDER_Y_MM
from .fixed_generator import (
    FIXED_HEIGHT_MM,
    FIXED_WIDTH_MM,
    fixed_anchor_local_point,
    fixed_assembly_anchors,
    fixed_physical_contours,
    fixed_pieces,
)
from .noise import add_vertex_noise, noise_parameters
from .random_generator import random_pieces
from .shuffle import shuffle_and_place


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_case(
    mode: str,
    seed: int,
    noise_mm: float,
    output: str | Path,
    piece_count: int | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    topology: str | None = None,
    noise_model: str = "disk_bounded_vertex",
    generation_mode: str = "mixed",
    render: bool = True,
    surface_mode: str = "white",
) -> Path:
    if mode not in {"fixed", "random"}:
        raise ValueError("mode 必须为 fixed 或 random")
    if noise_mm < 0:
        raise ValueError("noise_mm 不得为负数")
    if surface_mode not in {"white", "patterned"}:
        raise ValueError("surface_mode must be white or patterned")
    rng = np.random.default_rng(seed)
    if mode == "fixed":
        pieces, width, height = fixed_pieces(), FIXED_WIDTH_MM, FIXED_HEIGHT_MM
        partition_metadata = None
    else:
        pieces, width, height, partition_metadata = random_pieces(
            rng,
            piece_count,
            target_width,
            target_height,
            seed,
            topology,
            generation_mode=generation_mode,
            return_metadata=True,
        )
    shuffled = shuffle_and_place(pieces, rng)
    root = Path(output)
    public_dir, truth_dir = root / "public", root / "truth"
    public_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    public_pieces: list[dict[str, Any]] = []
    truth_pieces: list[dict[str, Any]] = []
    for item in shuffled:
        observed, noise = add_vertex_noise(item.clean_vertices, noise_mm, rng, noise_model)
        public_pieces.append(
            {
                "id": item.public_id,
                "vertices_mm": observed,
                "vertex_order": "boundary_order",
                "appearance_features": None,
            }
        )
        truth_pieces.append(
            {
                "public_id": item.public_id,
                "source_id": item.source_id,
                "target_vertices_local_mm": item.source_vertices,
                "clean_shuffled_vertices_mm": item.clean_vertices,
                "observed_vertices_mm": observed,
                "vertex_noise_mm": noise,
                "shuffle_rotation_deg_clockwise": item.shuffle_rotation_deg,
                "shuffle_translation_a4_mm": item.shuffle_translation,
            }
        )
    observations = {
        "schema_version": "1.0",
        "coordinate_system": COORDINATE_SYSTEM,
        "page": {
            "width_mm": A4_WIDTH_MM,
            "height_mm": A4_HEIGHT_MM,
            "divider_y_mm": DIVIDER_Y_MM,
        },
        "mode": mode,
        "surface_mode": surface_mode,
        "noise_bound_mm": float(noise_mm),
        "measurement_error_radius_mm": float(noise_mm),
        "noise_model": noise_model,
        "noise_level": float(noise_mm),
        "noise_parameters": noise_parameters(noise_model, noise_mm),
        "pieces": public_pieces,
    }
    truth = {
        "schema_version": "1.0",
        "seed": int(seed),
        "mode": mode,
        "surface_mode": surface_mode,
        "noise_model": noise_model,
        "noise_level": float(noise_mm),
        "noise_parameters": noise_parameters(noise_model, noise_mm),
        "target_rectangle": {"width_mm": width, "height_mm": height},
        "pieces": truth_pieces,
    }
    if mode == "fixed":
        truth["fixed_template_metadata"] = {
            "physical_contours": [
                {
                    "piece_template_id": item.piece_template_id,
                    "corner_vertices": [
                        list(point) for point in item.corner_vertices
                    ],
                    "physical_edge_count": len(item.physical_edges),
                }
                for item in fixed_physical_contours()
            ],
            "virtual_assembly_anchors": [
                {
                    "anchor_id": anchor.anchor_id,
                    "host_piece_template_id": (
                        anchor.host_piece_template_id
                    ),
                    "host_physical_edge_index": (
                        anchor.host_physical_edge_index
                    ),
                    "edge_parameter_t": anchor.edge_parameter_t,
                    "local_point_mm": list(
                        fixed_anchor_local_point(anchor)
                    ),
                    "incident_piece_template_ids": list(
                        anchor.incident_piece_template_ids
                    ),
                    "semantic_role": anchor.semantic_role,
                }
                for anchor in fixed_assembly_anchors()
            ],
        }
    if mode == "random":
        truth["generator_mode"] = partition_metadata["generator_mode"]
        truth["topology"] = partition_metadata.get(
            "template_topology", partition_metadata["generator_mode"]
        )
        truth["partition_metadata"] = partition_metadata
    _dump(public_dir / "observations.json", observations)
    _dump(truth_dir / "ground_truth.json", truth)

    if render:
        from ..visualization.render_input import render_generated_case
        render_generated_case(root, observations, truth)
    return root
