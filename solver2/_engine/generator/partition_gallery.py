"""批量生成并验证随机铁皮分割效果图，不运行散放或求解。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch, Rectangle
import numpy as np

from .random_generator import random_pieces
from .validator import check_partition

TOPOLOGIES = [
    ("two_polyline", 2),
    ("three_y", 3),
    ("four_fan", 4),
    ("guillotine", None),
    ("nonparallel", 4),
    ("difficult_similar", 4),
]


def _render_partition(
    path: Path,
    pieces,
    width: float,
    height: float,
    topology: str,
    seed: int,
) -> None:
    colors = plt.get_cmap("Set2").colors
    fig_width = 6.0
    fig_height = max(3.5, fig_width * height / width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for index, piece in enumerate(pieces):
        vertices = np.asarray(piece.vertices, dtype=float)
        ax.add_patch(
            PolygonPatch(
                vertices,
                closed=True,
                facecolor=colors[index % len(colors)],
                edgecolor="black",
                linewidth=2.0,
                alpha=0.88,
            )
        )
        center = vertices.mean(axis=0)
        ax.text(
            center[0], center[1], f"P{index}",
            ha="center", va="center", fontsize=11, weight="bold",
        )
    ax.add_patch(
        Rectangle((0.0, 0.0), width, height, fill=False, edgecolor="black", linewidth=2.5)
    )
    margin = 0.04 * max(width, height)
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(height + margin, -margin)
    ax.set_aspect("equal")
    ax.set_title(f"{topology} | seed={seed} | {width:.1f} x {height:.1f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(False)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_partition_gallery(
    count: int,
    seed: int,
    output: str | Path,
) -> dict:
    if count <= 0:
        raise ValueError("count 必须大于 0")
    root = Path(output)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    topology_counts: Counter[str] = Counter()
    piece_counts: Counter[int] = Counter()
    cases = []
    for index in range(count):
        case_seed = seed + index
        topology, fixed_count = TOPOLOGIES[index % len(TOPOLOGIES)]
        piece_count = fixed_count
        if topology == "guillotine":
            piece_count = 2 + ((index // len(TOPOLOGIES)) % 3)
        rng = np.random.default_rng(case_seed)
        pieces, width, height = random_pieces(
            rng,
            piece_count=piece_count,
            seed=case_seed,
            topology=topology,
        )
        validation = check_partition(pieces, width, height)
        if not validation.valid:
            raise AssertionError(
                f"seed={case_seed} topology={topology} 验证失败: {validation.errors}"
            )
        filename = f"{index + 1:04d}_{topology}_seed_{case_seed:06d}.png"
        _render_partition(images / filename, pieces, width, height, topology, case_seed)
        topology_counts[topology] += 1
        piece_counts[len(pieces)] += 1
        cases.append(
            {
                "index": index + 1,
                "seed": case_seed,
                "topology": topology,
                "piece_count": len(pieces),
                "width_mm": width,
                "height_mm": height,
                "image": f"images/{filename}",
                "valid": True,
                "overlap_area_mm2": validation.overlap_area_mm2,
                "union_area_error_mm2": abs(validation.union_area_mm2 - width * height),
            }
        )
    summary = {
        "count": count,
        "seed": seed,
        "all_valid": True,
        "topology_counts": dict(sorted(topology_counts.items())),
        "piece_count_distribution": {
            str(key): value for key, value in sorted(piece_counts.items())
        },
        "cases": cases,
    }
    (root / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

