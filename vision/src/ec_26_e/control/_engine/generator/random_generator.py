"""随机合法拼图生成器的兼容入口。

旧六模板仍可通过 ``generation_mode="template"`` 或 ``topology=...`` 使用；
默认 ``mixed`` 按 60/25/5/10 混合 recursive/arrangement/template/adversarial。
"""

from __future__ import annotations

import numpy as np

from ..models import Piece
from .general_partition_generator import (
    PartitionQualityConfig,
    classify_partition,
    generate_general_partition,
)
from .validator import check_partition

TEMPLATE_TOPOLOGIES = (
    "two_polyline",
    "three_y",
    "four_fan",
    "guillotine",
    "nonparallel",
    "difficult_similar",
)
GENERATION_MODES = ("template", "recursive", "arrangement", "adversarial", "mixed")


def _cuts(total: float, count: int, rng: np.random.Generator) -> np.ndarray:
    slack = total - 20.0 * count
    if slack < 0:
        raise ValueError("目标宽度不足以满足每段 20 mm")
    weights = rng.dirichlet(np.linspace(1.1, 2.0, count))
    widths = 20.0 + slack * weights
    return np.concatenate(([0.0], np.cumsum(widths)))


def _guillotine(
    width: float,
    height: float,
    count: int,
    rng: np.random.Generator,
) -> list[Piece]:
    cuts = _cuts(width, count, rng)
    return [
        Piece(
            f"guillotine_piece_{index}",
            [
                (float(cuts[index]), 0.0),
                (float(cuts[index + 1]), 0.0),
                (float(cuts[index + 1]), height),
                (float(cuts[index]), height),
            ],
        )
        for index in range(count)
    ]


def _two_polyline(
    width: float,
    height: float,
    rng: np.random.Generator,
) -> list[Piece]:
    top = (float(rng.uniform(22.0, width - 22.0)), 0.0)
    bottom = (float(rng.uniform(22.0, width - 22.0)), height)
    center = (
        float(rng.uniform(0.38 * width, 0.62 * width)),
        float(rng.uniform(0.35 * height, 0.65 * height)),
    )
    return [
        Piece(
            "two_polyline_piece_0",
            [(0.0, 0.0), top, center, bottom, (0.0, height)],
        ),
        Piece(
            "two_polyline_piece_1",
            [top, (width, 0.0), (width, height), bottom, center],
        ),
    ]


def _three_y(
    width: float,
    height: float,
    rng: np.random.Generator,
) -> list[Piece]:
    top = (float(rng.uniform(0.35 * width, 0.65 * width)), 0.0)
    right = (width, float(rng.uniform(0.35 * height, 0.65 * height)))
    bottom = (float(rng.uniform(0.35 * width, 0.65 * width)), height)
    center = (
        float(rng.uniform(0.38 * width, 0.62 * width)),
        float(rng.uniform(0.35 * height, 0.65 * height)),
    )
    return [
        Piece("three_y_piece_0", [top, (width, 0.0), right, center]),
        Piece("three_y_piece_1", [right, (width, height), bottom, center]),
        Piece(
            "three_y_piece_2",
            [bottom, (0.0, height), (0.0, 0.0), top, center],
        ),
    ]


def _four_fan(
    width: float,
    height: float,
    rng: np.random.Generator,
    difficult: bool = False,
) -> list[Piece]:
    if difficult:
        top = (width / 2, 0.0)
        right = (width, height / 2)
        bottom = (width / 2, height)
        left = (0.0, height / 2)
        center = (width / 2, height / 2)
        prefix = "difficult_similar"
    else:
        top = (float(rng.uniform(0.30 * width, 0.70 * width)), 0.0)
        right = (width, float(rng.uniform(0.30 * height, 0.70 * height)))
        bottom = (float(rng.uniform(0.30 * width, 0.70 * width)), height)
        left = (0.0, float(rng.uniform(0.30 * height, 0.70 * height)))
        center = (
            float(rng.uniform(0.38 * width, 0.62 * width)),
            float(rng.uniform(0.35 * height, 0.65 * height)),
        )
        prefix = "four_fan"
    boundary = [top, right, bottom, left]
    corners = [
        (width, 0.0),
        (width, height),
        (0.0, height),
        (0.0, 0.0),
    ]
    return [
        Piece(
            f"{prefix}_piece_{index}",
            [boundary[index], corners[index], boundary[(index + 1) % 4], center],
        )
        for index in range(4)
    ]


def _template_partition(
    rng: np.random.Generator,
    count: int,
    target_width: float | None,
    target_height: float | None,
    topology: str | None,
    seed: int | None,
) -> tuple[list[Piece], float, float, dict]:
    last_errors: list[str] = []
    for _attempt in range(1000):
        width = float(
            target_width
            if target_width is not None
            else rng.uniform(max(90.0, 20.0 * count), 120.0)
        )
        height = float(
            target_height
            if target_height is not None
            else rng.uniform(50.0, 90.0)
        )
        compatible = {
            2: ["two_polyline", "guillotine"],
            3: ["three_y", "guillotine"],
            4: [
                "four_fan",
                "nonparallel",
                "difficult_similar",
                "guillotine",
            ],
        }[count]
        chosen = topology or str(rng.choice(compatible))
        if chosen not in compatible:
            raise ValueError(
                f"模板 {chosen!r} 与 piece_count={count} 不兼容"
            )
        if chosen == "two_polyline":
            pieces = _two_polyline(width, height, rng)
        elif chosen == "three_y":
            pieces = _three_y(width, height, rng)
        elif chosen in {"four_fan", "nonparallel"}:
            pieces = _four_fan(width, height, rng)
            if chosen == "nonparallel":
                for index, piece in enumerate(pieces):
                    piece.piece_id = f"nonparallel_piece_{index}"
        elif chosen == "difficult_similar":
            pieces = _four_fan(width, height, rng, difficult=True)
        else:
            pieces = _guillotine(width, height, count, rng)
        result = check_partition(pieces, width, height)
        if result.valid:
            metadata = classify_partition(pieces, width, height)
            metadata["generator_mode"] = "template"
            metadata["template_topology"] = chosen
            return pieces, width, height, metadata
        last_errors = result.errors
    raise RuntimeError(
        f"模板采样 1000 次失败：seed={seed}, piece_count={count}, "
        f"target={target_width}x{target_height}, 最后错误={last_errors}"
    )


def random_pieces(
    rng: np.random.Generator,
    piece_count: int | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    seed: int | None = None,
    topology: str | None = None,
    generation_mode: str = "mixed",
    quality_config: PartitionQualityConfig | None = None,
    return_metadata: bool = False,
):
    count = int(
        piece_count if piece_count is not None else rng.integers(2, 5)
    )
    if not 2 <= count <= 4:
        raise ValueError("piece_count 必须为 2～4")
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"generation_mode 必须为 {GENERATION_MODES}")
    if topology is not None:
        generation_mode = "template"
    selected_mode = generation_mode
    if generation_mode == "mixed":
        selected_mode = str(
            rng.choice(
                ["recursive", "arrangement", "template", "adversarial"],
                p=[0.60, 0.25, 0.05, 0.10],
            )
        )
    if selected_mode == "template":
        result = _template_partition(
            rng,
            count,
            target_width,
            target_height,
            topology,
            seed,
        )
    else:
        result = generate_general_partition(
            rng,
            count,
            target_width,
            target_height,
            selected_mode,
            quality_config,
        )
    return result if return_metadata else result[:3]
