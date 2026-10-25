"""限制拼图来源区，避免右侧白色执行器被误认为碎片。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_SOURCE_MAX_X_RATIO = 0.52


@dataclass(frozen=True, slots=True)
class SourceRegionFilterResult:
    accepted: tuple[Any, ...]
    rejected: tuple[Any, ...]


def filter_left_source_pieces(
    pieces: tuple[Any, ...],
    *,
    max_x_ratio: float = DEFAULT_SOURCE_MAX_X_RATIO,
) -> SourceRegionFilterResult:
    """只接收完整落在校正 A4 左侧来源区的纸片。

    以全部顶点而不是质心作判定：跨过边界的物体也会被拒绝，防止右侧
    执行器、吸头或其阴影只露出一部分时混入稳定器。默认阈值 0.52，
    即允许碎片跨过中线 2%，同时保留右侧执行器过滤区。
    """

    if not 0.0 < max_x_ratio <= 0.6:
        raise ValueError("来源区最大 X 比例必须位于 (0, 0.6]")
    accepted: list[Any] = []
    rejected: list[Any] = []
    for piece in pieces:
        corners = getattr(piece, "corners_relative", None)
        if not isinstance(corners, tuple) or len(corners) < 3:
            rejected.append(piece)
            continue
        try:
            max_x = max(float(point[0]) for point in corners)
        except (TypeError, ValueError, IndexError):
            rejected.append(piece)
            continue
        if max_x <= max_x_ratio:
            accepted.append(piece)
        else:
            rejected.append(piece)
    return SourceRegionFilterResult(tuple(accepted), tuple(rejected))

