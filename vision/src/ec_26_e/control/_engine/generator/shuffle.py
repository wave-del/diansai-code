"""将目标碎片独立旋转并拒绝采样到 A4 上半区。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, box

from ..models import Piece
from ..transforms import transform_points


@dataclass
class ShuffledPiece:
    public_id: str
    source_id: str
    clean_vertices: list[tuple[float, float]]
    source_vertices: list[tuple[float, float]]
    shuffle_rotation_deg: float
    shuffle_translation: tuple[float, float]


def shuffle_and_place(pieces: list[Piece], rng: np.random.Generator) -> list[ShuffledPiece]:
    page_region = box(5.0, 5.0, 205.0, 143.5)
    # 贪心散放可能把最后一片堵死；整套失败时重新选择片序和前序姿态，
    # 不能把可生成的拼图误报成 solver 失败。
    for _layout_attempt in range(40):
        placed: list[Polygon] = []
        interim: list[ShuffledPiece] = []
        source_order = list(range(len(pieces)))
        rng.shuffle(source_order)
        layout_valid = True
        for source_index in source_order:
            piece = pieces[source_index]
            success = False
            for _ in range(5000):
                angle = float(rng.uniform(-180.0, 180.0))
                rotated = transform_points(piece.vertices, angle)
                minx, miny = rotated.min(axis=0)
                maxx, maxy = rotated.max(axis=0)
                if maxx - minx > 200.0 or maxy - miny > 138.5:
                    continue
                dx = float(rng.uniform(5.0 - minx, 205.0 - maxx))
                dy = float(rng.uniform(5.0 - miny, 143.5 - maxy))
                coords = transform_points(piece.vertices, angle, (dx, dy))
                poly = Polygon(coords)
                if not page_region.covers(poly):
                    continue
                if any(poly.distance(other) < 2.0 or poly.intersects(other) for other in placed):
                    continue
                public_id = f"piece_{len(interim)}"
                interim.append(
                    ShuffledPiece(
                        public_id,
                        piece.piece_id,
                        [(float(x), float(y)) for x, y in coords],
                        list(piece.vertices),
                        angle,
                        (dx, dy),
                    )
                )
                placed.append(poly)
                success = True
                break
            if not success:
                layout_valid = False
                break
        if layout_valid:
            return interim
    raise RuntimeError("40 次整套重试后仍无法将碎片无重叠摆放到 A4 上半区")
