"""全局 SE(2) 优化使用的纯 NumPy 光滑残差。

这里故意不导入 Shapely。精确 overlap/gap/outside 只在优化完成后评分。
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..transforms import transform_points


@dataclass(frozen=True)
class SmoothConstraintModel:
    records: tuple[dict, ...]
    resolved_matches: tuple[tuple[int, int, float, float, int, int, float, float], ...]
    outer_assignments: tuple[tuple[int, int, str], ...]
    confidence_weights: tuple[float, ...]

    def transformed(self, params: np.ndarray) -> list[np.ndarray]:
        return [
            transform_points(
                record["observed"],
                params[index * 3],
                params[index * 3 + 1:index * 3 + 3],
            )
            for index, record in enumerate(self.records)
        ]

    def residual(self, params: np.ndarray) -> np.ndarray:
        transformed = self.transformed(params)
        width, height = params[-2:]
        rows: list[float] = []
        for (
            first, left_edge, left_t0, left_t1,
            second, right_edge, right_t0, right_t1,
        ) in self.resolved_matches:
            left, right = transformed[first], transformed[second]
            la, lb = left[left_edge], left[(left_edge + 1) % len(left)]
            ra, rb = right[right_edge], right[(right_edge + 1) % len(right)]
            left_segment = (
                la + left_t0 * (lb - la),
                la + left_t1 * (lb - la),
            )
            right_segment = (
                ra + right_t0 * (rb - ra),
                ra + right_t1 * (rb - ra),
            )
            weight = np.sqrt(
                max(
                    min(
                        self.confidence_weights[first],
                        self.confidence_weights[second],
                    ),
                    1e-3,
                )
            )
            # 反向端点、共同支撑线和相反方向。
            rows.extend((weight * (left_segment[0] - right_segment[1])).tolist())
            rows.extend((weight * (left_segment[1] - right_segment[0])).tolist())
            left_direction = left_segment[1] - left_segment[0]
            right_direction = right_segment[1] - right_segment[0]
            normal = np.array([-left_direction[1], left_direction[0]])
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            rows.append(0.35 * weight * float(np.dot(
                right_segment[0] - left_segment[0], normal
            )))
            cross_value = (
                left_direction[0] * right_direction[1]
                - left_direction[1] * right_direction[0]
            )
            rows.append(
                0.15 * weight * float(cross_value)
                / max(float(np.linalg.norm(left_direction)), 1e-12)
            )
        for piece_index, edge_index, side in self.outer_assignments:
            points = transformed[piece_index]
            weight = np.sqrt(max(self.confidence_weights[piece_index], 1e-3))
            endpoints = (points[edge_index], points[(edge_index + 1) % len(points)])
            for point in endpoints:
                if side == "left":
                    value = point[0]
                elif side == "right":
                    value = point[0] - width
                elif side == "top":
                    value = point[1]
                else:
                    value = point[1] - height
                rows.append(weight * float(value))
        # 光滑的矩形合法性 hinge；不代替最终精确合法性检查。
        rows.extend((
            5.0 * max(0.0, 90.0 - width),
            5.0 * max(0.0, width - 120.0),
            5.0 * max(0.0, 50.0 - height),
            5.0 * max(0.0, height - 90.0),
        ))
        return np.asarray(rows, dtype=np.float64)
