"""候选刚体位姿的全局 soft_l1 联合优化。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from .exact_candidate_scoring import (
    exact_geometry_metrics,
    exact_geometry_objective,
)
from .smooth_residuals import SmoothConstraintModel
from ..transforms import normalize_angle, transform_points


def robust_least_squares(fun, x0: np.ndarray):
    return least_squares(fun, x0, loss="soft_l1")


def optimize_template_poses(records: list[dict]) -> tuple[list[dict], dict]:
    """联合优化全部碎片 SE(2)，模板公共顶点自然落到同一目标坐标。

    每片仍只有 theta/tx/ty，不允许缩放、剪切或镜像。
    """
    x0 = np.asarray(
        [
            value
            for record in records
            for value in (record["angle"], record["translation"][0], record["translation"][1])
        ],
        dtype=float,
    )

    def residual(params: np.ndarray) -> np.ndarray:
        rows = []
        for index, record in enumerate(records):
            angle, tx, ty = params[index * 3 : index * 3 + 3]
            structural_observed = np.asarray(
                record.get("structural_observed_corners", record["observed"]),
                dtype=float,
            )
            template = np.asarray(
                record.get("ideal_template_target", record["template_target"]),
                dtype=float,
            )
            fitted_structural = transform_points(
                structural_observed, angle, (tx, ty)
            )
            rows.extend((fitted_structural - template).ravel())
            fitted_raw = transform_points(
                record["observed"], angle, (tx, ty)
            )
            for edge_index, group in enumerate(
                record.get("edge_sample_groups", ())
            ):
                start = template[edge_index]
                end = template[(edge_index + 1) % len(template)]
                line = end - start
                length = max(float(np.linalg.norm(line)), 1e-9)
                direction = line / length
                normal = np.asarray([-direction[1], direction[0]])
                sample_distances = (
                    fitted_raw[list(group)] - start
                ) @ normal
                rows.extend(sample_distances.tolist())
                observed_edge = (
                    fitted_structural[(edge_index + 1) % len(template)]
                    - fitted_structural[edge_index]
                )
                observed_length = max(
                    float(np.linalg.norm(observed_edge)), 1e-9
                )
                observed_direction = observed_edge / observed_length
                rows.append(
                    0.35 * float(
                        observed_direction[0] * direction[1]
                        - observed_direction[1] * direction[0]
                    ) * length
                )
                rows.append(0.10 * (observed_length - length))
            for edge in record.get("edges", []):
                samples = edge.get("sample_points_mm")
                if not samples:
                    continue
                start = int(edge["start_vertex_index"])
                end = int(edge["end_vertex_index"])
                line = template[end] - template[start]
                line /= max(float(np.linalg.norm(line)), 1e-9)
                normal = np.asarray([-line[1], line[0]])
                transformed_samples = transform_points(samples, angle, (tx, ty))
                confidence = float(edge.get("confidence", 1.0))
                fit_rms = max(float(edge.get("fit_rms_mm", 1.0)), 0.1)
                distances = (transformed_samples - template[start]) @ normal
                rows.extend((distances * np.sqrt(confidence) / fit_rms).tolist())
        return np.asarray(rows, dtype=float)

    before = float(np.sqrt(np.mean(residual(x0) ** 2)))
    result = least_squares(residual, x0, loss="soft_l1", f_scale=2.0, max_nfev=1000)
    after = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    accepted = bool(result.success and after <= before + 1e-9)
    chosen = result.x if accepted else x0
    optimized = []
    for index, record in enumerate(records):
        angle, tx, ty = chosen[index * 3 : index * 3 + 3]
        updated = dict(record)
        updated["angle"] = normalize_angle(float(angle))
        updated["translation"] = np.asarray([tx, ty], dtype=float)
        updated["target"] = transform_points(
            record["observed"],
            updated["angle"],
            updated["translation"],
        )
        updated["structural_target"] = transform_points(
            record.get("structural_observed_corners", record["observed"]),
            updated["angle"],
            updated["translation"],
        )
        optimized.append(updated)
    return optimized, {
        "optimization_success": bool(result.success),
        "optimization_accepted": accepted,
        "before_rms_mm": before,
        "after_rms_mm": after,
        "improvement_mm": before - after,
        "nfev": int(result.nfev),
        "residual_semantics": (
            "physical_corners_and_edge_samples_to_support_lines"
        ),
        "virtual_anchor_point_correspondences_used": False,
    }


def optimize_strip_candidate(records: list[dict], width: float, height: float) -> tuple[list[dict], float, float, dict]:
    """对条带候选联合优化 SE(2)、W/H；目标方向和平移由局部规范坐标固定。

    规范化去除了全局 SE(2) 的三个自由度；之后由 rectangle_fit 整体搬到 A4。
    """
    x0 = np.asarray(
        [
            *[
                value
                for record in records
                for value in (record["angle"], record["translation"][0], record["translation"][1])
            ],
            width,
            height,
        ],
        dtype=float,
    )

    def residual(params: np.ndarray) -> np.ndarray:
        transformed = []
        for index, record in enumerate(records):
            angle, tx, ty = params[index * 3 : index * 3 + 3]
            transformed.append(transform_points(record["observed"], angle, (tx, ty)))
        w, h = params[-2:]
        rows: list[float] = []
        # 每片选定上边的两点贴 y=0，对边两点贴 y=H。
        for record, points in zip(records, transformed):
            edge = int(record["top_edge_index"])
            n = len(points)
            rows.extend([points[edge, 1], points[(edge + 1) % n, 1]])
            rows.extend([points[(edge + 2) % n, 1] - h, points[(edge - 1) % n, 1] - h])
        # 相邻公共侧边两个端点联合对齐。
        for index in range(len(records) - 1):
            left_record, right_record = records[index], records[index + 1]
            left, right = transformed[index], transformed[index + 1]
            le, re = int(left_record["top_edge_index"]), int(right_record["top_edge_index"])
            rows.extend((left[(le + 1) % len(left)] - right[re]).tolist())
            rows.extend((left[(le + 2) % len(left)] - right[(re - 1) % len(right)]).tolist())
        first_edge = int(records[0]["top_edge_index"])
        last_edge = int(records[-1]["top_edge_index"])
        rows.extend(
            [
                transformed[0][first_edge, 0],
                transformed[0][(first_edge - 1) % len(transformed[0]), 0],
                transformed[-1][(last_edge + 1) % len(transformed[-1]), 0] - w,
                transformed[-1][(last_edge + 2) % len(transformed[-1]), 0] - w,
            ]
        )
        # 合法宽高的平滑边界惩罚。
        rows.extend(
            [
                5.0 * max(0.0, 90.0 - w),
                5.0 * max(0.0, w - 120.0),
                5.0 * max(0.0, 50.0 - h),
                5.0 * max(0.0, h - 90.0),
            ]
        )
        return np.asarray(rows, dtype=float)

    before = float(np.sqrt(np.mean(residual(x0) ** 2)))
    result = least_squares(residual, x0, loss="soft_l1", f_scale=2.0, max_nfev=1200)
    after = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    accepted = bool(result.success and after <= before + 1e-9)
    chosen = result.x if accepted else x0
    optimized = []
    for index, record in enumerate(records):
        angle, tx, ty = chosen[index * 3 : index * 3 + 3]
        updated = dict(record)
        updated["angle"] = normalize_angle(float(angle))
        updated["translation"] = np.asarray([tx, ty], dtype=float)
        updated["target"] = transform_points(record["observed"], updated["angle"], updated["translation"])
        optimized.append(updated)
    return optimized, float(chosen[-2]), float(chosen[-1]), {
        "optimization_success": bool(result.success),
        "optimization_accepted": accepted,
        "before_rms_mm": before,
        "after_rms_mm": after,
        "improvement_mm": before - after,
        "nfev": int(result.nfev),
    }


def optimize_generic_candidate(
    records: list[dict],
    width: float,
    height: float,
    matches: list[tuple[str, int, str, int]] | None = None,
    interfaces: list | None = None,
    discovered_interfaces: list[dict] | None = None,
    max_nfev_override: int | None = None,
    dimension_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[list[dict], float, float, dict]:
    """对搜索得到的链式公共边候选做通用联合优化。

    除公共边端点/共线约束外，将初始未匹配外边分配给最近的矩形边界，
    联合优化每片 SE(2) 与 W/H。局部矩形规范等价于消去全局方向和平移，
    求解完成后再统一放入 A4。
    """
    initial_targets = [np.asarray(record["target"], dtype=float) for record in records]
    minx = min(float(points[:, 0].min()) for points in initial_targets)
    miny = min(float(points[:, 1].min()) for points in initial_targets)
    maxx = max(float(points[:, 0].max()) for points in initial_targets)
    maxy = max(float(points[:, 1].max()) for points in initial_targets)
    id_to_index = {record["piece_id"]: index for index, record in enumerate(records)}
    resolved_matches: list[
        tuple[int, int, float, float, int, int, float, float]
    ] = []
    if interfaces:
        for item in interfaces:
            first = item.first_segment
            second = item.second_segment
            resolved_matches.append(
                (
                    id_to_index[item.first_piece],
                    int(first.edge_index),
                    float(first.t_start),
                    float(first.t_end),
                    id_to_index[item.second_piece],
                    int(second.edge_index),
                    float(second.t_start),
                    float(second.t_end),
                )
            )
    elif matches:
        resolved_matches = [
            (
                id_to_index[first], int(edge_a), 0.0, 1.0,
                id_to_index[second], int(edge_b), 0.0, 1.0,
            )
            for first, edge_a, second, edge_b in matches
        ]
    else:
        for index in range(len(records) - 1):
            if records[index].get("matched_next_edge") is not None and records[index + 1].get("matched_prev_edge") is not None:
                resolved_matches.append(
                    (
                        index, int(records[index]["matched_next_edge"]), 0.0, 1.0,
                        index + 1, int(records[index + 1]["matched_prev_edge"]), 0.0, 1.0,
                    )
                )
    existing_resolved = {
        (
            first, edge_a, second, edge_b
        )
        for first, edge_a, _a0, _a1, second, edge_b, _b0, _b1
        in resolved_matches
    }
    for item in discovered_interfaces or []:
        first = id_to_index.get(str(item["piece_a"]))
        second = id_to_index.get(str(item["piece_b"]))
        if first is None or second is None:
            continue
        edge_a, edge_b = int(item["edge_a"]), int(item["edge_b"])
        if (
            (first, edge_a, second, edge_b) in existing_resolved
            or (second, edge_b, first, edge_a) in existing_resolved
        ):
            continue
        interval_a = item.get("interval_a", [0.0, 1.0])
        interval_b = item.get("interval_b", [0.0, 1.0])
        resolved_matches.append((
            first, edge_a, float(interval_a[0]), float(interval_a[1]),
            second, edge_b, float(interval_b[0]), float(interval_b[1]),
        ))
        existing_resolved.add((first, edge_a, second, edge_b))
    # 粗一对多布局只声明长边上的主拼缝；其余短片之间的闭环在带噪
    # 初值中可能相差不到 1 mm，却尚未形成可被接触发现器确认的线段。
    # 若不补充这些近闭合整边，下面会把它们误当作矩形外边，导致优化
    # 模型自相矛盾。这里贪心加入端点反向接近、长度相容且不复用物理边
    # 的候选；优化后的正式准入仍必须重新证实接口、外框与无重叠。
    already_used = {
        (first, edge_a)
        for first, edge_a, _a0, _a1, _second, _edge_b, _b0, _b1
        in resolved_matches
    } | {
        (second, edge_b)
        for _first, _edge_a, _a0, _a1, second, edge_b, _b0, _b1
        in resolved_matches
    }
    near_closures: list[tuple[float, int, int, int, int]] = []
    for first in range(len(initial_targets)):
        for second in range(first + 1, len(initial_targets)):
            first_points = initial_targets[first]
            second_points = initial_targets[second]
            for edge_a in range(len(first_points)):
                if (first, edge_a) in already_used:
                    continue
                segment_a = np.asarray([
                    first_points[edge_a],
                    first_points[(edge_a + 1) % len(first_points)],
                ])
                length_a = float(np.linalg.norm(segment_a[1] - segment_a[0]))
                for edge_b in range(len(second_points)):
                    if (second, edge_b) in already_used:
                        continue
                    segment_b = np.asarray([
                        second_points[edge_b],
                        second_points[(edge_b + 1) % len(second_points)],
                    ])
                    length_b = float(
                        np.linalg.norm(segment_b[1] - segment_b[0])
                    )
                    relative_length_error = abs(length_a - length_b) / max(
                        length_a, length_b, 1e-12
                    )
                    if relative_length_error > 0.08:
                        continue
                    endpoint_rms = float(np.sqrt(np.mean(np.sum(
                        (segment_a - segment_b[::-1]) ** 2,
                        axis=1,
                    ))))
                    if endpoint_rms <= 1.5:
                        near_closures.append((
                            endpoint_rms, first, edge_a, second, edge_b
                        ))
    for _rms, first, edge_a, second, edge_b in sorted(near_closures):
        if (first, edge_a) in already_used or (second, edge_b) in already_used:
            continue
        resolved_matches.append((
            first, edge_a, 0.0, 1.0,
            second, edge_b, 0.0, 1.0,
        ))
        already_used.add((first, edge_a))
        already_used.add((second, edge_b))
    # 记录已匹配边，剩余边才允许分配到矩形外边界。
    matched: set[tuple[int, int]] = set()
    for first, edge_a, _a0, _a1, second, edge_b, _b0, _b1 in resolved_matches:
        matched.add((first, edge_a))
        matched.add((second, edge_b))
    outer_assignments: list[tuple[int, int, str]] = []
    for piece_index, points in enumerate(initial_targets):
        for edge_index in range(len(points)):
            if (piece_index, edge_index) in matched:
                continue
            midpoint = (points[edge_index] + points[(edge_index + 1) % len(points)]) / 2.0
            distances = {
                "left": abs(midpoint[0] - minx),
                "right": abs(midpoint[0] - maxx),
                "top": abs(midpoint[1] - miny),
                "bottom": abs(midpoint[1] - maxy),
            }
            outer_assignments.append((piece_index, edge_index, min(distances, key=distances.get)))
    x0 = np.asarray(
        [
            *[
                value
                for record in records
                for value in (record["angle"], record["translation"][0] - minx, record["translation"][1] - miny)
            ],
            width,
            height,
        ],
        dtype=float,
    )

    confidence_weights = tuple(
        float(record.get("piece_confidence", 1.0)) for record in records
    )
    model = SmoothConstraintModel(
        records=tuple(records),
        resolved_matches=tuple(resolved_matches),
        outer_assignments=tuple(outer_assignments),
        confidence_weights=confidence_weights,
    )
    residual_call_count = 0

    def residual(params: np.ndarray) -> np.ndarray:
        nonlocal residual_call_count
        residual_call_count += 1
        return model.residual(params)

    bounds = (-np.inf, np.inf)
    if dimension_bounds is not None:
        lower, upper = np.full_like(x0, -np.inf), np.full_like(x0, np.inf)
        lower[-2:], upper[-2:] = dimension_bounds
        x0[-2:] = np.clip(x0[-2:], lower[-2:], upper[-2:])
        bounds = (lower, upper)
    before = float(np.sqrt(np.mean(residual(x0) ** 2)))
    exact_before = exact_geometry_metrics(
        model.transformed(x0), float(x0[-2]), float(x0[-1])
    )
    max_nfev = min(800, max(180, 60 * len(records) + 20 * len(resolved_matches)))
    if max_nfev_override is not None:
        max_nfev = max(20, min(max_nfev, int(max_nfev_override)))
    result = least_squares(
        residual, x0, loss="soft_l1", f_scale=2.0, max_nfev=max_nfev, bounds=bounds
    )
    after = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    exact_after = exact_geometry_metrics(
        model.transformed(result.x),
        float(result.x[-2]), float(result.x[-1]),
    )
    accepted = bool(
        result.success
        and after <= before + 1e-9
        and exact_geometry_objective(exact_after)
        <= exact_geometry_objective(exact_before)
    )
    chosen = result.x if accepted else x0
    optimized = []
    for index, record in enumerate(records):
        angle, tx, ty = chosen[index * 3 : index * 3 + 3]
        updated = dict(record)
        updated["angle"] = normalize_angle(float(angle))
        updated["translation"] = np.asarray([tx, ty], dtype=float)
        updated["target"] = transform_points(record["observed"], updated["angle"], updated["translation"])
        optimized.append(updated)
    return optimized, float(chosen[-2]), float(chosen[-1]), {
        "optimization_success": bool(result.success),
        "optimization_accepted": accepted,
        "before_rms_mm": before,
        "after_rms_mm": after,
        "improvement_mm": before - after,
        "nfev": int(result.nfev),
        "njev": int(result.njev or 0),
        "residual_call_count": residual_call_count,
        "jacobian_call_count": int(result.njev or 0),
        "exact_geometry_before": exact_before,
        "exact_geometry_after": exact_after,
        "max_nfev": max_nfev,
    }
