"""允许读取 truth 的独立评估器；求解器不导入本模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from ..geometry import edge_lengths, polygon
from ..transforms import estimate_rigid, normalize_angle, transform_points
from .metrics import best_correspondence, union_metrics

PHYSICAL_OVERLAP_TOLERANCE_MM2 = 1e-5


def _rotate_polygon(poly: Polygon, center: np.ndarray, angle_deg: float) -> Polygon:
    points = np.asarray(poly.exterior.coords[:-1], dtype=float)
    return Polygon(transform_points(points - center, angle_deg) + center)


def _cyclic_vertex_distances(predicted: np.ndarray, expected: np.ndarray) -> list[float]:
    if len(predicted) != len(expected):
        # Sampling density is not a physical vertex correspondence. Compare
        # complete boundaries when a contour contains extra collinear samples.
        predicted_boundary = Polygon(predicted).boundary
        expected_boundary = Polygon(expected).boundary
        distances = [
            float(Point(point).distance(expected_boundary)) for point in predicted
        ]
        distances.extend(
            float(Point(point).distance(predicted_boundary)) for point in expected
        )
        distances.append(
            float(predicted_boundary.hausdorff_distance(expected_boundary))
        )
        return distances
    choices: list[np.ndarray] = []
    for candidate in (expected, expected[::-1]):
        choices.extend(
            np.linalg.norm(predicted - np.roll(candidate, shift, axis=0), axis=1)
            for shift in range(len(candidate))
        )
    best = min(choices, key=lambda values: float(np.mean(values**2)))
    return [float(value) for value in best]


def orientation_preserved(
    source_vertices: list | np.ndarray,
    target_vertices: list | np.ndarray,
    tolerance: float = 1e-12,
) -> bool:
    """刚体旋转必须保持有向面积符号；镜像会翻转符号。"""
    source = np.asarray(source_vertices, dtype=float)
    target = np.asarray(target_vertices, dtype=float)
    source_area2 = float(
        np.sum(
            source[:, 0] * np.roll(source[:, 1], -1)
            - np.roll(source[:, 0], -1) * source[:, 1]
        )
    )
    target_area2 = float(
        np.sum(
            target[:, 0] * np.roll(target[:, 1], -1)
            - np.roll(target[:, 0], -1) * target[:, 1]
        )
    )
    return source_area2 * target_area2 > tolerance


def _variant_metrics(
    solved: dict[str, Polygon],
    expected: dict[str, Polygon],
    expected_piece_records: dict[str, dict],
    move_by_public: dict[str, dict],
    center: np.ndarray,
    global_angle_deg: float,
) -> dict[str, Any]:
    rotated_expected = {
        source_id: _rotate_polygon(poly, center, global_angle_deg)
        for source_id, poly in expected.items()
    }
    solved_union = unary_union(list(solved.values()))
    expected_union = unary_union(list(rotated_expected.values()))
    denominator = solved_union.union(expected_union).area
    union_iou = (
        float(solved_union.intersection(expected_union).area / denominator)
        if denominator
        else 0.0
    )
    mapping, mean_piece_iou = best_correspondence(solved, rotated_expected)
    vertex_distances: list[float] = []
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for public_id, source_id in mapping.items():
        predicted = np.asarray(solved[public_id].exterior.coords[:-1], dtype=float)
        expected_vertices = np.asarray(
            rotated_expected[source_id].exterior.coords[:-1], dtype=float
        )
        vertex_distances.extend(
            _cyclic_vertex_distances(predicted, expected_vertices)
        )
        source = expected_piece_records[source_id]
        if len(source["clean_shuffled_vertices_mm"]) == len(expected_vertices):
            ideal_angle, ideal_translation, _ = estimate_rigid(
                source["clean_shuffled_vertices_mm"], expected_vertices
            )
            output_transform = move_by_public[public_id]["rigid_transform"]
            rotation_errors.append(
                abs(
                    normalize_angle(
                        float(output_transform["rotation_deg_clockwise"])
                        - ideal_angle
                    )
                )
            )
            translation_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(
                            output_transform["translation_a4_mm"], dtype=float
                        )
                        - np.asarray(ideal_translation, dtype=float)
                    )
                )
            )
    return {
        "global_angle_deg": float(global_angle_deg),
        "union_iou": union_iou,
        "mapping": mapping,
        "mean_piece_iou": float(mean_piece_iou),
        "vertex_distances": vertex_distances,
        "max_vertex_error_mm": max(vertex_distances) if vertex_distances else float("inf"),
        "mean_vertex_error_mm": (
            float(np.mean(vertex_distances)) if vertex_distances else float("inf")
        ),
        "rotation_error_deg_mean": (
            float(np.mean(rotation_errors)) if rotation_errors else None
        ),
        "translation_error_mm_mean": (
            float(np.mean(translation_errors)) if translation_errors else None
        ),
    }


def _variant_sort_key(item: dict[str, Any]) -> tuple[float, ...]:
    """按任务定义选择全局等价旋转，避免矩形 IoU 打平时固定选 0°。"""
    rotation_error = item["rotation_error_deg_mean"]
    return (
        round(item["union_iou"], 9),
        -round(item["max_vertex_error_mm"], 6),
        -(rotation_error if rotation_error is not None else float("inf")),
        round(item["mean_piece_iou"], 9),
    )


def evaluate_solution(
    truth_path: str | Path, solution_path: str | Path
) -> dict[str, Any]:
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    solution = json.loads(Path(solution_path).read_text(encoding="utf-8"))
    if not solution.get("success"):
        return {
            "success": False,
            "reason": solution.get("failure_reason", "solver failed"),
            "failure_type": solution.get("failure_type"),
            "geometry_success": False,
            "competition_white_success": False,
            "strict_truth_success": False,
            "appearance_success": None,
        }

    truth_by_public = {piece["public_id"]: piece for piece in truth["pieces"]}
    truth_by_source = {piece["source_id"]: piece for piece in truth["pieces"]}
    solved_polygons: dict[str, Polygon] = {}
    observed_final_polygons: dict[str, Polygon] = {}
    move_by_public: dict[str, dict] = {}
    pick_inside: list[bool] = []
    place_correct: list[float] = []
    min_edge_ok: list[bool] = []
    reflection_ok: list[bool] = []
    for move in solution["moves"]:
        source = truth_by_public[move["piece_id"]]
        transform = move["rigid_transform"]
        angle = float(transform["rotation_deg_clockwise"])
        translation = transform["translation_a4_mm"]
        final = transform_points(
            source["clean_shuffled_vertices_mm"], angle, translation
        )
        solved_polygons[move["piece_id"]] = Polygon(final)
        observed_final_polygons[move["piece_id"]] = Polygon(
            transform_points(
                move["observed_vertices_a4_mm"],
                angle,
                translation,
            )
        )
        move_by_public[move["piece_id"]] = move
        computed_place = transform_points(
            [move["pick_xy_a4_mm"]], angle, translation
        )[0]
        place_correct.append(
            float(
                np.linalg.norm(
                    computed_place - np.asarray(move["place_xy_a4_mm"], dtype=float)
                )
            )
        )
        pick_inside.append(
            polygon(move["observed_vertices_a4_mm"]).covers(
                Point(move["pick_xy_a4_mm"])
            )
        )
        min_edge_ok.append(bool(edge_lengths(final).min() >= 19.0))
        reflection_ok.append(
            orientation_preserved(
                source["clean_shuffled_vertices_mm"], final
            )
        )

    target_declared = solution["target_rectangle"]
    center = np.asarray(target_declared["center_a4_mm"], dtype=float)
    declared_w = float(target_declared["width_mm"])
    declared_h = float(target_declared["height_mm"])
    target_polygon = box(
        center[0] - declared_w / 2.0,
        center[1] - declared_h / 2.0,
        center[0] + declared_w / 2.0,
        center[1] + declared_h / 2.0,
    )
    union = union_metrics(list(solved_polygons.values()), target_polygon)
    ground_truth_actual_overlap = float(union["overlap_area_mm2"])
    observed_overlap = float(sum(
        observed_final_polygons[first].intersection(
            observed_final_polygons[second]
        ).area
        for index, first in enumerate(observed_final_polygons)
        for second in list(observed_final_polygons)[index + 1 :]
    ))
    actual_pairwise_clearances = [
        float(solved_polygons[first].distance(solved_polygons[second]))
        for index, first in enumerate(solved_polygons)
        for second in list(solved_polygons)[index + 1 :]
    ]

    truth_w = float(truth["target_rectangle"]["width_mm"])
    truth_h = float(truth["target_rectangle"]["height_mm"])
    truth_ox = center[0] - truth_w / 2.0
    truth_oy = center[1] - truth_h / 2.0
    expected = {
        piece["source_id"]: Polygon(
            [
                (float(x) + truth_ox, float(y) + truth_oy)
                for x, y in piece["target_vertices_local_mm"]
            ]
        )
        for piece in truth["pieces"]
    }

    angles = (0.0,) if truth.get("mode") == "fixed" else (0.0, 90.0, 180.0, 270.0)
    variants = [
        _variant_metrics(
            solved_polygons,
            expected,
            truth_by_source,
            move_by_public,
            center,
            angle,
        )
        for angle in angles
    ]
    strict_variant = variants[0]
    white_variant = max(variants, key=_variant_sort_key)

    # 严格 truth 还要按 public_id 检查原碎片的绝对目标位置，不能靠换片规避。
    strict_public_vertex_distances: list[float] = []
    strict_public_rotation_errors: list[float] = []
    strict_public_translation_errors: list[float] = []
    for public_id, source in truth_by_public.items():
        predicted = np.asarray(
            solved_polygons[public_id].exterior.coords[:-1], dtype=float
        )
        expected_vertices = np.asarray(
            expected[source["source_id"]].exterior.coords[:-1], dtype=float
        )
        strict_public_vertex_distances.extend(
            _cyclic_vertex_distances(predicted, expected_vertices)
        )
        if len(source["clean_shuffled_vertices_mm"]) == len(expected_vertices):
            ideal_angle, ideal_translation, _ = estimate_rigid(
                source["clean_shuffled_vertices_mm"], expected_vertices
            )
            output_transform = move_by_public[public_id]["rigid_transform"]
            strict_public_rotation_errors.append(
                abs(
                    normalize_angle(
                        float(output_transform["rotation_deg_clockwise"])
                        - ideal_angle
                    )
                )
            )
            strict_public_translation_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(
                            output_transform["translation_a4_mm"], dtype=float
                        )
                        - np.asarray(ideal_translation, dtype=float)
                    )
                )
            )

    # 按真实公共顶点聚合最终位置，量化接缝。
    common_positions: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for public_id, source in truth_by_public.items():
        move = move_by_public[public_id]
        tf = move["rigid_transform"]
        final = transform_points(
            source["clean_shuffled_vertices_mm"],
            tf["rotation_deg_clockwise"],
            tf["translation_a4_mm"],
        )
        for local, actual in zip(source["target_vertices_local_mm"], final):
            key = (round(float(local[0]), 6), round(float(local[1]), 6))
            common_positions.setdefault(key, []).append(
                (float(actual[0]), float(actual[1]))
            )
    seam_distances: list[float] = []
    for positions in common_positions.values():
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                seam_distances.append(
                    float(
                        np.linalg.norm(
                            np.asarray(positions[i]) - np.asarray(positions[j])
                        )
                    )
                )

    truth_ids = list(truth_by_public)
    expected_adjacency: set[frozenset[str]] = set()
    recovered_expected_adjacency: set[frozenset[str]] = set()
    for index, first_id in enumerate(truth_ids):
        first_truth = Polygon(
            truth_by_public[first_id]["target_vertices_local_mm"]
        )
        for second_id in truth_ids[index + 1 :]:
            second_truth = Polygon(
                truth_by_public[second_id]["target_vertices_local_mm"]
            )
            pair = frozenset((first_id, second_id))
            if (
                first_truth.boundary.intersection(second_truth.boundary).length
                > 1e-6
            ):
                expected_adjacency.add(pair)
                if (
                    solved_polygons[first_id].distance(solved_polygons[second_id])
                    <= 10.0
                ):
                    recovered_expected_adjacency.add(pair)
    strict_adjacency_accuracy = (
        len(recovered_expected_adjacency) / len(expected_adjacency)
        if expected_adjacency
        else 1.0
    )
    expected_source_adjacency = {
        frozenset(
            (
                truth_by_public[first]["source_id"],
                truth_by_public[second]["source_id"],
            )
        )
        for first, second in (
            tuple(pair) for pair in expected_adjacency
        )
    }
    reverse_white_mapping = {
        source_id: public_id
        for public_id, source_id in white_variant["mapping"].items()
    }
    recovered_white_adjacency = 0
    for source_pair in expected_source_adjacency:
        source_ids = tuple(source_pair)
        if (
            len(source_ids) == 2
            and all(item in reverse_white_mapping for item in source_ids)
            and solved_polygons[
                reverse_white_mapping[source_ids[0]]
            ].distance(
                solved_polygons[reverse_white_mapping[source_ids[1]]]
            )
            <= 10.0
        ):
            recovered_white_adjacency += 1
    adjacency_accuracy = (
        recovered_white_adjacency / len(expected_source_adjacency)
        if expected_source_adjacency else 1.0
    )

    strict_mapping = strict_variant["mapping"]
    strict_piece_accuracy = (
        sum(
            truth_by_public[public_id]["source_id"] == source_id
            for public_id, source_id in strict_mapping.items()
        )
        / max(len(strict_mapping), 1)
    )
    no_reflection = all(reflection_ok)
    all_moves = len(solution["moves"]) == len(truth["pieces"])
    place_ok = max(place_correct, default=float("inf")) <= 1e-6
    basic_execution_ok = (
        all_moves and no_reflection and all(pick_inside) and place_ok
    )
    declared_dimensions_legal = (
        90.0 - 1e-5 <= declared_w <= 120.0 + 1e-5
        and 50.0 - 1e-5 <= declared_h <= 90.0 + 1e-5
    ) or (
        90.0 - 1e-5 <= declared_h <= 120.0 + 1e-5
        and 50.0 - 1e-5 <= declared_w <= 90.0 + 1e-5
    )
    # The white competition mapping permits one global equivalent rotation
    # and geometrically equivalent piece assignment.  Its maximum is over all
    # corresponding vertices, so it is conservative relative to checking only
    # adjacent seam vertices.  The strict-public seam metric remains below as
    # an independent research diagnostic.
    max_adjacent_vertex_distance = float(
        white_variant["max_vertex_error_mm"]
    )
    topology_correct = adjacency_accuracy >= 1.0 - 1e-9
    geometry_success = bool(
        basic_execution_ok
        and declared_dimensions_legal
        and topology_correct
        and max_adjacent_vertex_distance <= 20.0
        and ground_truth_actual_overlap
        <= PHYSICAL_OVERLAP_TOLERANCE_MM2
    )
    competition_white_success = bool(
        geometry_success
    )
    strict_max_vertex = max(
        strict_public_vertex_distances, default=float("inf")
    )
    strict_rotation_mean = (
        float(np.mean(strict_public_rotation_errors))
        if strict_public_rotation_errors
        else None
    )
    strict_translation_mean = (
        float(np.mean(strict_public_translation_errors))
        if strict_public_translation_errors
        else None
    )
    strict_truth_success = bool(
        geometry_success
        and union["exact_connected"]
        and strict_piece_accuracy >= 1.0 - 1e-9
        and strict_max_vertex <= 20.0
        and (strict_rotation_mean is None or strict_rotation_mean <= 10.0)
    )

    metadata = truth.get("partition_metadata", {})
    result: dict[str, Any] = {
        "success": True,
        "piece_correspondence": white_variant["mapping"],
        "piece_correspondence_accuracy": float(strict_piece_accuracy),
        "white_piece_correspondence": white_variant["mapping"],
        "adjacency_accuracy": float(adjacency_accuracy),
        "strict_adjacency_accuracy": float(strict_adjacency_accuracy),
        "evaluation_global_alignment_deg": white_variant["global_angle_deg"],
        "equivalent_global_rotation": bool(
            competition_white_success
            and not strict_truth_success
            and abs(white_variant["global_angle_deg"]) > 1e-9
        ),
        "mean_piece_iou": white_variant["mean_piece_iou"],
        "rotation_error_deg_mean": white_variant["rotation_error_deg_mean"],
        "translation_error_mm_mean": white_variant[
            "translation_error_mm_mean"
        ],
        "strict_rotation_error_deg_mean": strict_rotation_mean,
        "strict_translation_error_mm_mean": strict_translation_mean,
        "common_vertex_max_distance_mm": max(seam_distances)
        if seam_distances
        else 0.0,
        "common_vertex_mean_distance_mm": float(np.mean(seam_distances))
        if seam_distances
        else 0.0,
        "common_edge_rms_mm": float(
            np.sqrt(np.mean(np.square(seam_distances)))
        )
        if seam_distances
        else 0.0,
        "corresponding_vertex_max_distance_mm": white_variant[
            "max_vertex_error_mm"
        ],
        "corresponding_vertex_mean_distance_mm": white_variant[
            "mean_vertex_error_mm"
        ],
        "strict_corresponding_vertex_max_distance_mm": strict_max_vertex,
        "strict_corresponding_vertex_mean_distance_mm": (
            float(np.mean(strict_public_vertex_distances))
            if strict_public_vertex_distances
            else None
        ),
        "pick_points_inside": all(pick_inside),
        "max_place_formula_error_mm": max(place_correct, default=0.0),
        "no_reflection": no_reflection,
        "all_edges_at_least_20_mm": all(min_edge_ok),
        "solver_elapsed_seconds": solution.get("elapsed_seconds"),
        "solver_truth_read_detected": False,
        "observed_overlap_area_mm2": observed_overlap,
        "estimated_physical_overlap_area_mm2": float(
            solution.get(
                "estimated_physical_overlap_area_mm2",
                0.0,
            )
        ),
        "robust_buffer_overlap_area_mm2": float(
            solution.get("robust_buffer_overlap_area_mm2", 0.0)
        ),
        "ground_truth_actual_overlap_area_mm2": (
            ground_truth_actual_overlap
        ),
        "minimum_pairwise_clearance_mm": min(
            actual_pairwise_clearances, default=0.0
        ),
        "maximum_pairwise_clearance_mm": max(
            actual_pairwise_clearances, default=0.0
        ),
        "max_corresponding_vertex_distance_mm": (
            max_adjacent_vertex_distance
        ),
        "clearance_target_mm": float(
            solution.get("clearance_target_mm", 0.0)
        ),
        "clearance_violation_mm": float(max(
            0.0,
            float(solution.get("clearance_target_mm", 0.0))
            - min(actual_pairwise_clearances, default=0.0),
        )),
        "physical_overlap_tolerance_mm2": (
            PHYSICAL_OVERLAP_TOLERANCE_MM2
        ),
        "topology_correct": topology_correct,
        "geometry_success": geometry_success,
        "competition_white_success": competition_white_success,
        "strict_truth_success": strict_truth_success,
        "appearance_success": None,
        "positive_length_adjacency_connected": metadata.get(
            "positive_length_adjacency_connected", True
        ),
        "positive_length_component_count": metadata.get(
            "positive_length_component_count", 1
        ),
        "point_contact_count": metadata.get("point_contact_count", 0),
        "point_contact_locations": metadata.get("point_contact_locations", []),
        **union,
    }
    # 旧字段仅作兼容；competition gate 从此明确等于白色几何比赛指标。
    result["passes_recommended_thresholds"] = competition_white_success
    result["passes_operational_thresholds"] = competition_white_success
    result["passes_competition_gate"] = competition_white_success
    result["passes_strict_exact"] = strict_truth_success
    return result
