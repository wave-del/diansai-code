"""在已放置的统一坐标中直接发现真实公共区间、T节点和点接触。"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from ..geometry_relations import robust_segment_overlap


def discover_interfaces_in_placed_state(
    placed_vertices: dict[str, np.ndarray],
    tolerance_mm: float,
    length_epsilon_mm: float = 1e-6,
    parallel_tolerance: float | None = None,
) -> dict[str, Any]:
    interfaces: list[dict[str, Any]] = []
    point_contacts: list[dict[str, Any]] = []
    occupied: dict[str, list[tuple[float, float]]] = {}
    graph_edges: set[tuple[str, str]] = set()
    ids = sorted(placed_vertices)
    for first_id, second_id in combinations(ids, 2):
        first = np.asarray(placed_vertices[first_id], dtype=float)
        second = np.asarray(placed_vertices[second_id], dtype=float)
        pair_length = 0.0
        for first_edge in range(len(first)):
            for second_edge in range(len(second)):
                overlap = robust_segment_overlap(
                    first[first_edge],
                    first[(first_edge + 1) % len(first)],
                    second[second_edge],
                    second[(second_edge + 1) % len(second)],
                    line_tolerance_mm=tolerance_mm,
                    parallel_tolerance=(
                        max(1e-8, tolerance_mm * 1e-3)
                        if parallel_tolerance is None
                        else float(parallel_tolerance)
                    ),
                )
                if overlap is None or overlap.length_mm <= length_epsilon_mm:
                    continue
                pair_length += overlap.length_mm
                graph_edges.add((first_id, second_id))
                key_a = f"{first_id}:edge_{first_edge}"
                key_b = f"{second_id}:edge_{second_edge}"
                occupied.setdefault(key_a, []).append(
                    overlap.first_parameters
                )
                occupied.setdefault(key_b, []).append(
                    overlap.second_parameters
                )
                interfaces.append(
                    {
                        "piece_a": first_id,
                        "edge_a": first_edge,
                        "interval_a": list(overlap.first_parameters),
                        "piece_b": second_id,
                        "edge_b": second_edge,
                        "interval_b": list(overlap.second_parameters),
                        "length_mm": overlap.length_mm,
                        "start_point": list(overlap.start_point),
                        "end_point": list(overlap.end_point),
                        "support_line_distance_mm": (
                            overlap.support_line_distance_mm
                        ),
                        "parallel_error": overlap.parallel_error,
                    }
                )
        if pair_length <= length_epsilon_mm:
            poly_a = Polygon(first)
            poly_b = Polygon(second)
            intersection = poly_a.boundary.intersection(poly_b.boundary)
            if not intersection.is_empty and intersection.length <= length_epsilon_mm:
                point_contacts.append(
                    {
                        "pieces": [first_id, second_id],
                        "geometry_type": intersection.geom_type,
                        "wkt": intersection.wkt,
                    }
                )
    endpoint_groups: dict[tuple[int, int], set[str]] = {}
    scale = max(tolerance_mm, 1e-7)
    for interface in interfaces:
        for point in (interface["start_point"], interface["end_point"]):
            key = (
                round(float(point[0]) / scale),
                round(float(point[1]) / scale),
            )
            endpoint_groups.setdefault(key, set()).update(
                (interface["piece_a"], interface["piece_b"])
            )
    junctions = [
        {
            "junction_type": (
                "THREE_WAY_JUNCTION"
                if len(pieces) == 3
                else "MULTI_T_JUNCTION"
            ),
            "participating_pieces": sorted(pieces),
            "quantized_location": list(key),
        }
        for key, pieces in endpoint_groups.items()
        if len(pieces) >= 3
    ]
    return {
        "interfaces": interfaces,
        "junctions": junctions,
        "point_contacts": point_contacts,
        "occupied_intervals": occupied,
        "interface_graph_edges": [list(edge) for edge in sorted(graph_edges)],
        "cycle_residual_mm": max(
            (
                max(
                    interface["support_line_distance_mm"],
                    interface["parallel_error"],
                )
                for interface in interfaces
            ),
            default=0.0,
        ),
    }
