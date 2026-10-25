"""通用理想直线分割生成器，以及生成后的拓扑/几何分类。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, permutations
import math
from typing import Iterable

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box
from shapely.ops import snap, split, unary_union

from ..config import (
    GEOMETRY_TOLERANCE_MM,
    SHARED_EDGE_LENGTH_TOLERANCE_MM,
)
from ..geometry import internal_angles
from ..geometry_relations import robust_segment_overlap
from ..models import Piece


@dataclass(frozen=True)
class PartitionQualityConfig:
    min_interior_angle_deg: float = 20.0
    max_aspect_ratio: float = 8.0
    min_piece_area_mm2: float = 400.0
    min_edge_length_mm: float = 20.0
    collinear_merge_tolerance_deg: float = 0.25
    min_junction_separation_mm: float = 20.0
    require_positive_edge_connected: bool = False


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return math.degrees(
        math.acos(float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))
    )


def merge_collinear_vertices(
    vertices: Iterable[Iterable[float]],
    tolerance_deg: float = 0.25,
) -> list[tuple[float, float]]:
    """先删除重复点，再删除连续共线点。"""
    points = [np.asarray(point, dtype=float) for point in vertices]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-7:
        points.pop()
    changed = True
    while changed and len(points) > 3:
        changed = False
        kept: list[np.ndarray] = []
        count = len(points)
        for index, point in enumerate(points):
            previous = points[(index - 1) % count]
            following = points[(index + 1) % count]
            first = point - previous
            second = following - point
            turn = _angle_between(first, second)
            if min(turn, abs(180.0 - turn)) <= tolerance_deg and np.dot(first, second) >= 0:
                changed = True
                continue
            if not kept or np.linalg.norm(point - kept[-1]) > 1e-7:
                kept.append(point)
        points = kept
    return [(float(point[0]), float(point[1])) for point in points]


def _extract_polygons(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [
            item for item in geometry.geoms
            if isinstance(item, Polygon) and item.area > 1e-7
        ]
    return []


def _split_polygon(
    poly: Polygon,
    a: tuple[float, float],
    b: tuple[float, float],
) -> list[Polygon]:
    if np.linalg.norm(np.asarray(a) - np.asarray(b)) < 1e-6:
        return []
    try:
        parts = _extract_polygons(split(poly, LineString([a, b])))
    except Exception:
        return []
    if len(parts) != 2 or abs(sum(item.area for item in parts) - poly.area) > 1e-5:
        return []
    return parts


def _edge_records(poly: Polygon, width: float, height: float) -> list[dict]:
    coords = list(poly.exterior.coords)[:-1]
    records = []
    for index, first in enumerate(coords):
        second = coords[(index + 1) % len(coords)]
        line = LineString([first, second])
        midpoint = line.interpolate(0.5, normalized=True)
        exterior = (
            abs(midpoint.x) < 1e-6
            or abs(midpoint.y) < 1e-6
            or abs(midpoint.x - width) < 1e-6
            or abs(midpoint.y - height) < 1e-6
        )
        records.append(
            {
                "a": first,
                "b": second,
                "line": line,
                "length": line.length,
                "exterior": exterior,
            }
        )
    return records


def _sample_on_edge(
    record: dict,
    rng: np.random.Generator,
    allow_vertex: bool,
) -> tuple[float, float]:
    if allow_vertex and rng.random() < 0.24:
        point = record["a"] if rng.random() < 0.5 else record["b"]
        return float(point[0]), float(point[1])
    margin = min(0.30, 10.0 / max(record["length"], 1e-9))
    fraction = float(rng.uniform(margin, 1.0 - margin))
    point = record["line"].interpolate(fraction, normalized=True)
    return float(point.x), float(point.y)


def _candidate_recursive_split(
    poly: Polygon,
    width: float,
    height: float,
    rng: np.random.Generator,
    config: PartitionQualityConfig,
    prefer_internal_endpoint: bool,
) -> list[Polygon]:
    eligible = [
        edge for edge in _edge_records(poly, width, height)
        if edge["length"] >= config.min_edge_length_mm * 1.15
    ]
    pairs = list(combinations(eligible, 2))
    rng.shuffle(pairs)
    if prefer_internal_endpoint:
        pairs.sort(key=lambda pair: not (not pair[0]["exterior"] or not pair[1]["exterior"]))
    for first, second in pairs[:24]:
        for _ in range(4):
            a = _sample_on_edge(first, rng, allow_vertex=True)
            b = _sample_on_edge(second, rng, allow_vertex=True)
            parts = _split_polygon(poly, a, b)
            if len(parts) != 2:
                continue
            cleaned = [
                merge_collinear_vertices(item.exterior.coords, config.collinear_merge_tolerance_deg)
                for item in parts
            ]
            if all(
                len(vertices) <= 5
                and Polygon(vertices).area >= config.min_piece_area_mm2 - 1e-6
                for vertices in cleaned
            ):
                return parts
    return []


def _piece_from_polygon(
    poly: Polygon,
    index: int,
    prefix: str,
    config: PartitionQualityConfig,
) -> Piece:
    vertices = merge_collinear_vertices(
        poly.exterior.coords, config.collinear_merge_tolerance_deg
    )
    return Piece(f"{prefix}_piece_{index}", vertices)


def _finalize(
    polygons: list[Polygon],
    width: float,
    height: float,
    prefix: str,
    config: PartitionQualityConfig,
) -> list[Piece]:
    from .validator import check_partition

    ordered = sorted(polygons, key=lambda item: (item.centroid.y, item.centroid.x))
    pieces = [
        _piece_from_polygon(poly, index, prefix, config)
        for index, poly in enumerate(ordered)
    ]
    result = check_partition(
        pieces,
        width,
        height,
        min_edge_mm=config.min_edge_length_mm,
        min_area_mm2=config.min_piece_area_mm2,
        min_interior_angle_deg=config.min_interior_angle_deg,
        max_aspect_ratio=config.max_aspect_ratio,
        collinear_merge_tolerance_deg=config.collinear_merge_tolerance_deg,
    )
    if not result.valid:
        return []
    metadata = classify_partition(pieces, width, height)
    if (
        config.require_positive_edge_connected
        and not metadata["positive_length_adjacency_connected"]
    ):
        return []
    nodes = [
        np.asarray(point, dtype=float)
        for point in metadata["internal_node_positions_mm"]
    ]
    if any(
        np.linalg.norm(first - second) < config.min_junction_separation_mm - 1e-6
        for first, second in combinations(nodes, 2)
    ):
        return []
    return pieces


def recursive_random_partition(
    rng: np.random.Generator,
    piece_count: int,
    width: float,
    height: float,
    config: PartitionQualityConfig | None = None,
    *,
    require_t_junction: bool = False,
    require_multiple_internal_nodes: bool = False,
) -> list[Piece]:
    """递归选择可分区域，并连接其任意两条边界边。"""
    cfg = config or PartitionQualityConfig()
    for _attempt in range(1600):
        regions = [box(0.0, 0.0, width, height)]
        while len(regions) < piece_count:
            candidates = [
                index for index, region in enumerate(regions)
                if region.area >= 2.0 * cfg.min_piece_area_mm2
            ]
            if not candidates:
                break
            weights = np.asarray([regions[index].area for index in candidates], dtype=float)
            selected = int(rng.choice(candidates, p=weights / weights.sum()))
            prefer_internal = len(regions) >= 2 and (
                require_t_junction or rng.random() < 0.68
            )
            parts = _candidate_recursive_split(
                regions[selected], width, height, rng, cfg, prefer_internal
            )
            if not parts:
                break
            regions[selected:selected + 1] = parts
        if len(regions) != piece_count:
            continue
        pieces = _finalize(regions, width, height, "recursive", cfg)
        if not pieces:
            continue
        metadata = classify_partition(pieces, width, height)
        if require_t_junction and not metadata["has_t_junction"]:
            continue
        if require_multiple_internal_nodes and not metadata["has_multiple_internal_nodes"]:
            continue
        return pieces
    raise RuntimeError(
        f"recursive 无法生成 {piece_count} 片；"
        f"T={require_t_junction}, multi={require_multiple_internal_nodes}"
    )


def _long_random_chord(
    width: float,
    height: float,
    rng: np.random.Generator,
    existing_angles: list[float],
) -> LineString:
    boundary_edges = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    for _ in range(200):
        edge_a, edge_b = rng.choice(4, size=2, replace=False)
        first, second = boundary_edges[int(edge_a)], boundary_edges[int(edge_b)]
        ta = float(rng.uniform(0.22, 0.78))
        tb = float(rng.uniform(0.22, 0.78))
        a = tuple(np.asarray(first[0]) + ta * (np.asarray(first[1]) - np.asarray(first[0])))
        b = tuple(np.asarray(second[0]) + tb * (np.asarray(second[1]) - np.asarray(second[0])))
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        if all(
            min(abs(angle - old), 180.0 - abs(angle - old)) >= 18.0
            for old in existing_angles
        ):
            return LineString([a, b])
    return LineString([(width * 0.25, 0.0), (width * 0.75, height)])


def _atomic_regions_from_lines(
    rectangle: Polygon,
    lines: list[LineString],
) -> list[Polygon]:
    regions = [rectangle]
    for line in lines:
        updated: list[Polygon] = []
        for region in regions:
            pieces = _extract_polygons(split(region, line))
            updated.extend(pieces if len(pieces) > 1 else [region])
        regions = updated
    return regions


def _merge_atomic_regions(
    atoms: list[Polygon],
    target_count: int,
    rng: np.random.Generator,
) -> list[Polygon]:
    groups = [{index} for index in range(len(atoms))]
    while len(groups) > target_count:
        candidates = []
        for i, j in combinations(range(len(groups)), 2):
            left = unary_union([atoms[index] for index in groups[i]])
            right = unary_union([atoms[index] for index in groups[j]])
            if (
                left.boundary.intersection(right.boundary).length
                <= SHARED_EDGE_LENGTH_TOLERANCE_MM
            ):
                continue
            merged = left.union(right)
            if isinstance(merged, Polygon) and not merged.interiors:
                candidates.append((i, j))
        if not candidates:
            return []
        i, j = candidates[int(rng.integers(len(candidates)))]
        groups[i] |= groups[j]
        groups.pop(j)
    result = [
        unary_union([atoms[index] for index in group])
        for group in groups
    ]
    return (
        result
        if all(isinstance(poly, Polygon) and not poly.interiors for poly in result)
        else []
    )


def line_arrangement_merge(
    rng: np.random.Generator,
    piece_count: int,
    width: float,
    height: float,
    config: PartitionQualityConfig | None = None,
    *,
    require_nonparallel: bool = True,
) -> list[Piece]:
    """相交直线切出原子区，再随机合并相邻原子区。"""
    cfg = config or PartitionQualityConfig()
    for _attempt in range(1800):
        line_count = int(rng.integers(2 if require_nonparallel else 1, 4))
        lines: list[LineString] = []
        angles: list[float] = []
        for _ in range(line_count):
            line = _long_random_chord(width, height, rng, angles)
            delta = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
            angles.append(math.degrees(math.atan2(delta[1], delta[0])) % 180.0)
            lines.append(line)
        atoms = _atomic_regions_from_lines(box(0.0, 0.0, width, height), lines)
        if not piece_count <= len(atoms) <= 8:
            continue
        polygons = _merge_atomic_regions(atoms, piece_count, rng)
        if not polygons:
            continue
        pieces = _finalize(polygons, width, height, "arrangement", cfg)
        if not pieces:
            continue
        metadata = classify_partition(pieces, width, height)
        directions = metadata["common_edge_direction_deg"]
        if require_nonparallel and len(directions) >= 2:
            if not any(
                min(abs(a - b), 180.0 - abs(a - b)) >= 10.0
                for a, b in combinations(directions, 2)
            ):
                continue
        return pieces
    raise RuntimeError(f"arrangement 无法生成 {piece_count} 片")


def _shared_lines(pieces: list[Piece]) -> list[tuple[int, int, LineString]]:
    """按最终保存几何的真实正长度交集分类，不用 snap 扩大接触。"""
    result: list[tuple[int, int, LineString]] = []
    for i, j in combinations(range(len(pieces)), 2):
        first = np.asarray(pieces[i].vertices, dtype=float)
        second = np.asarray(pieces[j].vertices, dtype=float)
        first_inside = np.asarray(
            Polygon(first).representative_point().coords[0], dtype=float
        )
        second_inside = np.asarray(
            Polygon(second).representative_point().coords[0], dtype=float
        )
        raw_lines: list[LineString] = []
        for edge_a in range(len(first)):
            for edge_b in range(len(second)):
                overlap = robust_segment_overlap(
                    first[edge_a],
                    first[(edge_a + 1) % len(first)],
                    second[edge_b],
                    second[(edge_b + 1) % len(second)],
                    line_tolerance_mm=GEOMETRY_TOLERANCE_MM,
                    parallel_tolerance=1e-7,
                )
                if (
                    overlap is not None
                    and overlap.length_mm
                    > SHARED_EDGE_LENGTH_TOLERANCE_MM
                ):
                    direction = first[(edge_a + 1) % len(first)] - first[edge_a]
                    direction = direction / max(
                        float(np.linalg.norm(direction)), 1e-12
                    )
                    normal = np.asarray([-direction[1], direction[0]])
                    midpoint = (
                        np.asarray(overlap.start_point)
                        + np.asarray(overlap.end_point)
                    ) / 2.0
                    side_a = float(np.dot(first_inside - midpoint, normal))
                    side_b = float(np.dot(second_inside - midpoint, normal))
                    if side_a * side_b >= -GEOMETRY_TOLERANCE_MM**2:
                        continue
                    raw_lines.append(
                        LineString([overlap.start_point, overlap.end_point])
                    )
        merged = unary_union(raw_lines) if raw_lines else GeometryCollection()
        if isinstance(merged, LineString):
            lines = [merged]
        elif hasattr(merged, "geoms"):
            lines = [
                item
                for item in merged.geoms
                if isinstance(item, LineString)
            ]
        else:
            lines = []
        result.extend(
            (i, j, line)
            for line in lines
            if line.length > SHARED_EDGE_LENGTH_TOLERANCE_MM
        )
    return result


def _canonical_graph_signature(
    node_count: int,
    edges: list[tuple[int, int]],
) -> str:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    encodings = []
    for order in permutations(range(node_count)):
        bits = [
            "1" if tuple(sorted((order[i], order[j]))) in edge_set else "0"
            for i, j in combinations(range(node_count), 2)
        ]
        encodings.append("".join(bits))
    return f"n{node_count}:{min(encodings)}"


def _cluster_points(
    points: list[np.ndarray],
    tolerance: float = 1e-5,
) -> list[np.ndarray]:
    clusters: list[list[np.ndarray]] = []
    for point in points:
        for cluster in clusters:
            if np.linalg.norm(point - np.mean(cluster, axis=0)) <= tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    return [np.mean(cluster, axis=0) for cluster in clusters]


def _junction_degree(
    point: np.ndarray,
    pieces: list[Piece],
    tolerance: float = 1e-5,
) -> int:
    ray_angles: list[float] = []
    for piece in pieces:
        vertices = np.asarray(piece.vertices, dtype=float)
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            line = LineString([first, second])
            if line.distance(Point(point)) > tolerance:
                continue
            for endpoint in (first, second):
                vector = endpoint - point
                if np.linalg.norm(vector) <= tolerance:
                    continue
                angle = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
                if all(
                    min(abs(angle - old), 360.0 - abs(angle - old)) > 0.5
                    for old in ray_angles
                ):
                    ray_angles.append(angle)
    return len(ray_angles)


def _minimum_aspect_ratio(poly: Polygon) -> float:
    rectangle = poly.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords[:4], dtype=float)
    lengths = np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1)
    positive = lengths[lengths > 1e-8]
    return float(positive.max() / positive.min()) if len(positive) else float("inf")


def classify_partition(pieces: list[Piece], width: float, height: float) -> dict:
    """从最终几何计算真值分类，不信任生成器预先声明的名称。"""
    shared = _shared_lines(pieces)
    adjacency_edges = sorted(
        {tuple(sorted((i, j))) for i, j, _line in shared}
    )
    adjacency_sets = [set() for _piece in pieces]
    for first, second in adjacency_edges:
        adjacency_sets[first].add(second)
        adjacency_sets[second].add(first)
    component_ids = [-1] * len(pieces)
    component_count = 0
    for start in range(len(pieces)):
        if component_ids[start] >= 0:
            continue
        pending = [start]
        component_ids[start] = component_count
        while pending:
            current = pending.pop()
            for neighbor in adjacency_sets[current]:
                if component_ids[neighbor] < 0:
                    component_ids[neighbor] = component_count
                    pending.append(neighbor)
        component_count += 1
    polygons = [Polygon(piece.vertices) for piece in pieces]
    point_contacts: list[dict] = []
    shared_pairs = {tuple(edge) for edge in adjacency_edges}
    for first, second in combinations(range(len(pieces)), 2):
        if (first, second) in shared_pairs:
            continue
        intersection = polygons[first].boundary.intersection(
            polygons[second].boundary
        )
        points = []
        if isinstance(intersection, Point):
            points = [intersection]
        elif hasattr(intersection, "geoms"):
            points = [
                item for item in intersection.geoms
                if isinstance(item, Point)
            ]
        for point in points:
            point_contacts.append(
                {
                    "pieces": [first, second],
                    "location_mm": [float(point.x), float(point.y)],
                }
            )
    common_lengths = [float(line.length) for _i, _j, line in shared]
    shared_length_by_pair: dict[tuple[int, int], float] = {}
    shared_interval_count_by_pair: dict[tuple[int, int], int] = {}
    for first, second, line in shared:
        pair = tuple(sorted((first, second)))
        shared_length_by_pair[pair] = (
            shared_length_by_pair.get(pair, 0.0) + float(line.length)
        )
        shared_interval_count_by_pair[pair] = (
            shared_interval_count_by_pair.get(pair, 0) + 1
        )
    directions = []
    for _i, _j, line in shared:
        coords = np.asarray(line.coords, dtype=float)
        delta = coords[-1] - coords[0]
        directions.append(
            float(math.degrees(math.atan2(delta[1], delta[0])) % 180.0)
        )

    interior_candidates: list[np.ndarray] = []
    for piece in pieces:
        for vertex in piece.vertices:
            point = np.asarray(vertex, dtype=float)
            if (
                1e-5 < point[0] < width - 1e-5
                and 1e-5 < point[1] < height - 1e-5
            ):
                interior_candidates.append(point)
    interior_nodes = _cluster_points(interior_candidates)
    degrees = [_junction_degree(point, pieces) for point in interior_nodes]
    meaningful_nodes = [
        (point, degree)
        for point, degree in zip(interior_nodes, degrees)
        if degree >= 3
    ]
    degree_values = [degree for _point, degree in meaningful_nodes]

    parallel_pairs = sum(
        min(abs(first - second), 180.0 - abs(first - second)) <= 5.0
        for first, second in combinations(directions, 2)
    )
    equal_pairs = sum(
        abs(first - second) <= max(2.0, 0.05 * max(first, second))
        for first, second in combinations(common_lengths, 2)
    )
    piece_areas = [float(poly.area) for poly in polygons]
    piece_edges = [len(piece.vertices) for piece in pieces]
    all_angles = [
        float(angle)
        for piece in pieces
        for angle in internal_angles(piece.vertices)
    ]
    min_angle = min(all_angles) if all_angles else 0.0
    aspect_ratios = [_minimum_aspect_ratio(poly) for poly in polygons]
    diagonal = math.hypot(width, height)
    descriptor = {
        "graph": _canonical_graph_signature(len(pieces), adjacency_edges),
        "junction_degrees": sorted(degree_values),
        "piece_edge_counts": sorted(piece_edges),
        "area_fractions": sorted(
            round(area / (width * height), 3) for area in piece_areas
        ),
        "common_length_fractions": sorted(
            round(length / diagonal, 3) for length in common_lengths
        ),
        "common_direction_bins_5deg": sorted(
            int(round(direction / 5.0)) % 36 for direction in directions
        ),
    }
    symmetry_risk = (
        equal_pairs > 0
        or any(
            abs(a - b) <= 0.02 * width * height
            for a, b in combinations(piece_areas, 2)
        )
    )
    return {
        "adjacency_graph": {
            "nodes": list(range(len(pieces))),
            "edges": [list(edge) for edge in adjacency_edges],
        },
        "adjacency_canonical_signature": descriptor["graph"],
        "positive_length_adjacency_connected": component_count <= 1,
        "positive_length_component_count": component_count,
        "positive_length_component_ids": component_ids,
        "point_contact_count": len(point_contacts),
        "point_contact_locations": [
            item["location_mm"] for item in point_contacts
        ],
        "point_contacts": point_contacts,
        "interior_node_count": len(degree_values),
        "interior_node_degrees": degree_values,
        "internal_node_positions_mm": [
            [float(point[0]), float(point[1])]
            for point, _degree in meaningful_nodes
        ],
        "common_edge_count": len(shared),
        "robust_shared_length_mm": [
            {"pieces": list(pair), "length_mm": length}
            for pair, length in sorted(shared_length_by_pair.items())
        ],
        "shared_interval_count": [
            {"pieces": list(pair), "count": count}
            for pair, count in sorted(shared_interval_count_by_pair.items())
        ],
        "piece_edge_counts": piece_edges,
        "minimum_interior_angle_deg": min_angle,
        "piece_aspect_ratios": aspect_ratios,
        "near_parallel_common_edge_pair_count": parallel_pairs,
        "near_equal_common_edge_pair_count": equal_pairs,
        "has_t_junction": 3 in degree_values,
        "has_multiple_internal_nodes": len(degree_values) >= 2,
        "geometric_symmetry_or_ambiguity_risk": symmetry_risk,
        "common_edge_direction_deg": directions,
        "common_edge_length_mm": common_lengths,
        "normalized_geometry_descriptor": descriptor,
    }


def descriptor_key(metadata: dict) -> str:
    descriptor = metadata["normalized_geometry_descriptor"]
    return repr(
        (
            descriptor["graph"],
            tuple(descriptor["junction_degrees"]),
            tuple(descriptor["piece_edge_counts"]),
            tuple(descriptor["area_fractions"]),
            tuple(descriptor["common_length_fractions"]),
            tuple(descriptor["common_direction_bins_5deg"]),
        )
    )


def generate_general_partition(
    rng: np.random.Generator,
    piece_count: int | None = None,
    target_width: float | None = None,
    target_height: float | None = None,
    mode: str = "mixed",
    config: PartitionQualityConfig | None = None,
) -> tuple[list[Piece], float, float, dict]:
    """生成 recursive/arrangement/adversarial/mixed 分割。"""
    cfg = config or PartitionQualityConfig()
    count = int(piece_count if piece_count is not None else rng.integers(2, 5))
    if not 2 <= count <= 4:
        raise ValueError("piece_count 必须为 2～4")
    if mode not in {"recursive", "arrangement", "adversarial", "mixed"}:
        raise ValueError(f"未知通用生成模式: {mode}")
    selected_mode = mode
    if mode == "mixed":
        selected_mode = str(
            rng.choice(
                ["recursive", "arrangement", "adversarial"],
                p=[60.0 / 95.0, 25.0 / 95.0, 10.0 / 95.0],
            )
        )
    last_error = ""
    for _attempt in range(300):
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
        try:
            if selected_mode == "recursive":
                pieces = recursive_random_partition(rng, count, width, height, cfg)
            elif selected_mode == "arrangement":
                pieces = line_arrangement_merge(rng, count, width, height, cfg)
            else:
                relaxed = PartitionQualityConfig(
                    min_interior_angle_deg=max(
                        12.0, cfg.min_interior_angle_deg * 0.7
                    ),
                    max_aspect_ratio=max(cfg.max_aspect_ratio, 10.0),
                    min_piece_area_mm2=cfg.min_piece_area_mm2,
                    min_edge_length_mm=cfg.min_edge_length_mm,
                    collinear_merge_tolerance_deg=cfg.collinear_merge_tolerance_deg,
                    min_junction_separation_mm=cfg.min_junction_separation_mm,
                    require_positive_edge_connected=cfg.require_positive_edge_connected,
                )
                pieces = line_arrangement_merge(rng, count, width, height, relaxed)
            metadata = classify_partition(pieces, width, height)
            if selected_mode == "adversarial" and not (
                metadata["geometric_symmetry_or_ambiguity_risk"]
                or metadata["near_equal_common_edge_pair_count"] > 0
            ):
                continue
            metadata["generator_mode"] = selected_mode
            metadata["quality_config"] = asdict(cfg)
            return pieces, width, height, metadata
        except RuntimeError as exc:
            last_error = str(exc)
    raise RuntimeError(f"{mode} 通用分割生成失败: {last_error}")
