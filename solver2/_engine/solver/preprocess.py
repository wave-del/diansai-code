"""公开观测多边形的校验、方向统一和特征预处理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ..geometry import clean_vertices, edge_angles, edge_lengths, internal_angles, polygon


@dataclass
class ProcessedPiece:
    piece_id: str
    vertices: np.ndarray
    area: float
    centroid: np.ndarray
    edge_lengths: np.ndarray
    edge_angles: np.ndarray
    internal_angles: np.ndarray
    vertex_confidence: np.ndarray
    vertex_covariance_mm2: np.ndarray | None = None
    edges: list[dict[str, Any]] = field(default_factory=list)
    piece_confidence: float = 1.0
    vertex_estimated_sigma_mm: np.ndarray | None = None


def preprocess_piece(
    piece_id: str,
    vertices: list[list[float]],
    vertex_confidence: list[float] | None = None,
    edges: list[dict[str, Any]] | None = None,
    piece_confidence: float | None = None,
    vertex_covariance_mm2: list[list[list[float]]] | None = None,
    vertex_estimated_sigma_mm: list[float] | None = None,
) -> ProcessedPiece:
    pts = clean_vertices(vertices)
    poly = polygon(pts)
    confidence = np.ones(len(pts), dtype=float)
    if vertex_confidence is not None:
        if len(vertex_confidence) != len(pts):
            raise ValueError(f"{piece_id}: vertex_confidence 数量必须与顶点数相同")
        confidence = np.clip(np.asarray(vertex_confidence, dtype=float), 0.0, 1.0)
    edge_data = list(edges or [])
    for edge in edge_data:
        start = int(edge["start_vertex_index"])
        end = int(edge["end_vertex_index"])
        if not (0 <= start < len(pts) and 0 <= end < len(pts)):
            raise ValueError(f"{piece_id}: edge 顶点索引越界")
    covariance = None
    if vertex_covariance_mm2 is not None:
        covariance = np.asarray(vertex_covariance_mm2, dtype=np.float64)
        if covariance.shape != (len(pts), 2, 2):
            raise ValueError(
                f"{piece_id}: vertex_covariance_mm2 必须为 N×2×2"
            )
        if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-10):
            raise ValueError(f"{piece_id}: 顶点协方差必须对称")
        if np.any(np.linalg.eigvalsh(covariance) < -1e-10):
            raise ValueError(f"{piece_id}: 顶点协方差必须半正定")
    sigma = None
    if vertex_estimated_sigma_mm is not None:
        sigma = np.asarray(vertex_estimated_sigma_mm, dtype=float)
        if sigma.shape != (len(pts),):
            raise ValueError(
                f"{piece_id}: vertex_estimated_sigma_mm must have N values"
            )
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError(
                f"{piece_id}: vertex estimated sigma must be finite and nonnegative"
            )
    return ProcessedPiece(
        piece_id=piece_id,
        vertices=pts,
        area=float(poly.area),
        centroid=np.asarray([poly.centroid.x, poly.centroid.y]),
        edge_lengths=edge_lengths(pts),
        edge_angles=edge_angles(pts),
        internal_angles=internal_angles(pts),
        vertex_confidence=confidence,
        vertex_covariance_mm2=covariance,
        edges=edge_data,
        piece_confidence=float(1.0 if piece_confidence is None else piece_confidence),
        vertex_estimated_sigma_mm=sigma,
    )


def preprocess_observations(data: dict) -> list[ProcessedPiece]:
    if data.get("schema_version") != "1.0":
        raise ValueError("不支持的 observations schema_version")
    pieces = data.get("pieces")
    if not isinstance(pieces, list) or not 2 <= len(pieces) <= 4:
        raise ValueError("公开输入必须包含 2～4 片")
    ids = [item.get("id") for item in pieces]
    if len(set(ids)) != len(ids):
        raise ValueError("piece id 重复")
    result = []
    for item in pieces:
        vertex_records = item.get("vertices")
        raw_vertices = item.get("vertices_mm")
        inferred_confidence = None
        inferred_sigma = None
        if (
            isinstance(vertex_records, list)
            and vertex_records
            and isinstance(vertex_records[0], dict)
        ):
            raw_vertices = [
                [vertex["x_mm"], vertex["y_mm"]]
                for vertex in vertex_records
            ]
            inferred_confidence = [
                float(vertex.get("confidence", 1.0))
                for vertex in vertex_records
            ]
            values = [
                vertex.get("estimated_sigma_mm")
                for vertex in vertex_records
            ]
            if all(value is not None for value in values):
                inferred_sigma = values
        if raw_vertices is None:
            raise ValueError("piece 必须提供 vertices_mm 或结构化 vertices")
        result.append(preprocess_piece(
            piece_id=str(item["id"]),
            vertices=raw_vertices,
            vertex_confidence=(
                item.get("vertex_confidence") or inferred_confidence
            ),
            edges=item.get("edges"),
            piece_confidence=item.get("piece_confidence"),
            vertex_covariance_mm2=item.get("vertex_covariance_mm2"),
            vertex_estimated_sigma_mm=(
                item.get("vertex_estimated_sigma_mm") or inferred_sigma
            ),
        ))
    return result
