from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon


@dataclass
class PuzzlePiece:
    id: int
    polygon: Polygon


@dataclass
class Candidate:
    poses: dict[int, np.ndarray]
    score: float
    metrics: dict[str, float]
    seams: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SolveResult:
    candidates: list[Candidate]
    elapsed_s: float
    status: str
    expanded_states: int
