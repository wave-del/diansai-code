"""边描述子。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .preprocess import ProcessedPiece


@dataclass
class EdgeDescriptor:
    piece_id: str
    edge_index: int
    start: list[float]
    end: list[float]
    length: float
    direction_deg: float
    previous_length: float
    next_length: float
    start_internal_angle_deg: float
    end_internal_angle_deg: float
    adjacent_direction_delta_deg: float
    possible_outer_edge: bool
    local_signature: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def describe_edges(piece: ProcessedPiece) -> list[EdgeDescriptor]:
    n = len(piece.vertices)
    median = float(np.median(piece.edge_lengths))
    result = []
    for i in range(n):
        result.append(
            EdgeDescriptor(
                piece.piece_id,
                i,
                piece.vertices[i].tolist(),
                piece.vertices[(i + 1) % n].tolist(),
                float(piece.edge_lengths[i]),
                float(piece.edge_angles[i]),
                float(piece.edge_lengths[i - 1]),
                float(piece.edge_lengths[(i + 1) % n]),
                float(piece.internal_angles[i]),
                float(piece.internal_angles[(i + 1) % n]),
                float((piece.edge_angles[(i + 1) % n] - piece.edge_angles[i] + 180) % 360 - 180),
                bool(piece.edge_lengths[i] <= median),
                [
                    float(piece.edge_lengths[i]),
                    float(piece.edge_lengths[i - 1]),
                    float(piece.edge_lengths[(i + 1) % n]),
                    float(piece.internal_angles[i]),
                    float(piece.internal_angles[(i + 1) % n]),
                ],
            )
        )
    return result

