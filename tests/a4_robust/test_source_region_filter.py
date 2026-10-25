"""左侧拼图来源区过滤的纯离线回归测试。"""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from source_region_filter import filter_left_source_pieces


@dataclass(frozen=True)
class _Piece:
    name: str
    corners_relative: tuple[tuple[float, float], ...]


class SourceRegionFilterTests(unittest.TestCase):
    def test_accepts_piece_completely_in_left_source_region(self) -> None:
        left = _Piece("left", ((0.10, 0.1), (0.52, 0.1), (0.52, 0.3)))
        result = filter_left_source_pieces((left,), max_x_ratio=0.52)
        self.assertEqual(result.accepted, (left,))
        self.assertEqual(result.rejected, ())

    def test_rejects_right_and_boundary_crossing_objects(self) -> None:
        right = _Piece("right", ((0.60, 0.1), (0.80, 0.1), (0.80, 0.3)))
        crossing = _Piece("crossing", ((0.42, 0.1), (0.53, 0.1), (0.42, 0.3)))
        result = filter_left_source_pieces((right, crossing), max_x_ratio=0.52)
        self.assertEqual(result.accepted, ())
        self.assertEqual(result.rejected, (right, crossing))


if __name__ == "__main__":
    unittest.main()
