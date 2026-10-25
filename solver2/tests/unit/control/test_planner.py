"""验证固定与随机拼图、刚体闭环、错误分类和无文件副作用。

该文件不依赖相机或外部文件，可通过
``python -m unittest tests.unit.control.test_planner`` 独立运行。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from shapely.geometry import Polygon

from solver2 import (
    PlanningError,
    PlanningOptions,
    PuzzlePiece,
    plan_sheet_moves,
)
from solver2._engine.solver.interface_matching import (
    build_shared_interface_candidates,
)
from solver2._engine.solver.observation_uncertainty import (
    ObservationUncertaintyProfile,
)
from solver2._engine.solver.preprocess import preprocess_piece


FIXED_PIECES = (
    PuzzlePiece(
        "p0",
        ((0, 0), (20, 0), (36, 12), (0, 20)),
    ),
    PuzzlePiece(
        "p1",
        ((0, 20), (36, 12), (76, 42), (0, 30)),
    ),
    PuzzlePiece(
        "p2",
        ((0, 30), (76, 42), (100, 60), (0, 60)),
    ),
    PuzzlePiece(
        "p3",
        ((20, 0), (100, 0), (100, 60), (76, 42), (36, 12)),
    ),
)


def transform_points(
    points: tuple[tuple[float, float], ...],
    angle_deg_clockwise: float,
    translation: tuple[float, float],
) -> np.ndarray:
    """在测试中独立计算 y 向下坐标系的二维刚体变换。"""

    radians = math.radians(angle_deg_clockwise)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )
    return (
        np.asarray(points, dtype=float) @ rotation.T
        + np.asarray(translation, dtype=float)
    )


class FixedPlannerTest(unittest.TestCase):
    """验证固定模板求解和每片纸的控制闭环。"""

    @classmethod
    def setUpClass(cls) -> None:
        """只执行一次固定模板求解供本类全部断言复用。"""

        cls.plan = plan_sheet_moves(
            FIXED_PIECES,
            PlanningOptions(mode="fixed"),
        )

    def test_fixed_plan_is_complete_and_on_lower_half(self) -> None:
        """固定四片应形成 100×60 mm 矩形并位于下半区。"""

        self.assertEqual(len(self.plan.moves), 4)
        self.assertAlmostEqual(self.plan.target_width_mm, 100.0)
        self.assertAlmostEqual(self.plan.target_height_mm, 60.0)
        self.assertEqual(self.plan.target_center_a4_mm, (105.0, 222.75))
        self.assertEqual(
            [move.order for move in self.plan.moves],
            [1, 2, 3, 4],
        )
        self.assertGreaterEqual(
            min(
                y
                for move in self.plan.moves
                for _, y in move.target_vertices_a4_mm
            ),
            148.5,
        )
        self.assertTrue(
            all(move.template_id is not None for move in self.plan.moves)
        )
        self.assertTrue(
            all(0.0 <= move.rotation_deg_clockwise <= 360.0
                for move in self.plan.moves)
        )

    def test_each_move_satisfies_rigid_transform_closure(self) -> None:
        """每片目标轮廓和放置点都应满足同一个 SE(2) 变换。"""

        for move in self.plan.moves:
            transformed = transform_points(
                move.observed_vertices_a4_mm,
                move.rotation_deg_clockwise,
                move.translation_a4_mm,
            )
            np.testing.assert_allclose(
                transformed,
                move.target_vertices_a4_mm,
                atol=1e-6,
            )
            transformed_pick = transform_points(
                (move.pick_xy_a4_mm,),
                move.rotation_deg_clockwise,
                move.translation_a4_mm,
            )[0]
            np.testing.assert_allclose(
                transformed_pick,
                move.place_xy_a4_mm,
                atol=1e-6,
            )
            self.assertGreater(move.pick_clearance_mm, 0.0)

    def test_fixed_plan_uses_gapped_non_overlapping_final_layout(self) -> None:
        """固定模板最终落位应保留约 6 mm 接缝且绝不产生面积重叠。"""

        polygons = [
            Polygon(move.target_vertices_a4_mm)
            for move in self.plan.moves
        ]
        distances = []
        for first, polygon in enumerate(polygons):
            for other in polygons[first + 1 :]:
                self.assertLessEqual(
                    polygon.intersection(other).area,
                    1e-6,
                )
                distances.append(polygon.distance(other))
        self.assertGreaterEqual(min(distances), 5.0)

    def test_result_serializes_without_mutating_input(self) -> None:
        """方案序列化应成功且不得改变不可变输入纸片。"""

        snapshot = tuple(piece.vertices_mm for piece in FIXED_PIECES)

        serialized = self.plan.to_dict()

        json.dumps(serialized, ensure_ascii=False)
        self.assertEqual(
            tuple(piece.vertices_mm for piece in FIXED_PIECES),
            snapshot,
        )

    def test_planning_has_no_file_side_effect(self) -> None:
        """纯控制求解不得在当前目录创建任何文件。"""

        original_directory = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                plan_sheet_moves(
                    FIXED_PIECES,
                    PlanningOptions(mode="fixed"),
                )
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(original_directory)


class RandomPlannerTest(unittest.TestCase):
    """验证未知拓扑搜索保留参考算法的完整矩形能力。"""

    def test_random_mode_solves_static_unknown_topology_case(self) -> None:
        """相同四片在 random 模式下也应恢复 100×60 mm 矩形。"""

        plan = plan_sheet_moves(
            FIXED_PIECES,
            PlanningOptions(mode="random"),
        )

        self.assertEqual(len(plan.moves), 4)
        self.assertAlmostEqual(plan.target_width_mm, 100.0, places=6)
        self.assertAlmostEqual(plan.target_height_mm, 60.0, places=6)
        self.assertTrue(
            all(move.template_id is None for move in plan.moves)
        )

    def test_random_interface_pool_does_not_depend_on_hard_radius(self) -> None:
        """未知观测误差不得改变随机拓扑候选的生成和早期剪裁。"""

        pieces = [
            preprocess_piece(
                piece.piece_id,
                [list(vertex) for vertex in piece.vertices_mm],
            )
            for piece in FIXED_PIECES
        ]

        def placement_signatures(radius_mm: float) -> set[tuple]:
            """用不同兼容字段值生成候选接口的离散签名。"""

            profile = ObservationUncertaintyProfile(
                hard_error_radius_mm=radius_mm,
                typical_sigma_mm=None,
                calibration_rms_mm=None,
                edge_fit_rms_mm=None,
                mean_vertex_confidence=None,
                source="unit_test",
                quality_level="medium",
                absolute_measurement_sigma_known=False,
            )
            _candidates, placements, diagnostics = (
                build_shared_interface_candidates(
                    pieces,
                    include_generic_partial=True,
                    uncertainty_profile=profile,
                )
            )
            self.assertEqual(
                diagnostics["generic_partial_candidate_count"],
                sum(
                    placement.interface_type in {
                        "partial_to_full", "full_to_partial",
                        "partial_to_partial",
                    }
                    for placement in placements
                ),
            )
            return {
                (
                    placement.first_piece,
                    placement.first_segment.edge_index,
                    round(placement.first_segment.t_start, 7),
                    round(placement.first_segment.t_end, 7),
                    placement.second_piece,
                    placement.second_segment.edge_index,
                    round(placement.second_segment.t_start, 7),
                    round(placement.second_segment.t_end, 7),
                )
                for placement in placements
            }

        self.assertEqual(
            placement_signatures(0.1),
            placement_signatures(50.0),
        )

    def test_random_topology_pool_bounds_one_to_two_and_skips_free_partials(
        self,
    ) -> None:
        """随机接口只保留可解释的 T 拓扑，不枚举自由区间互贴。"""

        pieces = [
            preprocess_piece(
                piece.piece_id,
                [list(vertex) for vertex in piece.vertices_mm],
            )
            for piece in FIXED_PIECES
        ]
        _candidates, placements, diagnostics = (
            build_shared_interface_candidates(
                pieces, include_generic_partial=True
            )
        )
        one_to_two_ids = {
            placement.composite_id
            for placement in placements
            if placement.composite_id is not None
        }
        self.assertLessEqual(
            len(one_to_two_ids),
            sum(len(piece.edge_lengths) for piece in pieces)
            * diagnostics["one_to_two_budget_per_long_edge"],
        )
        self.assertEqual(
            0,
            diagnostics["partial_to_partial_candidate_count"],
        )


class PlannerValidationTest(unittest.TestCase):
    """验证结构错误和合法但无解输入使用不同异常类型。"""

    def test_rejects_invalid_sequence_and_mode_piece_count(self) -> None:
        """入口应拒绝松散字典、空输入和错误 fixed 数量。"""

        with self.assertRaises(TypeError):
            plan_sheet_moves([{"id": "p"}])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            plan_sheet_moves(())
        with self.assertRaises(ValueError):
            plan_sheet_moves(
                FIXED_PIECES[:2],
                PlanningOptions(mode="fixed"),
            )

    def test_rejects_duplicate_ids_and_invalid_polygons(self) -> None:
        """入口应拒绝重复 ID、重复闭合点、自交和退化轮廓。"""

        duplicate = (
            PuzzlePiece("p", ((0, 0), (10, 0), (0, 10))),
            PuzzlePiece("p", ((20, 0), (30, 0), (20, 10))),
        )
        with self.assertRaises(ValueError):
            plan_sheet_moves(duplicate)

        repeated_closure = (
            PuzzlePiece("a", ((0, 0), (10, 0), (0, 10), (0, 0))),
            PuzzlePiece("b", ((20, 0), (30, 0), (20, 10))),
        )
        with self.assertRaises(ValueError):
            plan_sheet_moves(repeated_closure)

        self_intersecting = (
            PuzzlePiece("a", ((0, 0), (10, 10), (0, 10), (10, 0))),
            PuzzlePiece("b", ((20, 0), (30, 0), (20, 10))),
        )
        with self.assertRaises(ValueError):
            plan_sheet_moves(self_intersecting)

        degenerate = (
            PuzzlePiece("a", ((0, 0), (5, 0), (10, 0))),
            PuzzlePiece("b", ((20, 0), (30, 0), (20, 10))),
        )
        with self.assertRaises(ValueError):
            plan_sheet_moves(degenerate)

    def test_valid_but_unsolved_input_raises_planning_error(self) -> None:
        """互不匹配的合法三角形应报告无解而不是格式错误。"""

        pieces = (
            PuzzlePiece("a", ((0, 0), (10, 0), (0, 10))),
            PuzzlePiece("b", ((30, 0), (40, 0), (30, 10))),
        )

        with self.assertRaises(PlanningError):
            plan_sheet_moves(
                pieces,
                PlanningOptions(mode="random"),
            )


if __name__ == "__main__":
    unittest.main()
