"""冻结控制输入、配置和输出契约，并验证其严格校验行为。

该文件不依赖相机或外部文件，可通过
``python -m unittest tests.unit.control.test_contracts`` 独立运行。
"""

from __future__ import annotations

from dataclasses import fields
import json
import math
import unittest
from typing import get_type_hints

import solver2 as public_control
from solver2 import (
    MovePlan,
    PlannedMove,
    PlanningError,
    PlanningOptions,
    PuzzlePiece,
)


class ControlContractTest(unittest.TestCase):
    """验证控制公共类型不暴露视觉置信度或松散字典入口。"""

    def test_public_exports_are_complete_and_minimal(self) -> None:
        """控制包应只公开计划要求的强类型接口。"""

        self.assertEqual(
            set(public_control.__all__),
            {
                "MovePlan",
                "PlannedMove",
                "PlanningError",
                "PlanningOptions",
                "PuzzlePiece",
                "plan_sheet_moves",
            },
        )
        self.assertIs(public_control.PuzzlePiece, PuzzlePiece)
        self.assertIs(public_control.PlanningOptions, PlanningOptions)
        self.assertIs(public_control.PlanningError, PlanningError)
        self.assertFalse(hasattr(public_control, "plan_from_observation"))

    def test_input_fields_exclude_all_visual_confidence_data(self) -> None:
        """纸片输入只包含 ID 和 A4 毫米轮廓。"""

        self.assertEqual(
            tuple(field.name for field in fields(PuzzlePiece)),
            ("piece_id", "vertices_mm"),
        )
        self.assertEqual(
            get_type_hints(PuzzlePiece),
            {
                "piece_id": str,
                "vertices_mm": tuple[tuple[float, float], ...],
            },
        )
        forbidden = {
            "piece_confidence",
            "vertex_confidence",
            "edges",
            "appearance_edge_matches",
        }
        self.assertTrue(
            forbidden.isdisjoint(
                field.name for field in fields(PuzzlePiece)
            )
        )

    def test_planning_options_have_safe_explicit_defaults(self) -> None:
        """默认配置应保持随机精确求解和明确吸头起点。"""

        options = PlanningOptions()
        self.assertEqual(options.mode, "random")
        self.assertEqual(options.noise_bound_mm, 0.0)
        self.assertEqual(options.search_top_n, 100)
        self.assertEqual(options.start_xy_a4_mm, (10.0, 10.0))
        self.assertEqual(options.minimum_execution_confidence, 0.6)

    def test_piece_rejects_invalid_identity_structure_and_coordinates(
        self,
    ) -> None:
        """纸片契约应拒绝空 ID、松散列表和非法 A4 坐标。"""

        with self.assertRaises(TypeError):
            PuzzlePiece(1, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PuzzlePiece(" ", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(TypeError):
            PuzzlePiece("p", [(0.0, 0.0)] * 3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PuzzlePiece("p", ((0.0, 0.0), (1.0, 0.0)))
        with self.assertRaises(ValueError):
            PuzzlePiece(
                "p",
                ((0.0, 0.0), (math.nan, 1.0), (0.0, 1.0)),
            )
        with self.assertRaises(ValueError):
            PuzzlePiece(
                "p",
                ((0.0, 0.0), (211.0, 1.0), (0.0, 1.0)),
            )

    def test_options_reject_invalid_values(self) -> None:
        """规划配置应拒绝非法模式、噪声、搜索规模和起点。"""

        with self.assertRaises(ValueError):
            PlanningOptions(mode="other")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PlanningOptions(noise_bound_mm=-0.1)
        with self.assertRaises(TypeError):
            PlanningOptions(search_top_n=True)
        with self.assertRaises(ValueError):
            PlanningOptions(search_top_n=0)
        with self.assertRaises(TypeError):
            PlanningOptions(minimum_execution_confidence=True)
        with self.assertRaises(ValueError):
            PlanningOptions(minimum_execution_confidence=1.01)
        with self.assertRaises(ValueError):
            PlanningOptions(start_xy_a4_mm=(211.0, 10.0))

    def test_move_plan_serialization_is_json_compatible(self) -> None:
        """控制输出应使用列表坐标且可直接序列化为 JSON。"""

        move = PlannedMove(
            order=1,
            piece_id="p0",
            pick_xy_a4_mm=(10.0, 20.0),
            pick_clearance_mm=3.0,
            rotation_deg_clockwise=90.0,
            place_xy_a4_mm=(100.0, 200.0),
            translation_a4_mm=(120.0, 190.0),
            observed_vertices_a4_mm=(
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ),
            target_vertices_a4_mm=(
                (100.0, 200.0),
                (100.0, 201.0),
                (99.0, 200.0),
            ),
        )
        plan = MovePlan(
            mode="random",
            confidence=0.8,
            ambiguous=False,
            target_width_mm=100.0,
            target_height_mm=60.0,
            target_center_a4_mm=(105.0, 222.75),
            moves=(move,),
        )

        serialized = plan.to_dict()

        self.assertIsInstance(
            serialized["target_rectangle"]["center_a4_mm"],  # type: ignore[index]
            list,
        )
        self.assertIsInstance(
            serialized["moves"][0]["pick_xy_a4_mm"],  # type: ignore[index]
            list,
        )
        json.dumps(serialized, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
