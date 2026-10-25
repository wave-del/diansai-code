"""冻结视觉公共数据对象、坐标语义和序列化契约。

该文件只使用内存合成图像，不依赖相机或外部文件。可通过
``python -m unittest tests.unit.vision.test_result_contract`` 独立运行。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import json
import unittest
from typing import get_type_hints

from ec_26_e import vision as public_vision
from ec_26_e.vision import (
    Detect1Result,
    PieceDetection,
)
from ec_26_e.vision.geometry import relative_point


PIECE_FIELD_NAMES = (
    "group_id",
    "corners_px",
    "center_px",
    "corners_relative",
    "center_relative",
    "area_px",
    "area_relative",
    "polygon_iou",
    "line_residual",
    "confidence",
)
RESULT_FIELD_NAMES = (
    "image_size",
    "pieces",
    "foreground_threshold",
    "workspace_area_px",
    "coordinate_system",
)
PIECE_DICT_KEYS = (
    "group_id",
    "corners_px",
    "center_px",
    "corners_relative",
    "center_relative",
    "area_px",
    "area_relative",
    "polygon_iou",
    "line_residual",
    "confidence",
)
RESULT_DICT_KEYS = (
    "image_size",
    "coordinate_system",
    "piece_count",
    "groups",
    "diagnostics",
)


class VisionResultPublicContractTest(unittest.TestCase):
    """验证任务后续继续依赖的冻结视觉结果契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        """构造一次确定性结果供全部契约断言复用。"""

        width, height = 764, 540
        corners = (
            (100.0, 100.0),
            (220.0, 100.0),
            (220.0, 200.0),
            (100.0, 200.0),
        )
        center = (160.0, 150.0)
        piece = PieceDetection(
            group_id=0,
            corners_px=corners,
            center_px=center,
            corners_relative=tuple(
                relative_point(point, width, height)
                for point in corners
            ),
            center_relative=relative_point(center, width, height),
            area_px=12000.0,
            area_relative=round(12000.0 / (width * height), 6),
            polygon_iou=0.99,
            line_residual=0.2,
            confidence=0.95,
        )
        cls.result = Detect1Result(
            (width, height),
            (piece,),
            0.15,
            width * height,
        )

    def test_public_exports_and_field_types_are_frozen(self) -> None:
        """公共导出、字段顺序和类型提示必须保持不变。"""

        self.assertIs(public_vision.PieceDetection, PieceDetection)
        self.assertIs(public_vision.Detect1Result, Detect1Result)
        self.assertEqual(
            tuple(field.name for field in fields(PieceDetection)),
            PIECE_FIELD_NAMES,
        )
        self.assertEqual(
            tuple(field.name for field in fields(Detect1Result)),
            RESULT_FIELD_NAMES,
        )
        self.assertEqual(
            get_type_hints(PieceDetection),
            {
                "group_id": int,
                "corners_px": tuple[tuple[float, float], ...],
                "center_px": tuple[float, float],
                "corners_relative": tuple[tuple[float, float], ...],
                "center_relative": tuple[float, float],
                "area_px": float,
                "area_relative": float,
                "polygon_iou": float,
                "line_residual": float,
                "confidence": float,
            },
        )
        self.assertEqual(
            get_type_hints(Detect1Result),
            {
                "image_size": tuple[int, int],
                "pieces": tuple[PieceDetection, ...],
                "foreground_threshold": float,
                "workspace_area_px": int,
                "coordinate_system": str,
            },
        )

    def test_result_objects_are_immutable_slot_dataclasses(self) -> None:
        """结果和纸片均应使用不可变元组及 frozen/slots 数据对象。"""

        self.assertIsInstance(self.result.pieces, tuple)
        self.assertFalse(hasattr(self.result, "__dict__"))
        self.assertFalse(hasattr(self.result.pieces[0], "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            self.result.coordinate_system = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            self.result.pieces[0].group_id = 99  # type: ignore[misc]

    def test_coordinate_system_and_polygon_order_are_canonical(self) -> None:
        """坐标系、顺时针环绕和起始角点规则必须保持稳定。"""

        self.assertEqual(
            self.result.coordinate_system,
            "a4_landscape_top_left",
        )
        for piece in self.result.pieces:
            points = piece.corners_px
            signed_area = 0.5 * sum(
                x * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * y
                for index, (x, y) in enumerate(points)
            )
            self.assertGreater(signed_area, 0.0)
            self.assertEqual(
                points[0],
                min(points, key=lambda point: (point[1], point[0])),
            )

    def test_relative_coordinates_derive_from_rounded_pixel_coordinates(
        self,
    ) -> None:
        """相对坐标应继续由两位像素坐标按闭区间端点归一化。"""

        width, height = self.result.image_size
        for piece in self.result.pieces:
            expected_corners = tuple(
                (
                    round(x / (width - 1), 6),
                    round(y / (height - 1), 6),
                )
                for x, y in piece.corners_px
            )
            expected_center = (
                round(piece.center_px[0] / (width - 1), 6),
                round(piece.center_px[1] / (height - 1), 6),
            )
            self.assertEqual(piece.corners_relative, expected_corners)
            self.assertEqual(piece.center_relative, expected_center)

    def test_serialized_shape_is_exact_and_json_compatible(self) -> None:
        """字典键、诊断结构和 JSON 兼容性必须保持不变。"""

        serialized = self.result.to_dict()
        self.assertEqual(tuple(serialized), RESULT_DICT_KEYS)
        self.assertEqual(serialized["image_size"], [764, 540])
        self.assertEqual(
            serialized["coordinate_system"],
            "a4_landscape_top_left",
        )
        self.assertEqual(serialized["piece_count"], len(self.result.pieces))
        groups = serialized["groups"]
        self.assertIsInstance(groups, list)
        self.assertEqual(tuple(groups[0]), PIECE_DICT_KEYS)
        self.assertEqual(
            tuple(serialized["diagnostics"]),
            ("foreground_threshold", "workspace_area_px"),
        )
        json.dumps(serialized, ensure_ascii=False)

    def test_repeated_serialization_does_not_mutate_source(self) -> None:
        """重复序列化应返回独立容器且不改变原始 dataclass。"""

        original_corners = self.result.pieces[0].corners_px
        first = self.result.to_dict()
        second = self.result.to_dict()

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first["groups"], second["groups"])
        first["groups"][0]["corners_px"][0][0] = -1.0
        self.assertEqual(self.result.pieces[0].corners_px, original_corners)
        self.assertNotEqual(first, self.result.to_dict())

if __name__ == "__main__":
    unittest.main()
