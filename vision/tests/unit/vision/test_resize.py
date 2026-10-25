"""验证蓝色 A4 四边拟合、角点遮挡恢复和透视校正。

该文件只使用内存合成图像，不依赖相机或外部文件。可通过
``python -m unittest tests.unit.vision.test_resize`` 独立运行。
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from ec_26_e.vision import Resize, ResizeConfig


SOURCE_CORNERS = np.array(
    [[176, 78], [974, 69], [989, 638], [176, 646]],
    dtype=np.float32,
)


def create_perspective_scene(
    *,
    corner_occluded: bool = False,
    edge_occluded: bool = False,
    shadowed: bool = False,
    external_blue_object: bool = False,
    include_pieces: bool = True,
    background_color: tuple[int, int, int] = (24, 30, 25),
    workspace_color: tuple[int, int, int] = (180, 100, 35),
    top_edge_glare: bool = False,
) -> np.ndarray:
    """创建固定黑底上的蓝色 A4、可选扰动和内部纸片。"""

    image = np.full(
        (720, 1280, 3),
        background_color,
        dtype=np.uint8,
    )
    if external_blue_object:
        cv2.rectangle(
            image,
            (420, 675),
            (900, 719),
            (150, 145, 65),
            thickness=-1,
        )
    cv2.fillPoly(
        image,
        [SOURCE_CORNERS.astype(np.int32)],
        workspace_color,
    )
    if top_edge_glare:
        target = np.array(
            [[0, 0], [761, 0], [761, 539], [0, 539]],
            dtype=np.float32,
        )
        inverse = cv2.getPerspectiveTransform(target, SOURCE_CORNERS)
        glare = np.array(
            [[0, 0], [761, 0], [761, 75], [0, 75]],
            dtype=np.float32,
        )
        mapped_glare = cv2.perspectiveTransform(
            glare.reshape(1, -1, 2),
            inverse,
        )[0]
        cv2.fillPoly(
            image,
            [np.rint(mapped_glare).astype(np.int32)],
            (250, 250, 235),
        )
    if include_pieces:
        target = np.array(
            [[0, 0], [761, 0], [761, 539], [0, 539]],
            dtype=np.float32,
        )
        inverse = cv2.getPerspectiveTransform(target, SOURCE_CORNERS)
        pieces = (
            np.array(
                [[100, 100], [250, 100], [250, 210], [100, 210]],
                dtype=np.float32,
            ),
            np.array(
                [[450, 300], [590, 300], [520, 430]],
                dtype=np.float32,
            ),
        )
        for piece in pieces:
            mapped = cv2.perspectiveTransform(
                piece.reshape(1, -1, 2),
                inverse,
            )[0]
            cv2.fillPoly(
                image,
                [np.rint(mapped).astype(np.int32)],
                (235, 235, 235),
            )
    if corner_occluded:
        occluder = np.array(
            [
                [900, 35],
                [1030, 30],
                [1040, 180],
                [990, 175],
                [955, 135],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [occluder], (235, 235, 235))
    if shadowed:
        shadow = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            shadow,
            [
                np.array(
                    [[130, 230], [720, 120], [780, 560], [180, 650]],
                    dtype=np.int32,
                )
            ],
            255,
        )
        image[shadow > 0] = np.clip(
            image[shadow > 0].astype(np.float32) * 0.55,
            0,
            255,
        ).astype(np.uint8)
    if edge_occluded:
        cv2.rectangle(
            image,
            (300, 600),
            (850, 719),
            (120, 165, 205),
            thickness=-1,
        )
    return image


class ResizeConfigTest(unittest.TestCase):
    """验证工作区定位和横向 A4 尺寸配置约束。"""

    def test_default_config_is_valid(self) -> None:
        """默认配置应使用 640×360 定位和 762×540 输出。"""

        config = ResizeConfig()

        self.assertEqual(
            (config.detect_width, config.detect_height),
            (640, 360),
        )
        self.assertEqual(
            (config.target_width, config.target_height),
            (762, 540),
        )
        self.assertEqual(len(config.reference_corners_relative), 4)
        self.assertEqual(config.search_band_ratio, 0.14)

    def test_invalid_a4_ratio_is_rejected(self) -> None:
        """横向输出偏离 A4 比例超过容差时应立即失败。"""

        with self.assertRaisesRegex(ValueError, "292:207"):
            ResizeConfig(target_width=800, target_height=500)

    def test_portrait_target_is_rejected(self) -> None:
        """目标宽度不大于高度时应拒绝竖向坐标系。"""

        with self.assertRaisesRegex(ValueError, "横向"):
            ResizeConfig(target_width=540, target_height=764)

    def test_reference_corners_are_frozen_from_json_lists(self) -> None:
        """JSON 风格列表参考四角应转换为不可变元组。"""

        config = ResizeConfig(
            reference_corners_relative=[
                [0.14, 0.07],
                [0.78, 0.08],
                [0.77, 0.87],
                [0.14, 0.84],
            ]
        )

        self.assertIsInstance(config.reference_corners_relative, tuple)
        self.assertTrue(
            all(
                isinstance(point, tuple)
                for point in config.reference_corners_relative
            )
        )

    def test_invalid_reference_corner_order_is_rejected(self) -> None:
        """非顺时针参考四角不能进入边缘搜索。"""

        with self.assertRaisesRegex(ValueError, "顺时针"):
            ResizeConfig(
                reference_corners_relative=(
                    (0.14, 0.07),
                    (0.14, 0.84),
                    (0.77, 0.87),
                    (0.78, 0.08),
                )
            )


class ResizeTest(unittest.TestCase):
    """验证四边拟合结果、单应矩阵和输入边界。"""

    def test_rectifies_perspective_workspace(self) -> None:
        """正常场景应恢复源四角并输出配置尺寸。"""

        image = create_perspective_scene()
        source = image.copy()

        result = Resize().resize(image)

        self.assertEqual(result.image_bgr.shape, (540, 762, 3))
        np.testing.assert_allclose(
            np.asarray(result.source_corners_px),
            SOURCE_CORNERS,
            atol=2.0,
        )
        np.testing.assert_array_equal(image, source)

    def test_homography_maps_source_to_target_corners(self) -> None:
        """正向单应矩阵应把四个源角映射到 A4 图四角。"""

        result = Resize().resize(create_perspective_scene())
        source = np.asarray(
            result.source_corners_px,
            dtype=np.float32,
        ).reshape(1, -1, 2)
        mapped = cv2.perspectiveTransform(source, result.homography)[0]
        expected = np.array(
            [[0, 0], [761, 0], [761, 539], [0, 539]],
            dtype=np.float32,
        )

        np.testing.assert_allclose(mapped, expected, atol=0.1)

    def test_recovers_partially_occluded_corner_from_four_lines(self) -> None:
        """右上角被局部遮挡时应由剩余边段交点恢复真实角点。"""

        result = Resize().resize(
            create_perspective_scene(corner_occluded=True)
        )

        np.testing.assert_allclose(
            np.asarray(result.source_corners_px),
            SOURCE_CORNERS,
            atol=3.0,
        )

    def test_missing_workspace_uses_reference_fallback(self) -> None:
        """没有可用边线时仍应输出固定参考工作区。"""

        image = np.full((720, 1280, 3), (0, 0, 255), dtype=np.uint8)

        result = Resize().resize(image)

        self.assertEqual(result.image_bgr.shape, (540, 762, 3))
        self.assertEqual(result.line_residual, 0.0)

    def test_workspace_touching_frame_border_uses_fallback(self) -> None:
        """边界接触画框时不应中断后续工作区输出。"""

        image = np.full((720, 1280, 3), (180, 100, 35), dtype=np.uint8)

        result = Resize().resize(image)

        self.assertEqual(result.image_bgr.shape, (540, 762, 3))
        self.assertEqual(result.line_residual, 0.0)

    def test_moderate_brightness_changes_preserve_corner_accuracy(
        self,
    ) -> None:
        """普通变暗和变亮不应明显移动拟合四角。"""

        image = create_perspective_scene()
        for alpha in (0.55, 1.30):
            adjusted = np.clip(
                image.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

            result = Resize().resize(adjusted)

            np.testing.assert_allclose(
                np.asarray(result.source_corners_px),
                SOURCE_CORNERS,
                atol=2.0,
            )

    def test_partial_shadow_preserves_corner_accuracy(self) -> None:
        """跨过蓝纸内部和部分边缘的软阴影不应改变透视四角。"""

        result = Resize().resize(
            create_perspective_scene(shadowed=True)
        )

        np.testing.assert_allclose(
            np.asarray(result.source_corners_px),
            SOURCE_CORNERS,
            atol=2.5,
        )

    def test_dark_blue_cast_black_base_is_not_treated_as_blue_paper(
        self,
    ) -> None:
        """暗黑底红通道截断时仍应依靠明度边界识别青蓝卡纸。"""

        result = Resize().resize(
            create_perspective_scene(
                background_color=(28, 22, 2),
                workspace_color=(251, 200, 91),
            )
        )

        np.testing.assert_allclose(
            np.asarray(result.source_corners_px),
            SOURCE_CORNERS,
            atol=2.0,
        )

    def test_local_glare_does_not_veto_other_blue_paper_regions(
        self,
    ) -> None:
        """单侧局部强光褪去纸色时其他有效区域应支持工作区。"""

        result = Resize().resize(
            create_perspective_scene(top_edge_glare=True)
        )

        np.testing.assert_allclose(
            np.asarray(result.source_corners_px),
            SOURCE_CORNERS,
            atol=2.0,
        )

    def test_non_blue_bright_quadrilateral_uses_reference_fallback(self) -> None:
        """非蓝候选不能令管线中断，应回退固定参考工作区。"""

        result = Resize().resize(
            create_perspective_scene(
                include_pieces=False,
                workspace_color=(170, 170, 170),
            )
        )

        self.assertEqual(result.image_bgr.shape, (540, 762, 3))
        self.assertEqual(result.line_residual, 0.0)

    def test_external_blue_object_is_ignored(self) -> None:
        """机器下方青色物体不能参与固定 A4 四边搜索。"""

        result = Resize().resize(
            create_perspective_scene(external_blue_object=True)
        )

        np.testing.assert_allclose(
            np.asarray(result.source_corners_px),
            SOURCE_CORNERS,
            atol=2.0,
        )

    def test_large_edge_occlusion_uses_fallback(self) -> None:
        """大面积遮挡底边时应以断续边线继续输出工作区。"""

        result = Resize().resize(create_perspective_scene(edge_occluded=True))

        self.assertEqual(result.image_bgr.shape, (540, 762, 3))
        self.assertEqual(result.line_residual, 0.0)

    def test_detect_size_cannot_exceed_input(self) -> None:
        """定位图不能通过放大输入图生成。"""

        image = np.zeros((360, 640, 3), dtype=np.uint8)
        resizer = Resize(
            ResizeConfig(detect_width=1280, detect_height=720)
        )

        with self.assertRaisesRegex(ValueError, "不能大于"):
            resizer.resize(image)

    def test_detect_size_must_preserve_source_aspect(self) -> None:
        """定位分辨率改变输入宽高比时应在分割前失败。"""

        resizer = Resize(
            ResizeConfig(detect_width=640, detect_height=400)
        )

        with self.assertRaisesRegex(ValueError, "保持输入图像宽高比"):
            resizer.resize(create_perspective_scene())


if __name__ == "__main__":
    unittest.main()
