"""验证 Task1 两秒周期、A4 毫米跳变、纸片轨迹和 medoid。

该文件只运行纯内存状态机，不依赖视觉、相机或文件。可通过
``python -m unittest tests.unit.tasks.test_task1_stability`` 运行。
"""

from __future__ import annotations

import unittest

from ec_26_e.control import PlanningOptions, PuzzlePiece
from ec_26_e.tasks.task1 import Task1Options
from ec_26_e.tasks.task1.review import ReviewedFrame
from ec_26_e.tasks.task1.stability import StableSceneReviewer


BASE_WORKSPACE = (
    (100.0, 80.0),
    (1180.0, 80.0),
    (1180.0, 640.0),
    (100.0, 640.0),
)
BASE_SQUARE = (
    (20.0, 20.0),
    (40.0, 20.0),
    (40.0, 40.0),
    (20.0, 40.0),
)
BASE_TRIANGLE = (
    (100.0, 80.0),
    (120.0, 80.0),
    (110.0, 100.0),
)


def shifted_vertices(
    vertices: tuple[tuple[float, float], ...],
    offset: float,
) -> tuple[tuple[float, float], ...]:
    """把测试轮廓沿 x 轴平移指定毫米数。"""

    return tuple((x + offset, y) for x, y in vertices)


def make_reviewed_frame(
    sequence_id: int,
    received_at_ns: int,
    pieces: tuple[PuzzlePiece, ...],
    *,
    workspace_offset_px: float = 0.0,
) -> ReviewedFrame:
    """构造一张带原图 A4 四角的已审查帧。"""

    workspace = tuple(
        (x + workspace_offset_px, y) for x, y in BASE_WORKSPACE
    )
    return ReviewedFrame(
        sequence_id=sequence_id,
        received_at_ns=received_at_ns,
        workspace_corners_source_px=workspace,
        pieces=pieces,
    )


def two_pieces(
    first_id: int,
    second_id: int,
    *,
    offset: float = 0.0,
    cyclic_shift: int = 0,
) -> tuple[PuzzlePiece, ...]:
    """构造两个稳定形状并允许角点起点循环变化。"""

    square = shifted_vertices(BASE_SQUARE, offset)
    triangle = shifted_vertices(BASE_TRIANGLE, offset)
    if cyclic_shift:
        square = square[cyclic_shift:] + square[:cyclic_shift]
        triangle = triangle[cyclic_shift:] + triangle[:cyclic_shift]
    return (
        PuzzlePiece(str(first_id), square),
        PuzzlePiece(str(second_id), triangle),
    )


def four_pieces(sequence_id: int) -> tuple[PuzzlePiece, ...]:
    """构造 fixed 模式所需的四条相互分离稳定轨迹。"""

    return tuple(
        PuzzlePiece(
            str(sequence_id * 10 + index),
            shifted_vertices(BASE_SQUARE, index * 45.0),
        )
        for index in range(4)
    )


class StableSceneReviewerTest(unittest.TestCase):
    """验证周期审查不跨周期累计并正确排除跳变观测。"""

    def test_geometry_matching_medoid_and_transient_piece_filtering(
        self,
    ) -> None:
        """ID、循环顶点和逐帧跳变假纸片不得污染真实 medoid。"""

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=2.0,
                minimum_stable_frames=11,
                piece_vertex_tolerance_mm=2.0,
                acquisition_timeout_seconds=5.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        offsets = (
            -0.9,
            -0.7,
            -0.5,
            -0.3,
            -0.1,
            0.0,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        )
        result = None
        for index, offset in enumerate(offsets):
            stable = two_pieces(
                100 + index,
                200 + index,
                offset=offset,
                cyclic_shift=index % 2,
            )
            false_piece = PuzzlePiece(
                str(300 + index),
                shifted_vertices(BASE_TRIANGLE, 35.0 + 3.0 * index),
            )
            result = reviewer.add_frame(
                make_reviewed_frame(
                    index,
                    1_000_000_000 + index * 200_000_000,
                    stable + (false_piece,),
                )
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(
            tuple(piece.piece_id for piece in result.pieces),
            ("105", "205"),
        )

    def test_workspace_uses_previous_accepted_frame_in_a4_mm(
        self,
    ) -> None:
        """单帧尖峰应只记一次，恢复帧仍相对上一合格帧通过。"""

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=4,
                allowed_jump_frames=1,
                acquisition_timeout_seconds=3.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        result = None
        for index, offset_px in enumerate((0.0, 10.0, 0.0, 0.0, 0.0)):
            result = reviewer.add_frame(
                make_reviewed_frame(
                    index,
                    1_000_000_000 + index * 250_000_000,
                    two_pieces(index * 2, index * 2 + 1),
                    workspace_offset_px=offset_px,
                )
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics.workspace_jump_frames, 1)
        self.assertEqual(result.diagnostics.workspace_accepted_frames, 4)

    def test_six_workspace_jumps_reject_and_next_cycle_restarts(
        self,
    ) -> None:
        """第六张 A4 跳变帧应丢弃周期且不得污染下一周期。"""

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                allowed_jump_frames=5,
                acquisition_timeout_seconds=5.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        rejected = None
        for index, offset_px in enumerate(
            (0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        ):
            rejected = reviewer.add_frame(
                make_reviewed_frame(
                    index,
                    1_000_000_000 + round(index * 1_000_000_000 / 6),
                    two_pieces(index * 2, index * 2 + 1),
                    workspace_offset_px=offset_px,
                )
            )

        self.assertIsNotNone(rejected)
        assert rejected is not None
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.diagnostics.workspace_jump_frames, 6)

        self.assertIsNone(
            reviewer.add_frame(
                make_reviewed_frame(
                    20,
                    3_000_000_000,
                    two_pieces(20, 21),
                    workspace_offset_px=10.0,
                )
            )
        )
        accepted = reviewer.add_frame(
            make_reviewed_frame(
                21,
                4_000_000_000,
                two_pieces(22, 23),
                workspace_offset_px=10.0,
            )
        )
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.diagnostics.cycle_index, 2)

    def test_processing_failures_consume_every_object_allowance(
        self,
    ) -> None:
        """五张处理失败可容忍，第六张必须使 A4 周期失败。"""

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                allowed_jump_frames=5,
                acquisition_timeout_seconds=3.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        self.assertIsNone(
            reviewer.add_frame(
                make_reviewed_frame(
                    0,
                    1_000_000_000,
                    two_pieces(0, 1),
                )
            )
        )
        for index in range(5):
            self.assertIsNone(
                reviewer.note_processing_failure(
                    1_100_000_000 + index * 100_000_000
                )
            )
        accepted = reviewer.add_frame(
            make_reviewed_frame(
                1,
                2_000_000_000,
                two_pieces(2, 3),
            )
        )
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.diagnostics.processing_failure_frames, 5)
        self.assertTrue(
            all(count == 5 for _, count in accepted.diagnostics.piece_jump_frames)
        )

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                allowed_jump_frames=5,
                acquisition_timeout_seconds=3.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        reviewer.add_frame(
            make_reviewed_frame(10, 3_000_000_000, two_pieces(10, 11))
        )
        for index in range(6):
            result = reviewer.note_processing_failure(
                3_200_000_000 + index * 160_000_000
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.accepted)
        self.assertEqual(result.diagnostics.processing_failure_frames, 6)

    def test_piece_missing_frame_uses_own_jump_allowance(
        self,
    ) -> None:
        """单片一次漏检应排除该观测但保留其他纸片有效数据。"""

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=3,
                allowed_jump_frames=1,
                acquisition_timeout_seconds=3.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        result = None
        for index in range(4):
            pieces = two_pieces(index * 2, index * 2 + 1)
            if index == 1:
                pieces = pieces[:1]
            result = reviewer.add_frame(
                make_reviewed_frame(
                    index,
                    1_000_000_000 + round(index * 1_000_000_000 / 3),
                    pieces,
                )
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics.common_frame_count, 3)
        self.assertIn(1, dict(result.diagnostics.piece_jump_frames).values())

    def test_fixed_mode_requires_exactly_four_cycle_tracks(self) -> None:
        """fixed 模式不足四片时应拒绝，恰好四片时通过。"""

        options = Task1Options(
            review_cycle_seconds=1.0,
            minimum_stable_frames=2,
            acquisition_timeout_seconds=4.0,
            planning_options=PlanningOptions(mode="fixed"),
        )
        reviewer = StableSceneReviewer(options)
        reviewer.add_frame(
            make_reviewed_frame(0, 1_000_000_000, two_pieces(0, 1))
        )
        rejected = reviewer.add_frame(
            make_reviewed_frame(1, 2_000_000_000, two_pieces(2, 3))
        )
        self.assertIsNotNone(rejected)
        assert rejected is not None
        self.assertFalse(rejected.accepted)

        reviewer.add_frame(
            make_reviewed_frame(2, 3_000_000_000, four_pieces(2))
        )
        accepted = reviewer.add_frame(
            make_reviewed_frame(3, 4_000_000_000, four_pieces(3))
        )
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertTrue(accepted.accepted)
        self.assertEqual(len(accepted.pieces), 4)

    def test_random_mode_selects_stable_tracks_before_confidence(self) -> None:
        """超额随机候选应优先排除跳变更多的高置信度纸片。"""

        reviewer = StableSceneReviewer(
            Task1Options(
                review_cycle_seconds=1.0,
                minimum_stable_frames=2,
                allowed_jump_frames=1,
                acquisition_timeout_seconds=4.0,
                planning_options=PlanningOptions(mode="random"),
            )
        )
        pieces = tuple(
            PuzzlePiece(
                str(index),
                shifted_vertices(BASE_SQUARE, index * 30.0),
            )
            for index in range(5)
        )
        result = None
        for frame_index in range(3):
            visible = pieces if frame_index != 1 else pieces[:4]
            confidences = (
                (0.1, 0.2, 0.3, 0.4, 1.0)
                if frame_index != 1
                else (0.1, 0.2, 0.3, 0.4)
            )
            result = reviewer.add_frame(
                ReviewedFrame(
                    sequence_id=frame_index,
                    received_at_ns=1_000_000_000 + frame_index * 500_000_000,
                    workspace_corners_source_px=(
                        (100.0, 80.0),
                        (1180.0, 80.0),
                        (1180.0, 640.0),
                        (100.0, 640.0),
                    ),
                    pieces=visible,
                    piece_confidences=confidences,
                )
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics.stable_piece_count, 5)
        self.assertEqual(
            result.diagnostics.selected_piece_track_ids,
            (3, 2, 1, 0),
        )
        self.assertEqual(len(result.pieces), 4)


if __name__ == "__main__":
    unittest.main()
