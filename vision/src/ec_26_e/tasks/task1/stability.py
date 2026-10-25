"""按固定周期审查 A4 与纸片毫米几何，并选择共同合格代表帧。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ec_26_e.control import PuzzlePiece
from ec_26_e.tasks.task1.contracts import Task1Options
from ec_26_e.tasks.task1.review import ReviewedFrame


A4_LANDSCAPE_WIDTH_MM = 292.0
A4_LANDSCAPE_HEIGHT_MM = 207.0
_A4_CORNERS_MM = np.asarray(
    (
        (0.0, 0.0),
        (A4_LANDSCAPE_WIDTH_MM, 0.0),
        (A4_LANDSCAPE_WIDTH_MM, A4_LANDSCAPE_HEIGHT_MM),
        (0.0, A4_LANDSCAPE_HEIGHT_MM),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class CycleReviewDiagnostics:
    """描述一个完整或正在采集的审查周期现场。"""

    cycle_index: int
    processing_failure_frames: int
    workspace_jump_frames: int
    workspace_accepted_frames: int
    piece_jump_frames: tuple[tuple[int, int], ...]
    stable_piece_count: int
    selected_piece_track_ids: tuple[int, ...]
    selected_track_scores: tuple[
        tuple[int, int, int, float, float], ...
    ]
    common_frame_count: int
    reason: str

    def to_text(self) -> str:
        """转换为适合超时错误和运行日志的紧凑中文文本。"""

        piece_text = (
            ",".join(
                f"track-{track_id}:{count}"
                for track_id, count in self.piece_jump_frames
            )
            or "无"
        )
        selection_text = (
            ",".join(
                "track-"
                f"{track_id}(跳变={jumps},观测={observations},"
                f"抖动={jitter:.3f},置信度={confidence:.3f})"
                for (
                    track_id,
                    jumps,
                    observations,
                    jitter,
                    confidence,
                ) in self.selected_track_scores
            )
            or "无"
        )
        return (
            f"周期={self.cycle_index}，"
            f"处理失败帧={self.processing_failure_frames}，"
            f"A4跳变帧={self.workspace_jump_frames}，"
            f"A4合格帧={self.workspace_accepted_frames}，"
            f"纸片跳变={piece_text}，"
            f"合格纸片={self.stable_piece_count}，"
            f"选中轨迹={self.selected_piece_track_ids or '无'}，"
            f"选中分数={selection_text}，"
            f"共同合格帧={self.common_frame_count}，"
            f"周期原因={self.reason}"
        )


@dataclass(frozen=True, slots=True)
class CycleReviewResult:
    """保存一个已结束周期的通过状态、代表纸片和诊断。"""

    accepted: bool
    pieces: tuple[PuzzlePiece, ...]
    diagnostics: CycleReviewDiagnostics


@dataclass(frozen=True, slots=True)
class _CycleEvent:
    """保存周期内一张有效 ReviewedFrame 或一次处理失败。"""

    observed_at_ns: int
    frame: ReviewedFrame | None


@dataclass(slots=True)
class _PieceObservation:
    """保存一张帧内的原始纸片和跨帧对齐顶点。"""

    piece: PuzzlePiece
    aligned_vertices: NDArray[np.float64]
    confidence: float


@dataclass(slots=True)
class _PieceTrack:
    """保存周期内一片纸的有效观测和跳变/缺失帧数。"""

    track_id: int
    observations: dict[int, _PieceObservation] = field(default_factory=dict)
    jump_frames: int = 0

    @property
    def last_vertices(self) -> NDArray[np.float64]:
        """返回最近一张合格观测的对齐顶点。"""

        return next(reversed(self.observations.values())).aligned_vertices


class StableSceneReviewer:
    """按互不重叠周期累计事件并统一审查 A4 与纸片稳定性。"""

    def __init__(self, options: Task1Options) -> None:
        """保存阈值并初始化第一个空周期。"""

        if not isinstance(options, Task1Options):
            raise TypeError("options 必须是 Task1Options")
        self.options = options
        self._cycle_index = 1
        self._cycle_started_at_ns: int | None = None
        self._events: list[_CycleEvent] = []
        self._last_diagnostics: CycleReviewDiagnostics | None = None

    @property
    def cycle_index(self) -> int:
        """返回当前等待采集或正在采集的周期序号。"""

        return self._cycle_index

    def reset(self) -> None:
        """清空全部周期状态并重新从第一个周期开始。"""

        self._cycle_index = 1
        self._cycle_started_at_ns = None
        self._events.clear()
        self._last_diagnostics = None

    def note_processing_failure(
        self,
        observed_at_ns: int,
    ) -> CycleReviewResult | None:
        """把相机、Resize、Detect1 或单帧审查失败加入当前周期。"""

        self._validate_timestamp(observed_at_ns)
        return self._append_event(
            _CycleEvent(observed_at_ns=observed_at_ns, frame=None)
        )

    def add_frame(
        self,
        frame: ReviewedFrame,
    ) -> CycleReviewResult | None:
        """加入一张有效帧，并在周期结束时返回统一审查结果。"""

        if not isinstance(frame, ReviewedFrame):
            raise TypeError("frame 必须是 ReviewedFrame")
        self._validate_timestamp(frame.received_at_ns)
        if any(
            event.frame is not None
            and event.frame.sequence_id == frame.sequence_id
            for event in self._events
        ):
            raise ValueError("同一审查周期不能重复加入帧序号")
        return self._append_event(
            _CycleEvent(
                observed_at_ns=frame.received_at_ns,
                frame=frame,
            )
        )

    def diagnostics_text(self) -> str:
        """返回当前周期或最近结束周期的诊断文本。"""

        if self._events:
            return self._evaluate_events(
                self._events,
                completed=False,
            ).diagnostics.to_text()
        if self._last_diagnostics is not None:
            return self._last_diagnostics.to_text()
        return (
            f"周期={self._cycle_index}，处理失败帧=0，A4跳变帧=0，"
            "A4合格帧=0，纸片跳变=无，合格纸片=0，"
            "共同合格帧=0，周期原因=尚未开始采集"
        )

    @staticmethod
    def _validate_timestamp(observed_at_ns: int) -> None:
        """校验周期事件使用正整数单调时钟时间戳。"""

        if isinstance(observed_at_ns, bool) or not isinstance(
            observed_at_ns,
            int,
        ):
            raise TypeError("observed_at_ns 必须是整数")
        if observed_at_ns <= 0:
            raise ValueError("observed_at_ns 必须大于 0")

    def _append_event(
        self,
        event: _CycleEvent,
    ) -> CycleReviewResult | None:
        """加入事件，并在到达周期边界时结算并开启下一周期。"""

        if self._cycle_started_at_ns is None:
            self._cycle_started_at_ns = event.observed_at_ns
        self._events.append(event)
        elapsed_seconds = (
            event.observed_at_ns - self._cycle_started_at_ns
        ) / 1e9
        if elapsed_seconds + 1e-12 < self.options.review_cycle_seconds:
            return None

        result = self._evaluate_events(self._events, completed=True)
        self._last_diagnostics = result.diagnostics
        self._events = []
        self._cycle_started_at_ns = None
        self._cycle_index += 1
        return result

    def _workspace_corner_errors_mm(
        self,
        reference: ReviewedFrame,
        current: ReviewedFrame,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """把当前四角投影到参考 A4 平面并返回有向和绝对毫米误差。"""

        reference_corners = np.asarray(
            reference.workspace_corners_source_px,
            dtype=np.float32,
        )
        current_corners = np.asarray(
            current.workspace_corners_source_px,
            dtype=np.float32,
        )
        if reference_corners.shape != (4, 2) or current_corners.shape != (
            4,
            2,
        ):
            raise ValueError("A4 源图角点必须是 4×2 坐标")
        homography = cv2.getPerspectiveTransform(
            reference_corners,
            _A4_CORNERS_MM,
        )
        projected = cv2.perspectiveTransform(
            current_corners.reshape(1, 4, 2),
            homography,
        )[0].astype(np.float64)
        signed = projected - _A4_CORNERS_MM.astype(np.float64)
        if not np.isfinite(signed).all():
            raise ValueError("A4 毫米投影产生非有限结果")
        return signed, np.abs(signed)

    def _workspace_is_jump(
        self,
        reference: ReviewedFrame,
        current: ReviewedFrame,
    ) -> bool:
        """判断当前任一 A4 角的 X、Y 或二维毫米误差是否超限。"""

        try:
            signed, absolute = self._workspace_corner_errors_mm(
                reference,
                current,
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            return True
        distances = np.linalg.norm(signed, axis=1)
        return bool(
            np.any(
                absolute[:, 0]
                > self.options.workspace_corner_x_tolerance_mm + 1e-12
            )
            or np.any(
                absolute[:, 1]
                > self.options.workspace_corner_y_tolerance_mm + 1e-12
            )
            or np.any(
                distances
                > self.options.workspace_corner_distance_tolerance_mm + 1e-12
            )
        )

    def _accepted_workspace_frames(
        self,
        frames: list[ReviewedFrame],
    ) -> tuple[list[ReviewedFrame], int]:
        """按上一合格帧筛出 A4 非跳变帧并返回跳变数量。"""

        accepted: list[ReviewedFrame] = []
        jump_frames = 0
        reference: ReviewedFrame | None = None
        for frame in frames:
            if reference is not None and self._workspace_is_jump(
                reference,
                frame,
            ):
                jump_frames += 1
                continue
            accepted.append(frame)
            reference = frame
        return accepted, jump_frames

    def _alignment_for_track(
        self,
        track: _PieceTrack,
        piece: PuzzlePiece,
    ) -> tuple[float, NDArray[np.float64]] | None:
        """相对上一合格观测寻找不反向的最佳循环顶点对齐。"""

        candidate = np.asarray(piece.vertices_mm, dtype=float)
        previous = track.last_vertices
        if len(candidate) != len(previous):
            return None

        best_cost = float("inf")
        best_vertices: NDArray[np.float64] | None = None
        for shift in range(len(candidate)):
            aligned = np.roll(candidate, -shift, axis=0)
            cost = float(
                np.max(np.linalg.norm(aligned - previous, axis=1))
            )
            if cost < best_cost:
                best_cost = cost
                best_vertices = aligned
        if (
            best_vertices is None
            or best_cost > self.options.piece_vertex_tolerance_mm + 1e-12
        ):
            return None
        return best_cost, best_vertices

    def _piece_assignments(
        self,
        tracks: list[_PieceTrack],
        pieces: tuple[PuzzlePiece, ...],
    ) -> dict[int, tuple[int, NDArray[np.float64]]]:
        """用全局最小代价关联既有轨迹和当前纸片。"""

        track_count = len(tracks)
        piece_count = len(pieces)
        if track_count == 0 or piece_count == 0:
            return {}

        size = track_count + piece_count
        unmatched_cost = self.options.piece_vertex_tolerance_mm + 1e-9
        impossible_cost = 1e9
        costs = np.full((size, size), impossible_cost, dtype=float)
        costs[:track_count, piece_count:] = unmatched_cost
        costs[track_count:, :piece_count] = unmatched_cost
        costs[track_count:, piece_count:] = 0.0
        alignments: dict[
            tuple[int, int],
            NDArray[np.float64],
        ] = {}

        for track_index, track in enumerate(tracks):
            for piece_index, piece in enumerate(pieces):
                result = self._alignment_for_track(track, piece)
                if result is None:
                    continue
                cost, aligned = result
                costs[track_index, piece_index] = cost
                alignments[(track_index, piece_index)] = aligned

        rows, columns = linear_sum_assignment(costs)
        assignments: dict[int, tuple[int, NDArray[np.float64]]] = {}
        for row, column in zip(rows, columns):
            key = int(row), int(column)
            if (
                row < track_count
                and column < piece_count
                and key in alignments
            ):
                assignments[int(row)] = (
                    int(column),
                    alignments[key],
                )
        return assignments

    def _piece_tracks(
        self,
        frames: list[ReviewedFrame],
    ) -> list[_PieceTrack]:
        """在 A4 合格帧内建立纸片轨迹并累计缺失和顶点跳变。"""

        tracks: list[_PieceTrack] = []
        next_track_id = 0
        for frame_index, frame in enumerate(frames):
            assignments = self._piece_assignments(tracks, frame.pieces)
            matched_piece_indexes: set[int] = set()
            for track_index, track in enumerate(tracks):
                assignment = assignments.get(track_index)
                if assignment is None:
                    track.jump_frames += 1
                    continue
                piece_index, aligned = assignment
                track.observations[frame.sequence_id] = _PieceObservation(
                    frame.pieces[piece_index],
                    aligned,
                    self._piece_confidence(frame, piece_index),
                )
                matched_piece_indexes.add(piece_index)

            for piece_index, piece in enumerate(frame.pieces):
                if piece_index in matched_piece_indexes:
                    continue
                vertices = np.asarray(piece.vertices_mm, dtype=float)
                tracks.append(
                    _PieceTrack(
                        track_id=next_track_id,
                        observations={
                            frame.sequence_id: _PieceObservation(
                                piece,
                                vertices,
                                self._piece_confidence(frame, piece_index),
                            )
                        },
                        # 首次出现前的 A4 合格帧均视为该候选漏检。
                        jump_frames=frame_index,
                    )
                )
                next_track_id += 1
        return tracks

    @staticmethod
    def _piece_confidence(
        frame: ReviewedFrame,
        piece_index: int,
    ) -> float:
        """返回审查帧内纸片置信度，并兼容旧的直接测试构造。"""

        if not frame.piece_confidences:
            return 0.0
        return frame.piece_confidences[piece_index]

    def _stable_tracks(
        self,
        tracks: list[_PieceTrack],
        processing_failure_frames: int,
    ) -> list[_PieceTrack]:
        """筛选跳变总数和合格观测数均满足周期门槛的轨迹。"""

        return [
            track
            for track in tracks
            if (
                track.jump_frames + processing_failure_frames
                <= self.options.allowed_jump_frames
                and len(track.observations)
                >= self.options.minimum_stable_frames
            )
        ]

    def _stable_track_count_is_valid(self, count: int) -> bool:
        """按控制模式判断周期内合格纸片数量。"""

        if self.options.planning_options.mode == "fixed":
            return count == 4
        return count >= 2

    @staticmethod
    def _track_jitter(track: _PieceTrack) -> float:
        """计算轨迹相对逐顶点中位几何的中位最大顶点偏差。"""

        vertices = np.stack(
            [
                observation.aligned_vertices
                for observation in track.observations.values()
            ]
        )
        center = np.median(vertices, axis=0)
        deviations = np.max(
            np.linalg.norm(vertices - center, axis=2),
            axis=1,
        )
        return float(np.median(deviations))

    def _track_selection_score(
        self,
        track: _PieceTrack,
        processing_failure_frames: int,
    ) -> tuple[int, int, float, float, int]:
        """构造随机任务多片择优使用的确定性字典序分数。"""

        confidence = float(
            np.median(
                [
                    observation.confidence
                    for observation in track.observations.values()
                ]
            )
        )
        return (
            track.jump_frames + processing_failure_frames,
            -len(track.observations),
            self._track_jitter(track),
            -confidence,
            track.track_id,
        )

    def _selected_stable_tracks(
        self,
        stable_tracks: list[_PieceTrack],
        processing_failure_frames: int,
    ) -> list[_PieceTrack]:
        """为随机任务从超额稳定轨迹中选择稳定性最高的四片。"""

        if self.options.planning_options.mode == "fixed":
            return stable_tracks
        return sorted(
            stable_tracks,
            key=lambda track: self._track_selection_score(
                track,
                processing_failure_frames,
            ),
        )[:4]

    @staticmethod
    def _complete_frame_ids(
        tracks: list[_PieceTrack],
    ) -> list[int]:
        """求全部合格轨迹同时存在有效观测的帧序号。"""

        id_sets = [set(track.observations) for track in tracks]
        if not id_sets:
            return []
        return sorted(reduce(set.intersection, id_sets))

    def _workspace_pair_distance(
        self,
        first: ReviewedFrame,
        second: ReviewedFrame,
    ) -> float:
        """计算两帧 A4 四角经三种毫米阈值归一化的对称距离。"""

        values: list[float] = []
        for reference, current in ((first, second), (second, first)):
            signed, absolute = self._workspace_corner_errors_mm(
                reference,
                current,
            )
            values.append(
                max(
                    float(np.max(absolute[:, 0]))
                    / self.options.workspace_corner_x_tolerance_mm,
                    float(np.max(absolute[:, 1]))
                    / self.options.workspace_corner_y_tolerance_mm,
                    float(np.max(np.linalg.norm(signed, axis=1)))
                    / self.options.workspace_corner_distance_tolerance_mm,
                )
            )
        return max(values)

    def _frame_distance(
        self,
        first_id: int,
        second_id: int,
        tracks: list[_PieceTrack],
        frames_by_id: dict[int, ReviewedFrame],
    ) -> float:
        """计算两张共同合格帧的 A4 与全部纸片总距离。"""

        workspace_distance = self._workspace_pair_distance(
            frames_by_id[first_id],
            frames_by_id[second_id],
        )
        piece_distance = sum(
            float(
                np.max(
                    np.linalg.norm(
                        track.observations[first_id].aligned_vertices
                        - track.observations[second_id].aligned_vertices,
                        axis=1,
                    )
                )
            )
            / self.options.piece_vertex_tolerance_mm
            for track in tracks
        )
        return workspace_distance + piece_distance

    def _medoid_pieces(
        self,
        frame_ids: list[int],
        tracks: list[_PieceTrack],
        frames_by_id: dict[int, ReviewedFrame],
    ) -> tuple[PuzzlePiece, ...]:
        """选择到其他共同帧总距离最小的真实帧纸片。"""

        medoid_id = min(
            frame_ids,
            key=lambda candidate_id: (
                sum(
                    self._frame_distance(
                        candidate_id,
                        other_id,
                        tracks,
                        frames_by_id,
                    )
                    for other_id in frame_ids
                ),
                frames_by_id[candidate_id].received_at_ns,
                candidate_id,
            ),
        )
        pieces = [
            track.observations[medoid_id].piece for track in tracks
        ]
        return tuple(sorted(pieces, key=lambda piece: int(piece.piece_id)))

    def _evaluate_events(
        self,
        events: list[_CycleEvent],
        *,
        completed: bool,
    ) -> CycleReviewResult:
        """统一评估周期内 A4、纸片、共同帧和代表帧。"""

        processing_failure_frames = sum(
            event.frame is None for event in events
        )
        reviewed_frames = [
            event.frame for event in events if event.frame is not None
        ]
        accepted_frames, workspace_jump_frames = (
            self._accepted_workspace_frames(reviewed_frames)
        )
        tracks = self._piece_tracks(accepted_frames)
        stable_tracks = self._stable_tracks(
            tracks,
            processing_failure_frames,
        )
        selected_tracks = self._selected_stable_tracks(
            stable_tracks,
            processing_failure_frames,
        )
        complete_frame_ids = self._complete_frame_ids(selected_tracks)
        frames_by_id = {
            frame.sequence_id: frame for frame in accepted_frames
        }

        workspace_bad_frames = (
            processing_failure_frames + workspace_jump_frames
        )
        if not completed:
            reason = "周期采集中"
        elif workspace_bad_frames > self.options.allowed_jump_frames:
            reason = "A4 跳变与处理失败帧超过允许数量"
        elif (
            len(accepted_frames) < self.options.minimum_stable_frames
        ):
            reason = "A4 合格帧数量不足"
        elif not self._stable_track_count_is_valid(len(stable_tracks)):
            reason = "合格纸片数量不满足规划模式"
        elif (
            len(complete_frame_ids)
            < self.options.minimum_stable_frames
        ):
            reason = "全部合格纸片的共同帧数量不足"
        else:
            reason = "周期审查通过"

        accepted = completed and reason == "周期审查通过"
        pieces = (
            self._medoid_pieces(
                complete_frame_ids,
                selected_tracks,
                frames_by_id,
            )
            if accepted
            else ()
        )
        diagnostics = CycleReviewDiagnostics(
            cycle_index=self._cycle_index,
            processing_failure_frames=processing_failure_frames,
            workspace_jump_frames=workspace_jump_frames,
            workspace_accepted_frames=len(accepted_frames),
            piece_jump_frames=tuple(
                (
                    track.track_id,
                    track.jump_frames + processing_failure_frames,
                )
                for track in tracks
            ),
            stable_piece_count=len(stable_tracks),
            selected_piece_track_ids=tuple(
                track.track_id for track in selected_tracks
            ),
            selected_track_scores=tuple(
                (
                    track.track_id,
                    track.jump_frames + processing_failure_frames,
                    len(track.observations),
                    self._track_jitter(track),
                    float(
                        np.median(
                            [
                                observation.confidence
                                for observation in track.observations.values()
                            ]
                        )
                    ),
                )
                for track in selected_tracks
            ),
            common_frame_count=len(complete_frame_ids),
            reason=reason,
        )
        return CycleReviewResult(
            accepted=accepted,
            pieces=pieces,
            diagnostics=diagnostics,
        )
