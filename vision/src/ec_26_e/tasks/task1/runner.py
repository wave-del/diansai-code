"""编排相机、视觉稳定审查和纯控制规划的 Task1 入口。"""

from __future__ import annotations

import time

from ec_26_e.cam import (
    CameraFrame,
    CameraSource,
    CameraTimeoutError,
)
from ec_26_e.control import MovePlan, PuzzlePiece, plan_sheet_moves
from ec_26_e.tasks.task1.contracts import (
    Detect1Processor,
    PuzzlePlanner,
    ResizeProcessor,
    Task1FrameObserver,
    Task1Error,
    Task1Options,
    Task1TimeoutError,
)
from ec_26_e.tasks.contracts import TaskCancelledError
from ec_26_e.tasks.task1.review import ReviewedFrame, review_visual_frame
from ec_26_e.tasks.task1.stability import StableSceneReviewer
from ec_26_e.vision import Detect1Result


def _validate_dependencies(
    camera: CameraSource,
    resizer: ResizeProcessor,
    detector: Detect1Processor,
    planner: PuzzlePlanner,
    observer: Task1FrameObserver | None,
) -> None:
    """在访问硬件前确认注入对象提供 Task1 所需最小接口。"""

    for name, dependency, method_names in (
        ("camera", camera, ("start", "read_latest", "close")),
        ("resizer", resizer, ("resize",)),
        ("detector", detector, ("detect",)),
    ):
        for method_name in method_names:
            if not callable(getattr(dependency, method_name, None)):
                raise TypeError(
                    f"{name} 必须提供可调用的 {method_name} 方法"
                )
    if not callable(planner):
        raise TypeError("planner 必须是可调用对象")
    if observer is not None:
        for method_name in (
            "start",
            "show_detection",
            "show_failure",
            "close",
        ):
            if not callable(getattr(observer, method_name, None)):
                raise TypeError(
                    f"observer 必须提供可调用的 {method_name} 方法"
                )


def _format_timeout(
    options: Task1Options,
    processed_frames: int,
    invalid_events: int,
    reviewer: StableSceneReviewer,
    last_reason: str | None,
    task_label: str,
) -> Task1TimeoutError:
    """创建包含稳定审查现场信息的任务超时异常。"""

    reason = (
        last_reason
        or "尚未形成满足数量与持续时间的稳定碎片集合"
    )
    return Task1TimeoutError(
        f"{task_label} 在 "
        f"{options.acquisition_timeout_seconds:.3f} 秒内未获得稳定结果；"
        f"已处理帧={processed_frames}，"
        f"异常事件={invalid_events}，"
        f"最后原因={reason}；"
        f"{reviewer.diagnostics_text()}"
    )


def _acquire_stable_pieces(
    camera: CameraSource,
    resizer: ResizeProcessor,
    detector: Detect1Processor,
    options: Task1Options,
    observer: Task1FrameObserver | None,
    task_label: str,
) -> tuple[PuzzlePiece, ...]:
    """在五秒采集窗口内返回稳定结果，或最后一张可审查帧。"""

    reviewer = StableSceneReviewer(options)
    deadline = time.monotonic() + options.acquisition_timeout_seconds
    after_sequence: int | None = None
    processed_frames = 0
    invalid_events = 0
    last_reason: str | None = None
    last_reviewed: ReviewedFrame | None = None
    last_complete_pieces: tuple[PuzzlePiece, ...] | None = None

    def _last_available_pieces() -> tuple[PuzzlePiece, ...] | None:
        """超时回退到最近完整 fixed 帧，避免瞬时漏检覆盖它。"""

        if (
            options.planning_options.mode == "fixed"
            and last_complete_pieces is not None
        ):
            return last_complete_pieces
        if last_reviewed is None:
            return None
        return last_reviewed.pieces

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fallback_pieces = _last_available_pieces()
            if fallback_pieces is not None:
                return fallback_pieces
            raise _format_timeout(
                options,
                processed_frames,
                invalid_events,
                reviewer,
                last_reason,
                task_label,
            )
        try:
            frame = camera.read_latest(
                after_sequence=after_sequence,
                timeout=min(options.frame_read_timeout_seconds, remaining),
            )
        except CameraTimeoutError as error:
            invalid_events += 1
            last_reason = str(error)
            cycle_result = reviewer.note_processing_failure(
                time.monotonic_ns()
            )
            if cycle_result is not None:
                last_reason = cycle_result.diagnostics.to_text()
                if cycle_result.accepted:
                    return cycle_result.pieces
            if time.monotonic() >= deadline:
                fallback_pieces = _last_available_pieces()
                if fallback_pieces is not None:
                    return fallback_pieces
            continue

        if not isinstance(frame, CameraFrame):
            raise Task1Error("camera.read_latest 必须返回 CameraFrame")
        if after_sequence is not None and frame.sequence_id <= after_sequence:
            raise Task1Error("相机必须返回序号严格递增的新帧")
        if time.monotonic() > deadline:
            fallback_pieces = _last_available_pieces()
            if fallback_pieces is not None:
                return fallback_pieces
            raise _format_timeout(
                options,
                processed_frames,
                invalid_events,
                reviewer,
                last_reason,
                task_label,
            )

        after_sequence = frame.sequence_id
        processed_frames += 1
        resize_result = None
        try:
            resize_result = resizer.resize(frame.image_bgr)
        except ValueError as error:
            invalid_events += 1
            last_reason = str(error)
            cycle_result = reviewer.note_processing_failure(
                frame.received_at_ns
            )
            if cycle_result is not None:
                last_reason = cycle_result.diagnostics.to_text()
            if (
                observer is not None
                and observer.show_failure(
                    frame,
                    stage=(
                        "stability"
                        if cycle_result is not None
                        and not cycle_result.accepted
                        else "workspace"
                    ),
                )
            ):
                raise TaskCancelledError(
                    "用户通过预览窗口中止任务"
                )
            if cycle_result is not None and cycle_result.accepted:
                return cycle_result.pieces
            continue

        try:
            stable_result = detector.detect(resize_result.image_bgr)
        except ValueError as error:
            invalid_events += 1
            last_reason = str(error)
            cycle_result = reviewer.note_processing_failure(
                frame.received_at_ns
            )
            if cycle_result is not None:
                last_reason = cycle_result.diagnostics.to_text()
            if (
                observer is not None
                and observer.show_failure(
                    frame,
                    stage=(
                        "stability"
                        if cycle_result is not None
                        and not cycle_result.accepted
                        else "detect1"
                    ),
                    resize_result=resize_result,
                )
            ):
                raise TaskCancelledError(
                    "用户通过预览窗口中止任务"
                )
            if cycle_result is not None and cycle_result.accepted:
                return cycle_result.pieces
            continue

        # 与实时录像诊断保持一致：稳定器尚未确认时，使用同一帧的
        # 单帧精修结果继续审查。这样单帧已完整识别的场景不会被稳定器
        # 的短暂空结果直接丢弃。
        detect_result = stable_result
        single_result = getattr(detector, "last_single_result", None)
        if not detect_result.pieces and isinstance(
            single_result,
            Detect1Result,
        ):
            detect_result = single_result

        try:
            reviewed = review_visual_frame(
                frame,
                resize_result,
                detect_result,
                options.landscape_rotation,
            )
        except ValueError as error:
            invalid_events += 1
            last_reason = str(error)
            cycle_result = reviewer.note_processing_failure(
                frame.received_at_ns
            )
            if cycle_result is not None:
                last_reason = cycle_result.diagnostics.to_text()
            if (
                observer is not None
                and observer.show_failure(
                    frame,
                    stage=(
                        "stability"
                        if cycle_result is not None
                        and not cycle_result.accepted
                        else "review"
                    ),
                    resize_result=resize_result,
                )
            ):
                raise TaskCancelledError(
                    "用户通过预览窗口中止任务"
                )
            if cycle_result is not None and cycle_result.accepted:
                return cycle_result.pieces
            continue

        # 即使本周期因稳定性门未通过，该帧仍已具备可供控制层使用的
        # 坐标和简单多边形契约；采集窗口耗尽时按用户要求交给下一步。
        last_reviewed = reviewed
        if (
            options.planning_options.mode == "fixed"
            and len(reviewed.pieces) == 4
        ):
            last_complete_pieces = reviewed.pieces

        cycle_result = reviewer.add_frame(reviewed)
        if cycle_result is not None and not cycle_result.accepted:
            invalid_events += 1
            last_reason = cycle_result.diagnostics.to_text()
            if (
                observer is not None
                and observer.show_failure(
                    frame,
                    stage="stability",
                    resize_result=resize_result,
                )
            ):
                raise TaskCancelledError(
                    "用户通过预览窗口中止任务"
                )
            continue
        if (
            observer is not None
            and observer.show_detection(
                frame,
                resize_result,
                detect_result,
            )
        ):
            raise TaskCancelledError(
                "用户通过预览窗口中止任务"
            )
        if cycle_result is not None:
            return cycle_result.pieces
        last_reason = "当前两秒周期仍在采集中"


def run_puzzle_task(
    camera: CameraSource,
    resizer: ResizeProcessor,
    detector: Detect1Processor,
    *,
    options: Task1Options = Task1Options(),
    planner: PuzzlePlanner = plan_sheet_moves,
    observer: Task1FrameObserver | None = None,
    task_label: str = "Task1",
) -> MovePlan:
    """运行一次共享拼图任务，并在释放相机后同步返回纯控制移动方案。

    本入口不读取配置文件、不序列化中间结果、不写文件，也不
    访问相机之外的硬件。视觉场景错误会在采集期限内重试，
    控制规划错误原样向上传播。observer 为 ``None`` 时不会触发
    任何图像复制、绘制或窗口操作。
    """

    if not isinstance(options, Task1Options):
        raise TypeError("options 必须是 Task1Options 的任务选项")
    if not isinstance(task_label, str) or not task_label.strip():
        raise ValueError("task_label 必须是非空字符串")
    _validate_dependencies(
        camera,
        resizer,
        detector,
        planner,
        observer,
    )

    try:
        if observer is not None:
            observer.start()
        camera.start()
        pieces = _acquire_stable_pieces(
            camera,
            resizer,
            detector,
            options,
            observer,
            task_label,
        )
    finally:
        try:
            camera.close()
        finally:
            if observer is not None:
                observer.close()

    return planner(pieces, options.planning_options)


def run_task1(
    camera: CameraSource,
    resizer: ResizeProcessor,
    detector: Detect1Processor,
    *,
    options: Task1Options = Task1Options(),
    planner: PuzzlePlanner = plan_sheet_moves,
    observer: Task1FrameObserver | None = None,
) -> MovePlan:
    """运行一次 Task1，并在释放相机后同步返回纯控制移动方案。"""

    return run_puzzle_task(
        camera,
        resizer,
        detector,
        options=options,
        planner=planner,
        observer=observer,
        task_label="Task1",
    )
