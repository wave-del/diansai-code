"""以真实录像和 A733 NPU 验证正式统一视觉装配，不访问运动硬件。

独立运行：
``poetry run python tests/external/vision/main_pipeline_dry_run.py``。
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.config import load_task1_config
from ec_26_e.control import MovePlan, PlanningOptions, PuzzlePiece
from ec_26_e.tasks import TaskContext
from ec_26_e.tasks.task1.application import execute_task1
from ec_26_e.vision import (
    Detect1Config,
    Detect1Result,
    StableVisionDetector,
    create_production_detector,
)
from ec_26_e.vision.evaluation import percentile_summary
from tests.external.task1_video_serial import VideoCamera
from tests.unit.tasks.task1_helpers import make_move_plan


app = typer.Typer(
    add_completion=False,
    help="运行正式 Task1 统一视觉链路并输出无运动硬件的耗时报告。",
)


class TimedProductionDetector:
    """包装正式检测器并记录首个稳定结果、阶段耗时和线程归属。"""

    def __init__(self, legacy_config: Detect1Config) -> None:
        """在当前线程创建正式 9 帧窗口、6 帧确认的稳定检测器。"""

        self.detector: StableVisionDetector = create_production_detector(
            legacy_config
        )
        self.created_thread = threading.get_ident()
        self.closed_thread: int | None = None
        self.detect_threads: set[int] = set()
        self.started_at = time.perf_counter()
        self.first_stable_seconds: float | None = None
        self.frame_ms: list[float] = []
        self.stage_ms: dict[str, list[float]] = defaultdict(list)

    def detect(self, image_bgr: np.ndarray) -> Detect1Result:
        """调用正式检测器并累计不含显示、录像和旧 Detect1 对照的耗时。"""

        self.detect_threads.add(threading.get_ident())
        started = time.perf_counter()
        result = self.detector.detect(image_bgr)
        self.frame_ms.append((time.perf_counter() - started) * 1000.0)
        diagnostics = self.detector.detector.last_diagnostics
        for name, value in diagnostics.native_timing.to_dict().items():
            self.stage_ms[name].append(value)
        for name, value in diagnostics.stage_ms.items():
            self.stage_ms[name].append(value)
        self.stage_ms["vision_stabilizer"].append(
            self.detector.last_stabilizer_ms
        )
        if result.pieces and self.first_stable_seconds is None:
            self.first_stable_seconds = time.perf_counter() - self.started_at
        return result

    def close(self) -> None:
        """记录释放线程并幂等关闭正式检测器。"""

        self.closed_thread = threading.get_ident()
        self.detector.close()


class RecordingPlanner:
    """记录进入规划的实测纸片，但返回无移动的合法测试方案。"""

    def __init__(self) -> None:
        """初始化规划调用记录。"""

        self.called_at: float | None = None
        self.pieces: tuple[PuzzlePiece, ...] = ()

    def __call__(
        self,
        pieces: tuple[PuzzlePiece, ...],
        options: PlanningOptions,
    ) -> MovePlan:
        """保存正式审查输出，避免真实求解差异影响视觉接入验证。"""

        self.called_at = time.perf_counter()
        self.pieces = tuple(pieces)
        return make_move_plan(options.mode)


class EmptyActuator:
    """消费内存 MovePlan 且明确不访问串口、GPIO 或电机。"""

    def __init__(self) -> None:
        """创建未调用状态。"""

        self.called = False

    def execute(self, plan: MovePlan) -> None:
        """只记录合法方案到达执行边界。"""

        if not isinstance(plan, MovePlan):
            raise TypeError("plan 必须是 MovePlan")
        self.called = True


@app.command()
def run(
    video: Path = typer.Option(
        Path("outputs/cam/detection_20260730_180643_595024_raw.mp4"),
        "--video",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("outputs/vision/main_pipeline_dry_run.json"),
        "--output",
    ),
) -> None:
    """执行正式装配并保存首次稳定、首次可规划和分阶段性能。"""

    config = load_task1_config()
    detector_holder: list[TimedProductionDetector] = []
    planner = RecordingPlanner()
    actuator = EmptyActuator()
    started = time.perf_counter()

    def detector_factory(legacy_config: Detect1Config) -> TimedProductionDetector:
        """创建并暴露本次唯一的正式检测器包装。"""

        detector = TimedProductionDetector(legacy_config)
        detector_holder.append(detector)
        return detector

    execute_task1(
        config,
        TaskContext(draw_enabled=False),
        camera_factory=lambda unused: VideoCamera(video),
        detector_factory=detector_factory,
        planner=planner,
        actuator_factory=lambda unused: actuator,
    )
    elapsed = time.perf_counter() - started
    detector = detector_holder[0]
    thread_ids = {
        detector.created_thread,
        detector.closed_thread,
        *detector.detect_threads,
    }
    report = {
        "video": str(video),
        "processed_frames": len(detector.frame_ms),
        "accepted_pieces": len(planner.pieces),
        "first_stable_seconds": detector.first_stable_seconds,
        "first_plannable_seconds": (
            None if planner.called_at is None else planner.called_at - started
        ),
        "total_seconds": round(elapsed, 4),
        "complete_fps": round(len(detector.frame_ms) / elapsed, 4),
        "frame_ms": percentile_summary(detector.frame_ms),
        "stage_ms": {
            name: percentile_summary(values)
            for name, values in detector.stage_ms.items()
        },
        "single_thread_lifecycle": len(thread_ids) == 1,
        "empty_actuator_called": actuator.called,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
