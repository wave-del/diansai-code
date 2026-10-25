"""验证正式 Task1 配置可装配真实组件并驱动纯内存任务入口。

该文件只构造 FFmpegCamera，不启动设备；任务运行使用内存 Fake，
不创建输出文件。可通过
``python -m unittest tests.integration.tasks.test_task1_config`` 运行。
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ec_26_e.cam import FFmpegCamera
from ec_26_e.config import (
    DEFAULT_TASK1_CONFIG_PATH,
    load_task1_config,
)
from ec_26_e.tasks.task1 import run_task1
from ec_26_e.vision import Resize
from tests.unit.control.test_planner import FIXED_PIECES
from tests.unit.tasks.task1_helpers import (
    ConstantDetector,
    FakeCamera,
    RecordingPlanner,
    SequenceResizer,
    make_camera_frame,
    make_detect_result,
    portrait_vertices_to_landscape_relative,
)


class Task1ConfigIntegrationTest(unittest.TestCase):
    """验证配置对象与真实构造器、Task1 入口的类型边界一致。"""

    def test_official_config_builds_components_without_side_effects(
        self,
    ) -> None:
        """加载和构造真实组件不得访问设备或创建输出。"""

        config_path = DEFAULT_TASK1_CONFIG_PATH.resolve()
        original_directory = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = load_task1_config(
                    config_path,
                    environ={"MOTION_MAGNET_GPIO_LINE": "12"},
                    env_path=None,
                )
                camera = FFmpegCamera(config.camera)
                resizer = Resize(config.resize)

                self.assertIs(camera.config, config.camera)
                self.assertIs(resizer.config, config.resize)
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(original_directory)

    def test_official_options_drive_task1_without_dictionary_bridge(
        self,
    ) -> None:
        """正式 fixed 配置应直接传入 run_task1 并筛选四片。"""

        config = load_task1_config(
            environ={"MOTION_MAGNET_GPIO_LINE": "12"},
            env_path=None,
        )
        detections = make_detect_result(
            tuple(
                (
                    index,
                    portrait_vertices_to_landscape_relative(
                        piece.vertices_mm
                    ),
                )
                for index, piece in enumerate(FIXED_PIECES)
            )
        )
        frames = tuple(
            make_camera_frame(
                index,
                1_000_000_000
                + round(index * 2_000_000_000 / 9),
            )
            for index in range(10)
        )
        camera = FakeCamera(frames)
        planner = RecordingPlanner(camera)

        plan = run_task1(
            camera,
            SequenceResizer(frames[0]),
            ConstantDetector(detections),
            options=config.options,
            planner=planner,
        )

        self.assertEqual(plan.mode, "fixed")
        self.assertEqual(len(planner.calls), 1)
        pieces, planning_options = planner.calls[0]
        self.assertEqual(len(pieces), 4)
        self.assertIs(planning_options, config.options.planning_options)


if __name__ == "__main__":
    unittest.main()
