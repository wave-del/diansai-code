"""验证摄像头诊断 CLI 从环境读取设备与分辨率及命令行覆盖顺序。

该文件使用命令替身，不访问真实摄像头、显示器或 FFmpeg。
可通过 ``python -m unittest tests.unit.cam.test_camera_cli_config`` 运行。
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from ec_26_e.config import load_task1_config, load_task2_config
import tests.external.cam.record_camera as record_camera_module
import tests.external.cam.record_detected_video as detected_video_module


class CameraCliConfigTest(unittest.TestCase):
    """验证相机、工作区定位和 A4 输出配置来源。"""

    def test_detected_video_matches_registered_task_configuration(
        self,
    ) -> None:
        """Task1/Task2 视觉诊断配置应与正式任务加载结果完全相等。"""

        loaders = {
            "task1": load_task1_config,
            "task2": load_task2_config,
        }
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / "machine.env"
            for task_name, loader in loaders.items():
                with self.subTest(task=task_name):
                    task_path = (
                        detected_video_module.PROJECT_ROOT
                        / "configs"
                        / "tasks"
                        / f"{task_name}.json"
                    )
                    env_path.write_text(
                        f"EC_TASK={task_name}\n"
                        f"EC_TASK_CONFIG={task_path}\n"
                        "CAMERA_DEVICE_PATH=/dev/video8\n"
                        "CAMERA_WIDTH=1280\n"
                        "CAMERA_HEIGHT=720\n"
                        "WORKSPACE_DETECT_WIDTH=640\n"
                        "WORKSPACE_DETECT_HEIGHT=360\n"
                        "A4_RESIZE_WIDTH=764\n"
                        "A4_RESIZE_HEIGHT=540\n"
                        "MOTION_MAGNET_GPIO_LINE=12\n",
                        encoding="utf-8",
                    )

                    diagnostic = (
                        detected_video_module.load_detection_runtime_config(
                            env_path,
                            environ={},
                        )
                    )
                    production = loader(
                        task_path,
                        env_path=env_path,
                        environ={},
                    )

                    self.assertEqual(diagnostic.task_name, task_name)
                    self.assertEqual(diagnostic.camera, production.camera)
                    self.assertEqual(diagnostic.resize, production.resize)
                    self.assertEqual(diagnostic.detect1, production.detect1)

    def test_record_camera_reads_device_and_resolution_from_environment(
        self,
    ) -> None:
        """普通录像脚本应把环境设备路径和宽高转换为强类型配置。"""

        summary = Mock()
        summary.to_dict.return_value = {}
        with (
            patch.dict(
                os.environ,
                {
                    "CAMERA_DEVICE_PATH": "/dev/video8",
                    "CAMERA_WIDTH": "1280",
                    "CAMERA_HEIGHT": "720",
                },
                clear=False,
            ),
            patch.object(
                record_camera_module,
                "record_camera",
                return_value=summary,
            ) as record_camera,
        ):
            result = CliRunner().invoke(record_camera_module.app, [])

        self.assertEqual(result.exit_code, 0, result.output)
        config = record_camera.call_args.args[0]
        self.assertEqual(str(config.device_path), "/dev/video8")
        self.assertEqual((config.width, config.height), (1280, 720))

    def test_detected_video_explicit_task_overrides_env_selection(self) -> None:
        """显式任务和 JSON 应覆盖 .env，供桌面双任务预览入口使用。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / "machine.env"
            task1_path = (
                detected_video_module.PROJECT_ROOT
                / "configs/tasks/task1.json"
            )
            task2_path = (
                detected_video_module.PROJECT_ROOT
                / "configs/tasks/task2.json"
            )
            env_path.write_text(
                "EC_TASK=task1\n"
                f"EC_TASK_CONFIG={task1_path}\n"
                "MOTION_MAGNET_GPIO_LINE=12\n",
                encoding="utf-8",
            )

            config = detected_video_module.load_detection_runtime_config(
                env_path,
                environ={},
                task_name="task2",
                task_config_path=task2_path,
            )

        self.assertEqual(config.task_name, "task2")
        self.assertEqual(config.task_config_path, task2_path)
        self.assertEqual(
            config.task_config.options.planning_options.mode,
            "random",
        )

    def test_detected_video_reads_device_and_resolutions_from_environment(
        self,
    ) -> None:
        """检测录像脚本应从环境构造相机和 Resize 强类型配置。"""

        camera_info = Mock()
        camera_info.to_dict.return_value = {}
        summary = Mock()
        summary.to_dict.return_value = {}
        with (
            patch.dict(
                os.environ,
                {
                    "CAMERA_DEVICE_PATH": "/dev/video8",
                    "CAMERA_WIDTH": "1280",
                    "CAMERA_HEIGHT": "720",
                    "WORKSPACE_DETECT_WIDTH": "640",
                    "WORKSPACE_DETECT_HEIGHT": "360",
                    "A4_RESIZE_WIDTH": "764",
                    "A4_RESIZE_HEIGHT": "540",
                    "DETECT1_PIECE_MODE": "1",
                },
                clear=False,
            ),
            patch.object(
                detected_video_module,
                "query_camera_info",
                return_value=camera_info,
            ) as query_camera_info,
            patch.object(
                detected_video_module,
                "record_detection_video",
                return_value=summary,
            ) as record_video,
        ):
            result = CliRunner().invoke(
                detected_video_module.app,
                ["--no-preview"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        camera_config = record_video.call_args.args[0]
        resize_config = record_video.call_args.args[1]
        detect1_config = record_video.call_args.args[2]
        self.assertEqual(str(camera_config.device_path), "/dev/video8")
        query_camera_info.assert_called_once_with(
            camera_config.device_path,
            v4l2_ctl_executable="v4l2-ctl",
        )
        self.assertEqual(
            (camera_config.width, camera_config.height),
            (1280, 720),
        )
        self.assertEqual(
            (resize_config.detect_width, resize_config.detect_height),
            (640, 360),
        )
        self.assertEqual(
            (resize_config.target_width, resize_config.target_height),
            (764, 540),
        )
        self.assertEqual(detect1_config.piece_mode, 1)

    def test_detected_video_cli_overrides_environment_resolutions(self) -> None:
        """显式参数应覆盖相机、定位和 A4 环境配置。"""

        camera_info = Mock()
        camera_info.to_dict.return_value = {}
        summary = Mock()
        summary.to_dict.return_value = {}
        with (
            patch.dict(
                os.environ,
                {"CAMERA_DEVICE_PATH": "/dev/video8"},
                clear=False,
            ),
            patch.object(
                detected_video_module,
                "query_camera_info",
                return_value=camera_info,
            ) as query_camera_info,
            patch.object(
                detected_video_module,
                "record_detection_video",
                return_value=summary,
            ) as record_video,
        ):
            result = CliRunner().invoke(
                detected_video_module.app,
                [
                    "--device",
                    "/dev/video9",
                    "--width",
                    "640",
                    "--height",
                    "480",
                    "--workspace-detect-width",
                    "320",
                    "--workspace-detect-height",
                    "240",
                    "--a4-resize-width",
                    "636",
                    "--a4-resize-height",
                    "450",
                    "--no-preview",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        camera_config = record_video.call_args.args[0]
        resize_config = record_video.call_args.args[1]
        self.assertEqual(str(camera_config.device_path), "/dev/video9")
        query_camera_info.assert_called_once_with(
            camera_config.device_path,
            v4l2_ctl_executable="v4l2-ctl",
        )
        self.assertEqual(
            (camera_config.width, camera_config.height),
            (640, 480),
        )
        self.assertEqual(
            (resize_config.detect_width, resize_config.detect_height),
            (320, 240),
        )
        self.assertEqual(
            (resize_config.target_width, resize_config.target_height),
            (636, 450),
        )

    def test_invalid_a4_ratio_fails_before_device_query(self) -> None:
        """非法 A4 配置应在访问真实摄像头前失败。"""

        with patch.object(
            detected_video_module,
            "query_camera_info",
        ) as query:
            result = CliRunner().invoke(
                detected_video_module.app,
                [
                    "--a4-resize-width",
                    "800",
                    "--a4-resize-height",
                    "500",
                    "--no-preview",
                ],
            )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("292:207", result.output)
        query.assert_not_called()

    def test_detected_video_help_has_no_duration_option(self) -> None:
        """无时长录像 CLI 不应再暴露 duration 参数。"""

        result = CliRunner().invoke(detected_video_module.app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("--duration", result.output)


if __name__ == "__main__":
    unittest.main()
