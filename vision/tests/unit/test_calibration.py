"""验证现场标定配置的初始化、校验、原子发布和历史恢复。

该文件只使用临时目录和正式配置加载器，不访问相机、GPIO 或串口。
可通过 ``python -m unittest tests.unit.test_calibration`` 独立运行。
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ec_26_e.config.calibration import (
    CalibrationPaths,
    active_generation,
    initialize_calibration,
    publish_draft,
    restore_previous,
    validate_draft,
)
from ec_26_e.config.env import load_env_file
from tools.calibration.console import PreviewTask, preview_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CalibrationConfigurationTest(unittest.TestCase):
    """验证整套配置只有通过两个任务校验后才会生效。"""

    def _initialize(self, directory: str) -> CalibrationPaths:
        """在临时目录中创建使用完整环境模板的标定工作区。"""

        source_env = Path(directory) / "source.env"
        source_env.write_text(
            "EC_TASK=task1\n"
            "EC_TASK_CONFIG=configs/tasks/task1.json\n"
            "CAMERA_DEVICE_PATH=/dev/video8\n"
            "MOTION_MAGNET_GPIO_LINE=12\n",
            encoding="utf-8",
        )
        paths = CalibrationPaths(Path(directory) / "calibration")
        initialize_calibration(
            paths,
            source_env=source_env,
            env_template=PROJECT_ROOT / ".env.example",
            source_task1=PROJECT_ROOT / "configs/tasks/task1.json",
            source_task2=PROJECT_ROOT / "configs/tasks/task2.json",
        )
        return paths

    def test_initialize_preserves_values_and_exposes_all_template_fields(
        self,
    ) -> None:
        """首次初始化应保留现场值并补齐模板中的全部可调参数。"""

        with TemporaryDirectory() as directory:
            paths = self._initialize(directory)
            values: dict[str, str] = {}
            load_env_file(paths.editing_env, environ=values, override=True)
            validation = validate_draft(paths)

            self.assertEqual(values["CAMERA_DEVICE_PATH"], "/dev/video8")
            self.assertEqual(values["CAMERA_FPS"], "30")
            self.assertEqual(values["MOTION_MAGNET_GPIO_LINE"], "12")
            self.assertEqual(validation.selected_task, "task1")
            self.assertTrue(paths.active_link.is_symlink())
            self.assertEqual(
                values["EC_TASK_CONFIG"],
                str(paths.active_task("task1")),
            )

    def test_publish_switches_whole_generation_and_restore_updates_draft(
        self,
    ) -> None:
        """发布应原子切换整套配置，恢复后草稿应同步到上一版。"""

        with TemporaryDirectory() as directory:
            paths = self._initialize(directory)
            original = active_generation(paths)
            task1 = json.loads(paths.editing_task1.read_text(encoding="utf-8"))
            task1["planning"]["search_top_n"] = 99
            paths.editing_task1.write_text(
                json.dumps(task1, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            publish_draft(paths)
            published = active_generation(paths)
            active_task1 = json.loads(
                paths.active_task("task1").read_text(encoding="utf-8")
            )

            self.assertNotEqual(published, original)
            self.assertEqual(active_task1["planning"]["search_top_n"], 99)
            restored = restore_previous(paths)
            self.assertEqual(restored, original)
            restored_draft = json.loads(
                paths.editing_task1.read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                restored_draft["planning"]["search_top_n"],
                99,
            )

    def test_invalid_draft_cannot_replace_active_generation(self) -> None:
        """损坏 JSON 应使发布失败且 active 保持原版本。"""

        with TemporaryDirectory() as directory:
            paths = self._initialize(directory)
            original = active_generation(paths)
            paths.editing_task2.write_text("{", encoding="utf-8")

            with self.assertRaises(ValueError):
                publish_draft(paths)

            self.assertEqual(active_generation(paths), original)

    def test_preview_command_uses_active_task_without_recording(self) -> None:
        """桌面快速预览必须读取 active JSON 且明确关闭录像。"""

        with TemporaryDirectory() as directory:
            paths = self._initialize(directory)
            command = preview_command(paths, PreviewTask.task2)

        self.assertIn("--no-record", command)
        self.assertIn("--task", command)
        self.assertIn("task2", command)
        self.assertIn(str(paths.active_task("task2")), command)


if __name__ == "__main__":
    unittest.main()
