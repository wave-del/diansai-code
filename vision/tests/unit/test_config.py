"""验证项目级 ``.env`` 的读取、覆盖顺序和错误输入处理。

该文件不依赖外部环境，可通过
``python -m unittest tests.unit.test_config`` 独立运行。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ec_26_e.config import load_env_file


class EnvFileTest(unittest.TestCase):
    """验证基础 ``.env`` 语法和环境变量优先级。"""

    def test_loads_camera_and_a4_resolutions(self) -> None:
        """相机、定位和 A4 宽高应从文件写入环境映射。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "CAMERA_WIDTH=1280\n"
                "CAMERA_HEIGHT=720\n"
                "WORKSPACE_DETECT_WIDTH=640\n"
                "WORKSPACE_DETECT_HEIGHT=360\n"
                "A4_RESIZE_WIDTH=764\n"
                "A4_RESIZE_HEIGHT=540\n",
                encoding="utf-8",
            )
            environ: dict[str, str] = {}

            loaded = load_env_file(env_path, environ=environ)

        self.assertEqual(
            environ,
            {
                "CAMERA_WIDTH": "1280",
                "CAMERA_HEIGHT": "720",
                "WORKSPACE_DETECT_WIDTH": "640",
                "WORKSPACE_DETECT_HEIGHT": "360",
                "A4_RESIZE_WIDTH": "764",
                "A4_RESIZE_HEIGHT": "540",
            },
        )
        self.assertEqual(loaded, environ)

    def test_exported_environment_takes_priority(self) -> None:
        """已导出的环境变量默认不应被 ``.env`` 覆盖。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "CAMERA_WIDTH=1920\nCAMERA_HEIGHT=1080\n",
                encoding="utf-8",
            )
            environ = {"CAMERA_WIDTH": "3840"}

            loaded = load_env_file(env_path, environ=environ)

        self.assertEqual(environ["CAMERA_WIDTH"], "3840")
        self.assertEqual(environ["CAMERA_HEIGHT"], "1080")
        self.assertEqual(loaded, {"CAMERA_HEIGHT": "1080"})

    def test_invalid_line_is_rejected(self) -> None:
        """缺少等号的配置行应提供明确错误位置。"""

        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("CAMERA_WIDTH 1920\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "第 1 行"):
                load_env_file(env_path, environ={})


if __name__ == "__main__":
    unittest.main()
