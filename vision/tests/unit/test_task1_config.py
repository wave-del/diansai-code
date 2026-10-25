"""验证 Task1 JSON、环境来源、强类型装配和跨模块配置约束。

该文件只读写临时目录，不访问相机或输出目录。可通过
``python -m unittest tests.unit.test_task1_config`` 独立运行。
"""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import ec_26_e.config as public_config
from ec_26_e.config import (
    DEFAULT_ENV_PATH,
    DEFAULT_TASK_CONFIG_DIRECTORY,
    DEFAULT_TASK1_CONFIG_PATH,
    DETECT1_PIECE_MODE_ENV,
    DRAW_ENABLED_ENV,
    RuntimeConfig,
    TASK_CONFIG_PATH_ENV,
    TASK1_ALLOWED_JUMP_FRAMES_ENV,
    TASK1_REVIEW_CYCLE_SECONDS_ENV,
    TASK_NAME_ENV,
    Task1Config,
    load_task1_config,
    load_runtime_config,
)
from ec_26_e.control import PlanningOptions
from ec_26_e.tasks.task1 import Task1Options


def write_json(path: Path, value: object) -> None:
    """把测试对象写成标准 UTF-8 JSON。"""

    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def minimal_config(**sections: object) -> dict[str, object]:
    """构造只含 schema_version 和指定覆盖段的测试配置。"""

    return {
        "schema_version": 3,
        **sections,
    }


TEST_ENV = {"MOTION_MAGNET_GPIO_LINE": "12"}


def test_env(**values: str) -> dict[str, str]:
    """构造包含必需磁铁 line 的测试机器环境。"""

    return {**TEST_ENV, **values}


class Task1ConfigContractTest(unittest.TestCase):
    """冻结配置公共导出、字段结构和无副作用默认值。"""

    def test_public_exports_and_frozen_contracts(self) -> None:
        """配置包应兼容旧入口并公开 Task1 强类型 API。"""

        self.assertEqual(
            set(public_config.__all__),
            {
                "DEFAULT_ENV_PATH",
                "DEFAULT_TASK_CONFIG_DIRECTORY",
                "DEFAULT_TASK1_CONFIG_PATH",
                "DEFAULT_TASK2_CONFIG_PATH",
                "DETECT1_PIECE_MODE_ENV",
                "DRAW_ENABLED_ENV",
                "RuntimeConfig",
                "TASK_CONFIG_PATH_ENV",
                "TASK1_ALLOWED_JUMP_FRAMES_ENV",
                "TASK1_REVIEW_CYCLE_SECONDS_ENV",
                "TASK2_ALLOWED_JUMP_FRAMES_ENV",
                "TASK2_DETECT1_PIECE_MODE_ENV",
                "TASK2_REVIEW_CYCLE_SECONDS_ENV",
                "TASK_NAME_ENV",
                "Task1Config",
                "Task2Config",
                "load_env_file",
                "load_runtime_config",
                "load_task1_config",
                "load_task2_config",
            },
        )
        self.assertEqual(DEFAULT_ENV_PATH, Path(".env"))
        self.assertEqual(
            DEFAULT_TASK_CONFIG_DIRECTORY,
            Path("configs/tasks"),
        )
        self.assertEqual(TASK_NAME_ENV, "EC_TASK")
        self.assertEqual(TASK_CONFIG_PATH_ENV, "EC_TASK_CONFIG")
        self.assertEqual(DRAW_ENABLED_ENV, "EC_DRAW_ENABLED")
        self.assertEqual(
            DETECT1_PIECE_MODE_ENV,
            "DETECT1_PIECE_MODE",
        )
        self.assertEqual(
            TASK1_REVIEW_CYCLE_SECONDS_ENV,
            "TASK1_REVIEW_CYCLE_SECONDS",
        )
        self.assertEqual(
            TASK1_ALLOWED_JUMP_FRAMES_ENV,
            "TASK1_ALLOWED_JUMP_FRAMES",
        )
        self.assertTrue(callable(load_runtime_config))
        self.assertTrue(issubclass(RuntimeConfig, object))
        self.assertEqual(
            DEFAULT_TASK1_CONFIG_PATH,
            Path("configs/tasks/task1.json"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Task1Config)),
            ("camera", "resize", "detect1", "options", "actuation"),
        )

    def test_official_json_is_complete_fixed_configuration(self) -> None:
        """仓库正式 JSON 应加载为 fixed、2 mm 和运动配置。"""

        raw = json.loads(
            DEFAULT_TASK1_CONFIG_PATH.read_text(encoding="utf-8")
        )
        config = load_task1_config(environ=TEST_ENV, env_path=None)

        self.assertEqual(
            set(raw),
            {
                "schema_version",
                "task",
                "resize",
                "detect1",
                "planning",
            },
        )
        self.assertEqual(raw["schema_version"], 3)
        self.assertEqual(len(raw["task"]), 8)
        self.assertEqual(len(raw["resize"]), 3)
        self.assertEqual(len(raw["detect1"]), 13)
        self.assertEqual(len(raw["planning"]), 5)
        self.assertIsInstance(config, Task1Config)
        self.assertEqual(config.options.planning_options.mode, "fixed")
        self.assertEqual(
            config.options.planning_options.noise_bound_mm,
            2.0,
        )
        self.assertEqual(
            config.options.planning_options.minimum_execution_confidence,
            0.0,
        )
        self.assertEqual(config.options.minimum_stable_frames, 5)
        self.assertEqual(config.options.review_cycle_seconds, 2.0)
        self.assertEqual(config.options.allowed_jump_frames, 5)
        self.assertEqual(
            config.options.workspace_corner_distance_tolerance_mm,
            2.5,
        )
        self.assertEqual(
            config.resize.reference_corners_relative,
            (
                (0.1375, 0.065),
                (0.778, 0.078),
                (0.771, 0.87),
                (0.1375, 0.844),
            ),
        )
        self.assertEqual(config.resize.search_band_ratio, 0.14)
        self.assertEqual(config.detect1.piece_mode, 2)
        self.assertEqual(config.detect1.short_edge_merge_mm, 20.0)
        self.assertEqual(config.detect1.minimum_physical_edge_mm, 20.0)
        self.assertEqual(config.detect1.minimum_corner_turn_deg, 12.0)
        self.assertEqual(config.detect1.minimum_corner_depth_mm, 4.0)
        self.assertEqual(config.detect1.minimum_polygon_iou, 0.85)
        self.assertEqual(config.actuation.magnet.line_offset, 12)
        self.assertEqual(config.actuation.xy_speed_mm_per_second, 80.0)
        self.assertEqual(config.actuation.xy_fixed_wait_seconds, 1.0)
        self.assertEqual(config.actuation.rotation_zero_wait_seconds, 1.0)
        self.assertEqual(
            config.actuation.rotation.clockwise_mechanical_sign,
            1,
        )
        self.assertEqual(
            (
                config.actuation.a4_origin_x_mm,
                config.actuation.a4_origin_y_mm,
            ),
            (14.5, 6.5),
        )

    def test_missing_sections_use_code_defaults(self) -> None:
        """省略行为段时应回落到现有模块和 Task1 代码默认值。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(path, minimal_config())

            config = load_task1_config(
                path,
                environ=TEST_ENV,
                env_path=None,
            )

        self.assertEqual(
            config.options,
            Task1Options(planning_options=PlanningOptions(mode="fixed")),
        )
        self.assertEqual(
            config.options.planning_options,
            PlanningOptions(mode="fixed"),
        )
        self.assertEqual(config.actuation.magnet.line_offset, 12)

    def test_json_execution_confidence_threshold_overrides_default(self) -> None:
        """Task1 JSON 的规划阈值应进入强类型执行配置。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(
                path,
                minimal_config(
                    planning={"minimum_execution_confidence": 0.75},
                ),
            )
            config = load_task1_config(
                path,
                environ=TEST_ENV,
                env_path=None,
            )

        self.assertEqual(
            config.options.planning_options.minimum_execution_confidence,
            0.75,
        )

    def test_json_configures_edge_thresholds(self) -> None:
        """Task1 JSON 应分别配置假短边合并和物理边下限。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(
                path,
                minimal_config(
                    detect1={
                        "short_edge_merge_mm": 9.0,
                        "minimum_physical_edge_mm": 5.5,
                    },
                ),
            )
            config = load_task1_config(
                path,
                environ=TEST_ENV,
                env_path=None,
            )

        self.assertEqual(config.detect1.short_edge_merge_mm, 9.0)
        self.assertEqual(config.detect1.minimum_physical_edge_mm, 5.5)


class Task1ConfigSourceTest(unittest.TestCase):
    """验证导出环境、.env、JSON 和默认值的组合来源。"""

    def test_exported_environment_overrides_env_without_mutation(
        self,
    ) -> None:
        """显式环境应覆盖 .env，且输入映射与文件保持不变。"""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            json_path = root / "task1.json"
            env_text = (
                "CAMERA_WIDTH=1280\n"
                "CAMERA_HEIGHT=720\n"
                "WORKSPACE_DETECT_WIDTH=320\n"
                "WORKSPACE_DETECT_HEIGHT=180\n"
                "CAMERA_FPS=25\n"
                "TASK1_REVIEW_CYCLE_SECONDS=3.0\n"
                "TASK1_ALLOWED_JUMP_FRAMES=4\n"
            )
            env_path.write_text(env_text, encoding="utf-8")
            write_json(
                json_path,
                minimal_config(task={"minimum_stable_frames": 12}),
            )
            environ = {
                **TEST_ENV,
                "CAMERA_WIDTH": "1920",
                "CAMERA_HEIGHT": "1080",
                "CAMERA_FPS": "30",
                "TASK1_REVIEW_CYCLE_SECONDS": "1.5",
                "TASK1_ALLOWED_JUMP_FRAMES": "5",
            }
            snapshot = dict(environ)

            config = load_task1_config(
                json_path,
                env_path=env_path,
                environ=environ,
            )

            self.assertEqual(environ, snapshot)
            self.assertEqual(env_path.read_text(encoding="utf-8"), env_text)

        self.assertEqual(
            (config.camera.width, config.camera.height, config.camera.fps),
            (1920, 1080, 30),
        )
        self.assertEqual(
            (config.resize.detect_width, config.resize.detect_height),
            (320, 180),
        )
        self.assertEqual(config.options.minimum_stable_frames, 12)
        self.assertEqual(config.options.review_cycle_seconds, 1.5)
        self.assertEqual(config.options.allowed_jump_frames, 5)

    def test_all_machine_environment_fields_are_typed(self) -> None:
        """全部机器字段应由环境映射进入底层契约。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(path, minimal_config())
            environ = {
                **TEST_ENV,
                "CAMERA_DEVICE_PATH": "/dev/video8",
                "CAMERA_WIDTH": "1280",
                "CAMERA_HEIGHT": "720",
                "CAMERA_FPS": "60",
                "CAMERA_INPUT_FORMAT": "yuyv422",
                "CAMERA_FFMPEG_EXECUTABLE": "/usr/bin/ffmpeg",
                "CAMERA_STARTUP_TIMEOUT_SECONDS": "7.5",
                "CAMERA_READ_TIMEOUT_SECONDS": "1.5",
                "WORKSPACE_DETECT_WIDTH": "640",
                "WORKSPACE_DETECT_HEIGHT": "360",
                "A4_RESIZE_WIDTH": "840",
                "A4_RESIZE_HEIGHT": "594",
                "TASK1_REVIEW_CYCLE_SECONDS": "2.5",
                "TASK1_ALLOWED_JUMP_FRAMES": "7",
                "DETECT1_PIECE_MODE": "1",
                "FOC_SERIAL_PORT": "/dev/ttyUSB7",
                "FOC_RPM": "18",
                "MOTION_ROTATION_CLOCKWISE_SIGN": "-1",
                "MOTION_ROTATION_ZERO_WAIT_SECONDS": "1.25",
            }

            config = load_task1_config(
                path,
                environ=environ,
                env_path=None,
            )

        self.assertEqual(config.camera.device_path, Path("/dev/video8"))
        self.assertEqual(config.camera.fps, 60)
        self.assertEqual(config.camera.input_format, "yuyv422")
        self.assertEqual(
            config.camera.ffmpeg_executable,
            "/usr/bin/ffmpeg",
        )
        self.assertEqual(config.camera.startup_timeout_seconds, 7.5)
        self.assertEqual(config.camera.read_timeout_seconds, 1.5)
        self.assertEqual(config.options.review_cycle_seconds, 2.5)
        self.assertEqual(config.options.allowed_jump_frames, 7)
        self.assertEqual(config.detect1.piece_mode, 1)
        self.assertEqual(config.actuation.rotation.port, "/dev/ttyUSB7")
        self.assertEqual(config.actuation.rotation.rpm, 18)
        self.assertEqual(
            config.actuation.rotation.clockwise_mechanical_sign,
            -1,
        )
        self.assertEqual(
            config.actuation.rotation_zero_wait_seconds,
            1.25,
        )
        self.assertEqual(
            (
                config.resize.target_width,
                config.resize.target_height,
            ),
            (840, 594),
        )

    def test_a4_origin_environment_overrides_default_calibration(self) -> None:
        """A4 原点标定应默认固定，并允许按机器环境单独覆盖。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(path, minimal_config())
            config = load_task1_config(
                path,
                environ=test_env(
                    MOTION_A4_ORIGIN_X_MM="15.25",
                    MOTION_A4_ORIGIN_Y_MM="7.75",
                ),
                env_path=None,
            )

        self.assertEqual(
            (
                config.actuation.a4_origin_x_mm,
                config.actuation.a4_origin_y_mm,
            ),
            (15.25, 7.75),
        )

    def test_invalid_environment_values_fail_strictly(self) -> None:
        """环境整数、实数、枚举和空设备路径不得被宽松转换。"""

        invalid_values: tuple[tuple[str, object], ...] = (
            ("CAMERA_WIDTH", "1280.0"),
            ("CAMERA_WIDTH", True),
            ("CAMERA_FPS", "thirty"),
            ("CAMERA_INPUT_FORMAT", "rgb"),
            ("CAMERA_STARTUP_TIMEOUT_SECONDS", "nan"),
            ("CAMERA_DEVICE_PATH", ""),
            ("TASK1_REVIEW_CYCLE_SECONDS", "nan"),
            ("TASK1_ALLOWED_JUMP_FRAMES", "5.0"),
            ("DETECT1_PIECE_MODE", "true"),
            ("DETECT1_PIECE_MODE", "0"),
            ("DETECT1_PIECE_MODE", "3"),
            ("MOTION_ROTATION_CLOCKWISE_SIGN", "0"),
            ("MOTION_ROTATION_ZERO_WAIT_SECONDS", "0"),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(path, minimal_config())
            for name, value in invalid_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        load_task1_config(
                            path,
                            environ={
                                **TEST_ENV,
                                name: value,
                            },  # type: ignore[dict-item]
                            env_path=None,
                        )

    def test_magnet_gpio_line_is_required(self) -> None:
        """缺少磁铁 line 时应在任何硬件访问前拒绝启动。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(path, minimal_config())
            with self.assertRaisesRegex(
                ValueError,
                "MOTION_MAGNET_GPIO_LINE",
            ):
                load_task1_config(path, environ={}, env_path=None)


class Task1JsonStrictnessTest(unittest.TestCase):
    """验证 JSON 结构、键、版本和数值类型均执行严格拒绝。"""

    def _assert_raw_rejected(
        self,
        raw: str,
        expected_exception: type[BaseException] | tuple[
            type[BaseException],
            ...,
        ] = (TypeError, ValueError),
    ) -> None:
        """写入原始 JSON 并断言加载失败。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(expected_exception):
                load_task1_config(path, environ=TEST_ENV, env_path=None)

    def test_missing_file_and_non_file_path_are_rejected(self) -> None:
        """正式 JSON 缺失或路径指向目录时必须立即失败。"""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_task1_config(
                    root / "missing.json",
                    environ=TEST_ENV,
                    env_path=None,
                )
            with self.assertRaisesRegex(ValueError, "不是文件"):
                load_task1_config(
                    root,
                    environ=TEST_ENV,
                    env_path=None,
                )

    def test_top_level_version_duplicates_and_nonfinite_are_rejected(
        self,
    ) -> None:
        """非对象、错误版本、重复键和非有限数均应失败。"""

        invalid_documents = (
            "[]",
            "{}",
            '{"schema_version": true}',
            '{"schema_version": 1.0}',
            '{"schema_version": 2}',
            '{"schema_version": 3, "mode": "other"}',
            '{"schema_version": 3, "schema_version": 3}',
            '{"schema_version": 3, "task": {"x": NaN}}',
            '{"schema_version": 3, "task": {"x": Infinity}}',
            (
                '{"schema_version": 3, "task": '
                '{"minimum_stable_frames": 10, '
                '"minimum_stable_frames": 11}}'
            ),
        )
        for raw in invalid_documents:
            with self.subTest(raw=raw):
                self._assert_raw_rejected(raw)

    def test_unknown_fields_and_non_object_sections_are_rejected(
        self,
    ) -> None:
        """任一层级的未知字段或非对象分段都应明确失败。"""

        documents = (
            minimal_config(extra={}),
            minimal_config(task={"unknown": 1}),
            minimal_config(resize={"detect_width": 640}),
            minimal_config(detect1=[]),
            minimal_config(planning=None),
            minimal_config(output={}),
        )
        for document in documents:
            with self.subTest(document=document):
                self._assert_raw_rejected(json.dumps(document))

    def test_string_numbers_booleans_and_invalid_points_are_rejected(
        self,
    ) -> None:
        """字符串、布尔数值和错误坐标不得被隐式转换。"""

        documents = (
            minimal_config(task={"minimum_stable_frames": "10"}),
            minimal_config(task={"minimum_stable_frames": True}),
            minimal_config(
                resize={
                    "reference_corners_relative": [[0.1, 0.1]]
                }
            ),
            minimal_config(resize={"search_band_ratio": 0.5}),
            minimal_config(planning={"noise_bound_mm": "2.0"}),
            minimal_config(planning={"start_xy_a4_mm": [10.0]}),
            minimal_config(planning={"start_xy_a4_mm": [10.0, True]}),
        )
        for document in documents:
            with self.subTest(document=document):
                self._assert_raw_rejected(json.dumps(document))


class Task1AggregateValidationTest(unittest.TestCase):
    """验证配置组合在任何设备访问之前排除必然失败场景。"""

    def _load(
        self,
        document: dict[str, object],
        environ: dict[str, str] | None = None,
    ) -> Task1Config:
        """从临时 JSON 和可选环境加载一个聚合配置。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "task1.json"
            write_json(path, document)
            return load_task1_config(
                path,
                environ=(
                    test_env()
                    if environ is None
                    else test_env(**environ)
                ),
                env_path=None,
            )

    def test_detect_resolution_must_fit_and_match_camera(self) -> None:
        """工作区定位分辨率不得放大且必须保持相机宽高比。"""

        invalid_environments = (
            {
                "CAMERA_WIDTH": "640",
                "CAMERA_HEIGHT": "360",
                "WORKSPACE_DETECT_WIDTH": "1280",
                "WORKSPACE_DETECT_HEIGHT": "720",
            },
            {
                "CAMERA_WIDTH": "1280",
                "CAMERA_HEIGHT": "720",
                "WORKSPACE_DETECT_WIDTH": "640",
                "WORKSPACE_DETECT_HEIGHT": "400",
            },
        )
        for environ in invalid_environments:
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    self._load(minimal_config(), environ)

    def test_fixed_mode_requires_four_pieces_and_four_to_five_vertices(
        self,
    ) -> None:
        """fixed 组合必须允许模板所需的纸片数和顶点范围。"""

        invalid_detect_sections = (
            {"max_pieces": 3},
            {"min_vertices": 5, "max_vertices": 6},
            {"max_vertices": 4},
        )
        for detect1 in invalid_detect_sections:
            with self.subTest(detect1=detect1):
                with self.assertRaisesRegex(ValueError, "fixed"):
                    self._load(
                        minimal_config(
                            planning={"mode": "fixed"},
                            detect1=detect1,
                        )
                    )

    def test_task1_rejects_random_mode(
        self,
    ) -> None:
        """Task1 只接受固定四片拼图模式。"""

        with self.assertRaisesRegex(ValueError, "仅支持 fixed"):
            self._load(
                minimal_config(
                    planning={"mode": "random"},
                )
            )


if __name__ == "__main__":
    unittest.main()
