"""严格加载 Task1 JSON 与机器环境并构造冻结聚合配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from ec_26_e.actuation import ActuationConfig, MagnetConfig
from ec_26_e.cam import CameraConfig
from ec_26_e.config.env import load_env_file
from ec_26_e.config.runtime import DEFAULT_ENV_PATH
from ec_26_e.control import PlanningOptions
from ec_26_e.motion import FocRotationConfig, MotionSerialConfig
from ec_26_e.tasks.task1 import Task1Options
from ec_26_e.vision import Detect1Config, ResizeConfig
from ec_26_e.vision.resize.contracts import (
    ASPECT_RATIO_TOLERANCE,
    aspect_error,
)


DEFAULT_TASK1_CONFIG_PATH = Path("configs/tasks/task1.json")
TASK1_REVIEW_CYCLE_SECONDS_ENV = "TASK1_REVIEW_CYCLE_SECONDS"
TASK1_ALLOWED_JUMP_FRAMES_ENV = "TASK1_ALLOWED_JUMP_FRAMES"
DETECT1_PIECE_MODE_ENV = "DETECT1_PIECE_MODE"
_SCHEMA_VERSION = 3
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "task",
        "resize",
        "detect1",
        "planning",
    }
)
_TASK_FIELDS = frozenset(
    {
        "minimum_stable_frames",
        "piece_vertex_tolerance_mm",
        "workspace_corner_x_tolerance_mm",
        "workspace_corner_y_tolerance_mm",
        "workspace_corner_distance_tolerance_mm",
        "acquisition_timeout_seconds",
        "frame_read_timeout_seconds",
        "landscape_rotation",
    }
)
_RESIZE_FIELDS = frozenset(
    {
        "min_workspace_ratio",
        "reference_corners_relative",
        "search_band_ratio",
    }
)
_DETECT1_FIELDS = frozenset(
    {
        "piece_mode",
        "workspace_erode_ratio",
        "min_piece_area_ratio",
        "max_piece_area_ratio",
        "polygon_epsilon_ratio",
        "short_edge_merge_mm",
        "minimum_physical_edge_mm",
        "minimum_corner_turn_deg",
        "minimum_corner_depth_mm",
        "minimum_polygon_iou",
        "min_vertices",
        "max_vertices",
        "max_pieces",
    }
)
_PLANNING_FIELDS = frozenset(
    {
        "mode",
        "noise_bound_mm",
        "search_top_n",
        "start_xy_a4_mm",
        "minimum_execution_confidence",
        "competition_rules",
    }
)
@dataclass(frozen=True, slots=True)
class Task1Config:
    """聚合 Task1 启动前需要的全部强类型配置。"""

    camera: CameraConfig
    resize: ResizeConfig
    detect1: Detect1Config
    options: Task1Options
    actuation: ActuationConfig

    def __post_init__(self) -> None:
        """校验模块类型及跨相机、视觉和规划边界的组合约束。"""

        expected_types = (
            ("camera", self.camera, CameraConfig),
            ("resize", self.resize, ResizeConfig),
            ("detect1", self.detect1, Detect1Config),
            ("options", self.options, Task1Options),
            ("actuation", self.actuation, ActuationConfig),
        )
        for name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} 必须是 {expected_type.__name__}")

        if (
            self.resize.detect_width > self.camera.width
            or self.resize.detect_height > self.camera.height
        ):
            raise ValueError(
                "工作区定位分辨率不能大于相机采集分辨率"
            )
        camera_ratio = self.camera.width / self.camera.height
        detect_ratio = (
            self.resize.detect_width / self.resize.detect_height
        )
        if (
            aspect_error(detect_ratio, camera_ratio)
            > ASPECT_RATIO_TOLERANCE
        ):
            raise ValueError("工作区定位分辨率必须保持相机宽高比")

        mode = self.options.planning_options.mode
        if mode != "fixed":
            raise ValueError("Task1 仅支持 fixed 拼图模式")
        if self.detect1.max_pieces < 4:
            raise ValueError("fixed 模式要求 Detect1 至少允许 4 片")
        if self.detect1.min_vertices > 4:
            raise ValueError("fixed 模式要求 Detect1 接受 4 顶点纸片")
        if self.detect1.max_vertices < 5:
            raise ValueError("fixed 模式要求 Detect1 接受 5 顶点纸片")


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """构造 JSON 对象并拒绝同一层级出现重复键。"""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Task1 JSON 包含重复字段：{key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    """拒绝 JSON 标准之外的 NaN 与 Infinity 常量。"""

    raise ValueError(f"Task1 JSON 不允许非有限数：{value}")


def _read_json_object(path: Path) -> dict[str, object]:
    """读取 UTF-8 JSON 文件并返回经过重复键检查的顶层对象。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    if not path.exists():
        raise FileNotFoundError(f"Task1 JSON 不存在：{path}")
    if not path.is_file():
        raise ValueError(f"Task1 JSON 路径不是文件：{path}")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Task1 JSON 格式无效：第 {error.lineno} 行第 {error.colno} 列"
        ) from error
    if not isinstance(raw, dict):
        raise TypeError("Task1 JSON 顶层必须是对象")
    return raw


def _reject_unknown_fields(
    data: Mapping[str, object],
    allowed: frozenset[str],
    location: str,
) -> None:
    """拒绝指定 JSON 对象中不属于契约的字段。"""

    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"{location} 包含未知字段：{', '.join(unknown)}"
        )


def _section(
    raw: Mapping[str, object],
    name: str,
    allowed: frozenset[str],
) -> dict[str, object]:
    """读取可缺省的 JSON 子对象并执行未知字段检查。"""

    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Task1 JSON.{name} 必须是对象")
    _reject_unknown_fields(value, allowed, f"Task1 JSON.{name}")
    return dict(value)


def _json_point(
    value: object,
    location: str,
) -> tuple[object, object]:
    """把已知二维 JSON 数组转换为交给底层契约校验的元组。"""

    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{location} 必须是长度为 2 的数组")
    return value[0], value[1]


def _env_text(
    environ: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    """读取一个环境字符串，并拒绝非字符串映射值。"""

    value = environ.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"环境变量 {name} 必须是字符串")
    return value


def _env_integer(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    """严格解析一个十进制整数环境变量。"""

    value = _env_text(environ, name, str(default)).strip()
    if not _INTEGER_PATTERN.fullmatch(value):
        raise ValueError(f"环境变量 {name} 必须是十进制整数")
    return int(value)


def _env_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    """解析一个浮点环境变量并保留底层有限性校验。"""

    value = _env_text(environ, name, str(default)).strip()
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"环境变量 {name} 必须是实数") from error


def _camera_from_env(environ: Mapping[str, str]) -> CameraConfig:
    """从环境映射构造相机强类型配置。"""

    defaults = CameraConfig()
    device_path = _env_text(
        environ,
        "CAMERA_DEVICE_PATH",
        str(defaults.device_path),
    ).strip()
    if not device_path:
        raise ValueError("环境变量 CAMERA_DEVICE_PATH 不能为空")
    return CameraConfig(
        device_path=Path(device_path),
        width=_env_integer(
            environ,
            "CAMERA_WIDTH",
            defaults.width,
        ),
        height=_env_integer(
            environ,
            "CAMERA_HEIGHT",
            defaults.height,
        ),
        fps=_env_integer(environ, "CAMERA_FPS", defaults.fps),
        input_format=_env_text(
            environ,
            "CAMERA_INPUT_FORMAT",
            defaults.input_format,
        ).strip(),
        ffmpeg_executable=_env_text(
            environ,
            "CAMERA_FFMPEG_EXECUTABLE",
            defaults.ffmpeg_executable,
        ).strip(),
        startup_timeout_seconds=_env_float(
            environ,
            "CAMERA_STARTUP_TIMEOUT_SECONDS",
            defaults.startup_timeout_seconds,
        ),
        read_timeout_seconds=_env_float(
            environ,
            "CAMERA_READ_TIMEOUT_SECONDS",
            defaults.read_timeout_seconds,
        ),
    )


def _actuation_from_env(environ: Mapping[str, str]) -> ActuationConfig:
    """从机器环境构造运动、软件限位、Z 和磁铁强类型配置。"""

    serial_defaults = MotionSerialConfig()
    device = _env_text(
        environ,
        "MOTION_SERIAL_DEVICE",
        str(serial_defaults.device_path),
    ).strip()
    if not device:
        raise ValueError("环境变量 MOTION_SERIAL_DEVICE 不能为空")
    line_text = _env_text(
        environ,
        "MOTION_MAGNET_GPIO_LINE",
        "",
    ).strip()
    if not _INTEGER_PATTERN.fullmatch(line_text):
        raise ValueError(
            "环境变量 MOTION_MAGNET_GPIO_LINE 必须是非负十进制整数"
        )
    chip_text = _env_text(
        environ,
        "MOTION_MAGNET_GPIO_CHIP",
        "/dev/gpiochip0",
    ).strip()
    if not chip_text:
        raise ValueError("环境变量 MOTION_MAGNET_GPIO_CHIP 不能为空")
    defaults = {
        "x_min_mm": 0.0,
        "x_max_mm": 230.0,
        "y_min_mm": -10.0,
        "y_max_mm": 360.0,
        "a4_origin_x_mm": 14.5,
        "a4_origin_y_mm": 6.5,
        "xy_speed_mm_per_second": 80.0,
        "xy_fixed_wait_seconds": 1.0,
        "z_boot_degrees": 0.0,
        "z_up_degrees": 90.0,
        "z_down_degrees": 0.0,
        "z_enable_wait_seconds": 1.0,
        "z_move_wait_seconds": 1.0,
        "magnet_attach_wait_seconds": 0.3,
        "magnet_release_wait_seconds": 0.3,
        "rotation_zero_wait_seconds": 1.0,
    }
    return ActuationConfig(
        serial=MotionSerialConfig(
            device_path=Path(device),
            baudrate=_env_integer(
                environ,
                "MOTION_SERIAL_BAUDRATE",
                serial_defaults.baudrate,
            ),
            read_timeout_seconds=_env_float(
                environ,
                "MOTION_SERIAL_READ_TIMEOUT_SECONDS",
                serial_defaults.read_timeout_seconds,
            ),
            write_timeout_seconds=_env_float(
                environ,
                "MOTION_SERIAL_WRITE_TIMEOUT_SECONDS",
                serial_defaults.write_timeout_seconds,
            ),
            reply_timeout_seconds=_env_float(
                environ,
                "MOTION_SERIAL_REPLY_TIMEOUT_SECONDS",
                serial_defaults.reply_timeout_seconds,
            ),
        ),
        magnet=MagnetConfig(
            chip_path=Path(chip_text),
            line_offset=int(line_text),
        ),
        rotation=FocRotationConfig.from_environ(environ),
        x_min_mm=_env_float(
            environ,
            "MOTION_X_MIN_MM",
            defaults["x_min_mm"],
        ),
        x_max_mm=_env_float(
            environ,
            "MOTION_X_MAX_MM",
            defaults["x_max_mm"],
        ),
        y_min_mm=_env_float(
            environ,
            "MOTION_Y_MIN_MM",
            defaults["y_min_mm"],
        ),
        y_max_mm=_env_float(
            environ,
            "MOTION_Y_MAX_MM",
            defaults["y_max_mm"],
        ),
        a4_origin_x_mm=_env_float(
            environ,
            "MOTION_A4_ORIGIN_X_MM",
            defaults["a4_origin_x_mm"],
        ),
        a4_origin_y_mm=_env_float(
            environ,
            "MOTION_A4_ORIGIN_Y_MM",
            defaults["a4_origin_y_mm"],
        ),
        xy_speed_mm_per_second=_env_float(
            environ,
            "MOTION_XY_SPEED_MM_PER_SECOND",
            defaults["xy_speed_mm_per_second"],
        ),
        xy_fixed_wait_seconds=_env_float(
            environ,
            "MOTION_XY_FIXED_WAIT_SECONDS",
            defaults["xy_fixed_wait_seconds"],
        ),
        z_boot_degrees=_env_float(
            environ,
            "MOTION_Z_BOOT_DEGREES",
            defaults["z_boot_degrees"],
        ),
        z_up_degrees=_env_float(
            environ,
            "MOTION_Z_UP_DEGREES",
            defaults["z_up_degrees"],
        ),
        z_down_degrees=_env_float(
            environ,
            "MOTION_Z_DOWN_DEGREES",
            defaults["z_down_degrees"],
        ),
        z_speed_level=_env_integer(
            environ,
            "MOTION_Z_SPEED_LEVEL",
            64,
        ),
        z_enable_wait_seconds=_env_float(
            environ,
            "MOTION_Z_ENABLE_WAIT_SECONDS",
            defaults["z_enable_wait_seconds"],
        ),
        z_move_wait_seconds=_env_float(
            environ,
            "MOTION_Z_MOVE_WAIT_SECONDS",
            defaults["z_move_wait_seconds"],
        ),
        magnet_attach_wait_seconds=_env_float(
            environ,
            "MOTION_MAGNET_ATTACH_WAIT_SECONDS",
            defaults["magnet_attach_wait_seconds"],
        ),
        magnet_release_wait_seconds=_env_float(
            environ,
            "MOTION_MAGNET_RELEASE_WAIT_SECONDS",
            defaults["magnet_release_wait_seconds"],
        ),
        rotation_zero_wait_seconds=_env_float(
            environ,
            "MOTION_ROTATION_ZERO_WAIT_SECONDS",
            defaults["rotation_zero_wait_seconds"],
        ),
    )


def _resize_from_sources(
    environ: Mapping[str, str],
    section: dict[str, object],
) -> ResizeConfig:
    """从环境分辨率和 JSON 行为字段构造 ResizeConfig。"""

    defaults = ResizeConfig()
    return ResizeConfig(
        detect_width=_env_integer(
            environ,
            "WORKSPACE_DETECT_WIDTH",
            defaults.detect_width,
        ),
        detect_height=_env_integer(
            environ,
            "WORKSPACE_DETECT_HEIGHT",
            defaults.detect_height,
        ),
        target_width=_env_integer(
            environ,
            "A4_RESIZE_WIDTH",
            defaults.target_width,
        ),
        target_height=_env_integer(
            environ,
            "A4_RESIZE_HEIGHT",
            defaults.target_height,
        ),
        **section,
    )


def _detect1_from_sources(
    environ: Mapping[str, str],
    section: dict[str, object],
    *,
    piece_mode_env: str = DETECT1_PIECE_MODE_ENV,
) -> Detect1Config:
    """按任务指定环境覆盖和 JSON 行为字段构造 Detect1 配置。"""

    defaults = Detect1Config()
    json_piece_mode = section.pop("piece_mode", defaults.piece_mode)
    if isinstance(json_piece_mode, bool) or not isinstance(
        json_piece_mode,
        int,
    ):
        raise TypeError("Detect1 JSON.piece_mode 必须是整数")
    return Detect1Config(
        piece_mode=_env_integer(
            environ,
            piece_mode_env,
            json_piece_mode,
        ),
        **section,
    )


def _task1_from_sources(
    raw: dict[str, object],
    environ: Mapping[str, str],
) -> Task1Config:
    """把严格原始 JSON 与最终环境映射装配为 Task1Config。"""

    _reject_unknown_fields(raw, _TOP_LEVEL_FIELDS, "Task1 JSON")
    version = raw.get("schema_version")
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise ValueError(
            f"Task1 JSON.schema_version 必须为 {_SCHEMA_VERSION}"
        )
    task_section = _section(raw, "task", _TASK_FIELDS)
    resize_section = _section(raw, "resize", _RESIZE_FIELDS)
    detect1_section = _section(raw, "detect1", _DETECT1_FIELDS)
    planning_section = _section(raw, "planning", _PLANNING_FIELDS)
    if "start_xy_a4_mm" in planning_section:
        planning_section["start_xy_a4_mm"] = _json_point(
            planning_section["start_xy_a4_mm"],
            "Task1 JSON.planning.start_xy_a4_mm",
        )

    # Task1 的默认路径固定使用四片模板；控制公共接口仍保留 random。
    planning_section.setdefault("mode", "fixed")
    planning = PlanningOptions(**planning_section)
    option_defaults = Task1Options()
    options = Task1Options(
        **task_section,
        review_cycle_seconds=_env_float(
            environ,
            TASK1_REVIEW_CYCLE_SECONDS_ENV,
            option_defaults.review_cycle_seconds,
        ),
        allowed_jump_frames=_env_integer(
            environ,
            TASK1_ALLOWED_JUMP_FRAMES_ENV,
            option_defaults.allowed_jump_frames,
        ),
        planning_options=planning,
    )
    return Task1Config(
        camera=_camera_from_env(environ),
        resize=_resize_from_sources(environ, resize_section),
        detect1=_detect1_from_sources(environ, detect1_section),
        options=options,
        actuation=_actuation_from_env(environ),
    )


def load_task1_config(
    path: Path = DEFAULT_TASK1_CONFIG_PATH,
    *,
    env_path: Path | None = DEFAULT_ENV_PATH,
    environ: Mapping[str, str] | None = None,
) -> Task1Config:
    """无副作用加载环境和 Task1 JSON，并返回完整强类型配置。"""

    if environ is not None and not isinstance(environ, Mapping):
        raise TypeError("environ 必须是字符串映射或 None")
    raw = _read_json_object(path)
    merged_environ = dict(
        os.environ if environ is None else environ
    )
    if env_path is not None:
        if not isinstance(env_path, Path):
            raise TypeError("env_path 必须是 pathlib.Path 或 None")
        load_env_file(env_path, environ=merged_environ, override=False)
    return _task1_from_sources(raw, merged_environ)
