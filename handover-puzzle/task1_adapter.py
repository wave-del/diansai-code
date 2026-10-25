"""把参考 Task1 的纯 ``MovePlan`` 适配到本机拼图硬件。

本模块刻意不修改 ``reference-vision`` 的视觉、算法和原运动协议。它只
处理两件事：

* 将参考工程的竖向标准 A4 坐标转换为本机蓝纸右上角坐标；
* 在连接任何硬件之前完成限位与旋转配置预检，随后复用
  :class:`PuzzleController` 的已验证取放和异常恢复流程。

参考计划中的旋转是视觉平面中的“顺时针刚体旋转”；F32C 的机械正方向取决于
安装，不能由软件推断。因此未填写 ``vision_clockwise_to_motor_sign`` 时，
本模块可以预览计划，但会拒绝执行包含旋转的计划。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol


REFERENCE_A4_WIDTH_MM = 210.0
REFERENCE_A4_HEIGHT_MM = 297.0
ROTATION_CALIBRATION_SCHEMA_VERSION = 1
# 与取放旋转使用同一精确到位标准。回零未达标时会对同一零位目标重发
# 两次；只要角度反馈仍有效，后续取件会带告警继续，而不会中止任务。
F32C_HOME_TOLERANCE_DEG = 1.5
F32C_HOME_SETTLE_RETRIES = 2
DEFAULT_SAFE_PARKING_CONFIG = (
    Path(__file__).resolve().parent.parent / "host" / "safe_parking.json"
)


class Task1AdaptationError(RuntimeError):
    """表示计划、坐标标定或执行前条件不满足。"""


@dataclass(frozen=True, slots=True)
class F32CRotationCalibration:
    """视觉顺时针和 F32C 机械正方向之间的现场标定结果。"""

    vision_clockwise_to_motor_sign: int
    probe_degrees: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.vision_clockwise_to_motor_sign, bool)
            or self.vision_clockwise_to_motor_sign not in (-1, 1)
        ):
            raise ValueError("旋转标定 sign 必须为 -1 或 1")
        if not 1.0 <= self.probe_degrees <= 90.0:
            raise ValueError("旋转标定 probe_degrees 必须在 1～90 度")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ROTATION_CALIBRATION_SCHEMA_VERSION,
            "vision_clockwise_to_motor_sign": self.vision_clockwise_to_motor_sign,
            "probe_degrees": self.probe_degrees,
        }


def save_rotation_calibration(
    path: Path,
    calibration: F32CRotationCalibration,
) -> None:
    """原子写入旋转方向标定文件。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    if not isinstance(calibration, F32CRotationCalibration):
        raise TypeError("calibration 必须是 F32CRotationCalibration")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_rotation_calibration(path: Path) -> F32CRotationCalibration:
    """读取并严格校验旋转方向标定文件。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise Task1AdaptationError(f"无法读取旋转标定文件 {path}：{error}") from error
    if not isinstance(raw, dict):
        raise Task1AdaptationError("旋转标定文件顶层必须是 JSON 对象")
    expected_fields = {
        "schema_version",
        "vision_clockwise_to_motor_sign",
        "probe_degrees",
    }
    if set(raw) != expected_fields:
        raise Task1AdaptationError("旋转标定文件字段不匹配")
    if raw["schema_version"] != ROTATION_CALIBRATION_SCHEMA_VERSION:
        raise Task1AdaptationError("旋转标定文件版本不受支持")
    try:
        return F32CRotationCalibration(
            vision_clockwise_to_motor_sign=raw["vision_clockwise_to_motor_sign"],
            probe_degrees=raw["probe_degrees"],
        )
    except (TypeError, ValueError) as error:
        raise Task1AdaptationError(f"旋转标定文件内容无效：{error}") from error


def load_safe_parking_config(path: Path) -> tuple[float, float]:
    """读取 host 记录的蓝纸安全停靠点。"""

    if not isinstance(path, Path):
        raise TypeError("path 必须是 pathlib.Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        point = raw["blue_paper_xy_mm"]
        x, y = float(point[0]), float(point[1])
    except FileNotFoundError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise Task1AdaptationError(f"无法读取安全停靠点配置 {path}：{error}") from error
    if raw.get("schema_version") != 1 or not math.isfinite(x) or not math.isfinite(y):
        raise Task1AdaptationError("安全停靠点配置无效")
    return x, y


@dataclass(frozen=True, slots=True)
class TangramAdapterConfig:
    """本机蓝纸、写字机和旋转方向的部署标定。

    ``writer_x/y_*`` 是传给 ``WriterController.move_to`` 前允许的业务
    坐标。它们按已知的 ``X-23.5``、``Y+16.45`` 写字机映射推导。比赛
    现场的视觉四角可能有少量漂移：有限坐标超出该范围时，适配层会显式
    限幅并记录日志，再交给底层执行；不会因可恢复的越界而放弃整题。
    """

    blue_width_mm: float = 206.5
    blue_height_mm: float = 292.0
    # 固定题沿用历史 210×297 参考坐标；随机题使用同事新版求解器的
    # 207×292 坐标。把参考尺寸放进部署配置，避免两个任务相互缩放污染。
    reference_width_mm: float = REFERENCE_A4_WIDTH_MM
    reference_height_mm: float = REFERENCE_A4_HEIGHT_MM
    writer_x_min_mm: float = 23.5
    writer_x_max_mm: float = 206.5
    writer_y_min_mm: float = 0.0
    writer_y_max_mm: float = 283.55
    safe_parking_x_mm: float = 0.0
    safe_parking_y_mm: float = -1.0
    # ``task1_yolo_flow`` 将横向校正图顺时针转为求解器的竖向 A4
    # 坐标时，须同时翻转两个轴，才能回到 example 已验证的写字机坐标系。
    # False 保持同事原来的逆时针视觉路径，True 用于本机默认的顺时针路径。
    reverse_reference_axes: bool = False
    # +1 表示视觉顺时针 = F32C 机械正方向；-1 表示相反。
    # None 是安全默认：可预览，不能执行带旋转的计划。
    vision_clockwise_to_motor_sign: int | None = None
    rotation_epsilon_deg: float = 0.01
    # 上位机正式拼图流程的 XY 进给速度，单位 mm/min。
    xy_feed_mm_min: float = 4500.0
    z_feed_mm_min: float = 1500.0
    safe_z_mm: float = -10.0
    pickup_z_mm: float = -2.0
    place_z_mm: float = -2.0
    rest_z_mm: float = -10.0
    continuous_transfers: bool = False
    magnet_pickup_dwell_seconds: float = 0.5
    magnet_release_dwell_seconds: float = 0.5
    rotation_rpm: int = 29

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "vision_clockwise_to_motor_sign":
                continue
            if name == "reverse_reference_axes":
                continue
            if name == "continuous_transfers":
                continue
            if name == "rotation_rpm":
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} 必须是实数")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数")
        if (
            self.blue_width_mm <= 0
            or self.blue_height_mm <= 0
            or self.reference_width_mm <= 0
            or self.reference_height_mm <= 0
        ):
            raise ValueError("蓝纸和参考坐标宽高必须大于 0")
        if not 0 <= self.writer_x_min_mm < self.writer_x_max_mm:
            raise ValueError("写字机 X 边界必须满足 0 <= min < max")
        if not 0 <= self.writer_y_min_mm < self.writer_y_max_mm:
            raise ValueError("写字机 Y 边界必须满足 0 <= min < max")
        if (
            isinstance(self.vision_clockwise_to_motor_sign, bool)
            or self.vision_clockwise_to_motor_sign not in (None, -1, 1)
        ):
            raise ValueError(
                "vision_clockwise_to_motor_sign 必须是 -1、1 或 None"
            )
        if not isinstance(self.reverse_reference_axes, bool):
            raise TypeError("reverse_reference_axes 必须是布尔值")
        if not isinstance(self.continuous_transfers, bool):
            raise TypeError("continuous_transfers 必须是布尔值")
        for name, value in (
            ("safe_parking_x_mm", self.safe_parking_x_mm),
            ("safe_parking_y_mm", self.safe_parking_y_mm),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} 必须是有限实数")
        if self.rotation_epsilon_deg < 0:
            raise ValueError("rotation_epsilon_deg 不能小于 0")
        if self.xy_feed_mm_min <= 0 or self.z_feed_mm_min <= 0:
            raise ValueError("XY/Z 速度必须大于 0")
        for name, value in (
            ("safe_z_mm", self.safe_z_mm),
            ("pickup_z_mm", self.pickup_z_mm),
            ("place_z_mm", self.place_z_mm),
            ("rest_z_mm", self.rest_z_mm),
        ):
            if not -15.0 <= value <= 0.0:
                raise ValueError(f"{name} 必须位于本机已验证的 [-15, 0] mm")
        if not self.safe_z_mm < min(self.pickup_z_mm, self.place_z_mm):
            raise ValueError("safe_z_mm 必须严格高于取放 Z")
        if self.rest_z_mm < self.safe_z_mm:
            raise ValueError("rest_z_mm 不得高于 safe_z_mm")
        if max(
            self.pickup_z_mm - self.safe_z_mm,
            self.place_z_mm - self.safe_z_mm,
            self.rest_z_mm - self.safe_z_mm,
        ) > 15.0:
            raise ValueError("本机已验证 Z 抬高范围不得超过 15 mm")
        if (
            self.magnet_pickup_dwell_seconds < 0
            or self.magnet_release_dwell_seconds < 0
        ):
            raise ValueError("电磁铁等待时间不能小于 0")
        if not isinstance(self.rotation_rpm, int) or not 1 <= self.rotation_rpm <= 30:
            raise ValueError("rotation_rpm 必须是 1～30 的整数")

    def reference_to_writer_xy(
        self,
        x_a4_mm: float,
        y_a4_mm: float,
    ) -> tuple[float, float]:
        """转换参考竖向 A4 坐标为本机蓝纸右上角毫米坐标。"""

        x = _finite("参考 X", x_a4_mm)
        y = _finite("参考 Y", y_a4_mm)
        # 此处也做一次无日志的兜底限幅，保证外部调用本方法时与正式
        # adapt_move_plan 的行为一致。正式链路会在 clamp_reference_xy
        # 中打印完整告警。
        x = min(self.reference_width_mm, max(0.0, x))
        y = min(self.reference_height_mm, max(0.0, y))
        scaled_x = x * self.blue_width_mm / self.reference_width_mm
        scaled_y = y * self.blue_height_mm / self.reference_height_mm
        if self.reverse_reference_axes:
            return (self.blue_width_mm - scaled_x, self.blue_height_mm - scaled_y)
        return (scaled_x, scaled_y)

    def clamp_reference_xy(self, label: str, x: float, y: float) -> tuple[float, float]:
        """将有限参考坐标限制到 A4 内，并显式记录可恢复的视觉越界。"""

        original_x, original_y = _finite(f"{label} 参考 X", x), _finite(f"{label} 参考 Y", y)
        bounded_x = min(self.reference_width_mm, max(0.0, original_x))
        bounded_y = min(self.reference_height_mm, max(0.0, original_y))
        if (bounded_x, bounded_y) != (original_x, original_y):
            print(
                f"⚠ 坐标限幅：{label} 参考坐标 "
                f"({original_x:.3f}, {original_y:.3f}) mm → "
                f"({bounded_x:.3f}, {bounded_y:.3f}) mm；继续执行。",
                flush=True,
            )
        return bounded_x, bounded_y

    def validate_writer_xy(self, label: str, x: float, y: float) -> tuple[float, float]:
        """将业务坐标限制到已标定可执行区，而非因越界拒绝整份计划."""

        original_x, original_y = _finite(f"{label} X", x), _finite(f"{label} Y", y)
        bounded_x = min(self.writer_x_max_mm, max(self.writer_x_min_mm, original_x))
        bounded_y = min(self.writer_y_max_mm, max(self.writer_y_min_mm, original_y))
        if (bounded_x, bounded_y) != (original_x, original_y):
            print(
                f"⚠ 坐标限幅：{label} 蓝纸坐标 "
                f"({original_x:.3f}, {original_y:.3f}) mm → "
                f"({bounded_x:.3f}, {bounded_y:.3f}) mm；继续执行。",
                flush=True,
            )
        return bounded_x, bounded_y


@dataclass(frozen=True, slots=True)
class TangramMove:
    """一片拼图在本机业务坐标中的冻结执行计划。"""

    order: int
    piece_id: str
    pickup_xy_mm: tuple[float, float]
    place_xy_mm: tuple[float, float]
    vision_rotation_clockwise_deg: float
    motor_rotation_delta_deg: float | None
    template_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "piece_id": self.piece_id,
            "pickup_xy_mm": list(self.pickup_xy_mm),
            "place_xy_mm": list(self.place_xy_mm),
            "vision_rotation_clockwise_deg": self.vision_rotation_clockwise_deg,
            "motor_rotation_delta_deg": self.motor_rotation_delta_deg,
            "template_id": self.template_id,
        }


@dataclass(frozen=True, slots=True)
class TangramPlan:
    """参考 ``MovePlan`` 的本机可预览版本。"""

    mode: str
    confidence: float
    ambiguous: bool
    moves: tuple[TangramMove, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "moves": [move.to_dict() for move in self.moves],
        }

    def require_executable(self) -> None:
        """在任何硬件连接前检查旋转标定是否完整。"""

        uncalibrated = [
            move.piece_id
            for move in self.moves
            if abs(move.vision_rotation_clockwise_deg) > 0.01
            and move.motor_rotation_delta_deg is None
        ]
        if uncalibrated:
            raise Task1AdaptationError(
                "计划包含旋转，但未标定视觉顺时针到 F32C 机械方向："
                + ", ".join(uncalibrated)
            )


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数")
    return result


def _normalise_rotation(delta_deg: float) -> float:
    """转为 F32C 绝对定位可表达的最短等价角度。"""

    result = (delta_deg + 180.0) % 360.0 - 180.0
    # 精确 180° 保留传入方向，便于计划与日志一致。
    if math.isclose(abs(result), 180.0, abs_tol=1e-9):
        return math.copysign(180.0, delta_deg)
    return result


def adapt_move_plan(plan: object, config: TangramAdapterConfig) -> TangramPlan:
    """把参考工程的 ``MovePlan`` 转换、量化并完成全部坐标预检。"""

    if not isinstance(config, TangramAdapterConfig):
        raise TypeError("config 必须是 TangramAdapterConfig")
    moves = getattr(plan, "moves", None)
    if not isinstance(moves, tuple) or not moves:
        raise Task1AdaptationError("参考 MovePlan 必须包含非空 moves 元组")

    converted: list[TangramMove] = []
    seen_ids: set[str] = set()
    for expected_order, move in enumerate(moves, start=1):
        order = getattr(move, "order", None)
        piece_id = getattr(move, "piece_id", None)
        if order != expected_order:
            raise Task1AdaptationError("MovePlan 的 order 必须从 1 连续递增")
        if not isinstance(piece_id, str) or not piece_id or piece_id in seen_ids:
            raise Task1AdaptationError("MovePlan 包含空或重复 piece_id")
        seen_ids.add(piece_id)
        pick = getattr(move, "pick_xy_a4_mm", None)
        place = getattr(move, "place_xy_a4_mm", None)
        if not isinstance(pick, tuple) or len(pick) != 2:
            raise Task1AdaptationError(f"{piece_id} 的拾取坐标无效")
        if not isinstance(place, tuple) or len(place) != 2:
            raise Task1AdaptationError(f"{piece_id} 的放置坐标无效")
        pickup_reference = config.clamp_reference_xy(f"{piece_id} 拾取", pick[0], pick[1])
        target_reference = config.clamp_reference_xy(f"{piece_id} 放置", place[0], place[1])
        pickup = config.reference_to_writer_xy(*pickup_reference)
        target = config.reference_to_writer_xy(*target_reference)
        pickup = config.validate_writer_xy(f"{piece_id} 拾取", *pickup)
        target = config.validate_writer_xy(f"{piece_id} 放置", *target)
        visual_rotation = _finite(
            f"{piece_id} rotation_deg_clockwise",
            getattr(move, "rotation_deg_clockwise", None),
        )
        if abs(visual_rotation) <= config.rotation_epsilon_deg:
            motor_delta: float | None = 0.0
        elif config.vision_clockwise_to_motor_sign is None:
            motor_delta = None
        else:
            motor_delta = _normalise_rotation(
                visual_rotation * config.vision_clockwise_to_motor_sign
            )
        template_id = getattr(move, "template_id", None)
        if template_id is not None and not isinstance(template_id, str):
            raise Task1AdaptationError(f"{piece_id} 的 template_id 必须是字符串或 None")
        converted.append(
            TangramMove(
                order=order,
                piece_id=piece_id,
                pickup_xy_mm=pickup,
                place_xy_mm=target,
                vision_rotation_clockwise_deg=visual_rotation,
                motor_rotation_delta_deg=motor_delta,
                template_id=template_id,
            )
        )
    return TangramPlan(
        mode=str(getattr(plan, "mode", "unknown")),
        confidence=_finite("MovePlan confidence", getattr(plan, "confidence", None)),
        ambiguous=bool(getattr(plan, "ambiguous", False)),
        moves=tuple(converted),
    )


@dataclass(frozen=True, slots=True)
class TangramHardwarePorts:
    """本机实际串口参数。"""

    writer_port: str = ""
    motor_port: str = ""
    motor_address: int = 2

    def __post_init__(self) -> None:
        if not self.writer_port or not self.motor_port:
            raise ValueError("写字机和 F32C 串口路径不能为空")
        if isinstance(self.motor_address, bool) or not 0 <= self.motor_address <= 255:
            raise ValueError("F32C 地址必须位于 0～255")


class _PuzzleController(Protocol):
    motor: Any

    def connect(self, writer_port: str, *, writer_baud: int, motor_port: str, motor_address: int) -> None: ...
    def establish_origin(self) -> None: ...
    def prepare_existing_origin(self) -> None: ...
    def transfer_piece(self, request: object) -> object: ...
    def park_safely(self) -> None: ...
    def close(self) -> None: ...


ControllerFactory = Callable[[TangramAdapterConfig], _PuzzleController]
RequestFactory = Callable[..., object]


def _default_controller_factory(config: TangramAdapterConfig) -> _PuzzleController:
    try:
        from .puzzle_controller import PuzzleConfig, PuzzleController
    except ImportError:
        from puzzle_controller import PuzzleConfig, PuzzleController
    return PuzzleController(
        PuzzleConfig(
            safe_z=config.safe_z_mm,
            pickup_z=config.pickup_z_mm,
            place_z=config.place_z_mm,
            rest_z=config.rest_z_mm,
            continuous_transfers=config.continuous_transfers,
            safe_parking_x=config.safe_parking_x_mm,
            safe_parking_y=config.safe_parking_y_mm,
            xy_feed=config.xy_feed_mm_min,
            z_feed=config.z_feed_mm_min,
            magnet_pickup_dwell=config.magnet_pickup_dwell_seconds,
            magnet_release_dwell=config.magnet_release_dwell_seconds,
            rotation_rpm=config.rotation_rpm,
            workspace_x_max=config.writer_x_max_mm,
            workspace_y_max=config.writer_y_max_mm,
        )
    )


def _default_request_factory(**kwargs: object) -> object:
    try:
        from .puzzle_controller import TransferRequest
    except ImportError:
        from puzzle_controller import TransferRequest
    return TransferRequest(**kwargs)


class TangramPlanExecutor:
    """以本机 ``PuzzleController`` 执行一个已经预检的 ``TangramPlan``。"""

    def __init__(
        self,
        config: TangramAdapterConfig,
        ports: TangramHardwarePorts,
        *,
        controller_factory: ControllerFactory = _default_controller_factory,
        request_factory: RequestFactory = _default_request_factory,
    ) -> None:
        self._config = config
        self._ports = ports
        self._controller_factory = controller_factory
        self._request_factory = request_factory

    def execute(
        self,
        plan: TangramPlan,
        *,
        reuse_machine_origin: bool = False,
        origin_ready_callback: Callable[[bool], None] | None = None,
        completion_callback: Callable[[], None] | None = None,
    ) -> tuple[object, ...]:
        """连接一次硬件，逐片执行，并始终调用底层安全关闭。

        ``reuse_machine_origin`` 仅限机构仍位于上一轮安全停靠点、且未被
        人工移动的连续任务；串口重连后会把当前位置重新锚定为安全停靠点，
        不能把该点误设为机械零点。
        """

        if not isinstance(plan, TangramPlan):
            raise TypeError("plan 必须是 TangramPlan")
        plan.require_executable()
        controller = self._controller_factory(self._config)
        results: list[object] = []
        try:
            controller.connect(
                self._ports.writer_port,
                writer_baud=115200,
                motor_port=self._ports.motor_port,
                motor_address=self._ports.motor_address,
            )
            if reuse_machine_origin:
                controller.prepare_existing_origin()
            else:
                controller.establish_origin()
            if origin_ready_callback is not None:
                origin_ready_callback(reuse_machine_origin)
            if self._config.continuous_transfers:
                # 连续模式以机械零位为每片旋转的共同基准。先同步回零一次，
                # 防止上次异常结束留下的角度被带入首片。
                controller.motor.return_to_mechanical_zero(
                    rpm=self._config.rotation_rpm,
                    wait=True,
                    tolerance_deg=F32C_HOME_TOLERANCE_DEG,
                    settle_retries=F32C_HOME_SETTLE_RETRIES,
                    allow_inexact_arrival=True,
                )
            pending_home: Future[object] | None = None
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="tangram-f32c-home",
            ) as home_executor:
                for index, move in enumerate(plan.moves):
                    if move.motor_rotation_delta_deg in (None, 0.0):
                        rotation_target: float | None = None
                    elif self._config.continuous_transfers:
                        # 前片的回零任务在下一片下降前由 barrier 确认；因此
                        # 这里可安全地用零位作为本片绝对旋转基准，不读取旋转中
                        # 的瞬时角度。
                        rotation_target = move.motor_rotation_delta_deg % 360.0
                    else:
                        current = _finite(
                            "F32C 当前机械角度", controller.motor.get_angle()
                        )
                        rotation_target = (
                            current + move.motor_rotation_delta_deg
                        ) % 360.0
                    barrier: Callable[[], None] | None = None
                    if pending_home is not None:
                        barrier = pending_home.result
                    request = self._request_factory(
                        pickup_x=move.pickup_xy_mm[0],
                        pickup_y=move.pickup_xy_mm[1],
                        place_x=move.place_xy_mm[0],
                        place_y=move.place_xy_mm[1],
                        rotation_angle_deg=rotation_target,
                        return_rotation_to_zero=False,
                        rotation_home_barrier=barrier,
                        piece_released_callback=(
                            completion_callback
                            if index == len(plan.moves) - 1
                            else None
                        ),
                    )
                    results.append(controller.transfer_piece(request))
                    if self._config.continuous_transfers:
                        # 此刻已释放铁片、Z 已抬到安全高度；F32C 可与下一片
                        # XY 行程并行。Future.result() 在下一片下降前完成回零
                        # 确认；重试后的可读残差会告警并按任务策略继续。
                        pending_home = home_executor.submit(
                            controller.motor.return_to_mechanical_zero,
                            rpm=self._config.rotation_rpm,
                            wait=True,
                            tolerance_deg=F32C_HOME_TOLERANCE_DEG,
                            settle_retries=F32C_HOME_SETTLE_RETRIES,
                            allow_inexact_arrival=True,
                        )
                if pending_home is not None:
                    # 最后一片没有下一次下降屏障，仍需确认回零后才能关闭电机。
                    pending_home.result()
            if self._config.continuous_transfers:
                controller.park_safely()
            return tuple(results)
        finally:
            controller.close()

    def park_safely(
        self,
        *,
        origin_ready_callback: Callable[[bool], None] | None = None,
    ) -> None:
        """复用本次上电坐标，释放机构并返回记录的安全停靠点。"""

        controller = self._controller_factory(self._config)
        try:
            controller.connect(
                self._ports.writer_port,
                writer_baud=115200,
                motor_port=self._ports.motor_port,
                motor_address=self._ports.motor_address,
            )
            controller.prepare_existing_origin()
            if origin_ready_callback is not None:
                origin_ready_callback(True)
            controller.park_safely()
        finally:
            controller.close()

    def initialize_and_park(
        self,
        *,
        origin_ready_callback: Callable[[bool], None] | None = None,
    ) -> None:
        """从人工机械限位角建立本次原点，并前往固定安全停靠点。"""

        controller = self._controller_factory(self._config)
        try:
            controller.connect(
                self._ports.writer_port,
                writer_baud=115200,
                motor_port=self._ports.motor_port,
                motor_address=self._ports.motor_address,
            )
            controller.establish_origin()
            if origin_ready_callback is not None:
                origin_ready_callback(False)
            controller.park_safely()
        finally:
            controller.close()


def project_root() -> Path:
    """返回仓库根目录，供启动脚本定位参考工程。"""

    return Path(__file__).resolve().parents[1]
