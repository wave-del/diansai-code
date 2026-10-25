"""拼图装置联动控制接口。

本模块只编排已经验证过的三个底层控制器：

* ``WriterController``：XY 搬运与 Z 轴升降；
* ``F32CMotor``：拼图件绝对角度；
* ``ElectromagnetController``：高电平吸合、低电平释放。

坐标全部使用毫米和角度，不在联动层引入厘米或像素坐标。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Callable

try:
    from .electromagnet import ElectromagnetController
    from .f32c_controller import F32CMotor
    from .writer_controller import WriterController
except ImportError:
    from electromagnet import ElectromagnetController
    from f32c_controller import F32CMotor
    from writer_controller import WriterController


StatusCallback = Callable[[str, str], None]
MAX_VALIDATED_Z_COMMAND_MM = 15.0
# F32C 角度反馈存在约 1°量化/回读偏差。取放姿态以 ±1.5°判定精确
# 到位；未达标时由底层对同一目标重发两次。仍有有效反馈的残差只告警，
# 任务按现场要求继续完成放件，绝不因该软阈值中断在半途。
F32C_TARGET_TOLERANCE_DEG = 1.5
F32C_TARGET_SETTLE_RETRIES = 2


class PuzzleError(RuntimeError):
    """拼图联动控制的基础异常。"""


class PuzzleStateError(PuzzleError):
    """调用顺序或控制器状态不正确。"""


class PuzzleTransferError(PuzzleError):
    """搬运流程中途失败，并包含失败阶段和恢复结果。"""

    def __init__(
        self,
        stage: str,
        cause: BaseException,
        recovery_errors: tuple[str, ...] = (),
    ):
        message = f"拼图搬运在“{stage}”阶段失败：{cause}"
        if recovery_errors:
            message += "；安全恢复异常：" + "；".join(recovery_errors)
        super().__init__(message)
        self.stage = stage
        self.cause = cause
        self.recovery_errors = recovery_errors


@dataclass(frozen=True)
class PuzzleConfig:
    """一次联动进程使用的机械参数。

    所有长度单位均为 mm，速度单位为 mm/min，时间单位为秒，电机速度
    单位为 RPM。
    """

    # Z0 是自然下垂的下方机械限位；负方向抬高。
    safe_z: float = -10.0
    pickup_z: float = -2.0
    place_z: float = -2.0
    rest_z: float = -10.0
    # False 时每片成功后回安全停靠点；True 时留在放件点的安全 Z，
    # 由下一片直接前往取件点。无论该开关为何，异常恢复均回安全停靠点。
    continuous_transfers: bool = False
    # 蓝纸业务坐标；任务结束和异常恢复的视觉安全停靠点。
    safe_parking_x: float = 0.0
    safe_parking_y: float = -1.0
    # 正式拼图 XY 进给速度：4500 mm/min（75 mm/s）。
    xy_feed: float = 4500.0
    z_feed: float = 1500.0
    magnet_pickup_dwell: float = 0.5
    magnet_release_dwell: float = 0.5
    rotation_rpm: int = 29
    workspace_x_max: float | None = None
    workspace_y_max: float | None = None

    def __post_init__(self) -> None:
        finite_values = {
            "safe_z": self.safe_z,
            "pickup_z": self.pickup_z,
            "place_z": self.place_z,
            "rest_z": self.rest_z,
            "xy_feed": self.xy_feed,
            "z_feed": self.z_feed,
            "magnet_pickup_dwell": self.magnet_pickup_dwell,
            "magnet_release_dwell": self.magnet_release_dwell,
            "safe_parking_x": self.safe_parking_x,
            "safe_parking_y": self.safe_parking_y,
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数字")
        if self.pickup_z <= self.safe_z:
            raise ValueError("pickup_z 必须大于 safe_z（本机 Z+ 向下）")
        if self.place_z <= self.safe_z:
            raise ValueError("place_z 必须大于 safe_z（本机 Z+ 向下）")
        for name, value in (
            ("safe_z", self.safe_z),
            ("pickup_z", self.pickup_z),
            ("place_z", self.place_z),
            ("rest_z", self.rest_z),
        ):
            if not -MAX_VALIDATED_Z_COMMAND_MM <= value <= 0.0:
                raise ValueError(
                    f"{name} 必须位于本机已验证的 "
                    f"[-{MAX_VALIDATED_Z_COMMAND_MM:g}, 0] mm"
                )
        if self.rest_z < self.safe_z:
            raise ValueError("rest_z 不得高于 safe_z（本机 Z+ 向下）")
        commanded_lifts = (
            self.pickup_z - self.safe_z,
            self.place_z - self.safe_z,
            self.rest_z - self.safe_z,
        )
        if max(commanded_lifts) > MAX_VALIDATED_Z_COMMAND_MM:
            raise ValueError(
                "Z 命令超出本机已验证范围；当前跨度不得超过 "
                f"{MAX_VALIDATED_Z_COMMAND_MM:g}mm"
            )
        if self.xy_feed <= 0 or self.z_feed <= 0:
            raise ValueError("XY 和 Z 速度必须大于 0")
        if self.magnet_pickup_dwell < 0 or self.magnet_release_dwell < 0:
            raise ValueError("电磁铁等待时间不能小于 0")
        if not isinstance(self.continuous_transfers, bool):
            raise ValueError("continuous_transfers 必须是布尔值")
        if (
            int(self.rotation_rpm) != self.rotation_rpm
            or not 1 <= int(self.rotation_rpm) <= 30
        ):
            raise ValueError("rotation_rpm 必须是 1～30 的整数")
        for name, value in (
            ("workspace_x_max", self.workspace_x_max),
            ("workspace_y_max", self.workspace_y_max),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0
            ):
                raise ValueError(f"{name} 省略或设置为大于 0 的有限毫米值")


@dataclass(frozen=True)
class TransferRequest:
    """一个拼图件的取放任务。

    ``rotation_angle_deg=None`` 表示保持当前无刷电机角度；给出数值时，
    抓起并抬高后转到驱动板机械零点坐标中的绝对角度。
    """

    pickup_x: float
    pickup_y: float
    place_x: float
    place_y: float
    rotation_angle_deg: float | None = None
    return_rotation_to_zero: bool = False
    # 可选的同步栅栏：下一片 XY 已到取件点后、Z 下降前调用。连续搬运时
    # 它代表 F32C 回零任务已由角度反馈确认完成。
    rotation_home_barrier: Callable[[], None] | None = None
    # 仅最后一片使用：电磁铁已经释放且完成释放驻留后调用。该通知不参与
    # 运动控制，用于让上位机在回安全停靠点之前发出比赛完成声光信号。
    piece_released_callback: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("pickup_x", self.pickup_x),
            ("pickup_y", self.pickup_y),
            ("place_x", self.place_x),
            ("place_y", self.place_y),
        ):
            coordinate = float(value)
            if not math.isfinite(coordinate):
                raise ValueError(f"{name} 必须是有限毫米值")
            bounded = max(0.0, coordinate)
            if bounded != coordinate:
                object.__setattr__(self, name, bounded)
                print(
                    f"⚠ 坐标限幅：{name}={coordinate:.3f} mm → 0.000 mm；继续执行。",
                    flush=True,
                )
        angle = self.rotation_angle_deg
        if angle is not None and (
            not math.isfinite(float(angle)) or not 0 <= float(angle) < 360
        ):
            raise ValueError("rotation_angle_deg 必须省略或处于 [0, 360)")
        if not isinstance(self.return_rotation_to_zero, bool):
            raise ValueError("return_rotation_to_zero 必须是布尔值")
        if (
            self.rotation_home_barrier is not None
            and not callable(self.rotation_home_barrier)
        ):
            raise ValueError("rotation_home_barrier 必须是可调用对象或 None")
        if (
            self.piece_released_callback is not None
            and not callable(self.piece_released_callback)
        ):
            raise ValueError("piece_released_callback 必须是可调用对象或 None")


@dataclass(frozen=True)
class TransferResult:
    """成功搬运后的状态快照。"""

    final_x: float
    final_y: float
    final_z: float
    magnet_engaged: bool
    requested_rotation_deg: float | None
    rotation_returned_to_zero: bool


class PuzzleController:
    """把写字机、无刷电机和电磁铁组合成同步取放接口。

    同一个实例只应由同一线程顺序调用。对象不会执行限位回零；外部机构
    到达右下角和安全 Z 基准后，必须调用 :meth:`establish_origin`。
    """

    def __init__(
        self,
        config: PuzzleConfig | None = None,
        *,
        writer: WriterController | None = None,
        motor: F32CMotor | None = None,
        magnet: ElectromagnetController | None = None,
        status_callback: StatusCallback | None = None,
    ):
        self.config = config or PuzzleConfig()
        self.writer = writer or WriterController()
        self.motor = motor
        self.magnet = magnet or ElectromagnetController()
        self._status_callback = status_callback
        self._writer_connected = False
        self._motor_connected = False
        self._magnet_open = False
        self._origin_ready = False
        self._stage = "尚未连接"

    @property
    def is_ready(self) -> bool:
        """三个硬件已连接且本次软件原点已建立时返回 ``True``。"""
        return (
            self._writer_connected
            and self._motor_connected
            and self._magnet_open
            and self._origin_ready
        )

    @property
    def stage(self) -> str:
        """当前或最近完成的流程阶段。"""
        return self._stage

    def connect(
        self,
        writer_port: str,
        *,
        writer_baud: int = 115200,
        motor_port: str = "",
        motor_address: int = 2,
    ) -> None:
        """连接三套硬件，并自动使能无刷电机。

        电磁铁最先打开并原子初始化为低电平。连接失败会释放电磁铁、
        失能无刷电机并关闭已经打开的设备。
        """
        if self._writer_connected or self._motor_connected or self._magnet_open:
            raise PuzzleStateError("控制器已经连接，不能重复 connect()")

        # Validate the F32C deployment before opening any actuator.
        if self.motor is None:
            self.motor = F32CMotor(port=motor_port, address=motor_address)
        self._set_stage("初始化电磁铁", "Q6A PIN_13 输出低电平（松开）")
        try:
            self.magnet.open()
            self._magnet_open = True

            self._set_stage("连接写字机", f"打开 {writer_port}")
            self.writer.connect(port=writer_port, baud=writer_baud)
            self._writer_connected = True
            # 校准流程会临时把$1设为25释放步进电机。任何正式联动
            # 一旦连接成功，先恢复全轴持续保持，避免中断遗留状态。
            self.writer.set_motor_hold(True)

            self._set_stage("连接无刷电机", f"打开 {motor_port} 并安全使能")
            self.motor.connect(verify=False)
            self._motor_connected = True
            self.motor.enable(rpm=self.config.rotation_rpm)
        except BaseException:
            try:
                self.close()
            except BaseException:
                # 连接阶段保留最初的硬件异常；close() 已经尽力逐项清理。
                pass
            raise

    def establish_origin(self) -> None:
        """在机构已经位于右下角和安全 Z 基准时建立 X0/Y0/Z0。

        此方法调用写字机 ``home()``，实际发送 ``G92``；它不会主动寻找
        限位，也不会把尚未归位的机构移动到右下角。按本装置约定，调用
        前 Z 应自然下垂到下方机械限位，当前位置由此定义为 Z0；随后本
        方法会把 Z 抬到配置的 ``safe_z``。
        """
        self._require_connected()
        self._set_stage("建立软件原点", "当前位置定义为 X0/Y0/Z0")
        self.writer.home()
        self.writer.configure_pen(
            up_z=self.config.safe_z,
            down_z=self.config.pickup_z,
            speed=self.config.z_feed,
        )
        self.writer.pen_up()
        self._origin_ready = True
        self._set_stage("准备完成", "三套硬件可执行取放任务")

    def prepare_existing_origin(self) -> None:
        """从安全停靠点恢复本次上电先前建立的坐标。

        每次重新打开本机写字机串口都会使 ESP32 GRBL 经 DTR 重启，原有
        G92 工作坐标会丢失。因此，不能简单“不发 G92”：必须把当前已知的
        安全停靠点重新锚定为其业务坐标，才能与首次机械原点的绝对坐标保持
        一致。调用前机构必须确实停在安全停靠点、Z 在 ``safe_z``，且未被
        人工移动；否则应重新让机构置于机械限位角并建立原点。
        """
        self._require_connected()
        self._set_stage("重锚安全停靠点", "串口重连后恢复本次上电坐标")
        self.writer.reanchor_at_a4_position(
            self.config.safe_parking_x,
            self.config.safe_parking_y,
            self.config.safe_z,
        )
        self.writer.configure_pen(
            up_z=self.config.safe_z,
            down_z=self.config.pickup_z,
            speed=self.config.z_feed,
        )
        self.writer.pen_up()
        self._origin_ready = True
        self._set_stage("准备完成", "复用本次上电坐标，三套硬件可执行任务")

    def transfer_piece(self, request: TransferRequest) -> TransferResult:
        """执行一次完整取件、搬运、放件流程。

        默认正常返回时电磁铁已松开、XY 位于 ``(0, 0)``，Z 位于
        ``rest_z``。连续搬运模式会保留在放件点的安全 Z，供下一片直接
        前往取件点。发生异常或 ``Ctrl+C`` 时，始终尝试抬高 Z、释放
        电磁铁和 XY 回原点；底层原始异常不会被静默忽略。
        """
        self._require_ready()
        request = self._validate_request(request)

        try:
            self._set_stage("安全抬高", f"Z → {self.config.safe_z:.3f} mm")
            self.writer.pen_up()

            self._set_stage(
                "移动到取件点",
                self._format_xy(request.pickup_x, request.pickup_y),
            )
            self.writer.move_to(
                request.pickup_x,
                request.pickup_y,
                feedrate=self.config.xy_feed,
            )

            if request.rotation_home_barrier is not None:
                self._set_stage(
                    "等待 F32C 回零确认",
                    "F32C 优先精确回零至 0°±1.5°；残差会告警并继续取件",
                )
                request.rotation_home_barrier()

            self._set_stage(
                "下降取件",
                f"Z → {self.config.pickup_z:.3f} mm",
            )
            self.writer.pen_move_to(
                self.config.pickup_z,
                feedrate=self.config.z_feed,
            )

            self._set_stage("吸合电磁铁", "PL6 输出高电平")
            self.magnet.engage()
            self._sleep(self.config.magnet_pickup_dwell)

            self._set_stage("携件抬高", f"Z → {self.config.safe_z:.3f} mm")
            self.writer.pen_up()

            if request.rotation_angle_deg is not None:
                self._set_stage(
                    "旋转拼图件",
                    f"机械绝对角度 → {request.rotation_angle_deg:.1f}°",
                )
                assert self.motor is not None
                self.motor.move_to(
                    request.rotation_angle_deg,
                    rpm=self.config.rotation_rpm,
                    wait=True,
                    tolerance_deg=F32C_TARGET_TOLERANCE_DEG,
                    settle_retries=F32C_TARGET_SETTLE_RETRIES,
                    allow_inexact_arrival=True,
                )

            self._set_stage(
                "移动到放件点",
                self._format_xy(request.place_x, request.place_y),
            )
            self.writer.move_to(
                request.place_x,
                request.place_y,
                feedrate=self.config.xy_feed,
            )

            self._set_stage(
                "下降放件",
                f"Z → {self.config.place_z:.3f} mm",
            )
            self.writer.pen_move_to(
                self.config.place_z,
                feedrate=self.config.z_feed,
            )

            self._set_stage("释放电磁铁", "PL6 输出低电平")
            self.magnet.release()
            self._sleep(self.config.magnet_release_dwell)
            if request.piece_released_callback is not None:
                try:
                    request.piece_released_callback()
                except Exception as error:
                    # 声光提示失败不能改变已经完成的放件动作，更不能让铁片
                    # 停在纸面上方。通知调用方记录该异常后继续安全抬高/停靠。
                    print(f"⚠ 最后一片完成通知失败：{error}", flush=True)

            self._set_stage("放件后抬高", f"Z → {self.config.safe_z:.3f} mm")
            self.writer.pen_up()

            if not self.config.continuous_transfers:
                self._set_stage(
                    "XY 回安全停靠点",
                    self._format_xy(
                        self.config.safe_parking_x,
                        self.config.safe_parking_y,
                    ),
                )
                self.writer.move_to(
                    self.config.safe_parking_x,
                    self.config.safe_parking_y,
                    feedrate=self.config.xy_feed,
                )

            if request.return_rotation_to_zero:
                self._set_stage("无刷电机回零", "机械绝对角度 → 0.0°")
                assert self.motor is not None
                self.motor.return_to_mechanical_zero(
                    rpm=self.config.rotation_rpm,
                    wait=True,
                    tolerance_deg=F32C_TARGET_TOLERANCE_DEG,
                    settle_retries=F32C_TARGET_SETTLE_RETRIES,
                    allow_inexact_arrival=True,
                )

            if not self.config.continuous_transfers:
                self._set_stage(
                    "Z 回初始落下状态",
                    f"Z → {self.config.rest_z:.3f} mm",
                )
                self.writer.pen_move_to(
                    self.config.rest_z,
                    feedrate=self.config.z_feed,
                )
            # $1 是全轴步进空闲保持时间。再次显式写入，确保最终 Z
            # 动作完成后不会因旧设置而自动失能。
            self._set_stage("保持步进使能", "$1 → 255")
            self.writer.send_gcode("$1=255")

            final_x, final_y = self.writer.position
            final_z = self.writer.pen_position
            magnet_engaged = self.magnet.is_engaged
            self._set_stage(
                "搬运完成",
                (
                    "拼图件已放下，XY 已回安全停靠点，Z 已恢复落下状态"
                    if not self.config.continuous_transfers
                    else "拼图件已放下，XY 保持放件点，Z 保持安全抬高"
                ),
            )
            return TransferResult(
                final_x=final_x,
                final_y=final_y,
                final_z=final_z,
                magnet_engaged=magnet_engaged,
                requested_rotation_deg=request.rotation_angle_deg,
                rotation_returned_to_zero=request.return_rotation_to_zero,
            )
        except BaseException as exc:
            failed_stage = self._stage
            recovery_errors = self._recover_after_failure()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise PuzzleTransferError(
                failed_stage,
                exc,
                recovery_errors,
            ) from exc

    def park_safely(self) -> None:
        """在连续搬运任务结束后，释放并停到记录的视觉安全停靠点。"""

        self._require_ready()
        self._set_stage("任务结束安全抬高", f"Z → {self.config.safe_z:.3f} mm")
        self.writer.pen_up()
        self._set_stage("任务结束释放电磁铁", "PL6 输出低电平")
        self.magnet.release()
        self._set_stage(
            "任务结束回安全停靠点",
            self._format_xy(
                self.config.safe_parking_x, self.config.safe_parking_y
            ),
        )
        self.writer.move_to(
            self.config.safe_parking_x,
            self.config.safe_parking_y,
            feedrate=self.config.xy_feed,
        )

    def close(self) -> None:
        """尽力释放电磁铁、失能无刷电机并关闭所有设备。

        本方法不主动移动 XY/Z。正常业务应先让 ``transfer_piece()`` 完成；
        它会安全抬高后回到安全停靠点，最后把 Z 恢复到 ``rest_z``。写字机
        已设置 ``$1=255``，关闭串口后步进电机仍保持使能；无刷电机在
        关闭前会失能。
        """
        errors: list[str] = []
        if self._magnet_open:
            try:
                self.magnet.close(release=True)
            except BaseException as exc:
                errors.append(f"关闭电磁铁失败：{exc}")
            finally:
                self._magnet_open = False

        if self._motor_connected and self.motor is not None:
            try:
                self.motor.close(disable=True)
            except BaseException as exc:
                errors.append(f"关闭无刷电机失败：{exc}")
            finally:
                self._motor_connected = False

        if self._writer_connected:
            try:
                self.writer.disconnect()
            except BaseException as exc:
                errors.append(f"关闭写字机串口失败：{exc}")
            finally:
                self._writer_connected = False

        self._origin_ready = False
        self._stage = "已关闭"
        if errors:
            raise PuzzleError("；".join(errors))

    def _recover_after_failure(self) -> tuple[str, ...]:
        errors: list[str] = []
        z_is_safe = False

        if self._writer_connected and self._origin_ready:
            try:
                self.writer.pen_up()
                z_is_safe = True
            except BaseException as exc:
                errors.append(f"抬高 Z 失败：{exc}")

        if self._magnet_open:
            try:
                self.magnet.release()
            except BaseException as exc:
                errors.append(f"释放电磁铁失败：{exc}")

        if self._writer_connected and self._origin_ready and z_is_safe:
            try:
                self.writer.move_to(
                    self.config.safe_parking_x,
                    self.config.safe_parking_y,
                    feedrate=self.config.xy_feed,
                )
            except BaseException as exc:
                errors.append(f"XY 回安全停靠点失败：{exc}")

        return tuple(errors)

    def _validate_request(self, request: TransferRequest) -> TransferRequest:
        """对有限取放坐标限幅到本机边界，保留不可恢复输入的错误。"""
        coordinates = {
            "pickup_x": request.pickup_x,
            "pickup_y": request.pickup_y,
            "place_x": request.place_x,
            "place_y": request.place_y,
        }
        for name, value in coordinates.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} 必须是大于等于 0 的有限毫米值")

        bounded = {
            "pickup_x": self._check_workspace(
                "pickup_x", request.pickup_x, self.config.workspace_x_max
            ),
            "place_x": self._check_workspace(
                "place_x", request.place_x, self.config.workspace_x_max
            ),
            "pickup_y": self._check_workspace(
                "pickup_y", request.pickup_y, self.config.workspace_y_max
            ),
            "place_y": self._check_workspace(
                "place_y", request.place_y, self.config.workspace_y_max
            ),
        }

        angle = request.rotation_angle_deg
        if angle is not None and (
            not math.isfinite(float(angle)) or not 0 <= float(angle) < 360
        ):
            raise ValueError("rotation_angle_deg 必须省略或处于 [0, 360)")
        if not isinstance(request.return_rotation_to_zero, bool):
            raise ValueError("return_rotation_to_zero 必须是布尔值")
        if any(float(getattr(request, name)) != value for name, value in bounded.items()):
            return replace(request, **bounded)
        return request

    @staticmethod
    def _check_workspace(
        name: str,
        value: float,
        maximum: float | None,
    ) -> float:
        coordinate = float(value)
        if maximum is not None and coordinate > float(maximum):
            print(
                f"⚠ 坐标限幅：{name}={coordinate:.3f} mm → "
                f"{float(maximum):.3f} mm（本机边界）；继续执行。",
                flush=True,
            )
            return float(maximum)
        return coordinate

    def _require_connected(self) -> None:
        if not (
            self._writer_connected
            and self._motor_connected
            and self._magnet_open
        ):
            raise PuzzleStateError("三套硬件尚未全部连接")

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise PuzzleStateError(
                "控制器未准备好；请先 connect()，确认机构在右下角后再 "
                "establish_origin()"
            )

    def _set_stage(self, stage: str, detail: str) -> None:
        self._stage = stage
        if self._status_callback is not None:
            self._status_callback(stage, detail)

    @staticmethod
    def _format_xy(x: float, y: float) -> str:
        return f"X → {float(x):.3f} mm，Y → {float(y):.3f} mm"

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def __enter__(self) -> "PuzzleController":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exc_type is None:
                raise
            self._set_stage("关闭设备异常", str(close_error))


__all__ = [
    "PuzzleConfig",
    "PuzzleController",
    "PuzzleError",
    "PuzzleStateError",
    "PuzzleTransferError",
    "TransferRequest",
    "TransferResult",
]
