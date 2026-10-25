"""执行或预览一份 MovePlan 到电机和电磁铁控制链路。

脚本内置一个符合 ``MovePlan`` 接口的单纸片 stub。默认会真实打开
串口和 GPIO；使用 ``--dry-run`` 时会注入记录型假串口、假磁铁和假
时钟，以相同的装配逻辑输出全部待发送帧，而不访问任何硬件。

独立运行：
``poetry run python tests/external/motion/move_plan_execution.py --dry-run``
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import typer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_26_e.actuation import (
    ActuationConfig,
    GpiodMagnetOutput,
    TransferCycle,
    compile_move_plan,
)
from ec_26_e.actuation.contracts import MagnetOutput
from ec_26_e.config import (
    DEFAULT_TASK1_CONFIG_PATH,
    load_env_file,
    load_task1_config,
)
from ec_26_e.motion import (
    DEFAULT_MOTION_BAUDRATE,
    DEFAULT_MOTION_DEVICE_PATH,
    DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
    DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
    DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
    MotionSerialConfig,
    MotorAddress,
    SerialWriteEvent,
    Servo42CSerial,
)
from ec_26_e.control import MovePlan, PlannedMove
from ec_26_e.tasks.task1.actuation import MovePlanActuator
from tests.external.motion.helpers import (
    SerialWriteEventReporter,
    create_serial_config,
    emit_summary,
)


STUB_PICK_XY_A4_MM = (105.0, 100.0)
STUB_PLACE_XY_A4_MM = (105.0, 197.0)


# 与其他外部 motion 脚本一致：使 CLI 的 envvar 选项也能读取项目 .env。
load_env_file(PROJECT_ROOT / ".env")

app = typer.Typer(
    add_completion=False,
    help="执行内置 MovePlan stub；--dry-run 不会打开串口、GPIO 或电机。",
)


class FakeClock:
    """以 sleep 调用推进的单调时钟，供 dry-run 消除真实等待。"""

    def __init__(self) -> None:
        self.seconds = 0.0

    def monotonic(self) -> float:
        return self.seconds

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep 秒数不能小于 0")
        self.seconds += seconds


class RecordingTransport:
    """只记录生命周期和帧的 NoReplyTransport 实现。"""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.events: list[str] = []

    def open(self) -> None:
        self.events.append("open")

    def send_frame(self, frame: bytes) -> None:
        self.frames.append(frame)

    def discard_replies(self) -> None:
        self.events.append("discard_replies")

    def take_reply(self) -> None:
        return None

    def close(self) -> None:
        self.events.append("close")


class RecordingMagnet:
    """只记录磁铁调用而不申请 GPIO 的 MagnetOutput 实现。"""

    def __init__(self) -> None:
        self.events: list[object] = []

    def open(self) -> None:
        self.events.append("open")

    def set_enabled(self, enabled: bool) -> None:
        self.events.append({"enabled": enabled})

    def close(self) -> None:
        self.events.append("close")


class RecordingRotation:
    """模拟旋转轴角度反馈，供 dry-run 完整覆盖运动生命周期。"""

    def __init__(self) -> None:
        """从机械零点开始，且不申请任何串口资源。"""

        self.angle_degrees = 0.0
        self.events: list[object] = []

    def open(self) -> None:
        """记录模拟旋转轴打开。"""

        self.events.append("open")

    def current_angle_deg(self) -> float:
        """返回当前模拟机械单圈角度。"""

        self.events.append("current_angle")
        return self.angle_degrees

    def move_clockwise_from_zero(
        self,
        target_degrees: float,
        *,
        wait: bool = True,
    ) -> float:
        """记录顺时针目标并更新模拟角度。"""

        self.events.append(
            {"clockwise_degrees": target_degrees, "wait": wait}
        )
        self.angle_degrees = float(target_degrees) % 360.0
        return -float(target_degrees)

    def return_counterclockwise_to_zero(
        self,
        target_degrees: float,
        *,
        wait: bool = True,
    ) -> float:
        """记录回零并把模拟反馈更新为零位。"""

        self.events.append(
            {"counterclockwise_degrees": target_degrees, "wait": wait}
        )
        self.angle_degrees = 0.0
        return 0.0

    def close(self) -> None:
        """记录模拟旋转轴关闭。"""

        self.events.append("close")


class ObservedMagnet:
    """在真实 GPIO 调用外记录磁铁生命周期，供执行摘要使用。"""

    def __init__(self, config: ActuationConfig) -> None:
        self._output = GpiodMagnetOutput(config.magnet)
        self.events: list[object] = []

    def open(self) -> None:
        self._output.open()
        self.events.append("open")

    def set_enabled(self, enabled: bool) -> None:
        self._output.set_enabled(enabled)
        self.events.append({"enabled": enabled})

    def close(self) -> None:
        self._output.close()
        self.events.append("close")


def create_move_plan_stub() -> MovePlan:
    """创建从 A4 上半区中央移动到下半区中央的单周期 MovePlan。"""

    observed = ((100.0, 95.0), (110.0, 95.0), (105.0, 105.0))
    target = ((100.0, 192.0), (110.0, 192.0), (105.0, 202.0))
    move = PlannedMove(
        order=1,
        piece_id="move-plan-stub-piece",
        pick_xy_a4_mm=STUB_PICK_XY_A4_MM,
        pick_clearance_mm=5.0,
        rotation_deg_clockwise=0.0,
        place_xy_a4_mm=STUB_PLACE_XY_A4_MM,
        translation_a4_mm=(0.0, 97.0),
        observed_vertices_a4_mm=observed,
        target_vertices_a4_mm=target,
        template_id="external-motion-stub",
    )
    return MovePlan(
        mode="fixed",
        confidence=1.0,
        ambiguous=False,
        target_width_mm=10.0,
        target_height_mm=10.0,
        target_center_a4_mm=STUB_PLACE_XY_A4_MM,
        moves=(move,),
    )


def _cycle_to_dict(cycle: TransferCycle) -> dict[str, object]:
    """把量化后的周期队列项转换为稳定 JSON。"""

    return {
        "order": cycle.order,
        "piece_id": cycle.piece_id,
        "pick_0p01mm": [cycle.pick_x_0p01mm, cycle.pick_y_0p01mm],
        "place_0p01mm": [cycle.place_x_0p01mm, cycle.place_y_0p01mm],
        "pick_mm": [cycle.pick_x_0p01mm / 100.0, cycle.pick_y_0p01mm / 100.0],
        "place_mm": [cycle.place_x_0p01mm / 100.0, cycle.place_y_0p01mm / 100.0],
    }


def _frame_to_dict(sequence: int, frame: bytes) -> dict[str, object]:
    """为 dry-run 中记录的裸帧补充地址与功能码诊断信息。"""

    try:
        address = MotorAddress(frame[0]).name
    except ValueError:
        address = None
    return {
        "sequence": sequence,
        "address": address,
        "address_hex": f"0x{frame[0]:02X}",
        "function_hex": f"0x{frame[1]:02X}" if len(frame) > 1 else None,
        "frame_hex": frame.hex(" ").upper(),
    }


def _result_summary(
    *,
    config: ActuationConfig,
    plan: MovePlan,
    cycles: tuple[TransferCycle, ...],
    result_state: str,
    completed_cycles: int,
    state_history: tuple[object, ...],
    execute: bool,
    magnet_events: list[object],
    frames: list[bytes] | None = None,
    serial_events: list[SerialWriteEvent] | None = None,
    simulated_elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """汇总相同执行器在真实和预览两条路径上的可比结果。"""

    frame_records = (
        [_frame_to_dict(index, frame) for index, frame in enumerate(frames, 1)]
        if frames is not None
        else [event.to_dict() for event in serial_events or []]
    )
    summary: dict[str, object] = {
        "operation": "move-plan-execution",
        "dry_run": not execute,
        "executed": execute,
        "serial": {
            "device_path": str(config.serial.device_path),
            "baudrate": config.serial.baudrate,
        },
        "move_plan": plan.to_dict(),
        "transfer_cycles": [_cycle_to_dict(cycle) for cycle in cycles],
        "result": {
            "state": result_state,
            "completed_cycles": completed_cycles,
            "state_history": [
                getattr(state, "value", str(state)) for state in state_history
            ],
        },
        "magnet_events": magnet_events,
        "serial_writes": frame_records,
    }
    if simulated_elapsed_seconds is not None:
        summary["simulated_elapsed_seconds"] = simulated_elapsed_seconds
    return summary


def run_move_plan(
    config: ActuationConfig,
    plan: MovePlan,
    *,
    execute: bool = True,
) -> dict[str, object]:
    """执行或预览传入的 MovePlan，且两者使用相同生命周期装配。"""

    if not isinstance(config, ActuationConfig):
        raise TypeError("config 必须是 ActuationConfig")
    if not isinstance(plan, MovePlan):
        raise TypeError("plan 必须是 MovePlan")
    if not isinstance(execute, bool):
        raise TypeError("execute 必须是 bool")

    cycles = compile_move_plan(plan, config)
    if not execute:
        clock = FakeClock()
        transport = RecordingTransport()
        magnet = RecordingMagnet()
        rotation = RecordingRotation()
        actuator = MovePlanActuator(
            config,
            transport_factory=lambda unused: transport,
            magnet_factory=lambda unused: magnet,
            rotation_factory=lambda unused: rotation,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        result = actuator.execute(plan)
        return _result_summary(
            config=config,
            plan=plan,
            cycles=cycles,
            result_state=result.state.value,
            completed_cycles=result.completed_cycles,
            state_history=result.state_history,
            execute=False,
            magnet_events=magnet.events,
            frames=transport.frames,
            simulated_elapsed_seconds=clock.seconds,
        )

    serial_events: list[SerialWriteEvent] = []
    observed_magnet: ObservedMagnet | None = None

    def create_magnet(actuation_config: ActuationConfig) -> MagnetOutput:
        nonlocal observed_magnet
        observed_magnet = ObservedMagnet(actuation_config)
        return observed_magnet

    with SerialWriteEventReporter() as reporter:
        def observe_write(event: SerialWriteEvent) -> None:
            serial_events.append(event)
            reporter.publish(event)

        actuator = MovePlanActuator(
            config,
            transport_factory=lambda actuation_config: Servo42CSerial(
                actuation_config.serial,
                write_observer=observe_write,
            ),
            magnet_factory=create_magnet,
        )
        result = actuator.execute(plan)

    return _result_summary(
        config=config,
        plan=plan,
        cycles=cycles,
        result_state=result.state.value,
        completed_cycles=result.completed_cycles,
        state_history=result.state_history,
        execute=True,
        magnet_events=([] if observed_magnet is None else observed_magnet.events),
        serial_events=serial_events,
    )


@app.command()
def main(
    task_config: Path = typer.Option(
        PROJECT_ROOT / DEFAULT_TASK1_CONFIG_PATH,
        "--task-config",
        help="读取运动限位、标定、Z 时序与磁铁配置的 Task1 JSON。",
    ),
    device_path: Path = typer.Option(
        DEFAULT_MOTION_DEVICE_PATH,
        "--device",
        envvar="MOTION_SERIAL_DEVICE",
        help="Servo42C 串口设备；可由 MOTION_SERIAL_DEVICE 配置。",
    ),
    baudrate: int = typer.Option(
        DEFAULT_MOTION_BAUDRATE,
        "--baudrate",
        envvar="MOTION_SERIAL_BAUDRATE",
        min=1,
        help="串口波特率。",
    ),
    read_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_READ_TIMEOUT_SECONDS,
        "--read-timeout",
        envvar="MOTION_SERIAL_READ_TIMEOUT_SECONDS",
        min=0.001,
        help="单次串口读取超时，单位秒。",
    ),
    write_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_WRITE_TIMEOUT_SECONDS,
        "--write-timeout",
        envvar="MOTION_SERIAL_WRITE_TIMEOUT_SECONDS",
        min=0.001,
        help="单次串口写入超时，单位秒。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_MOTION_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout",
        envvar="MOTION_SERIAL_REPLY_TIMEOUT_SECONDS",
        min=0.001,
        help="串口回包超时，单位秒。",
    ),
    execute: bool = typer.Option(
        True,
        "--execute/--dry-run",
        help="默认真实控制硬件；--dry-run 仅装配和输出报文。",
    ),
) -> None:
    """以内置 MovePlan stub 运行真实执行或无硬件预览。"""

    def callback() -> dict[str, object]:
        task1_config = load_task1_config(task_config, env_path=None)
        serial = create_serial_config(
            device_path,
            baudrate,
            read_timeout_seconds,
            write_timeout_seconds,
            reply_timeout_seconds,
        )
        config = replace(task1_config.actuation, serial=serial)
        return run_move_plan(
            config,
            create_move_plan_stub(),
            execute=execute,
        )

    try:
        emit_summary(callback())
    except KeyboardInterrupt:
        typer.echo("MovePlan 执行已由用户中断", err=True)
        raise typer.Exit(code=130) from None
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        typer.echo(f"MovePlan 执行失败：{error}", err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
