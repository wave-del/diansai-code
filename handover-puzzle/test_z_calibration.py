#!/usr/bin/env python3
"""Z 轴速度对照、距离线性和保持使能诊断。

本脚本不移动 XY、不控制无刷电机和电磁铁，也不会自动修改 $102。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import math
import os
import queue
import signal
import statistics
import sys
import threading
import time

try:
    from .writer_controller import WriterController
except ImportError:
    from writer_controller import WriterController


PREFERRED_WRITER_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
)
DEFAULT_STEPS_PER_MM = 50.0
MAX_TEST_DISTANCE_MM = 5.0


class CalibrationAbort(BaseException):
    """操作者中断诊断流程。"""


_INPUT_QUEUE: queue.Queue[str | None] = queue.Queue()
_INPUT_THREAD_STARTED = False


def _input_worker() -> None:
    while True:
        line = sys.stdin.readline()
        if line == "":
            _INPUT_QUEUE.put(None)
            return
        _INPUT_QUEUE.put(line.rstrip("\r\n"))


def _prompt(text: str) -> str:
    """后台线程读取stdin，主线程保持可处理中断和实时暂停。"""
    global _INPUT_THREAD_STARTED
    if not _INPUT_THREAD_STARTED:
        threading.Thread(
            target=_input_worker,
            name="z-calibration-input",
            daemon=True,
        ).start()
        _INPUT_THREAD_STARTED = True

    print(text, end="", flush=True)
    while True:
        try:
            line = _INPUT_QUEUE.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            raise CalibrationAbort("终端输入已关闭")
        return line


@dataclass(frozen=True)
class TrialResult:
    label: str
    commanded_mm: float
    speed_mm_min: float
    measured_mm: float | None

    @property
    def ratio(self) -> float | None:
        if self.measured_mm is None:
            return None
        return self.measured_mm / self.commanded_mm

    def suggested_steps_per_mm(
        self,
        current_steps_per_mm: float,
    ) -> float | None:
        if self.measured_mm is None or self.measured_mm <= 0:
            return None
        return (
            current_steps_per_mm
            * self.commanded_mm
            / self.measured_mm
        )


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return value


def _resolve_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.exists(PREFERRED_WRITER_PORT):
        return PREFERRED_WRITER_PORT
    candidates = sorted(glob.glob("/dev/serial/by-id/*")) + sorted(
        glob.glob("/dev/ttyUSB*")
    )
    if not candidates:
        raise FileNotFoundError(
            "未发现写字机串口；请接好设备或用 --port 指定"
        )
    return candidates[0]


def _parse_distances(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "距离应是逗号分隔的数字，例如 10,20,30,40"
        ) from exc
    if not values or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise argparse.ArgumentTypeError("每个距离都必须大于 0")
    if any(value > MAX_TEST_DISTANCE_MM for value in values):
        raise argparse.ArgumentTypeError(
            "高速20mm已撞上限；本脚本暂时禁止测试超过5mm"
        )
    return values


def _read_setting(
    controller: WriterController,
    name: str,
    fallback: float,
) -> float:
    controller.send_gcode("$$")
    prefix = f"{name}="
    for line in reversed(controller._cnc.recent_responses):
        if line.startswith(prefix):
            try:
                return float(line[len(prefix):])
            except ValueError:
                break
    return fallback


def _set_z_zero_and_hold(controller: WriterController) -> None:
    """当前位置必须已经是自然下垂机械零点。"""
    controller.send_gcode("$X")
    controller.send_gcode("$1=255")
    controller.send_gcode("G92 Z0")


def _install_signal_handlers() -> None:
    """Qt对象创建后重新安装，避免QCoreApplication覆盖SIGINT。"""
    def request_abort(signum, frame) -> None:
        raise CalibrationAbort(f"收到信号 {signum}")

    signal.signal(signal.SIGINT, request_abort)
    signal.signal(signal.SIGTERM, request_abort)
    # 同时关闭SA_RESTART；主线程还会通过_prompt()轮询输入队列，
    # 不再直接阻塞于终端readline。
    signal.siginterrupt(signal.SIGINT, True)
    signal.siginterrupt(signal.SIGTERM, True)


def _emergency_feed_hold(controller: WriterController) -> None:
    """只发GRBL实时暂停字符，不等待ok/Idle，避免清理路径再次阻塞。"""
    if not controller.is_connected():
        return
    try:
        controller._cnc._write_bytes(b"!")
        controller._cnc._serial.flush()
        controller._cnc._serial.waitForBytesWritten(100)
    except BaseException:
        pass


def _prime_z_hold(
    controller: WriterController,
    speed: float,
) -> None:
    """用可见度很低的安全动作明确激活 Z 驱动，最后回到 Z0。"""
    # 现场实测：300mm/min 时 Z 轴可能完全不动，因此该速度不能用于
    # “驱动已经被激活”的判定。使用本机已验证能稳定运动的 1500mm/min；
    # 如调用方传入更低的速度，仍以 1500 执行这个仅 0.2mm 的安全微动。
    prime_speed = max(float(speed), 1500.0)
    controller.pen_move_to(-0.2, feedrate=prime_speed)
    controller.pen_move_to(0.0, feedrate=prime_speed)
    controller.send_gcode("$1=255")


def _ask_hold_state(prompt: str) -> str:
    while True:
        answer = _prompt(prompt).strip().lower()
        if answer in ("y", "yes", "是"):
            return "是"
        if answer in ("n", "no", "否"):
            return "否"
        print("请输入 y 或 n。")


def _ask_measurement() -> float | None:
    while True:
        raw = _prompt(
            "请用尺子测量真实抬高距离，输入 mm；直接按 Enter 表示跳过: "
        ).strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("请输入数字，例如 12.5。")
            continue
        if not math.isfinite(value) or value < 0:
            print("实测距离必须是大于等于 0 的有限数字。")
            continue
        return value


def _release_and_rezero(
    controller: WriterController,
    label: str,
) -> None:
    """临时释放步进电机，让 Z 依靠重力回到可重复机械零点。"""
    print()
    print(f"准备 {label}：临时释放所有步进电机。")
    print("⚠ 此时不要触碰 X/Y；只让 Z 自然下垂到下方机械限位。")
    # 本机WriterController已有实测结论：$1=25运动后可手动推动；
    # 不再使用行为未验证且现场未释放成功的$1=0。
    try:
        controller.set_motor_hold(False)
        time.sleep(0.6)
        answer = _prompt(
            "必须看到 Z 已自然落到底。确认后输入 DOWN；"
            "其他输入立即取消: "
        )
        if answer.strip().upper() != "DOWN":
            raise CalibrationAbort("未确认真实下方机械零点")
    finally:
        # $1会写入EEPROM。无论确认、取消、Ctrl+C还是SIGTERM，
        # 都不能把临时的$1=25遗留到下次上电。
        controller.set_motor_hold(True)
    _set_z_zero_and_hold(controller)


def _run_trial(
    controller: WriterController,
    *,
    label: str,
    distance_mm: float,
    speed_mm_min: float,
) -> TrialResult:
    _release_and_rezero(controller, label)
    answer = _prompt(
        f"即将执行单条命令 Z=-{distance_mm:g}mm @ "
        f"{speed_mm_min:g}mm/min；输入 MOVE 执行，其他输入取消: "
    )
    if answer.strip().upper() != "MOVE":
        raise CalibrationAbort("操作者取消本次运动")
    print(
        f"▶ {label}: Z 0 → {-distance_mm:.3f}mm，"
        f"速度={speed_mm_min:.0f}mm/min"
    )
    controller.pen_move_to(-distance_mm, feedrate=speed_mm_min)
    reported = controller.pen_position
    print(
        f"固件开环坐标={reported:.3f}mm；"
        "该数值不是尺子或编码器的真实反馈。"
    )
    measured = _ask_measurement()
    return TrialResult(
        label=label,
        commanded_mm=distance_mm,
        speed_mm_min=speed_mm_min,
        measured_mm=measured,
    )


def _print_results(
    results: list[TrialResult],
    current_steps_per_mm: float,
) -> None:
    print()
    print("=" * 72)
    print("Z 轴测试结果")
    print("=" * 72)
    print(
        "测试                         命令/mm  速度/mm/min  "
        "实测/mm  实测÷命令  推算$102"
    )
    for result in results:
        measured = (
            "跳过"
            if result.measured_mm is None
            else f"{result.measured_mm:.2f}"
        )
        ratio = (
            "-"
            if result.ratio is None
            else f"{result.ratio:.3f}"
        )
        suggestion = result.suggested_steps_per_mm(
            current_steps_per_mm
        )
        suggested_text = (
            "-" if suggestion is None else f"{suggestion:.3f}"
        )
        print(
            f"{result.label:<28} "
            f"{result.commanded_mm:>7.1f} "
            f"{result.speed_mm_min:>12.0f} "
            f"{measured:>8} {ratio:>9} {suggested_text:>10}"
        )

    measured_results = [
        result for result in results if result.measured_mm is not None
    ]
    if len(measured_results) < 2:
        print("\n测量数据不足，无法判断速度影响或距离线性。")
        return

    low, high = results[0], results[1]
    if low.measured_mm is not None and high.measured_mm is not None:
        reference = max(low.measured_mm, high.measured_mm, 0.1)
        relative_difference = (
            abs(high.measured_mm - low.measured_mm) / reference
        )
        print()
        if relative_difference <= 0.10:
            print(
                "速度对照：同一距离的实测差异不超过 10%，"
                "目前不支持“低速明显丢步”的判断。"
            )
        elif high.measured_mm > low.measured_mm:
            print(
                "速度对照：高速实测明显大于低速，可能存在低速共振、"
                "静摩擦或驱动状态差异，但仍需重复测试确认。"
            )
        else:
            print(
                "速度对照：低速实测不小于高速，更符合步进电机高速"
                "扭矩下降或高速丢步的常见表现。"
            )

    ratios = [
        result.ratio
        for result in measured_results
        if result.ratio is not None and result.measured_mm > 0
    ]
    if len(ratios) >= 3:
        mean_ratio = statistics.fmean(ratios)
        spread = (
            max(ratios) - min(ratios)
        ) / max(abs(mean_ratio), 0.001)
        if spread <= 0.15:
            suggested_values = [
                result.suggested_steps_per_mm(current_steps_per_mm)
                for result in measured_results
            ]
            valid_suggestions = [
                value for value in suggested_values if value is not None
            ]
            print(
                "距离线性：各测试比例离散不超过 15%，可以考虑用"
                " steps/mm 做统一标定。"
            )
            if valid_suggestions:
                print(
                    "多组测量推算的 $102 中位数为 "
                    f"{statistics.median(valid_suggestions):.3f}；"
                    "本脚本没有写入该值。"
                )
        else:
            print(
                "距离线性：实测/命令比例变化较大，更像空行程、松弛、"
                "连杆非线性或间歇丢步，不建议直接改 $102。"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", help="写字机 USB 串口")
    parser.add_argument(
        "--compare-distance",
        type=_positive_float,
        default=2.0,
        help="低速/高速对照使用的相同命令距离，单位 mm",
    )
    parser.add_argument(
        "--low-speed",
        type=_positive_float,
        default=300.0,
        help="低速对照，单位 mm/min",
    )
    parser.add_argument(
        "--high-speed",
        type=_positive_float,
        default=1500.0,
        help="高速对照，单位 mm/min",
    )
    parser.add_argument(
        "--linear-distances",
        type=_parse_distances,
        default=(1.0, 2.0, 3.0, 5.0),
        help="高速线性测试距离，逗号分隔，单位 mm，最大 5",
    )
    parser.add_argument(
        "--skip-linearity",
        action="store_true",
        help="只做同距离速度对照，不做 1/2/3/5mm 线性测试",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    args = _build_parser().parse_args(argv)
    if args.compare_distance > MAX_TEST_DISTANCE_MM:
        print("⛔ 高速20mm已撞上限；对照距离暂时不得超过5mm。")
        return 2

    controller = WriterController()
    connected = False
    hold_before_close = "未检查"
    try:
        port = _resolve_port(args.port)
        print("=" * 72)
        print("Z 轴速度 / 距离 / 保持使能诊断")
        print("=" * 72)
        print(f"串口：{port} @ 115200")
        print(
            f"速度对照：{args.compare_distance:g}mm @ "
            f"{args.low_speed:g} 与 {args.high_speed:g}mm/min"
        )
        if not args.skip_linearity:
            distances = ", ".join(
                f"{distance:g}" for distance in args.linear_distances
            )
            print(
                f"线性测试：{distances}mm @ "
                f"{args.high_speed:g}mm/min"
            )
        print("已知禁区：高速20mm和60mm均会撞上限；本脚本禁止超过5mm。")
        print()
        print("每轮都会临时释放所有步进电机以重建自然下垂 Z0。")
        print("释放期间不要触碰 X/Y，也不要把手放在 Z 机构运动范围内。")
        answer = _prompt("确认后输入 RUN 开始（其他输入取消）: ")
        if answer.strip().upper() != "RUN":
            print("已取消，未连接硬件。")
            return 0

        controller.connect(port)
        connected = True
        # QCoreApplication在connect()中创建；必须在其后再次安装。
        _install_signal_handlers()
        current_steps = _read_setting(
            controller,
            "$102",
            DEFAULT_STEPS_PER_MM,
        )
        current_idle_lock = _read_setting(controller, "$1", 255.0)
        print(
            f"读取到 $102={current_steps:.3f} steps/mm，"
            f"$1={current_idle_lock:g}。"
        )
        if current_idle_lock != 255:
            print("脚本将在建立 Z0 时把 $1 恢复为 255。")

        print()
        print("首次建立 Z0。")
        _release_and_rezero(controller, "首次建立Z0")
        _prime_z_hold(controller, args.high_speed)
        initial_hold = _ask_hold_state(
            "Z 已做 -0.2/0mm 激活动作。轻触检查是否锁住？[y/n]: "
        )
        print(f"启动后 Z 保持检查：{initial_hold}")

        results = [
            _run_trial(
                controller,
                label="同距离低速",
                distance_mm=args.compare_distance,
                speed_mm_min=args.low_speed,
            ),
            _run_trial(
                controller,
                label="同距离高速",
                distance_mm=args.compare_distance,
                speed_mm_min=args.high_speed,
            ),
        ]

        if not args.skip_linearity:
            for distance in args.linear_distances:
                results.append(
                    _run_trial(
                        controller,
                        label=f"高速线性 {distance:g}mm",
                        distance_mm=distance,
                        speed_mm_min=args.high_speed,
                    )
                )

        _print_results(results, current_steps)

        print()
        print("恢复最终状态：释放后让 Z 自然落到机械 Z0。")
        _release_and_rezero(controller, "恢复最终Z0")
        _prime_z_hold(controller, args.high_speed)
        hold_before_close = _ask_hold_state(
            "串口关闭前，Z 是否已经锁住？[y/n]: "
        )
        print("即将关闭串口；$1 保持为 255，Z 位于 0mm。")
        controller.disconnect()
        connected = False
        hold_after_close = _ask_hold_state(
            "串口关闭后，再轻触检查 Z 是否仍锁住？[y/n]: "
        )

        print()
        print("=" * 72)
        print("保持使能诊断")
        print("=" * 72)
        print(f"首次 Z 激活动作后：{initial_hold}")
        print(f"串口关闭前：{hold_before_close}")
        print(f"串口关闭后：{hold_after_close}")
        if initial_hold == "否":
            print(
                "判断：$1=255 且 Z 已实际运动后仍不保持，"
                "需要检查 Z 驱动使能引脚、驱动板或固件轴使能逻辑。"
            )
        elif hold_before_close == "是" and hold_after_close == "否":
            print(
                "判断：关闭串口或控制板复位会释放 Z；"
                "需要继续检查串口 DTR/复位行为或修改固件启动逻辑。"
            )
        elif hold_after_close == "是":
            print("判断：Z 在被明确激活后能够持续保持。")
        print("\n✅ Z 诊断完成，最终 Z=0mm，$1=255。")
        return 0
    except (CalibrationAbort, KeyboardInterrupt) as exc:
        print(
            f"\n⛔ 已中断：{exc}。已发送GRBL实时暂停，不再等待串口应答。",
            file=sys.stderr,
        )
        if connected:
            _emergency_feed_hold(controller)
        return 130
    except Exception as exc:
        print(f"\n⛔ Z 诊断失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if connected:
            controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
