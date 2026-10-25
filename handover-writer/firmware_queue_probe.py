#!/usr/bin/env python3
"""
GRBL 固件串口队列实机探针。

本脚本绕过上层控制器，直接使用 QSerialPort 的阻塞 API 验证：
1. 单条命令的 ok/error 与运动完成状态；
2. 收到 ok 后立即发送下一条时，固件是否会规划连续运动；
3. 不等待 ok、一次盲发多条时，固件是否会丢失或拒绝命令。

安全约束：
- 不执行 $H；
- 不修改 EEPROM 参数；
- 只使用 X/Y，当前位置临时设为测试工作原点；
- 每组测试后返回测试原点。
"""

from __future__ import annotations

import argparse
import array
import fcntl
import os
import re
import select
import sys
import termios
import time
from dataclasses import dataclass

STATUS_COORD_RE = re.compile(r"(MPos|WPos|WCO):([^|>]+)")


@dataclass
class Status:
    raw: str
    state: str
    mpos: tuple[float, ...] | None = None
    wpos: tuple[float, ...] | None = None
    wco: tuple[float, ...] | None = None

    @property
    def work_xy(self) -> tuple[float, float] | None:
        if self.wpos and len(self.wpos) >= 2:
            return self.wpos[0], self.wpos[1]
        if (
            self.mpos
            and self.wco
            and len(self.mpos) >= 2
            and len(self.wco) >= 2
        ):
            return self.mpos[0] - self.wco[0], self.mpos[1] - self.wco[1]
        return None


def parse_status(line: str) -> Status | None:
    if not (line.startswith("<") and line.endswith(">")):
        return None
    state = line[1:-1].split("|", 1)[0].split(":", 1)[0]
    values: dict[str, tuple[float, ...]] = {}
    for name, coordinate_text in STATUS_COORD_RE.findall(line):
        try:
            values[name.lower()] = tuple(float(v) for v in coordinate_text.split(","))
        except ValueError:
            continue
    return Status(line, state, values.get("mpos"), values.get("wpos"), values.get("wco"))


class BlockingGrbl:
    def __init__(self, port: str, baud: int = 115200):
        baud_constants = {
            9600: termios.B9600,
            19200: termios.B19200,
            38400: termios.B38400,
            57600: termios.B57600,
            115200: termios.B115200,
            230400: termios.B230400,
        }
        if baud not in baud_constants:
            raise ValueError(f"不支持的波特率: {baud}")
        self.port = port
        self.baud = baud
        self.baud_constant = baud_constants[baud]
        self.fd: int | None = None
        self.rx = bytearray()
        self.last_status: Status | None = None

    def open(self) -> None:
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = self.baud_constant | termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = self.baud_constant
        attrs[5] = self.baud_constant
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def pulse_dtr(self) -> None:
        if self.fd is None:
            raise RuntimeError("串口未打开")
        dtr = array.array("i", [termios.TIOCM_DTR])
        fcntl.ioctl(self.fd, termios.TIOCMBIC, dtr, True)
        time.sleep(0.1)
        fcntl.ioctl(self.fd, termios.TIOCMBIS, dtr, True)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _write_all(self, payload: bytes, timeout_s: float = 3.0) -> None:
        if self.fd is None:
            raise RuntimeError("串口未打开")
        deadline = time.monotonic() + timeout_s
        offset = 0
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"串口发送超时，只写入 {offset}/{len(payload)} 字节"
                )
            _, writable, _ = select.select([], [self.fd], [], min(0.2, remaining))
            if writable:
                offset += os.write(self.fd, payload[offset:])
        termios.tcdrain(self.fd)

    def _read_available(self) -> list[str]:
        if self.fd is None:
            raise RuntimeError("串口未打开")
        while True:
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            self.rx.extend(chunk)
        lines: list[str] = []
        while b"\n" in self.rx:
            raw, _, remainder = self.rx.partition(b"\n")
            self.rx = bytearray(remainder)
            line = raw.rstrip(b"\r").decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
                status = parse_status(line)
                if status:
                    self.last_status = status
        return lines

    def read_lines(self, timeout_s: float) -> list[str]:
        if self.fd is None:
            raise RuntimeError("串口未打开")
        deadline = time.monotonic() + timeout_s
        lines: list[str] = []
        lines.extend(self._read_available())
        if lines:
            return lines
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            readable, _, _ = select.select([self.fd], [], [], min(0.1, remaining))
            if readable:
                lines.extend(self._read_available())
                if lines:
                    return lines
        return lines

    def send_realtime(self, payload: bytes) -> None:
        self._write_all(payload)

    def send_line(self, command: str, timeout_s: float = 3.0) -> list[str]:
        print(f"  → {command}")
        self._write_all((command.strip() + "\n").encode("ascii"))
        deadline = time.monotonic() + timeout_s
        received: list[str] = []
        while time.monotonic() < deadline:
            for line in self.read_lines(min(0.2, deadline - time.monotonic())):
                received.append(line)
                print(f"  ← {line}")
                if line == "ok" or line.startswith(("error:", "ALARM:")):
                    return received
        raise TimeoutError(f"等待命令应答超时: {command}")

    def query_status(self, timeout_s: float = 1.0) -> Status:
        self.send_realtime(b"?")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for line in self.read_lines(min(0.2, deadline - time.monotonic())):
                status = parse_status(line)
                if status:
                    return status
                print(f"  ← {line}")
        raise TimeoutError("等待状态汇报超时")

    def wait_idle(self, timeout_s: float = 20.0) -> tuple[Status, list[str]]:
        deadline = time.monotonic() + timeout_s
        states: list[str] = []
        while time.monotonic() < deadline:
            status = self.query_status()
            if not states or states[-1] != status.state:
                states.append(status.state)
                print(f"  ← 状态 {status.raw}")
            if status.state == "Idle":
                return status, states
            if status.state in ("Alarm", "Door"):
                raise RuntimeError(f"GRBL 无法运动，当前状态: {status.raw}")
            time.sleep(0.03)
        raise TimeoutError(f"等待 Idle 超时，状态历史: {states}")

    def stream_ack_driven(
        self, commands: list[str], timeout_s: float = 30.0
    ) -> tuple[int, list[str]]:
        """每收到一条 ok 才发送下一条，但不等待前一动作执行完。"""
        responses: list[str] = []
        for index, command in enumerate(commands, 1):
            lines = self.send_line(command, timeout_s=timeout_s)
            terminal = next(
                (
                    line
                    for line in reversed(lines)
                    if line == "ok" or line.startswith(("error:", "ALARM:"))
                ),
                "",
            )
            responses.append(terminal)
            if terminal != "ok":
                print(f"  第 {index} 条未被接受: {terminal or '无应答'}")
                break
        return responses.count("ok"), responses

    def stream_blind(
        self, commands: list[str], timeout_s: float = 30.0
    ) -> tuple[int, list[str]]:
        """一次写入全部命令，随后统计全部终结应答。"""
        payload = "".join(command.strip() + "\n" for command in commands).encode("ascii")
        print(f"  → 一次写入 {len(commands)} 条，共 {len(payload)} 字节")
        self._write_all(payload)
        deadline = time.monotonic() + timeout_s
        terminals: list[str] = []
        while time.monotonic() < deadline and len(terminals) < len(commands):
            for line in self.read_lines(min(0.2, deadline - time.monotonic())):
                if line == "ok" or line.startswith(("error:", "ALARM:")):
                    terminals.append(line)
                    if line != "ok":
                        print(f"  ← {line}")
        print(
            f"  ← 终结应答 {len(terminals)}/{len(commands)}，"
            f"ok={terminals.count('ok')}，"
            f"error/alarm={len(terminals) - terminals.count('ok')}"
        )
        return terminals.count("ok"), terminals


def format_xy(status: Status) -> str:
    xy = status.work_xy
    if xy is not None:
        return f"WPos X={xy[0]:.3f}, Y={xy[1]:.3f}"
    if status.mpos and len(status.mpos) >= 2:
        return f"MPos X={status.mpos[0]:.3f}, Y={status.mpos[1]:.3f}"
    return "坐标不可用"


def require_ok(lines: list[str], command: str) -> None:
    if "ok" not in lines:
        raise RuntimeError(f"命令没有成功应答: {command}; 收到: {lines}")


def return_origin(grbl: BlockingGrbl) -> Status:
    require_ok(grbl.send_line("G21 G90 G1 X0 Y0 F1200"), "return origin")
    status, _ = grbl.wait_idle()
    print(f"  ✓ 已回测试原点，{format_xy(status)}")
    return status


def run(port: str, baud: int, include_blind: bool, handshake_only: bool) -> int:
    grbl = BlockingGrbl(port, baud)
    origin_established = False
    try:
        print(f"打开 {port} @ {baud}", flush=True)
        grbl.open()
        print("脉冲 DTR 复位线", flush=True)
        grbl.pulse_dtr()
        time.sleep(1.0)
        for line in grbl.read_lines(0.5):
            print(f"  ← {line}")
        print("发送 GRBL 软复位（Ctrl-X），等待启动信息", flush=True)
        grbl.send_realtime(b"\x18")
        time.sleep(1.5)
        for line in grbl.read_lines(1.0):
            print(f"  ← {line}")

        initial = grbl.query_status()
        print(f"初始状态: {initial.raw}")
        if handshake_only:
            print("握手成功；未发送运动命令。")
            return 0

        print("\n[准备] 解锁、毫米制、绝对模式，并把当前位置设为测试原点")
        require_ok(grbl.send_line("$X"), "$X")
        require_ok(grbl.send_line("G21 G90"), "G21 G90")
        require_ok(grbl.send_line("G92 X0 Y0"), "G92 X0 Y0")
        origin_established = True
        origin = grbl.query_status()
        print(f"测试原点: {origin.raw}")

        print("\n[测试 1] 单条 X+5mm，确认基础运动和完成等待")
        start = time.monotonic()
        require_ok(grbl.send_line("G21 G91 G1 X5 Y0 F600"), "single move")
        status, states = grbl.wait_idle()
        elapsed = time.monotonic() - start
        print(f"  ✓ 用时 {elapsed:.3f}s，状态历史 {states}，{format_xy(status)}")
        return_origin(grbl)

        print("\n[测试 2] 20 条 X+1mm：每收到 ok 立即发下一条，不等运动完成")
        commands = ["G21 G91 G1 X1 Y0 F2000"] * 20
        start = time.monotonic()
        accepted, responses = grbl.stream_ack_driven(commands)
        status, states = grbl.wait_idle()
        elapsed = time.monotonic() - start
        print(
            f"  ✓ 接受 {accepted}/{len(commands)} 条，用时 {elapsed:.3f}s，"
            f"状态历史 {states}，{format_xy(status)}"
        )
        if accepted != len(commands) or any(r != "ok" for r in responses):
            raise RuntimeError("应答驱动流发送出现拒绝或缺失")
        return_origin(grbl)

        print("\n[测试 3] 10 条 Y+1mm：验证另一轴和连续规划")
        commands = ["G21 G91 G1 X0 Y1 F1500"] * 10
        start = time.monotonic()
        accepted, responses = grbl.stream_ack_driven(commands)
        status, states = grbl.wait_idle()
        elapsed = time.monotonic() - start
        print(
            f"  ✓ 接受 {accepted}/{len(commands)} 条，用时 {elapsed:.3f}s，"
            f"状态历史 {states}，{format_xy(status)}"
        )
        if accepted != len(commands) or any(r != "ok" for r in responses):
            raise RuntimeError("Y 轴应答驱动流发送出现拒绝或缺失")
        return_origin(grbl)

        if include_blind:
            print("\n[测试 4] 压力测试：一次盲发 40 条 X+1mm")
            commands = ["G21 G91 G1 X1 Y0 F2000"] * 40
            start = time.monotonic()
            accepted, terminals = grbl.stream_blind(commands)
            status, states = grbl.wait_idle()
            elapsed = time.monotonic() - start
            print(
                f"  结果: ok {accepted}/{len(commands)}，用时 {elapsed:.3f}s，"
                f"状态历史 {states}，{format_xy(status)}"
            )
            return_origin(grbl)
            if accepted != len(commands) or len(terminals) != len(commands):
                print("  ⚠ 盲发存在应答缺失/拒绝，不能使用无流控发送")
            else:
                print("  ✓ 本次 40 条盲发全部应答；生产代码仍应使用流控")

        print("\n全部探针测试完成，机器已回到测试原点。")
        return 0
    except Exception as exc:
        print(f"\n⛔ 探针失败: {exc}", file=sys.stderr)
        if origin_established:
            try:
                print("尝试恢复绝对模式并返回测试原点...")
                grbl.send_line("G21 G90", timeout_s=2.0)
                return_origin(grbl)
            except Exception as recovery_exc:
                print(f"⚠ 自动回原点失败: {recovery_exc}", file=sys.stderr)
        return 1
    finally:
        grbl.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--handshake-only",
        action="store_true",
        help="只做复位和状态握手，不发送运动命令",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="额外执行一次 40 条命令的无流控盲发压力测试",
    )
    args = parser.parse_args()
    return run(args.port, args.baud, args.blind, args.handshake_only)


if __name__ == "__main__":
    raise SystemExit(main())
