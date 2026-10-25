#!/usr/bin/env python3
"""不连接硬件的 F32C 交付包自检。"""

from __future__ import annotations

import math
import os
import pty
import select
import threading
import time

from f32c_controller import (
    F32CMotor,
    F32CProtocolError,
    FeedbackStreamParser,
    build_multi_turn_angle_frame,
    build_speed_frame,
    decode_feedback_frame,
    make_frame,
    shortest_delta_deg,
)
from electromagnet import ElectromagnetController


class SimulatedF32C:
    """供离线自检使用的最小伪终端电机。"""

    FRAME_LENGTHS = {
        0x00: 7,
        0x01: 7,
        0x02: 9,
        0x05: 5,
        0x06: 5,
        0x0E: 6,
    }

    def __init__(self, master_fd: int):
        self.master_fd = master_fd
        self.total_x10 = 0
        self.power_up_mechanical_x10 = 3000
        self.enabled = False
        self.disable_seen = False
        self.position_targets_x10: list[int] = []
        self.drop_total_queries = 0
        self.functions_seen: list[int] = []
        self.rx = bytearray()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    @property
    def mechanical_x10(self) -> int:
        # 与真实F32C实测一致：机械角度和上电多圈角度方向相反。
        return (self.power_up_mechanical_x10 - self.total_x10) % 3600

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                readable, _, _ = select.select(
                    [self.master_fd], [], [], 0.05
                )
                if not readable:
                    continue
                data = os.read(self.master_fd, 4096)
                if not data:
                    continue
                self.rx.extend(data)
                self._consume()
            except OSError:
                return

    def _consume(self) -> None:
        while True:
            try:
                header = self.rx.index(0x7A)
            except ValueError:
                self.rx.clear()
                return
            if header:
                del self.rx[:header]
            if len(self.rx) < 3:
                return
            frame_length = self.FRAME_LENGTHS.get(self.rx[2])
            if frame_length is None:
                del self.rx[0]
                continue
            if len(self.rx) < frame_length:
                return
            frame = bytes(self.rx[:frame_length])
            del self.rx[:frame_length]
            if frame[-1] != 0x7B or frame[-2] != _bcc(frame[:-2]):
                continue
            self._handle(frame)

    def _handle(self, frame: bytes) -> None:
        function = frame[2]
        self.functions_seen.append(function)
        if function == 0x06:
            self.enabled = True
        elif function == 0x05:
            self.enabled = False
            self.disable_seen = True
        elif function == 0x02 and self.enabled:
            self.total_x10 = int.from_bytes(frame[3:7], "big", signed=True)
            self.position_targets_x10.append(self.total_x10)
        elif function == 0x0E:
            feedback_type = frame[3]
            if feedback_type == 1:
                if self.drop_total_queries:
                    self.drop_total_queries -= 1
                    return
                payload = self.total_x10.to_bytes(4, "big", signed=True)
            elif feedback_type == 2:
                payload = self.mechanical_x10.to_bytes(
                    2, "big", signed=False
                )
            else:
                return
            os.write(
                self.master_fd,
                make_frame(2, feedback_type, payload),
            )


class SimulatedDigitalOutput:
    """供电磁铁逻辑离线自检使用，不访问真实GPIO。"""

    def __init__(self):
        self.opened = False
        self.value = False
        self.history: list[bool] = []

    def open(self, initial_value: bool) -> None:
        assert not self.opened
        self.opened = True
        self.value = bool(initial_value)
        self.history.append(self.value)

    def set_value(self, value: bool) -> None:
        assert self.opened
        self.value = bool(value)
        self.history.append(self.value)

    def get_value(self) -> bool:
        assert self.opened
        return self.value

    def close(self) -> None:
        self.opened = False


def _bcc(data: bytes) -> int:
    result = 0
    for value in data:
        result ^= value
    return result


def feedback_frame(
    feedback_type: int,
    value: int,
    length: int,
    signed: bool,
) -> bytes:
    payload = value.to_bytes(length, "big", signed=signed)
    return make_frame(2, feedback_type, payload)


def main() -> int:
    assert make_frame(2, 0x06) == bytes.fromhex("7A 02 06 7E 7B")
    assert make_frame(2, 0x05) == bytes.fromhex("7A 02 05 7D 7B")
    assert make_frame(2, 0x00, b"\x00\x01") == bytes.fromhex(
        "7A 02 00 00 01 79 7B"
    )
    assert build_speed_frame(15) == bytes.fromhex(
        "7A 02 01 00 0F 76 7B"
    )
    assert build_multi_turn_angle_frame(0) == bytes.fromhex(
        "7A 02 02 00 00 00 00 7A 7B"
    )
    assert build_multi_turn_angle_frame(360) == bytes.fromhex(
        "7A 02 02 00 00 0E 10 64 7B"
    )
    assert build_multi_turn_angle_frame(-100) == bytes.fromhex(
        "7A 02 02 FF FF FC 18 9E 7B"
    )
    for invalid_speed in (0, 15.5, 1001):
        try:
            build_speed_frame(invalid_speed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法速度未被拒绝: {invalid_speed!r}")
    try:
        make_frame(2.5, 0x06)
    except ValueError:
        pass
    else:
        raise AssertionError("小数地址未被拒绝")

    total_raw = feedback_frame(1, -1234, 4, True)
    mechanical_2_raw = feedback_frame(2, 3599, 2, False)
    mechanical_4_raw = feedback_frame(2, 125, 4, True)
    assert decode_feedback_frame(total_raw).angle_deg == -123.4
    assert decode_feedback_frame(mechanical_2_raw).angle_deg == 359.9
    assert decode_feedback_frame(mechanical_4_raw).angle_deg == 12.5

    parser = FeedbackStreamParser(2)
    assert parser.feed(b"\x00\xFF" + total_raw[:4]) == []
    parsed = parser.feed(total_raw[4:] + mechanical_2_raw)
    assert [item.angle_deg for item in parsed] == [-123.4, 359.9]

    broken = bytearray(total_raw)
    broken[-2] ^= 0x01
    try:
        decode_feedback_frame(bytes(broken))
    except F32CProtocolError:
        pass
    else:
        raise AssertionError("损坏的 BCC 没有被拒绝")

    assert shortest_delta_deg(350, 10) == 20
    assert shortest_delta_deg(10, 350) == -20
    assert shortest_delta_deg(0, 180) == 180
    assert math.isclose(shortest_delta_deg(359.9, 0), 0.1)

    gpio = SimulatedDigitalOutput()
    magnet = ElectromagnetController(_backend=gpio)
    magnet.open()
    assert not magnet.is_engaged
    magnet.engage()
    assert magnet.is_engaged
    magnet.release()
    assert not magnet.is_engaged
    magnet.pulse(0.001)
    assert not magnet.is_engaged
    magnet.close()
    assert not gpio.opened
    assert gpio.history == [False, True, False, True, False, False]

    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)
    os.close(slave_fd)
    simulator = SimulatedF32C(master_fd)
    motor = F32CMotor(
        port=slave_name,
        address=2,
        feedback_timeout=0.5,
    )
    try:
        motor.connect()
        # 先打开PTY从端再启动主端线程，避免Linux在从端尚未打开时
        # 对主端read()返回EIO造成测试线程提前退出。
        simulator.start()
        assert motor.get_total_angle() == 0.0
        assert motor.get_angle() == 300.0
        simulator.functions_seen.clear()
        motor.enable(rpm=15)
        assert simulator.functions_seen == [
            0x00,  # 失能状态预设多圈T型模式
            0x01,  # 失能状态预设1 RPM
            0x06,  # 使能
            0x0E,  # 查询当前位置
            0x02,  # 保持当前位置，清除残留目标
            0x01,  # 设置业务速度15 RPM
        ]
        # 使能后第一条位置目标必须保持当前多圈位置，清除残留目标。
        assert simulator.position_targets_x10 == [0]

        # 机械正向60°对应协议多圈负向60°，不是机械反向300°。
        assert motor.return_to_mechanical_zero() == -60.0
        assert motor.get_angle() == 0.0

        assert motor.move_to(90.0) == -150.0
        assert motor.get_angle() == 90.0
        try:
            motor.move_to(360.0)
        except ValueError:
            pass
        else:
            raise AssertionError("高层API错误接受了360°绝对目标")

        # 单次反馈丢包应重试恢复，不能直接失能和终止正常运动。
        simulator.drop_total_queries = 1
        assert motor.rotate_one_turn(direction=-1) == -510.0
        assert motor.get_angle() == 90.0
    finally:
        motor.close(disable=True)
        simulator.stop()
        os.close(master_fd)
    assert simulator.disable_seen

    print(
        "✅ 独立交付包自检通过"
        "（F32C协议/伪终端 + 电磁铁伪GPIO；未连接真实硬件）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
