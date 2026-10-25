#!/usr/bin/env python3
"""不访问真实GPIO的电磁铁交付包自检。"""

from __future__ import annotations

from electromagnet import ElectromagnetController


class SimulatedDigitalOutput:
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


def main() -> int:
    gpio = SimulatedDigitalOutput()
    magnet = ElectromagnetController(_backend=gpio)

    magnet.open()
    assert magnet.is_open
    assert not magnet.is_engaged

    magnet.engage()
    assert magnet.is_engaged

    magnet.release()
    assert not magnet.is_engaged

    magnet.pulse(0.001)
    assert not magnet.is_engaged

    magnet.close()
    assert not magnet.is_open
    assert not gpio.opened
    assert gpio.history == [False, True, False, True, False, False]

    try:
        ElectromagnetController(line_offset=6.5)
    except ValueError:
        pass
    else:
        raise AssertionError("小数GPIO line offset未被拒绝")

    print(
        "✅ 电磁铁独立交付包自检通过"
        "（伪GPIO状态机；未访问真实PL6）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
