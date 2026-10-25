"""使用 libgpiod 控制 A7Z 物理 pin33 上的完成提示 LED。"""

from __future__ import annotations

from pathlib import Path


# 本机 gpioinfo 映射：A7Z 物理 pin33 是 gpiochip1 的 line 35。
COMPLETION_LED_CHIP_PATH = Path("/dev/gpiochip1")
COMPLETION_LED_LINE_OFFSET = 35


class GpiodCompletionLedOutput:
    """持有完成提示 LED 的 GPIO，并在释放前主动拉低。"""

    def __init__(self) -> None:
        """创建尚未访问 GPIO 的 LED 输出。"""

        self._chip: object | None = None
        self._line: object | None = None

    def open(self) -> None:
        """申请物理 pin33，并把初始值固定为低电平。"""

        if self._line is not None:
            return
        try:
            import gpiod
        except ImportError as error:
            raise RuntimeError("未安装 gpiod Python 绑定") from error
        chip = None
        try:
            chip_factory = getattr(gpiod, "Chip", None)
            if chip_factory is not None:
                chip = chip_factory(str(COMPLETION_LED_CHIP_PATH))
                line = chip.get_line(COMPLETION_LED_LINE_OFFSET)
                line.request(
                    consumer="ec-26-e-completion-led",
                    type=gpiod.LINE_REQ_DIR_OUT,
                    default_vals=[0],
                )
            else:
                chip = gpiod.chip(str(COMPLETION_LED_CHIP_PATH))
                line = chip.get_line(COMPLETION_LED_LINE_OFFSET)
                request = gpiod.line_request()
                request.consumer = "ec-26-e-completion-led"
                request.request_type = gpiod.line_request.DIRECTION_OUTPUT
                line.request(request, default_val=0)
        except Exception as error:
            try:
                if chip is not None:
                    self._close_chip(chip)
            except Exception:
                pass
            raise RuntimeError(f"申请完成提示 LED GPIO 失败：{error}") from error
        self._chip = chip
        self._line = line

    def set_enabled(self, enabled: bool) -> None:
        """使用高电平点亮 LED、低电平熄灭 LED。"""

        if not isinstance(enabled, bool):
            raise TypeError("enabled 必须是 bool")
        if self._line is None:
            raise RuntimeError("完成提示 LED GPIO 尚未打开")
        try:
            self._line.set_value(1 if enabled else 0)
        except Exception as error:
            raise RuntimeError(f"设置完成提示 LED GPIO 失败：{error}") from error

    def close(self) -> None:
        """尽力熄灭 LED 后释放 line 和 chip；重复关闭安全。"""

        line = self._line
        chip = self._chip
        self._line = None
        self._chip = None
        errors: list[str] = []
        if line is not None:
            try:
                line.set_value(0)
            except Exception as error:
                errors.append(f"完成提示 LED 拉低失败：{error}")
            try:
                line.release()
            except Exception as error:
                errors.append(f"GPIO line 释放失败：{error}")
        if chip is not None:
            try:
                self._close_chip(chip)
            except Exception as error:
                errors.append(f"GPIO chip 关闭失败：{error}")
        if errors:
            raise RuntimeError("；".join(errors))

    @staticmethod
    def _close_chip(chip: object) -> None:
        """关闭不同 libgpiod Python 绑定中的 GPIO chip。"""

        close = getattr(chip, "close", None)
        if callable(close):
            close()
            return
        reset = getattr(chip, "reset", None)
        if callable(reset):
            reset()
            return
        raise RuntimeError("当前 gpiod chip 对象不支持关闭")
