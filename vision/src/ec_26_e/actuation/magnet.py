"""使用 libgpiod Python 绑定控制高电平上磁的 GPIO line。"""

from __future__ import annotations

from ec_26_e.actuation.contracts import MagnetConfig


class GpiodMagnetOutput:
    """持有 GPIO line 文件描述符并在关闭前主动输出低电平。"""

    def __init__(self, config: MagnetConfig) -> None:
        """保存已校验配置，构造阶段不访问 GPIO。"""

        if not isinstance(config, MagnetConfig):
            raise TypeError("config 必须是 MagnetConfig")
        self._config = config
        self._chip: object | None = None
        self._line: object | None = None

    def open(self) -> None:
        """申请输出 line，并把初始值固定为低电平。"""

        if self._line is not None:
            return
        try:
            import gpiod
        except ImportError as error:
            raise RuntimeError("未安装 gpiod Python 绑定") from error
        chip = None
        try:
            # 官方 libgpiod Python 绑定使用 ``Chip``；部分发行版提供的
            # 1.x C++ 包装则使用小写 ``chip``。两者都需要支持。
            chip_factory = getattr(gpiod, "Chip", None)
            if chip_factory is not None:
                chip = chip_factory(str(self._config.chip_path))
                line = chip.get_line(self._config.line_offset)
                line.request(
                    consumer="ec-26-e-magnet",
                    type=gpiod.LINE_REQ_DIR_OUT,
                    default_vals=[0],
                )
            else:
                chip = gpiod.chip(str(self._config.chip_path))
                line = chip.get_line(self._config.line_offset)
                request = gpiod.line_request()
                request.consumer = "ec-26-e-magnet"
                request.request_type = gpiod.line_request.DIRECTION_OUTPUT
                # 申请输出时直接写 0，避免先以高电平或浮空状态驱动电磁铁。
                line.request(request, default_val=0)
        except Exception as error:
            try:
                if chip is not None:
                    self._close_chip(chip)
            except Exception:
                pass
            raise RuntimeError(f"申请磁铁 GPIO 失败：{error}") from error
        self._chip = chip
        self._line = line

    def set_enabled(self, enabled: bool) -> None:
        """输出高电平上磁或低电平消磁。"""

        if not isinstance(enabled, bool):
            raise TypeError("enabled 必须是 bool")
        if self._line is None:
            raise RuntimeError("磁铁 GPIO 尚未打开")
        try:
            self._line.set_value(1 if enabled else 0)
        except Exception as error:
            raise RuntimeError(f"设置磁铁 GPIO 失败：{error}") from error

    def close(self) -> None:
        """尽力拉低后释放 line 和 chip；重复关闭安全。"""

        line = self._line
        chip = self._chip
        self._line = None
        self._chip = None
        errors: list[str] = []
        if line is not None:
            try:
                line.set_value(0)
            except Exception as error:
                errors.append(f"磁铁拉低失败：{error}")
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
