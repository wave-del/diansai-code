"""Q6A deployment settings; importing this module never opens devices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "q6a_config.json"
VISION_UNAVAILABLE = "Q6A HTP 后端已接入，可使用 q6a_image.py 验证本地图片；相机、工作区及运动尚未完成验收，现场任务暂不开放。"


def load_config() -> dict[str, str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("camera_path", "writer_port", "motor_port"):
        if not isinstance(config.get(key), str):
            raise ValueError(f"q6a_config.json: {key} 必须是字符串")
    return config


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", dest="hardware", action="store_false",
                       help="无外设模式（默认），不打开相机、串口或 GPIO")
    group.add_argument("--hardware", action="store_true",
                       help="显式允许使用已配置的外设")
    parser.set_defaults(hardware=False)


def validate_motor_port(port: str) -> str:
    if not port.strip():
        raise ValueError("F32C 串口未配置；请填写 handover-puzzle/q6a_config.json 的 motor_port 或传入 --motor-port")
    resolved = str(Path(port).resolve())
    consoles = {"/dev/console", "/dev/ttyMSM0"}
    try:
        for entry in Path("/proc/cmdline").read_text().split():
            if entry.startswith("console="):
                consoles.add("/dev/" + entry.split("=", 1)[1].split(",", 1)[0])
    except OSError:
        pass
    if resolved in consoles:
        raise ValueError(f"拒绝使用系统控制台作为 F32C 串口：{port}")
    return port
