"""现场可更换 USB 摄像头、写字机的安全设备解析。

配置文件中的稳定 ``by-id`` 路径始终是第一优先级。只有该路径不存在或
摄像头无法打开时，才会枚举本机设备；自动发现要求候选唯一，绝不在存在多个
候选时猜测并连接错误设备。F32C 板载 UART 和 GPIO 不属于本模块。
"""

from __future__ import annotations

import glob
import fcntl
import os
import re
import struct
from typing import Any


class DeviceDiscoveryError(RuntimeError):
    """配置设备不可用且无法唯一自动发现时抛出。"""


def is_capture_device(path: str) -> bool:
    """Query capabilities only: reject codec, output and metadata nodes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            capability = bytearray(104)  # struct v4l2_capability
            fcntl.ioctl(fd, 0x80685600, capability, True)  # VIDIOC_QUERYCAP
            caps, device_caps = struct.unpack_from("=II", capability, 84)
            if caps & 0x80000000:  # V4L2_CAP_DEVICE_CAPS
                caps = device_caps
            capture = caps & (0x00000001 | 0x00001000)
            mem_to_mem = caps & (0x00004000 | 0x00008000)
            return bool(capture and not mem_to_mem)
        finally:
            os.close(fd)
    except OSError:
        return False


def _unique_paths(patterns: tuple[str, ...]) -> list[str]:
    """按出现顺序返回存在的字符设备，合并 by-id 与实际节点的重复项。"""

    paths: list[str] = []
    real_paths: set[str] = set()
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if not os.path.exists(candidate):
                continue
            real_path = os.path.realpath(candidate)
            if real_path in real_paths:
                continue
            real_paths.add(real_path)
            paths.append(candidate)
    return paths


def discover_camera_paths(*, exclude: str | None = None) -> list[str]:
    """发现物理 V4L2 摄像头，每台只保留一个最适合采集的节点。

    一只 UVC 摄像头常同时创建 ``video-index0``（视频）和 ``video-index1``
    （UVC 元数据）。二者不是两只摄像头；按 by-id 前缀分组后优先选择编号最小
    的接口，通常就是视频采集的 index0。
    """

    by_id_paths = [
        path
        for path in sorted(glob.glob("/dev/v4l/by-id/*-video-index*"))
        if os.path.exists(path)
    ]
    by_id_real_paths = {os.path.realpath(path) for path in by_id_paths}
    groups: dict[str, list[str]] = {}
    for path in by_id_paths:
        match = re.match(r"^(.*)-video-index(\d+)$", path)
        if match is None:
            continue
        groups.setdefault(match.group(1), []).append(path)

    candidates: list[str] = []
    seen_real_paths: set[str] = set()
    for _identity, interfaces in sorted(groups.items()):
        ordered = sorted(interfaces, key=lambda path: int(re.search(r"-video-index(\d+)$", path).group(1)))
        selected = next((path for path in ordered if is_capture_device(path)), None)
        if selected is None:
            continue
        real_path = os.path.realpath(selected)
        if real_path not in seen_real_paths:
            seen_real_paths.add(real_path)
            candidates.append(selected)

    # 没有 by-id 链接的旧式设备再退回 /dev/videoN。若同一类设备暴露多个
    # 无身份接口，保留安全的“多个候选则拒绝猜测”策略。
    for path in _unique_paths(("/dev/video[0-9]*",)):
        real_path = os.path.realpath(path)
        if real_path in by_id_real_paths:
            continue
        if real_path not in seen_real_paths and is_capture_device(path):
            seen_real_paths.add(real_path)
            candidates.append(path)
    if exclude is None:
        return candidates
    excluded_real_path = os.path.realpath(exclude)
    return [path for path in candidates if os.path.realpath(path) != excluded_real_path]


def discover_writer_ports(*, exclude: str | None = None) -> list[str]:
    """发现 USB 写字机串口，优先保留稳定的 ``/dev/serial/by-id`` 名称。"""

    candidates = _unique_paths(
        (
            "/dev/serial/by-id/*",
            "/dev/ttyUSB[0-9]*",
            "/dev/ttyACM[0-9]*",
        )
    )
    if exclude is None:
        return candidates
    excluded_real_path = os.path.realpath(exclude)
    return [path for path in candidates if os.path.realpath(path) != excluded_real_path]


def resolve_writer_port(configured_path: str) -> str:
    """返回配置写字机串口，或在其不存在时返回唯一的 USB 串口候选。"""

    if os.path.exists(configured_path):
        print(f"写字机串口：使用配置路径 {configured_path}", flush=True)
        return configured_path
    candidates = discover_writer_ports(exclude=configured_path)
    if len(candidates) == 1:
        selected = candidates[0]
        print(
            "写字机串口：配置路径不存在，已自动发现唯一候选 " + selected,
            flush=True,
        )
        return selected
    if not candidates:
        raise DeviceDiscoveryError(
            "写字机串口配置路径不存在，且未发现 USB 串口："
            f"{configured_path}"
        )
    raise DeviceDiscoveryError(
        "写字机串口配置路径不存在，发现多个候选，拒绝猜测："
        + ", ".join(candidates)
    )


def open_camera_with_fallback(
    configured_path: str,
    cv2_module: Any,
) -> tuple[Any, str]:
    """打开配置相机；失效时只允许打开唯一自动发现的候选。

    返回 ``(cv2.VideoCapture, 实际路径)``。调用方继续负责设置分辨率、帧率
    和像素格式，以保持原有视觉链路行为不变。
    """

    configured_exists = os.path.exists(configured_path) and is_capture_device(configured_path)
    if configured_exists:
        capture = cv2_module.VideoCapture(configured_path, cv2_module.CAP_V4L2)
        if capture.isOpened():
            print(f"摄像头：使用配置路径 {configured_path}", flush=True)
            return capture, configured_path
        capture.release()
        reason = "无法打开"
    else:
        reason = "不存在"

    candidates = discover_camera_paths(exclude=configured_path)
    if len(candidates) != 1:
        if not candidates:
            detail = "未发现可用 V4L2 摄像头"
        else:
            detail = "发现多个候选，拒绝猜测：" + ", ".join(candidates)
        raise DeviceDiscoveryError(
            f"摄像头配置路径{reason}：{configured_path}；{detail}"
        )

    selected = candidates[0]
    capture = cv2_module.VideoCapture(selected, cv2_module.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise DeviceDiscoveryError(
            "摄像头配置路径"
            f"{reason}：{configured_path}；自动发现的唯一候选也无法打开：{selected}"
        )
    print(
        "摄像头：配置路径"
        f"{reason}，已自动发现唯一候选 {selected}",
        flush=True,
    )
    return capture, selected
