"""USB 摄像头、写字机自动发现的离线测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from hardware_discovery import (
    DeviceDiscoveryError,
    discover_camera_paths,
    open_camera_with_fallback,
    resolve_writer_port,
)


class _Capture:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.released = True


class _Cv2:
    CAP_V4L2 = 200

    def __init__(self, opened: dict[str, bool]) -> None:
        self.opened = opened
        self.calls: list[tuple[str, int]] = []

    def VideoCapture(self, path: str, backend: int) -> _Capture:
        self.calls.append((path, backend))
        return _Capture(self.opened.get(path, False))


class WriterDiscoveryTests(unittest.TestCase):
    def test_uses_existing_configured_port_without_scan(self) -> None:
        with patch("hardware_discovery.os.path.exists", return_value=True), patch(
            "hardware_discovery.discover_writer_ports"
        ) as discover:
            selected = resolve_writer_port("/dev/serial/by-id/configured")
        self.assertEqual(selected, "/dev/serial/by-id/configured")
        discover.assert_not_called()

    def test_falls_back_only_when_one_port_exists(self) -> None:
        with patch("hardware_discovery.os.path.exists", return_value=False), patch(
            "hardware_discovery.discover_writer_ports",
            return_value=["/dev/ttyUSB0"],
        ):
            selected = resolve_writer_port("/dev/serial/by-id/missing")
        self.assertEqual(selected, "/dev/ttyUSB0")

    def test_rejects_ambiguous_ports(self) -> None:
        with patch("hardware_discovery.os.path.exists", return_value=False), patch(
            "hardware_discovery.discover_writer_ports",
            return_value=["/dev/ttyUSB0", "/dev/ttyUSB1"],
        ):
            with self.assertRaisesRegex(DeviceDiscoveryError, "多个候选"):
                resolve_writer_port("/dev/serial/by-id/missing")


class CameraDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        probe = patch("hardware_discovery.is_capture_device", return_value=True)
        probe.start()
        self.addCleanup(probe.stop)

    def test_groups_multiple_interfaces_of_one_usb_camera(self) -> None:
        index0 = "/dev/v4l/by-id/usb-camera-video-index0"
        index1 = "/dev/v4l/by-id/usb-camera-video-index1"
        real_paths = {
            index0: "/dev/video0",
            index1: "/dev/video1",
            "/dev/video0": "/dev/video0",
            "/dev/video1": "/dev/video1",
        }

        def fake_glob(pattern: str) -> list[str]:
            if pattern == "/dev/v4l/by-id/*-video-index*":
                return [index0, index1]
            if pattern == "/dev/video[0-9]*":
                return ["/dev/video0", "/dev/video1"]
            return []

        with patch("hardware_discovery.glob.glob", side_effect=fake_glob), patch(
            "hardware_discovery.os.path.exists", return_value=True
        ), patch(
            "hardware_discovery.os.path.realpath",
            side_effect=lambda path: real_paths.get(path, path),
        ):
            candidates = discover_camera_paths()
        self.assertEqual(candidates, [index0])

    def test_uses_configured_camera_first(self) -> None:
        cv2 = _Cv2({"/dev/video4": True})
        with patch("hardware_discovery.os.path.exists", return_value=True), patch(
            "hardware_discovery.discover_camera_paths"
        ) as discover:
            capture, selected = open_camera_with_fallback("/dev/video4", cv2)
        self.assertTrue(capture.isOpened())
        self.assertEqual(selected, "/dev/video4")
        self.assertEqual(cv2.calls, [("/dev/video4", cv2.CAP_V4L2)])
        discover.assert_not_called()

    def test_uses_unique_camera_when_configured_path_is_missing(self) -> None:
        cv2 = _Cv2({"/dev/video2": True})
        with patch("hardware_discovery.os.path.exists", return_value=False), patch(
            "hardware_discovery.discover_camera_paths",
            return_value=["/dev/video2"],
        ):
            _capture, selected = open_camera_with_fallback("/dev/v4l/by-id/missing", cv2)
        self.assertEqual(selected, "/dev/video2")

    def test_rejects_ambiguous_cameras_without_opening_them(self) -> None:
        cv2 = _Cv2({"/dev/video2": True, "/dev/video4": True})
        with patch("hardware_discovery.os.path.exists", return_value=False), patch(
            "hardware_discovery.discover_camera_paths",
            return_value=["/dev/video2", "/dev/video4"],
        ):
            with self.assertRaisesRegex(DeviceDiscoveryError, "多个候选"):
                open_camera_with_fallback("/dev/v4l/by-id/missing", cv2)
        self.assertEqual(cv2.calls, [])


if __name__ == "__main__":
    unittest.main()
