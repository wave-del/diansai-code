"""Q6A isolation and device-selection regression tests; no peripherals used."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import importlib
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "vision/src", ROOT / "handover-puzzle", ROOT / "host"):
    sys.path.insert(0, str(directory))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from q6a_platform import add_mode_arguments, validate_motor_port
import hardware_discovery as discovery


class PlatformTests(unittest.TestCase):
    def test_unconfigured_motor_fails_before_any_actuator_is_opened(self):
        from puzzle_controller import PuzzleController
        from unittest.mock import Mock
        for port in ("", "/dev/ttyMSM0"):
            writer, magnet = Mock(), Mock()
            controller = PuzzleController(writer=writer, magnet=magnet)
            with self.subTest(port=port), self.assertRaises(ValueError):
                controller.connect("/dev/serial/by-id/writer", motor_port=port)
            magnet.open.assert_not_called()
            writer.connect.assert_not_called()

    def test_explicit_hardware_opt_in_and_conflicting_flags(self):
        parser = argparse.ArgumentParser()
        add_mode_arguments(parser)
        self.assertFalse(parser.parse_args([]).hardware)
        self.assertFalse(parser.parse_args(["--offline"]).hardware)
        self.assertTrue(parser.parse_args(["--hardware"]).hardware)
        with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            parser.parse_args(["--offline", "--hardware"])

    def test_console_and_empty_motor_port_rejected(self):
        for port in ("", " ", "/dev/ttyMSM0", "/dev/console"):
            with self.subTest(port=port), self.assertRaises(ValueError):
                validate_motor_port(port)
        self.assertEqual(validate_motor_port("/dev/serial/by-id/f32c"), "/dev/serial/by-id/f32c")
        with tempfile.TemporaryDirectory() as tmp:
            alias = Path(tmp) / "console"
            alias.symlink_to("/dev/ttyMSM0")
            with self.assertRaises(ValueError):
                validate_motor_port(str(alias))

    def test_runner_offline_rejects_execution_before_device_discovery(self):
        runner = importlib.import_module("run_task1_tangram")
        with patch.object(runner, "resolve_writer_port", side_effect=AssertionError("opened writer")), \
             patch.object(runner, "_capture_plan", side_effect=AssertionError("opened camera")), \
             patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            self.assertEqual(runner.main([]), 0)
            self.assertEqual(runner.main(["--offline", "--execute"]), 2)
            self.assertEqual(runner.main(["--initialize-and-park", "--execute"]), 2)
            self.assertEqual(runner.main(["--reuse-machine-origin"]), 2)
            self.assertEqual(runner.main(["--hardware", "--execute"]), 2)
            self.assertEqual(runner.main(["--hardware", "--execute", "--initialize-and-park"]), 2)

    def test_a733_library_is_never_loaded(self):
        from ec_26_e.vision.npu import NpuYolo11SegInferencer
        from ec_26_e.vision.contracts import VisionConfig
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("loaded native library")):
            with self.assertRaisesRegex(RuntimeError, "A733"):
                NpuYolo11SegInferencer(VisionConfig(model_path=Path("old.nb"),library_path=Path("libold.so")))


class CaptureCapabilityTests(unittest.TestCase):
    def probe(self, caps, device_caps=0):
        def ioctl(fd, request, buffer, mutate):
            self.assertEqual(request, 0x80685600)
            struct.pack_into("=II", buffer, 84, caps, device_caps)
        with patch.object(discovery.os, "open", return_value=42), \
             patch.object(discovery.os, "close") as close, \
             patch.object(discovery.fcntl, "ioctl", side_effect=ioctl):
            result = discovery.is_capture_device("/dev/video-test")
            close.assert_called_once_with(42)
            return result

    def test_camera_vs_codec_metadata_and_output(self):
        self.assertTrue(self.probe(0x1))
        self.assertTrue(self.probe(0x1000))
        self.assertFalse(self.probe(0x1001 | 0x4000))
        self.assertFalse(self.probe(0x1001 | 0x8000))
        self.assertFalse(self.probe(0x00800000))
        self.assertFalse(self.probe(0x2))
        self.assertFalse(self.probe(0x80000001, 0x4000))

    def test_inaccessible_camera_is_not_selected(self):
        with patch.object(discovery.os, "open", side_effect=PermissionError):
            self.assertFalse(discovery.is_capture_device("/dev/no-access"))

    def test_codec_never_reaches_opencv_even_if_configured(self):
        capture = unittest.mock.Mock()
        with patch.object(discovery.os.path, "exists", return_value=True), \
             patch.object(discovery, "is_capture_device", return_value=False), \
             patch.object(discovery, "discover_camera_paths", return_value=[]):
            with self.assertRaises(discovery.DeviceDiscoveryError):
                discovery.open_camera_with_fallback("/dev/video0", capture)
        capture.VideoCapture.assert_not_called()


class OfflineWindowTests(unittest.TestCase):
    def test_hardware_mode_still_requires_motor_configuration(self):
        from PySide6.QtWidgets import QApplication
        import tangram_host as host
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(host, "SETTINGS_PATH", Path(tmp) / "settings.json"))
            stack.enter_context(patch.object(host, "load_config", return_value={
                "camera_path": "", "writer_port": "", "motor_port": ""}))
            for target in ("hardware_discovery.discover_camera_paths", "hardware_discovery.discover_writer_ports"):
                stack.enter_context(patch(target, return_value=[]))
            stack.enter_context(patch("os.open", side_effect=AssertionError("opened device")))
            window = host.MainWindow(hardware=True)
            window.plan_only.setChecked(False)
            self.assertTrue(window.park_button.isEnabled())
            window._park_safely()
            self.assertIsNone(window._runner)
            self.assertIsNone(window._preview)
            self.assertIn("F32C 串口未配置", window.log.toPlainText())
            self.assertTrue(window.close())

    def test_start_event_loop_actions_and_exit_do_not_touch_hardware(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        import tangram_host as host
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            settings = Path(tmp) / "settings.json"
            settings.write_text('{"plan_only": false}')  # stale A7Z setting
            stack.enter_context(patch.object(host, "SETTINGS_PATH", settings))
            for target in ("cv2.VideoCapture", "serial.Serial", "periphery.GPIO", "ctypes.CDLL",
                           "hardware_discovery.discover_camera_paths", "hardware_discovery.discover_writer_ports",
                           "PySide6.QtSerialPort.QSerialPort.open"):
                stack.enter_context(patch(target, side_effect=AssertionError("unexpected hardware access: " + target)))
            original_open = os.open
            def guarded_open(path, *args, **kwargs):
                if os.fsdecode(path).startswith("/dev/"):
                    raise AssertionError("device open: " + str(path))
                return original_open(path, *args, **kwargs)
            stack.enter_context(patch("os.open", side_effect=guarded_open))
            window = host.MainWindow()
            window.show()
            self.assertFalse(hasattr(window, "_buttons"))
            self.assertFalse(hasattr(window, "_buzzer"))
            self.assertTrue(window.plan_only.isChecked())
            self.assertFalse(window.park_button.isEnabled())
            self.assertTrue(all(not b.isEnabled() for b in window._task_buttons.values()))
            window._start_task(1)
            window._park_safely()
            window._refresh_hardware_devices_from_ui()
            QTimer.singleShot(100, app.quit)
            app.exec()
            self.assertIsNone(window._runner)
            self.assertIsNone(window._preview)
            self.assertFalse(window._preview_retry_timer.isActive())
            self.assertIn("无外设模式", window.state_label.text())
            self.assertTrue(window.close())


if __name__ == "__main__":
    unittest.main()
