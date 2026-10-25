"""不连接串口，验证 F32C 的到位重试与任务放行边界。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from f32c_controller import F32CMotor, F32CTimeoutError


class F32CArrivalPolicyTest(unittest.TestCase):
    @staticmethod
    def _motor() -> tuple[F32CMotor, list[object]]:
        motor = F32CMotor(port="/dev/fake-f32c")
        motor._enabled = True
        sent: list[object] = []
        motor._require_enabled = lambda: None  # type: ignore[method-assign]
        motor._send = sent.append  # type: ignore[method-assign]
        motor.set_speed = lambda unused: None  # type: ignore[method-assign]
        return motor, sent

    def test_task_policy_retries_exact_target_then_continues_with_feedback(self) -> None:
        motor, sent = self._motor()
        angles = iter((0.0, 7.0))  # 当前角度、最终残差反馈。
        motor.get_total_angle = lambda: 0.0  # type: ignore[method-assign]
        motor.get_angle = lambda: next(angles)  # type: ignore[method-assign]
        attempts = 0

        def never_settles(*unused: object) -> None:
            nonlocal attempts
            attempts += 1
            raise F32CTimeoutError("not settled")

        motor._wait_for_total_target = never_settles  # type: ignore[method-assign]
        result = motor.move_to(
            10.0,
            wait=True,
            tolerance_deg=1.5,
            settle_retries=2,
            allow_inexact_arrival=True,
        )

        self.assertEqual(result, -10.0)
        self.assertEqual(attempts, 3)
        # 首次命令与两次修正命令完全相同，绝不按残差改写目标。
        self.assertEqual(len(sent), 3)
        self.assertTrue(all(frame == sent[0] for frame in sent))
        self.assertTrue(motor.is_enabled)

    def test_strict_policy_still_disables_on_unsettled_target(self) -> None:
        motor, unused_sent = self._motor()
        motor.get_total_angle = lambda: 0.0  # type: ignore[method-assign]
        motor.get_angle = lambda: 0.0  # type: ignore[method-assign]
        motor._wait_for_total_target = (  # type: ignore[method-assign]
            lambda *unused: (_ for _ in ()).throw(F32CTimeoutError("not settled"))
        )
        disabled: list[bool] = []
        motor._safe_disable = lambda: disabled.append(True)  # type: ignore[method-assign]

        with self.assertRaises(F32CTimeoutError):
            motor.move_to(10.0, wait=True, tolerance_deg=1.5)

        self.assertEqual(disabled, [True])


if __name__ == "__main__":
    unittest.main()
