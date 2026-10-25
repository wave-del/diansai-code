"""上位机的非阻塞蜂鸣器完成提示。"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
BUZZER_DIR = ROOT / "handover-buzzer"


class BuzzerNotifier:
    """每次提示独占 GPIO，避免阻塞 Qt 的触摸与状态刷新线程。"""

    def __init__(self, on_error: Callable[[str], None]) -> None:
        self._on_error = on_error
        self._lock = threading.Lock()

    def completion_double_beep(self) -> None:
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            path = str(BUZZER_DIR)
            if path not in sys.path:
                sys.path.insert(0, path)
            from buzzer import Buzzer

            with Buzzer() as buzzer:
                buzzer.beep(duration=0.12, times=2, interval=0.16)
        except Exception as error:
            self._on_error(f"蜂鸣器提示不可用：{error}")
        finally:
            self._lock.release()
