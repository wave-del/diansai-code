"""基于 tracemalloc 的可移植 Python 峰值内存跟踪。"""

from __future__ import annotations

from dataclasses import dataclass
import tracemalloc


@dataclass
class MemoryTracker:
    enabled: bool = True
    track_python_allocations: bool = False
    _owned: bool = False
    start_current: int = 0
    start_peak: int = 0

    def start(self) -> None:
        if not self.enabled:
            return
        if self.track_python_allocations and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._owned = True
        if tracemalloc.is_tracing():
            self.start_current, self.start_peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()

    def snapshot(self) -> dict[str, float]:
        if not self.enabled:
            return {
                "current_memory_mb": 0.0,
                "peak_memory_mb": 0.0,
                "tracked_by": "disabled",
            }
        if not tracemalloc.is_tracing():
            try:
                import psutil
                info = psutil.Process().memory_info()
                peak = getattr(info, "peak_wset", info.rss)
                return {
                    "current_memory_mb": info.rss / (1024.0 * 1024.0),
                    "peak_memory_mb": peak / (1024.0 * 1024.0),
                    "tracked_by": "process_rss_peak",
                }
            except Exception:
                return {
                    "current_memory_mb": 0.0,
                    "peak_memory_mb": 0.0,
                    "tracked_by": "unavailable",
                }
        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_memory_mb": current / (1024.0 * 1024.0),
            "peak_memory_mb": peak / (1024.0 * 1024.0),
            "tracked_by": "tracemalloc",
        }

    def stop(self) -> dict[str, float]:
        result = self.snapshot()
        if self._owned and tracemalloc.is_tracing():
            tracemalloc.stop()
        return result
