"""低开销、可嵌套的阶段计时器。"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

SOLVER_STAGES = (
    "input_read", "preprocess", "edge_feature_build",
    "observation_uncertainty",
    "atomic_interval_build", "interface_candidate_build",
    "interface_generation", "interface_ranking",
    "pose_hypothesis_build", "initial_search", "generic_partial_fallback",
    "search_expansion", "placed_interface_discovery", "canonicalization",
    "candidate_deduplication", "raw_geometry_check",
    "exact_geometry_scoring", "candidate_ranking", "local_optimization",
    "global_optimization", "soft_l1", "clearance",
    "uncertainty_estimation", "robust_safety", "motion_validation",
    "motion_planning", "anchor_switch", "recovery_overhead",
    "final_revalidation", "output",
    "json_output", "visualization", "total",
)


@dataclass
class StageTimer:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.totals[name] += perf_counter() - started
            self.calls[name] += 1

    def add(self, name: str, seconds: float, calls: int = 1) -> None:
        self.totals[name] += float(seconds)
        self.calls[name] += int(calls)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        names = list(SOLVER_STAGES)
        names.extend(sorted(set(self.totals) - set(names)))
        return {
            name: {
                "seconds": float(self.totals.get(name, 0.0)),
                "calls": int(self.calls.get(name, 0)),
            }
            for name in names
        }
