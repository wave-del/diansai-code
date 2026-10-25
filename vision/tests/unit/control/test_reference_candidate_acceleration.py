"""候选距离粗筛必须保留原最优解；浮点保护带与错误不能放宽执行检查。"""
from itertools import combinations
import numpy as np
import pytest
from shapely.geometry import Polygon

from ec_26_e.control import plan_sheet_moves, PlanningOptions, PlanningError
from ec_26_e.control.candidate_acceleration import candidate_acceleration, scan_candidate_distances, cannot_improve
from .test_reference_random_acceleration import pieces


class ReferenceBackend:
    def __init__(self, error=0, fault=None):
        self.error, self.fault, self.calls = error, fault, []

    def pair_distances(self, vertices, deltas):
        self.calls.append(len(deltas))
        if self.fault == 'timeout':
            raise TimeoutError('candidate test timeout')
        result = []
        for shifts in deltas:
            polygons = [Polygon(v+d) for v, d in zip(vertices, shifts)]
            result.append([a.distance(b) for a, b in combinations(polygons, 2)])
        result = np.maximum(0, np.array(result)+self.error)
        if self.fault == 'short':
            result = result[:-1]
        if self.fault == 'nan':
            result[0, 0] = np.nan
        return result


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('error', [-.02, 0, .02])
def test_public_plan_identical_including_guard_extremes(count, error):
    options = PlanningOptions(competition_rules=True)
    cpu = plan_sheet_moves(pieces(count), options)
    backend = ReferenceBackend(error)
    accelerated = plan_sheet_moves(pieces(count), options, candidate_backend=backend)
    assert accelerated.to_dict() == cpu.to_dict()
    assert backend.calls and all(n == 190 for n in backend.calls)


@pytest.mark.parametrize('fault', ['timeout', 'short', 'nan'])
def test_failure_blocks_plan_and_context_is_released(fault):
    backend = ReferenceBackend(fault=fault)
    with pytest.raises(PlanningError):
        plan_sheet_moves(pieces(2), candidate_backend=backend)
    assert scan_candidate_distances([], []) is None


def test_lower_bound_never_discards_a_better_distance_score():
    rng = np.random.default_rng(44)
    for _ in range(2000):
        best_gaps, gaps = rng.uniform(0, 8, (2, 6))
        def score(g):
            return (0, 0, float(np.sum(np.maximum(0, 2-g)**2+np.maximum(0, g-7)**2)),
                    float(np.sum((g-4)**2)), 0, 0)
        best = score(best_gaps)
        measured = np.maximum(0, gaps+rng.uniform(-.02, .02, 6))
        if cannot_improve(measured, list(range(6)), best, 2, 7, 4):
            assert score(gaps) >= best
    assert not cannot_improve(np.array([20]), [0], (1, 0, 0, 0), 2, 7, 4)
    assert not cannot_improve(np.array([20]), [0], (0, 1, 0, 0), 2, 7, 4)
