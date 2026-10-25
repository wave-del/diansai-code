"""Public random backend propagation, failures, numerical contracts and contest rules."""
import numpy as np
import pytest
from solver2 import PlanningOptions, PuzzlePiece, PlanningError, plan_sheet_moves
from solver2.acceleration import CpuRigidFitBackend
from solver2._engine.generator.random_generator import random_pieces


class RecordingBackend(CpuRigidFitBackend):
    def __init__(self, fault=None):
        self.sizes = []
        self.fault = fault

    def fit_batch(self, requests):
        self.sizes.append(len(requests))
        if self.fault == 'timeout':
            raise TimeoutError('test PL timeout')
        result = super().fit_batch(requests)
        return result[:-1] if self.fault == 'short' else result


def pieces(count):
    generated, _, _ = random_pieces(np.random.default_rng(2600+count), piece_count=count,
        target_width=100, target_height=70, topology={2:'two_polyline',3:'three_y',4:'four_fan'}[count])
    return tuple(PuzzlePiece(p.piece_id, tuple(map(tuple,p.vertices))) for p in generated)


@pytest.mark.parametrize('count', [2,3,4])
def test_public_recovery_path_reaches_batched_accelerator(count):
    backend = RecordingBackend()
    plan = plan_sheet_moves(pieces(count), PlanningOptions(competition_rules=True), rigid_backend=backend)
    assert len(plan.moves) == count
    assert backend.sizes and max(backend.sizes) <= 64
    assert plan.target_width_mm == pytest.approx(100, abs=.002)


@pytest.mark.parametrize('fault', ['timeout','short'])
def test_transport_failure_blocks_plan_and_clears_session(fault):
    backend = RecordingBackend(fault)
    with pytest.raises(PlanningError):
        plan_sheet_moves(pieces(2), rigid_backend=backend)
    calls = len(backend.sizes)
    assert len(plan_sheet_moves(pieces(2)).moves) == 2
    assert len(backend.sizes) == calls


def test_contest_short_edges_rejected_before_backend_access():
    backend = RecordingBackend()
    small = (PuzzlePiece('a',((0.,0.),(10.,0.),(10.,60.),(0.,60.))),
             PuzzlePiece('b',((10.,0.),(100.,0.),(100.,60.),(10.,60.))))
    with pytest.raises(PlanningError, match='20 mm'):
        plan_sheet_moves(small, PlanningOptions(competition_rules=True), rigid_backend=backend)
    assert not backend.sizes


def test_competition_dimensions_and_whole_outer_edge():
    from solver2._engine.runtime_config import RuntimeConfig
    from solver2._engine.solver.random_safety import assess_random_assembly
    from solver2._engine.solver.observation_uncertainty import ObservationUncertaintyProfile
    profile = ObservationUncertaintyProfile(1.0, None, None, None, None, 'test', 'high', False)
    config = RuntimeConfig(random_competition_rules=True)
    # Old compatibility mode allowed up to 130 mm; contest mode rejects 128 mm.
    geometry = {'a':np.array([[0,0],[64,0],[64,70],[0,70]],float),
                'b':np.array([[64,0],[128,0],[128,70],[64,70]],float)}
    result = assess_random_assembly(geometry,profile,config,expected_piece_count=2)
    assert not result.accepted
    assert 'illegal_target_dimensions' in result.rejection_reasons


def test_relaxed_short_edge_threshold_also_applies_to_whole_outer_edge():
    from solver2._engine.runtime_config import RuntimeConfig
    from solver2._engine.solver.random_safety import assess_random_assembly
    from solver2._engine.solver.observation_uncertainty import ObservationUncertaintyProfile
    profile=ObservationUncertaintyProfile(1.0,None,None,None,None,'test','high',False)
    # A 10 mm corner triangle and the remaining pentagon exactly fill 100 x 60.
    geometry={'corner':np.array([[0,0],[10,0],[0,10]],float),
              'rest':np.array([[10,0],[100,0],[100,60],[0,60],[0,10]],float)}
    strict=assess_random_assembly(geometry,profile,RuntimeConfig(random_competition_rules=True),expected_piece_count=2)
    relaxed=assess_random_assembly(geometry,profile,RuntimeConfig(
        random_competition_rules=True,minimum_piece_edge_mm=10.),expected_piece_count=2)
    assert 'piece_without_target_outer_edge' in strict.rejection_reasons
    assert relaxed.accepted,relaxed.rejection_reasons
    assert relaxed.rectangle_fill_ratio==pytest.approx(1.)


@pytest.mark.parametrize('surface,safe,executable,expected', [
    ('white', True, True, False),
    ('white', False, True, True),
    ('white', True, False, True),
    ('patterned', True, True, True),
])
def test_candidate_early_stop_requires_complete_shape_solution(surface, safe, executable, expected):
    from types import SimpleNamespace
    from solver2._engine.runtime_config import RuntimeConfig
    from solver2._engine.solver.recovery import RecoveryCoordinator
    coordinator = RecoveryCoordinator(0.0, RuntimeConfig(), surface_mode=surface,
                                      monotonic_fn=lambda: 0.0)
    coordinator.incumbent = SimpleNamespace(safe=safe, machine_executable=executable)
    assert coordinator.should_continue_candidates() is expected


def test_four_piece_search_uses_geometry_quality_over_float_tie_noise():
    from solver2._engine.solver.four_piece_topology import _stable_partial_rank
    good = (1e-8, .05, 2.)
    bad = (-1e-8, .30, 1.)
    assert _stable_partial_rank(good) < _stable_partial_rank(bad)
    assert _stable_partial_rank((0., .05, 2.00001)) == _stable_partial_rank(good)


def test_fully_valid_candidate_outranks_a_better_scoring_diagnostic_candidate():
    from types import SimpleNamespace
    from solver2._engine.runtime_config import RuntimeConfig
    from solver2._engine.solver.recovery import RecoveryCoordinator
    coordinator = RecoveryCoordinator(0.0, RuntimeConfig(), monotonic_fn=lambda: 0.0)
    diagnostic = SimpleNamespace(candidate_id='a', variant_id='raw', safe=False,
                                 machine_executable=True, quality_key=(0.,))
    accepted = SimpleNamespace(candidate_id='a', variant_id='optimized', safe=True,
                               machine_executable=True, quality_key=(1.,))
    coordinator.consider(diagnostic)
    assert coordinator.should_continue_candidates()
    coordinator.consider(accepted)
    coordinator.consider(diagnostic)
    assert coordinator.incumbent is accepted
    assert coordinator.safe_candidates[0] is accepted
    assert not coordinator.should_continue_candidates()
