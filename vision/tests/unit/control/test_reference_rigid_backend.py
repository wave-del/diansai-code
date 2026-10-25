"""公开批量后端与原有 SVD/鲁棒路径一致，不引入硬件依赖。"""
import json
from pathlib import Path
import numpy as np
import pytest
from ec_26_e.control.acceleration import CpuRigidFitBackend, RigidFitRequest
from ec_26_e.control._engine.solver.fixed_template_solver import solve_fixed
from ec_26_e.control._engine.solver.preprocess import preprocess_observations
from ec_26_e.control._engine.solver.robust_pose import weighted_kabsch
from ec_26_e.control._engine.transforms import estimate_rigid, transform_points


@pytest.mark.parametrize('angle', [-180., -179.999, 0., 179.999])
@pytest.mark.parametrize('weighted', [False, True])
def test_reference_matches_existing_svd_with_noise_and_outlier(angle, weighted):
    source = np.array([[0., 0.], [23., 2.], [16., 8.], [19., 32.], [0., 30.]])
    target = transform_points(source, angle, (40, 70))
    target[1] += (5, -3)
    weights = np.array([.2, .7, .9, .3, 1.]) if weighted else None
    actual = CpuRigidFitBackend().fit_batch([RigidFitRequest(source, target, weights)])[0]
    expected = weighted_kabsch(source, target, weights) if weighted else estimate_rigid(source, target)
    assert actual.angle == expected[0] and actual.rms == expected[2]
    np.testing.assert_array_equal(actual.translation, expected[1])


def test_reference_degenerate_and_reflection_do_not_invent_scaling():
    for source in (np.ones((4, 2)), np.array([[0., 0.], [10., 0.], [9., 8.], [0., 4.]])):
        target = source * [-1., 1.]
        expected = estimate_rigid(source, target)
        actual = CpuRigidFitBackend().fit_batch([RigidFitRequest(source, target)])[0]
        assert actual.angle == expected[0]
        np.testing.assert_array_equal(actual.translation, expected[1])


def test_fixed_backend_is_one_logical_batch_and_keeps_candidate_order():
    payload = json.loads((Path(__file__).resolve().parents[2] / 'fixtures/fixed4_reference.json').read_text())
    pieces = preprocess_observations(payload)
    calls = []

    class Recording(CpuRigidFitBackend):
        def fit_batch(self, requests):
            calls.append(len(requests))
            return super().fit_batch(requests)

    expected = solve_fixed(pieces)
    actual = solve_fixed(pieces, rigid_backend=Recording())
    assert len(calls) == 1 and calls[0] > 1
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.total == b.total
        for p, q in zip(a.poses, b.poses):
            assert (p.template_id, p.estimator) == (q.template_id, q.estimator)
            np.testing.assert_array_equal(p.target_vertices_local, q.target_vertices_local)


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_rejected(bad):
    source = np.array([[0., 0.], [bad, 1.]])
    with pytest.raises(ValueError):
        CpuRigidFitBackend().fit_batch([RigidFitRequest(source, source)])
