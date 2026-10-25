"""验证固定配对缓存与解析导数保持原有几何求解语义。"""
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
from scipy.optimize import least_squares

from ec_26_e.control._engine.solver import fixed_template_solver, robust_pose
from ec_26_e.control._engine.solver.preprocess import preprocess_observations
from ec_26_e.control._engine.transforms import estimate_rigid, transform_points


def test_cached_pairs_match_uncached_candidates_and_do_not_survive_a_solve():
    payload = json.loads((Path(__file__).parents[2] / 'fixtures/fixed4_reference.json').read_text())
    pieces = preprocess_observations(payload)
    calls = []
    original = fixed_template_solver._best_pose

    def counted(observed, template_id, template, uncertainty, **kwargs):
        calls.append((id(observed), template_id))
        return original(observed, template_id, template, uncertainty, **kwargs)

    with patch.object(fixed_template_solver, '_best_pose', side_effect=counted):
        candidates = fixed_template_solver.solve_fixed(pieces)
    assert len(calls) == len(set(calls))
    assert len(candidates) > 0
    for candidate in candidates:
        for observed, pose in zip(pieces, candidate.poses):
            template = next(item for item in fixed_template_solver.fixed_physical_contours()
                            if item.piece_template_id == pose.template_id)
            reference = original(observed, pose.template_id, np.asarray(template.corner_vertices), 0.0)
            assert reference is not None
            np.testing.assert_array_equal(pose.target_vertices_local, reference.target_vertices_local)
            assert pose.shape_score == reference.shape_score
    calls.clear()
    with patch.object(fixed_template_solver, '_best_pose', side_effect=counted):
        fixed_template_solver.solve_fixed(pieces, uncertainty_mm=0.7)
    assert len(calls) > 0
    assert len(calls) == len(set(calls))


def test_analytic_pose_jacobian_matches_central_differences():
    source = np.array([[0., 0.], [20., 0.], [36., 12.], [0., 20.]])
    target = transform_points(source, 73., (30., 40.))
    target[2] += (2.3, -1.7)
    weights = np.array([0.3, 0.9, 0.5, 1.2])

    def check(fun, x0, *, jac, **kwargs):
        for angle in (-180., -90., 0., 73., 179.9):
            params = np.array([angle, 12., -4.])
            step = 1e-4
            numeric = np.column_stack([
                (fun(params + np.eye(3)[i]*step) - fun(params - np.eye(3)[i]*step))/(2*step)
                for i in range(3)])
            np.testing.assert_allclose(jac(params), numeric, atol=1e-7, rtol=1e-7)
        return least_squares(fun, x0, jac=jac, **kwargs)

    with patch.object(robust_pose, 'least_squares', side_effect=check):
        robust_pose.soft_l1_pose(source, target, estimate_rigid(source, target), weights)


def test_analytic_pose_matches_finite_difference_optimizer_with_outliers():
    rng = np.random.default_rng(20261011)
    for count in (3, 4, 5):
        source = rng.uniform(0., 100., size=(count, 2))
        for angle in (-170., 0., 92.):
            target = transform_points(source, angle, (40., 50.)) + rng.normal(0., 0.3, (count, 2))
            target[0] += (4., -3.)
            weights = robust_pose.automatic_vertex_weights(source, target)
            initial = robust_pose.huber_irls_kabsch(source, target, weights)
            result = robust_pose.soft_l1_pose(source, target, initial, weights)

            def numeric(fun, x0, *, jac, **kwargs):
                return least_squares(fun, x0, jac='2-point', **kwargs)

            with patch.object(robust_pose, 'least_squares', side_effect=numeric):
                reference = robust_pose.soft_l1_pose(source, target, initial, weights)
            np.testing.assert_allclose(transform_points(source, result[0], result[1]),
                                       transform_points(source, reference[0], reference[1]), atol=1e-4)


def test_production_pose_matches_full_comparison_with_noise_and_outliers():
    rng = np.random.default_rng(20261013)
    for count in (3, 4, 6):
        source = rng.uniform(-30., 30., (count, 2))
        for outlier in (0., 2., 12.):
            target = transform_points(source, 73., (40., 50.)) + rng.normal(0., .1, (count, 2))
            target[0] += (outlier, -outlier)
            reference = robust_pose.compare_pose_estimators(source, target)
            actual = robust_pose.compare_pose_estimators(source, target, full_diagnostics=False)
            assert actual[0] == reference[0]
            np.testing.assert_array_equal(actual[1], reference[1])
            assert actual[2:4] == reference[2:4]
            if len(actual[4]) > 1:
                assert actual[4] == reference[4]


def test_production_skips_discarded_estimators_but_preserves_full_diagnostics():
    source = np.array([[0., 0.], [30., 0.], [27., 20.], [0., 20.]])
    target = transform_points(source, 39., (40., 50.))
    target += np.array([[.1,.1],[-.1,.1],[-.1,-.1],[.1,-.1]])
    with patch.object(robust_pose, 'soft_l1_pose', side_effect=AssertionError('unused optimization')):
        actual = robust_pose.compare_pose_estimators(source, target, full_diagnostics=False)
    assert actual[3] == 'kabsch'
    assert set(actual[4]) == {'kabsch'}
    assert len(robust_pose.compare_pose_estimators(source, target)[4]) == 5
