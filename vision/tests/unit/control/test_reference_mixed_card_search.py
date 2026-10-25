"""真实混合边数纸片：接近快速 IoU 筛选边界也必须及时完整验证。"""
import json
from pathlib import Path

import pytest
import numpy as np
from shapely.geometry import Polygon

from solver2 import PuzzlePiece, PlanningOptions, plan_sheet_moves
from solver2._engine.solver import search


@pytest.mark.parametrize('index', [0, 3, 25])
def test_mixed_real_cards_validate_before_expensive_partial_search(index, monkeypatch):
    rows = json.loads((Path(__file__).parents[2] / 'fixtures/poker_mixed_30.json').read_text())['observations']
    pieces = tuple(PuzzlePiece(p['id'], tuple(map(tuple, p['vertices_mm']))) for p in rows[index]['pieces'])
    original = search.enumerate_partition_candidates
    phases = []
    def enumerate_candidates(*args, **kwargs):
        phases.append(kwargs['strict_only'])
        return original(*args, **kwargs)
    monkeypatch.setattr(search, 'enumerate_partition_candidates', enumerate_candidates)
    plan = plan_sheet_moves(pieces, PlanningOptions(competition_rules=True, noise_bound_mm=2))
    assert len(plan.moves) == 4 and plan.confidence >= .6
    assert 90 <= plan.target_width_mm <= 120
    assert 50 <= plan.target_height_mm <= 90
    assert phases == [True]


def test_empty_strict_probe_resumes_general_topology_search(monkeypatch):
    from solver2._engine.solver.preprocess import preprocess_observations
    row = json.loads((Path(__file__).parents[2] / 'fixtures/poker_mixed_30.json').read_text())['observations'][0]
    pieces = preprocess_observations(dict(schema_version='1.0', pieces=row['pieces']))
    phases = []
    def empty(*args, **kwargs):
        phases.append(kwargs['strict_only'])
        return [], dict(partition_search_expanded_states=0, partition_search_generated_states=0,
                        partition_search_elapsed_seconds=0.)
    monkeypatch.setattr(search, 'enumerate_partition_candidates', empty)
    assert search.solve_random(pieces) == []
    assert phases == [True, False]


def test_vectorized_bounding_rectangle_matches_exact_geometry():
    from solver2._engine.solver.partition_search import _minimum_rectangle_sides
    rng = np.random.default_rng(2210)
    for _ in range(200):
        hull = Polygon(rng.normal(size=(15, 2))*40 + rng.uniform(-1e4, 1e4, (1,2))).convex_hull
        short, long = _minimum_rectangle_sides(hull)
        reference = hull.minimum_rotated_rectangle
        assert short*long == pytest.approx(reference.area, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize('fixture', ['poker_dimension_boundary.json', 'poker_candidate_priority.json'])
def test_dimension_boundary_is_constrained_in_optimizer_not_relaxed_in_gate(monkeypatch, fixture):
    row = json.loads((Path(__file__).parents[2] / 'fixtures' / fixture).read_text())
    pieces = tuple(PuzzlePiece(p['id'], tuple(map(tuple,p['vertices_mm']))) for p in row['pieces'])
    original = search.enumerate_partition_candidates
    def strict_first(*args, **kwargs):
        assert kwargs['strict_only'], 'Valid bounded fit should stop before general tree search'
        return original(*args, **kwargs)
    monkeypatch.setattr(search,'enumerate_partition_candidates',strict_first)
    plan = plan_sheet_moves(pieces, PlanningOptions(competition_rules=True,noise_bound_mm=2))
    assert len(plan.moves)==4
    assert 90 <= plan.target_width_mm <= 120
    assert 50 <= plan.target_height_mm <= 90
