"""Measured rectangular pieces must reach edge-chain search before corner products."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vision/src'),str(ROOT/'handover-puzzle')]
from solver2 import PuzzlePiece
from solver2._engine.runtime_config import RuntimeConfig
from solver2._engine.solver import corner_hypotheses as corner, search
from solver2._engine.solver.preprocess import preprocess_observations
from solver2._engine.transforms import transform_points
from random_white_solver_adapter import solve_random_white

FIXTURES=ROOT/'vision/tests/fixtures'


@pytest.mark.parametrize('filename', ['web_four_grid.json','web_four_strips.json','web_four_mixed.json'])
def test_safe_edge_chains_finish_without_building_corner_pool(filename,tmp_path,monkeypatch):
    data=json.loads((FIXTURES/filename).read_text())
    pieces=tuple(PuzzlePiece(p['id'],tuple(map(tuple,p['vertices_mm']))) for p in data['pieces'])
    def corners_not_needed(*args,**kwargs):
        raise AssertionError('whole-chain solution must be checked before corner enumeration')
    monkeypatch.setattr(corner,'rectangle_corner_candidates',corners_not_needed)
    result=solve_random_white(pieces,tmp_path,candidate_target=1,minimum_piece_edge_mm=10.)
    assert result['success'] and result['geometry_gate_passed'],result.get('failure_reason')
    assert result['rectangle_fill_ratio']>=.95
    assert {m['piece_id'] for m in result['moves']}=={p.piece_id for p in pieces}
    polygons=[]
    for move in result['moves']:
        np.testing.assert_allclose(transform_points(move['observed_vertices_a4_mm'],
            move['rotation_deg_clockwise'],move['rigid_transform']['translation_a4_mm']),
            move['target_vertices_a4_mm'],atol=1e-7)
        polygons.append(Polygon(move['target_vertices_a4_mm']))
    for i,p in enumerate(polygons):
        for q in polygons[:i]:assert p.intersection(q).area<=1e-6
    assert all(3.-1e-7<=m['nominal_normal_gap_mm']<=6.+1e-7
               for m in result['clearance']['interface_metrics'])


def test_rejected_chains_still_reach_supplementary_corner_proposals(monkeypatch):
    pieces=preprocess_observations(json.loads((FIXTURES/'web_four_grid.json').read_text()))
    events=[]
    def partition(*args,**kwargs):
        events.append('strict' if kwargs['strict_only'] else 'partial')
        return [],dict(partition_search_expanded_states=0,partition_search_generated_states=0,
                       partition_search_elapsed_seconds=0.)
    def corners(*args,**kwargs):events.append('corners');return []
    monkeypatch.setattr(search,'enumerate_partition_candidates',partition)
    monkeypatch.setattr(corner,'rectangle_corner_candidates',corners)
    result=search.solve_random(pieces,runtime_config=RuntimeConfig(
        unlimited_search=True,rectangle_corner_hypotheses=True,random_competition_rules=True))
    assert result==[]
    assert events==['strict','partial','corners']


def test_corner_products_reuse_geometry_but_preserve_material_rotations():
    pieces=preprocess_observations(json.loads((FIXTURES/'web_four_grid.json').read_text()))
    def fit(objective,initial,args,**kwargs):
        return SimpleNamespace(x=initial,fun=objective(initial,*args))
    with patch.object(corner,'minimize',side_effect=fit),patch.object(corner,'union_all',wraps=corner.union_all) as union:
        layouts=corner.rectangle_corner_candidates(pieces,maximum_results=4)
    assert layouts
    # The old 24 * 4**4 * 36 Cartesian product needed 221184 union operations.
    # Equivalent opposite-vertex anchors and global 180-degree rotations are
    # redundant proposals. Bound work, rather than asserting a wall-clock time.
    assert union.call_count<=12*2**4*36+4
    for layout in layouts:
        for piece in pieces:
            angle,translation,target=layout.poses[piece.piece_id]
            np.testing.assert_allclose(transform_points(piece.vertices,angle,translation),target,atol=1e-8)
    # Equal rectangular outlines can still have different printed orientations.
    # Sharing the objective must not collapse these distinct rigid transforms.
    assert any(
        all(Polygon(a.poses[p.piece_id][2]).hausdorff_distance(Polygon(b.poses[p.piece_id][2]))<1e-7
            for p in pieces)
        and any(abs(((a.poses[p.piece_id][0]-b.poses[p.piece_id][0])%360)-180)<1e-7 for p in pieces)
        for i,a in enumerate(layouts) for b in layouts[:i])


def test_corner_supplement_respects_an_expired_generation_deadline():
    pieces=preprocess_observations(json.loads((FIXTURES/'web_four_grid.json').read_text()))
    with patch.object(corner,'union_all',side_effect=AssertionError('deadline already expired')):
        assert corner.rectangle_corner_candidates(pieces,deadline=time.perf_counter()-1)==[]
