"""Real failed frame: input order, complete merge pruning and unlimited search."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from solver2 import PuzzlePiece, PlanningOptions, plan_sheet_moves
from solver2._engine.runtime_config import RuntimeConfig
from solver2._engine.solver import search, partition_search as partition
from solver2._engine.solver.four_piece_topology import _leaf
from solver2._engine.solver.output_quality import rectangle_fill_ratio
from solver2._engine.solver.preprocess import preprocess_observations
from solver2._engine.transforms import estimate_rigid

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures'


@pytest.mark.parametrize('order', [(0,1,2,3),(3,0,1,2),(2,3,0,1),(3,2,1,0)])
def test_real_frame_solves_without_changing_piece_geometry(order):
    data = json.loads((FIXTURES/'frame_20691.json').read_text())
    pieces = tuple(PuzzlePiece(p['id'],tuple(map(tuple,p['vertices_mm']))) for p in data['pieces'])
    plan = plan_sheet_moves(tuple(pieces[i] for i in order),
                            PlanningOptions(noise_bound_mm=2, competition_rules=True))
    assert {move.piece_id for move in plan.moves} == {'1','2','3','4'}
    assert 90 <= plan.target_width_mm <= 120
    assert 50 <= plan.target_height_mm <= 90


def test_final_merge_keeps_rectangle_that_local_cost_previously_discarded():
    pieces = preprocess_observations(json.loads((FIXTURES/'frame_20691.json').read_text()))
    reference = json.loads((FIXTURES/'frame_20691_reference.json').read_text())
    leaves = {p.piece_id:_leaf(p) for p in pieces}

    def error(component):
        actual = np.vstack([component.poses[k][2] for k in component.members])
        expected = np.vstack([reference[k]['vertices'] for k in component.members])
        return estimate_rigid(actual,expected)[2]

    def merge(a,b,limit,complete=False,partial=True):
        return partition._merge_options(a,b,3.,limit,allow_endpoint_partials=partial,
                                        preserve_leaf_edges=True,complete_assembly=complete)

    pair = min(merge(leaves['1'],leaves['2'],1000,partial=False),key=error)
    triple = min(merge(pair,leaves['3'],1000),key=error)
    local = merge(leaves['4'],triple,1000)
    good_index = min(range(len(local)),key=lambda i:error(local[i]))
    assert good_index >= 48  # Historical local-cost cutoff loses this assembly.
    complete = merge(leaves['4'],triple,48,complete=True)
    assert min(error(c) for c in complete) < 2
    assert rectangle_fill_ratio(p[2] for p in complete[0].poses.values()) >= .95


def test_unlimited_search_does_not_honor_an_expired_wall_clock(monkeypatch):
    pieces = preprocess_observations(json.loads((FIXTURES/'frame_20691.json').read_text()))
    deadlines=[]

    def empty(*args, **kwargs):
        deadlines.append(kwargs['deadline'])
        return [],dict(partition_search_expanded_states=0,partition_search_generated_states=0,
                       partition_search_elapsed_seconds=0.)

    monkeypatch.setattr(search,'enumerate_partition_candidates',empty)
    search.solve_random(pieces,deadline=perf_counter()-100,
                        runtime_config=RuntimeConfig(unlimited_search=True))
    assert deadlines == [None,None]


@pytest.mark.parametrize('fill,accepted',[(.94999,False),(.95,True)])
def test_engine_never_emits_moves_below_the_output_threshold(tmp_path,monkeypatch,fill,accepted):
    from solver2._engine.solver import solve
    monkeypatch.setattr(solve,'rectangle_fill_ratio',lambda vertices:fill)
    result=solve.solve_observations(
        FIXTURES/'fixed4_reference.json',tmp_path,print_plan=False,
        runtime_config=dict(execution_profile='production',unlimited_search=True),
    )
    assert result['success'] == accepted
    if not accepted:
        assert result['moves'] == []
        assert result['failure_type'] == 'rectangle_fill_below_minimum'
