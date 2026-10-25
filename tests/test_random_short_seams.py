"""实拍短边与多段共线接缝：随机路径保持刚体、95% 和 3～6 mm 留缝。"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'vision/src'), str(ROOT/'handover-puzzle')]

from ec_26_e.control import PuzzlePiece
from ec_26_e.control._engine.solver import search, solve
from ec_26_e.control._engine.solver.partition_search import _turn_residual
from ec_26_e.control._engine.transforms import rotation_matrix, transform_points
from random_white_solver_adapter import solve_random_white


def test_boundary_jog_within_observation_error_is_not_a_real_seam_corner():
    chain = np.array([[0.,0.],[20.,0.],[21.,-1.],[100.,0.]])
    straight = np.array([[0.,0.],[100.,0.]])
    original = chain.copy()
    assert _turn_residual(chain,straight) > 28
    assert _turn_residual(chain,straight,3.) == pytest.approx(0.)
    np.testing.assert_array_equal(chain,original)


def test_actual_polyline_corner_outside_observation_error_remains_incompatible():
    corner = np.array([[0.,0.],[20.,0.],[20.,20.]])
    straight = np.array([[0.,0.],[20.,20.]])
    assert _turn_residual(corner,straight,3.) > 28


def test_refinement_cannot_accept_an_infeasible_gap_and_endpoint_limit():
    from ec_26_e.control._engine.solver.random_clearance import plan_random_clearance
    from ec_26_e.control._engine.solver.random_safety import reconstruct_random_topology
    vertices={'left':np.array([[0,0],[50,0],[50,60],[0,60]],float),
              'right':np.array([[50,0],[100,0],[100,60],[50,60]],float)}
    topology=reconstruct_random_topology(vertices,tolerance_mm=1.)
    result=plan_random_clearance(vertices,topology,target_gap_mm=5.,minimum_gap_mm=3.,
        maximum_gap_mm=6.,maximum_vertex_distance_mm=.5,measurement_error_radius_mm=0.,
        machine_position_margin_mm=0.,overlap_tolerance_mm2=1e-6)
    # Opening a >=3 mm gap cannot keep corresponding endpoints <=0.5 mm apart.
    assert not result.feasible
    assert result.failure_reason=='random_joint_clearance_infeasible'


def test_repairable_high_fill_proposal_outranks_layout_below_output_gate():
    from ec_26_e.control._engine.solver.exact_candidate_scoring import exact_geometry_objective
    def metrics(iou,overlap):
        return dict(rectangle_legal=True,connected_with_tolerance_10mm=True,
                    target_area_mm2=10000.,iou=iou,overlap_area_mm2=overlap,
                    outside_area_mm2=0.,gap_area_mm2=(1.-iou)*10000.)
    incomplete=metrics(.9407,0.)
    repairable=metrics(.9838,110.)
    excessive_overlap=metrics(.99,500.)
    assert exact_geometry_objective(repairable) < exact_geometry_objective(incomplete)
    assert exact_geometry_objective(incomplete) < exact_geometry_objective(excessive_overlap)


@pytest.mark.parametrize('scrambled', [False,True])
@pytest.mark.parametrize('fixture', ['frame_17412_short_seams.json','frame_29121_short_seams.json'])
def test_real_short_seams_solve_randomly_without_fixed_template_fallback(tmp_path,monkeypatch,scrambled,fixture):
    data=json.loads((ROOT/'vision/tests/fixtures'/fixture).read_text())
    pieces=[]
    for index,piece in enumerate(data['pieces']):
        points=np.asarray(piece['vertices_mm'])
        if scrambled:
            center=points.mean(axis=0)
            points=(points-center)@rotation_matrix((12.,127.,-65.,203.)[index]).T+center
        piece_id=f'scrambled_{index}' if scrambled else piece['id']
        pieces.append(PuzzlePiece(piece_id,tuple(map(tuple,points))))
    if scrambled:
        pieces=[pieces[i] for i in (1,3,0,2)]
    def fixed_is_forbidden(*args,**kwargs):
        raise AssertionError('随机求解不能切换为固定模板')
    monkeypatch.setattr(solve,'solve_fixed',fixed_is_forbidden)
    original=search.enumerate_partition_candidates
    phases=[]
    def enumerate_candidates(*args,**kwargs):
        phases.append(kwargs['strict_only'])
        return original(*args,**kwargs)
    monkeypatch.setattr(search,'enumerate_partition_candidates',enumerate_candidates)
    result=solve_random_white(pieces,tmp_path,timeout_s=None,candidate_target=1,minimum_piece_edge_mm=10.)
    assert result['success'] and result['geometry_gate_passed'],result.get('failure_reason')
    assert result['rectangle_fill_ratio'] >= .95
    assert not result['forced_output']
    assert phases == [True]  # 首次边链阶段即完整验证，不再进入耗时兜底搜索。
    assert {m['piece_id'] for m in result['moves']} == {p.piece_id for p in pieces}
    polygons=[]
    for move in result['moves']:
        np.testing.assert_allclose(transform_points(
            move['observed_vertices_a4_mm'],move['rotation_deg_clockwise'],
            move['rigid_transform']['translation_a4_mm'],
        ),move['target_vertices_a4_mm'],atol=1e-6)
        polygons.append(Polygon(move['target_vertices_a4_mm']))
    for i,first in enumerate(polygons):
        for second in polygons[i+1:]:
            assert first.intersection(second).area <= 1e-6
    assert all(3.-1e-7 <= seam['nominal_normal_gap_mm'] <= 6.+1e-7
               for seam in result['clearance']['interface_metrics'])
