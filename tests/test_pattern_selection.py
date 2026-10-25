"""Actual failed card, physical placement diversity and conservative scoring."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vision/src'),str(ROOT)]

from solver.core.models import Candidate, PuzzlePiece
from solver.core.texture import SceneTextureRanker
from ec_26_e.control._engine.appearance.variant_pool import same_assembly,distinct_safe_variants
from ec_26_e.control._engine.runtime_config import RuntimeConfig
from ec_26_e.control._engine.solver.recovery import RecoveryCoordinator
from ec_26_e.control._engine.solver.corner_hypotheses import rectangle_corner_candidates
from ec_26_e.control._engine.solver.preprocess import preprocess_observations
from ec_26_e.control._engine.transforms import transform_points

FIXTURES=ROOT/'vision/tests/fixtures'


def test_actual_card_scores_correct_seams_above_the_false_first_solution():
    data=json.loads((FIXTURES/'frame2489.json').read_text())
    poses=json.loads((FIXTURES/'frame2489_poses.json').read_text())
    pieces=[PuzzlePiece(i,Polygon(p['vertices_mm'])) for i,p in enumerate(data['pieces'])]
    ranker=SceneTextureRanker(FIXTURES/'frame2489_pattern.png',pieces)
    measured={}
    for name,rows in poses.items():
        candidate=Candidate({int(k):np.asarray(v) for k,v in rows.items()},0.,{})
        ranker.score_candidate(candidate)
        measured[name]=candidate.metrics
    correct,wrong=measured['reference'],measured['current']
    assert correct['texture_continuity'] > wrong['texture_continuity']+.05
    assert correct['texture_continuity'] >= .45
    assert correct['texture_min_seam'] >= .4
    assert wrong['texture_min_seam'] < .4
    assert correct['texture_graph_connected'] == 1.


def variant(name,second_rotation=False,angle=0.,offset=(0.,0.),quality=0):
    a=np.array([[0.,0.],[20.,0.],[20.,20.],[0.,20.]])
    b=a+[20.,0.]
    if second_rotation:b=2*b.mean(axis=0)-b
    records=[dict(piece_id=key,target=transform_points(points,angle,offset)) for key,points in [('a',a),('b',b)]]
    return SimpleNamespace(candidate_id=name,variant_id='raw',records=records,
                           safe=True,machine_executable=True,quality_key=(quality,))


def test_same_assembly_collapses_global_rotation_and_small_refits():
    first=variant('first')
    second=variant('refit',angle=180.,offset=(.4,.3))
    assert same_assembly(first,second)
    assert len(distinct_safe_variants([first,second])) == 1


def test_rotating_one_symmetric_piece_is_preserved_for_pattern_comparison():
    first,flipped=variant('first'),variant('flipped',second_rotation=True)
    assert not same_assembly(first,flipped)
    assert len(distinct_safe_variants([first,flipped])) == 2


def test_recovery_continues_until_distinct_safe_pool_is_available():
    coordinator=RecoveryCoordinator(0.,RuntimeConfig(unlimited_search=True,minimum_safe_candidates=2))
    coordinator.consider(variant('first',quality=1))
    assert coordinator.should_continue_candidates()
    coordinator.consider(variant('refit',angle=180.,quality=2))
    assert coordinator.should_continue_candidates()
    better=variant('different',second_rotation=True,quality=0)
    coordinator.consider(better)
    assert not coordinator.should_continue_candidates()
    assert coordinator.incumbent is better


def test_corner_hypotheses_are_rigid_and_do_not_use_reference_poses():
    data=json.loads((FIXTURES/'frame2489.json').read_text())
    pieces=preprocess_observations(data)
    candidates=rectangle_corner_candidates(pieces,maximum_results=4)
    assert candidates
    for candidate in candidates:
        for piece in pieces:
            angle,translation,target=candidate.poses[piece.piece_id]
            np.testing.assert_allclose(transform_points(piece.vertices,angle,translation),target,atol=1e-8)
    data['pieces'][0]['vertices_mm']=[[20.,30.],[60.,30.],[40.,64.6410161514]]
    assert rectangle_corner_candidates(preprocess_observations(data)) == []
