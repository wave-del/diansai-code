"""正式随机入口的几何、执行门和现场花纹接线回归；不访问设备。"""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vision/src"), str(ROOT / "handover-puzzle"), str(ROOT)]

from solver2 import PlanningError, PlanningOptions, PuzzlePiece
from solver2._engine.transforms import transform_points
from random_white_solver_adapter import solve_random_white
import random_patterned_solver_adapter as patterned
from task1_yolo_flow import _plan_with_artifacts


PIECES = (
    PuzzlePiece("left", ((10., 10.), (60., 10.), (60., 70.), (10., 70.))),
    PuzzlePiece("right", ((100., 10.), (150., 10.), (150., 70.), (100., 70.))),
)


def assert_complete_plan(result):
    assert result["success"] and result["executable"]
    assert result["competition_gate_passed"] and not result["forced_output"]
    assert result["solver_backend"] == "directed_edge_chain_reference"
    assert {m["piece_id"] for m in result["moves"]} == {p.piece_id for p in PIECES}
    polygons = []
    for move in result["moves"]:
        observed = move["observed_vertices_a4_mm"]
        target = move["target_vertices_a4_mm"]
        angle = move["rotation_deg_clockwise"]
        translation = move["rigid_transform"]["translation_a4_mm"]
        np.testing.assert_allclose(transform_points(observed, angle, translation), target, atol=1e-6)
        polygons.append(Polygon(target))
    assert polygons[0].intersection(polygons[1]).area <= 1e-6
    assert polygons[0].distance(polygons[1]) >= 3.0 - 1e-6


def test_white_entry_uses_reference_engine_and_checks_rigid_moves(tmp_path):
    result = solve_random_white(PIECES, tmp_path, timeout_s=15)
    assert_complete_plan(result)
    saved = json.loads((tmp_path / "solution.json").read_text())
    assert saved["solver_backend"] == result["solver_backend"]


@pytest.mark.parametrize('kwargs,target', [({},2),({'candidate_target':1},1)])
def test_white_candidate_target_preserves_default_and_allows_web_early_stop(tmp_path, monkeypatch, kwargs, target):
    import random_white_solver_adapter as white
    calls=[]
    def solve(*args, **options):
        calls.append(options)
        return {'success':False,'moves':[]}
    monkeypatch.setattr(white,'solve_observations',solve)
    white.solve_random_white(PIECES,tmp_path,**kwargs)
    config=calls[0]['runtime_config']
    assert config['minimum_safe_candidates']==target
    assert config['unlimited_search']
    assert config['execution_policy']=='strict'
    assert calls[0]['appearance_scorer'] is None


@pytest.mark.parametrize('target', [0,-1,1.5,True])
def test_white_rejects_invalid_candidate_target_before_solving(tmp_path,target):
    with pytest.raises(ValueError,match='candidate_target'):
        solve_random_white(PIECES,tmp_path,candidate_target=target)


def test_task_entry_converts_reference_result_to_existing_move_plan(tmp_path):
    plan = _plan_with_artifacts(
        PIECES, PlanningOptions(mode="random", noise_bound_mm=2), tmp_path,
        use_new_random_solver=True,
    )
    assert len(plan.moves) == 2
    assert all(0 <= move.rotation_deg_clockwise < 360 for move in plan.moves)


def test_no_valid_rectangle_returns_failure_instead_of_transport_fallback(tmp_path):
    tiny = (
        PuzzlePiece("a", ((0., 0.), (20., 0.), (0., 20.))),
        PuzzlePiece("b", ((50., 0.), (70., 0.), (50., 20.))),
    )
    result = solve_random_white(tiny, tmp_path, timeout_s=8)
    assert not result["success"] and not result["executable"]
    assert result["moves"] == []


def test_contest_constraints_reject_short_edges_before_search(tmp_path):
    short = (PuzzlePiece("a", ((0., 0.), (5., 0.), (0., 5.))), PIECES[1])
    with pytest.raises(PlanningError, match="20 mm"):
        solve_random_white(short, tmp_path)


@pytest.mark.parametrize('length,noise,threshold,accepted', [
    (11.229,2.,20.,False), (11.229,2.,10.,True),
    (5.99,2.,10.,False), (6.,2.,10.,True),
    (9.99,0.,10.,False), (10.,0.,10.,True),
])
def test_configurable_short_edge_threshold_includes_endpoint_uncertainty(
    tmp_path,monkeypatch,length,noise,threshold,accepted,
):
    import random_white_solver_adapter as white
    short=(PuzzlePiece('short',((0.,0.),(length,0.),(length,60.),(0.,60.))),PIECES[1])
    calls=[]
    def solve(*args,**options):
        calls.append(options)
        return {'success':False,'moves':[]}
    monkeypatch.setattr(white,'solve_observations',solve)
    kwargs=dict(noise_bound_mm=noise,minimum_piece_edge_mm=threshold)
    if accepted:
        result=white.solve_random_white(short,tmp_path,**kwargs)
        assert calls[0]['runtime_config']['minimum_piece_edge_mm']==threshold
        assert result['minimum_piece_edge_mm']==threshold
    else:
        with pytest.raises(PlanningError,match=f'{threshold:g} mm'):
            white.solve_random_white(short,tmp_path,**kwargs)
        assert not calls


def test_missing_pattern_image_blocks_execution(tmp_path):
    result = patterned.solve_random_patterned(PIECES, tmp_path, scene_path=None, timeout_s=15)
    assert result["success"]
    assert not result["executable"]
    assert "appearance_ambiguity" in result["execution_block_reasons"]
    assert not result["pattern_texture"]["resolved"]


@pytest.mark.parametrize("min_segment,executable", [(0.8, True), (0.2, False)])
def test_pattern_scores_are_applied_before_final_execution_gate(tmp_path, monkeypatch, min_segment, executable):
    class Ranker:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def score_candidate(self, candidate):
            # Texture receives transforms, never a reflected/scaled proxy.
            for matrix in candidate.poses.values():
                assert np.linalg.det(matrix[:2, :2]) == pytest.approx(1)
            score = 0.95 - 0.2 * self.calls
            self.calls += 1
            candidate.metrics.update(texture_information=1., texture_seam_count=1.,
                                     texture_reference_continuity=score, texture_min_segment=min_segment)

        def render_planned_reconstruction(self, records, path, **kwargs):
            assert len(records) == 2
            assert Polygon(records[0]["target"]).distance(Polygon(records[1]["target"])) >= 3 - 1e-6
            cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
            return path

        def render_reconstruction(self, candidate, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
            return path

    monkeypatch.setattr(patterned, "SceneTextureRanker", Ranker)
    scene = tmp_path / "scene.png"
    cv2.imwrite(str(scene), np.zeros((8, 8, 3), np.uint8))
    result = patterned.solve_random_patterned(PIECES, tmp_path, scene_path=scene, timeout_s=15)
    if executable:
        assert_complete_plan(result)
    else:
        assert not result["executable"]
        assert "appearance_ambiguity" in result["execution_block_reasons"]
    assert result["pattern_texture"]["used"]
    assert result["pattern_texture"]["resolved"] == executable
    assert Path(result["pattern_reconstruction_path"]).is_file()


def test_real_blank_scene_does_not_claim_pattern_alignment(tmp_path):
    scene = tmp_path / "blank.png"
    cv2.imwrite(str(scene), np.full((584, 414, 3), 255, np.uint8))
    result = patterned.solve_random_patterned(PIECES, tmp_path, scene_path=scene, timeout_s=15)
    assert result["success"] and not result["executable"]
    assert not result["pattern_texture"]["resolved"]
    assert result["pattern_texture"]["reason"] == "现场拼缝花纹信息不足"


def test_pattern_selector_excludes_geometry_rejected_variants(monkeypatch):
    from solver2._engine.appearance import policy
    from solver2._engine.runtime_config import RuntimeConfig

    accepted = SimpleNamespace(safe=True, machine_executable=True, candidate_id="valid", quality_key=(1,))
    rejected = SimpleNamespace(safe=False, machine_executable=True, candidate_id="invalid", quality_key=(0,))
    monkeypatch.setattr(policy, "_descriptor", lambda variant: None)
    monkeypatch.setattr(policy, "detect_geometry_ambiguity", lambda *args, **kwargs: SimpleNamespace(detected=False))
    seen = []

    def scorer(variants):
        seen.extend(variants)
        return {"candidate_scores": {variant.candidate_id: 1.0 for variant in variants}}

    selected = policy.resolve_patterned_variants(
        {}, [rejected, accepted], rejected, RuntimeConfig(), appearance_scorer=scorer,
    )
    assert seen == [accepted]
    assert selected.selected_variant is accepted
