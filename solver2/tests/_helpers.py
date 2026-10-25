"""Exercise standalone solver2 with the same geometry options as the web worker."""
import json
import logging

from solver2._engine.solver.solve import solve_observations


def solve_random_white(pieces, output_dir, *, timeout_s=None, candidate_target=1,
                       minimum_piece_edge_mm=10.):
    assert timeout_s is None
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir/'observations.json'
    source.write_text(json.dumps(dict(schema_version='1.0',mode='random',surface_mode='white',
        noise_bound_mm=2.,search_top_n=100,pieces=[dict(id=p.piece_id,vertices_mm=p.vertices_mm) for p in pieces])))
    try:
        return solve_observations(source, output_dir, print_plan=False, runtime_config={
            'execution_profile':'production','execution_policy':'strict','top_n':100,
            'minimum_execution_confidence':.6,'random_competition_rules':True,
            'minimum_piece_edge_mm':minimum_piece_edge_mm,'unlimited_search':True,
            'minimum_safe_candidates':candidate_target,'rectangle_corner_hypotheses':True,
            'max_search_seconds':30.,'wall_time_guard_seconds':30.,
            'recovery_total_deadline_seconds':30.,'recovery_finalization_reserve_seconds':1.2,
            'recovery_initial_stage_seconds':12.,'appearance_resolution_reserve_seconds':0.,
            'minimum_interface_appearance_score':None,
        })
    finally:
        logger = logging.getLogger('solver2')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
