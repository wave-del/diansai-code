"""验证随机拼图的赛题拓扑、短边兼容和 5 mm 执行留缝。

该文件不依赖相机、运动硬件或外部文件，可通过
``python -m unittest tests.unit.control.test_random_solver`` 独立运行。
"""

from __future__ import annotations

import unittest
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon

from ec_26_e.control import (
    PlanningOptions,
    PuzzlePiece,
    plan_sheet_moves,
)
from ec_26_e.control._engine.generator.random_generator import random_pieces
from ec_26_e.control._engine.solver.interface_matching import (
    build_shared_interface_candidates,
)
from ec_26_e.control._engine.solver.partition_search import (
    enumerate_partition_candidates,
)
from ec_26_e.control._engine.solver.preprocess import preprocess_piece
from ec_26_e.control._engine.runtime_config import RuntimeConfig
from ec_26_e.control._engine.solver.observation_uncertainty import (
    ObservationUncertaintyProfile,
)
from ec_26_e.control._engine.solver.random_clearance import (
    plan_random_clearance,
)
from ec_26_e.control._engine.solver.random_safety import (
    RANDOM_TARGET_LONG_SIDE_MAX_MM,
    RANDOM_TARGET_SHORT_SIDE_MAX_MM,
    RandomCandidatePool,
    assess_random_assembly,
)
from ec_26_e.control._engine.solver.random_execution_gate import (
    evaluate_random_execution_gate,
)
from ec_26_e.control._engine.solver.search import solve_random


def _public_pieces(generated: list[object]) -> tuple[PuzzlePiece, ...]:
    """把内部生成器碎片转换为不变的公开输入契约。"""

    return tuple(
        PuzzlePiece(
            piece.piece_id,
            tuple((float(x), float(y)) for x, y in piece.vertices),
        )
        for piece in generated
    )


class RandomCompetitionSolverTest(unittest.TestCase):
    """覆盖 2～4 片代表拓扑和赛题允许范围附近的短边。"""

    def test_two_and_three_piece_topologies_recover_rectangle(self) -> None:
        """折线两片和三向节点均应恢复同一结构矩形。"""

        for count, topology in ((2, "two_polyline"), (3, "three_y")):
            with self.subTest(piece_count=count, topology=topology):
                pieces, _width, _height = random_pieces(
                    np.random.default_rng(2600 + count),
                    piece_count=count,
                    target_width=100.0,
                    target_height=70.0,
                    topology=topology,
                )
                plan = plan_sheet_moves(
                    _public_pieces(pieces),
                    PlanningOptions(mode="random"),
                )
                self.assertAlmostEqual(plan.target_width_mm, 100.0, places=4)
                self.assertAlmostEqual(plan.target_height_mm, 70.0, places=4)
                self.assertEqual(len(plan.moves), count)

    def test_short_edge_above_ten_mm_is_not_rejected(self) -> None:
        """自制碎片的 10～20 mm 真边必须参与求解而非硬拒绝。"""

        pieces = (
            PuzzlePiece(
                "left",
                ((0.0, 0.0), (50.0, 0.0), (45.0, 10.0),
                 (50.0, 60.0), (0.0, 60.0)),
            ),
            PuzzlePiece(
                "right",
                ((50.0, 0.0), (100.0, 0.0), (100.0, 60.0),
                 (50.0, 60.0), (45.0, 10.0)),
            ),
        )
        plan = plan_sheet_moves(pieces, PlanningOptions(mode="random"))

        self.assertAlmostEqual(plan.target_width_mm, 100.0, places=4)
        self.assertAlmostEqual(plan.target_height_mm, 60.0, places=4)

    def test_four_piece_fan_uses_reverse_partition_evidence(self) -> None:
        """多路结点应由通用边证据图恢复，并携带原始边区间图。"""

        generated, _width, _height = random_pieces(
            np.random.default_rng(2604),
            piece_count=4,
            target_width=100.0,
            target_height=70.0,
            topology="four_fan",
        )
        pieces = [
            preprocess_piece(piece.piece_id, piece.vertices)
            for piece in generated
        ]
        diagnostics: dict[str, object] = {}

        candidates = solve_random(
            pieces,
            diagnostics=diagnostics,
            runtime_config=RuntimeConfig(),
        )

        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(
            candidate.generation_method,
            "directed_edge_chain_reverse_partition",
        )
        self.assertGreaterEqual(len(candidate.partition_evidence), 5)
        self.assertEqual(candidate.topology_template, "edge_evidence_graph")
        self.assertEqual(
            candidate.partition_evidence[0],
            {
                "evidence_type": "topology_template",
                "template": "edge_evidence_graph",
            },
        )
        correspondence = candidate.partition_evidence[-1]
        self.assertEqual(
            correspondence["evidence_type"],
            "final_original_edge_correspondence_graph",
        )
        self.assertGreaterEqual(len(correspondence["interfaces"]), 4)
        self.assertFalse(diagnostics["old_candidate_generator_used"])
        self.assertFalse(diagnostics["generic_partial_fallback_used"])

    def test_piece_count_dispatches_to_independent_topology_branches(self) -> None:
        """2、3、4 片必须进入不同解析入口并各自拥有搜索预算。"""

        expected = {
            2: "two_piece_edge_chain",
            3: "three_piece_topology",
            4: "four_piece_edge_evidence_graph",
        }
        topologies = {2: "two_polyline", 3: "three_y", 4: "four_fan"}
        for count in (2, 3, 4):
            with self.subTest(piece_count=count):
                generated, _width, _height = random_pieces(
                    np.random.default_rng(2610 + count),
                    piece_count=count,
                    target_width=100.0,
                    target_height=70.0,
                    topology=topologies[count],
                )
                processed = [
                    preprocess_piece(piece.piece_id, piece.vertices)
                    for piece in generated
                ]
                candidates, diagnostics = enumerate_partition_candidates(
                    processed,
                    tolerance_mm=2.0,
                    deadline=perf_counter() + 8.0,
                    maximum_results=24,
                )

                self.assertTrue(candidates)
                self.assertEqual(
                    diagnostics["piece_count_branch"], expected[count]
                )

    def test_four_piece_binary_tree_shapes_use_one_evidence_graph(self) -> None:
        """不同二叉合并树应由同一边证据图闭合而非答案模板分类。"""

        observations = {
            "two_disjoint_pair_merges": {
                "a": ((0, 0), (50, 0), (50, 30), (0, 30)),
                "b": ((0, 30), (50, 30), (50, 70), (0, 70)),
                "c": ((50, 0), (100, 0), (100, 45), (50, 45)),
                "d": ((50, 45), (100, 45), (100, 70), (50, 70)),
            },
            "rooted_growth_merges": {
                "a": ((0, 0), (40, 0), (40, 70), (0, 70)),
                "b": ((40, 0), (100, 0), (100, 20), (40, 20)),
                "c": ((40, 20), (100, 20), (100, 45), (40, 45)),
                "d": ((40, 45), (100, 45), (100, 70), (40, 70)),
            },
        }
        for case_name, vertices in observations.items():
            with self.subTest(topology=case_name):
                pieces = [
                    preprocess_piece(piece_id, points)
                    for piece_id, points in vertices.items()
                ]
                candidates, diagnostics = enumerate_partition_candidates(
                    pieces,
                    tolerance_mm=2.0,
                    deadline=perf_counter() + 12.0,
                    maximum_results=32,
                )

                self.assertTrue(candidates)
                self.assertEqual(
                    candidates[0].topology_template,
                    "edge_evidence_graph",
                )
                self.assertEqual(
                    diagnostics["piece_count_branch"],
                    "four_piece_edge_evidence_graph",
                )

    def test_noisy_four_piece_examples_generalize_across_vertex_counts(
        self,
    ) -> None:
        """全四边形双 T 和混合边数实拍均应由同一通用逻辑恢复。"""

        observations = {
            "all_quadrilaterals": {
                "0": ((22.220829, 61.373728), (50.029416, 9.991656),
                      (83.353104, 19.937468), (74.850372, 50.17728)),
                "1": ((26.971479, 73.12264), (85.18878, 79.18164),
                      (100.346976, 100.830228), (32.19057, 110.16576)),
                "2": ((107.167626, 63.956176), (176.883156, 95.07082),
                      (168.146307, 114.083232), (109.694682, 107.107644)),
                "3": ((125.064018, 57.21448), (147.592242, 13.095908),
                      (184.14927, 31.02646), (184.20309, 68.054104)),
            },
            "mixed_triangle_to_pentagon": {
                "0": ((19.378926, 109.302316), (22.74309, 85.646812),
                      (68.935968, 102.4993), (46.922553, 131.687912)),
                "1": ((67.918356, 88.006756), (71.124993, 27.46552),
                      (105.109218, 29.318844), (118.105299, 59.094376),
                      (93.568554, 88.781724)),
                "2": ((109.932732, 95.423848), (150.894927, 97.652976),
                      (125.982063, 128.311224)),
                "3": ((131.638959, 46.520564), (159.113448, 11.480564),
                      (186.568893, 72.090712), (164.420928, 82.76156)),
            },
        }
        for case_name, vertices in observations.items():
            with self.subTest(case=case_name):
                pieces = [
                    preprocess_piece(piece_id, points)
                    for piece_id, points in vertices.items()
                ]
                diagnostics: dict[str, object] = {}

                candidates = solve_random(
                    pieces,
                    diagnostics=diagnostics,
                    runtime_config=RuntimeConfig(),
                    deadline=perf_counter() + 20.0,
                )

                self.assertTrue(candidates)
                self.assertTrue(candidates[0].admissibility.accepted)
                self.assertGreaterEqual(
                    candidates[0].admissibility.union_rectangle_iou_estimate,
                    0.95,
                )
                self.assertEqual(
                    candidates[0].topology_template,
                    "edge_evidence_graph",
                )
                self.assertEqual(
                    diagnostics["piece_count_branch"],
                    "four_piece_edge_evidence_graph",
                )

    def test_execution_layout_has_five_mm_gap_without_overlap(self) -> None:
        """结构解输出前应平移成约 5 mm 拼缝并保持刚体不重叠。"""

        pieces = (
            PuzzlePiece(
                "left",
                ((0.0, 0.0), (50.0, 0.0), (45.0, 10.0),
                 (50.0, 60.0), (0.0, 60.0)),
            ),
            PuzzlePiece(
                "right",
                ((50.0, 0.0), (100.0, 0.0), (100.0, 60.0),
                 (50.0, 60.0), (45.0, 10.0)),
            ),
        )
        plan = plan_sheet_moves(pieces, PlanningOptions(mode="random"))
        first, second = (
            Polygon(move.target_vertices_a4_mm) for move in plan.moves
        )

        self.assertLessEqual(first.intersection(second).area, 1e-7)
        self.assertGreaterEqual(first.distance(second), 4.5)
        self.assertLessEqual(first.distance(second), 5.5)
        self.assertLessEqual(first.distance(second), 20.0)

    def test_four_piece_one_to_three_interface_is_kept_atomically(self) -> None:
        """四片中一条长边邻接其余三片时必须完整保留三个半边。"""

        vertices = {
            "0": ((33.66237, 88.862697), (80.0142, 13.215015),
                  (86.89485, 122.532993)),
            "1": ((108.09351, 104.790807), (116.361, 68.006466),
                  (135.49095, 72.556803), (148.99101, 115.592697),
                  (130.82328, 131.30667)),
            "2": ((134.9922, 58.77333), (185.41551, 77.181093),
                  (172.85457, 113.767038)),
            "3": ((154.98693, 59.547906), (159.45972, 13.203432),
                  (187.20387, 32.673861), (191.95323, 47.484954)),
        }
        pieces = [
            preprocess_piece(piece_id, [list(point) for point in points])
            for piece_id, points in vertices.items()
        ]
        candidates, placements, _diagnostics = (
            build_shared_interface_candidates(
                pieces,
                noise_mm=2.0,
                include_generic_partial=True,
            )
        )
        composite = next(
            item for item in candidates
            if item.interface_type == "one_to_three"
            and item.piece_a == "0"
            and item.segments_a[0].edge_index == 1
            and tuple(
                (segment.piece_id, segment.edge_index)
                for segment in item.segments_b
            ) == (("3", 1), ("1", 1), ("2", 0))
        )
        parts = [
            item for item in placements
            if item.composite_id == composite.composite_id
        ]

        self.assertEqual(len(parts), 3)
        self.assertEqual(
            {item.interface_type for item in parts},
            {"one_to_three_part"},
        )

    def test_new_gate_reconstructs_five_seams_and_virtual_t_nodes(self) -> None:
        """安全门必须独立恢复五条拼缝及长边上的两个虚拟T结点。"""

        vertices = {
            "0": np.asarray(((0.0, 0.0), (100.0, 0.0), (0.0, 60.0))),
            "1": np.asarray(((0.0, 60.0), (50.0, 60.0), (33.333, 40.0))),
            "2": np.asarray((
                (33.333, 40.0), (50.0, 60.0), (100.0, 60.0),
                (100.0, 30.0), (66.667, 20.0),
            )),
            "3": np.asarray(((66.667, 20.0), (100.0, 30.0), (100.0, 0.0))),
        }
        profile = ObservationUncertaintyProfile(
            2.0, 0.5, None, None, None, "unit_test", "medium", False
        )
        assessment = assess_random_assembly(
            vertices,
            profile,
            RuntimeConfig(),
            expected_piece_count=4,
            # 故意只声明三条，证明新门不把搜索接口当成完整事实。
            declared_interface_count=3,
        )

        self.assertTrue(assessment.accepted, assessment.rejection_reasons)
        self.assertEqual(len(assessment.topology.adjacency_pairs), 5)
        self.assertEqual(len(assessment.topology.interfaces), 5)
        self.assertEqual(
            sum(
                junction.junction_type == "T_JUNCTION"
                for junction in assessment.topology.junctions
            ),
            2,
        )

        clearance = plan_random_clearance(
            vertices,
            assessment.topology,
            target_gap_mm=5.0,
            minimum_gap_mm=3.0,
            maximum_gap_mm=6.0,
            maximum_vertex_distance_mm=20.0,
            measurement_error_radius_mm=2.0,
            machine_position_margin_mm=1.0,
            overlap_tolerance_mm2=1e-7,
        )
        self.assertTrue(clearance.feasible, clearance.failure_reason)
        self.assertLessEqual(
            clearance.estimated_physical_overlap_area_mm2, 1e-7
        )
        self.assertLessEqual(
            clearance.max_corresponding_vertex_distance_mm, 20.0
        )

    def test_new_gate_rejects_disconnected_high_level_layout(self) -> None:
        """矩形外观不足以放行，重建后的碎片邻接图还必须连通。"""

        vertices = {
            "left": np.asarray(((0.0, 0.0), (40.0, 0.0),
                                (40.0, 60.0), (0.0, 60.0))),
            "right": np.asarray(((60.0, 0.0), (100.0, 0.0),
                                 (100.0, 60.0), (60.0, 60.0))),
        }
        profile = ObservationUncertaintyProfile(
            2.0, 0.5, None, None, None, "unit_test", "medium", False
        )
        assessment = assess_random_assembly(
            vertices, profile, RuntimeConfig(), expected_piece_count=2
        )

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "reconstructed_topology_disconnected",
            assessment.rejection_reasons,
        )

    def test_random_gate_uses_final_130_by_100_mm_upper_limits(self) -> None:
        """随机安全门上限应为长边130 mm、短边100 mm且不再叠加放宽量。"""

        profile = ObservationUncertaintyProfile(
            2.0, 0.5, None, None, None, "unit_test", "medium", False
        )

        def assessment(width: float, height: float):
            """构造沿中线切开的无缝两片矩形并执行正式随机安全评估。"""

            middle = width / 2.0
            return assess_random_assembly(
                {
                    "left": np.asarray((
                        (0.0, 0.0), (middle, 0.0),
                        (middle, height), (0.0, height),
                    )),
                    "right": np.asarray((
                        (middle, 0.0), (width, 0.0),
                        (width, height), (middle, height),
                    )),
                },
                profile,
                RuntimeConfig(),
                expected_piece_count=2,
            )

        self.assertEqual(RANDOM_TARGET_LONG_SIDE_MAX_MM, 130.0)
        self.assertEqual(RANDOM_TARGET_SHORT_SIDE_MAX_MM, 100.0)
        self.assertTrue(assessment(129.9, 99.9).accepted)
        self.assertIn(
            "illegal_target_dimensions",
            assessment(130.1, 90.0).rejection_reasons,
        )
        self.assertIn(
            "illegal_target_dimensions",
            assessment(120.0, 100.1).rejection_reasons,
        )

    def test_candidate_pool_keeps_best_rejected_fallback(self) -> None:
        """没有合规解时仍应保留最优退化解，有合规解时必须优先合规解。"""

        profile = ObservationUncertaintyProfile(
            2.0, 0.5, None, None, None, "unit_test", "medium", False
        )
        rejected = assess_random_assembly(
            {
                "left": np.asarray(((0.0, 0.0), (40.0, 0.0),
                                    (40.0, 60.0), (0.0, 60.0))),
                "right": np.asarray(((60.0, 0.0), (100.0, 0.0),
                                     (100.0, 60.0), (60.0, 60.0))),
            },
            profile,
            RuntimeConfig(),
            expected_piece_count=2,
        )
        accepted = assess_random_assembly(
            {
                "left": np.asarray(((0.0, 0.0), (50.0, 0.0),
                                    (50.0, 60.0), (0.0, 60.0))),
                "right": np.asarray(((50.0, 0.0), (100.0, 0.0),
                                     (100.0, 60.0), (50.0, 60.0))),
            },
            profile,
            RuntimeConfig(),
            expected_piece_count=2,
        )
        rejected_candidate = SimpleNamespace(
            admissibility=rejected, interfaces=[], poses=[], total=0.0
        )
        accepted_candidate = SimpleNamespace(
            admissibility=accepted, interfaces=[], poses=[], total=100.0
        )
        pool = RandomCandidatePool(capacity=2)

        self.assertTrue(pool.consider(rejected_candidate))
        self.assertIs(pool.items()[0], rejected_candidate)
        self.assertTrue(pool.consider(accepted_candidate))
        self.assertIs(pool.items()[0], accepted_candidate)

    def test_random_quality_failure_blocks_execution_even_if_machine_reachable(self) -> None:
        """即使机械路径可达，几何不完整或置信度不足仍必须阻断执行。"""

        profile = ObservationUncertaintyProfile(
            2.0, 0.5, None, None, None, "unit_test", "medium", False
        )
        rejected = assess_random_assembly(
            {
                "left": np.asarray(((0.0, 0.0), (40.0, 0.0),
                                    (40.0, 60.0), (0.0, 60.0))),
                "right": np.asarray(((60.0, 0.0), (100.0, 0.0),
                                     (100.0, 60.0), (60.0, 60.0))),
            },
            profile,
            RuntimeConfig(),
            expected_piece_count=2,
        )
        targets = {
            "left": np.asarray(((20.0, 160.0), (60.0, 160.0),
                                (60.0, 220.0), (20.0, 220.0))),
            "right": np.asarray(((80.0, 160.0), (120.0, 160.0),
                                 (120.0, 220.0), (80.0, 220.0))),
        }
        records = [
            {
                "piece_id": piece_id,
                "observed": vertices.copy(),
                "target": vertices.copy(),
                "angle": 0.0,
                "translation": np.zeros(2),
            }
            for piece_id, vertices in targets.items()
        ]
        moves = [
            SimpleNamespace(
                piece_id=piece_id,
                pick_xy_a4_mm=np.mean(vertices, axis=0),
                place_xy_a4_mm=np.mean(vertices, axis=0),
            )
            for piece_id, vertices in targets.items()
        ]
        gate_arguments = {
            "mode": "random",
            "surface_mode": "white",
            "ambiguous": False,
            "appearance_resolved": True,
            "confidence": 0.3,
            "predicted_max_vertex_error_mm": 2.0,
            "nominal_max_vertex_distance_mm": 5.0,
            "estimated_risk_vertex_distance_mm": 7.0,
            "robust_max_vertex_distance_mm": 9.0,
            "estimated_overlap_area_mm2": 0.0,
            "robust_overlap_area_mm2": 0.0,
            "nonadjacent_robust_overlap_area_mm2": 0.0,
            "clearance_plan": SimpleNamespace(feasible=True),
            "optimization": {},
            "search_diagnostics": {},
            "recovery_outcome": None,
            "candidate_margin": 0.0,
            "pose_records": records,
            "moves": moves,
            "transport_reasons": [],
            "visual_geometry_validated": True,
            "calibration": {},
            "target_width_mm": 100.0,
            "target_height_mm": 60.0,
            "config": RuntimeConfig(),
            "random_assessment": rejected,
        }
        gate = evaluate_random_execution_gate(**gate_arguments)

        self.assertTrue(gate.machine_executable)
        self.assertFalse(gate.executable)
        self.assertFalse(gate.forced_output)
        self.assertFalse(gate.competition_gate_passed)
        self.assertIn("reconstructed_topology_disconnected", gate.reasons)
        self.assertIn("low_confidence", gate.reasons)
        self.assertIn(
            "reconstructed_topology_disconnected",
            gate.solver_block_reasons,
        )

        invalid_moves = list(moves)
        invalid_moves[0] = SimpleNamespace(
            piece_id="left",
            pick_xy_a4_mm=np.asarray((-1.0, 190.0)),
            place_xy_a4_mm=np.asarray((-1.0, 190.0)),
        )
        machine_blocked = evaluate_random_execution_gate(
            **{**gate_arguments, "moves": invalid_moves}
        )
        self.assertFalse(machine_blocked.executable)
        self.assertIn("invalid_pick_point", machine_blocked.reasons)


if __name__ == "__main__":
    unittest.main()
