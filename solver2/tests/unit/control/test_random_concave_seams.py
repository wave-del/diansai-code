"""Regression cases from round 3: concave notches and folded seam clearance."""
import numpy as np
import pytest
from shapely.geometry import Polygon

from solver2._engine.solver.random_clearance import (
    _constrained_interface_solution, _interface_metrics, plan_random_clearance,
)
from solver2._engine.solver.random_safety import reconstruct_random_topology


NOTCH = {
    'a': np.array([[0., 0.], [100., 0.], [100., 27.843338613546734],
                   [45.450713416233775, 8.21513800969408], [0., 60.]]),
    'b': np.array([[100., 27.843338613546734], [100., 60.], [0., 60.],
                   [45.450713416233775, 8.21513800969408]]),
}


@pytest.mark.parametrize('reverse_a,reverse_b', [(False, False), (True, False), (False, True), (True, True)])
def test_both_notch_edges_are_found_independent_of_polygon_winding(reverse_a, reverse_b):
    vertices = {k: v[::-1] if reverse else v.copy()
                for (k, v), reverse in zip(NOTCH.items(), [reverse_a, reverse_b])}
    angle = np.deg2rad(73.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    vertices = {k: v @ rotation.T + [81., 173.] for k, v in vertices.items()}
    topology = reconstruct_random_topology(vertices, tolerance_mm=3.)
    assert topology.graph_connected
    assert len(topology.interfaces) == 2
    assert topology.adjacency_pairs == (('a', 'b'),)
    assert sum(i.length_mm for i in topology.interfaces) == pytest.approx(126.87485529076965)


def test_coincident_same_side_outer_edges_are_not_seams():
    rectangle = np.array([[0., 0.], [50., 0.], [50., 60.], [0., 60.]])
    topology = reconstruct_random_topology({'a': rectangle, 'b': rectangle[::-1]}, tolerance_mm=3.)
    assert not topology.interfaces and not topology.graph_connected


def test_long_edge_is_split_into_all_three_t_junction_interfaces():
    vertices = {'left': np.array([[0., 0.], [40., 0.], [40., 60.], [0., 60.]])}
    for i in range(3):
        y = i * 20.
        vertices[str(i)] = np.array([[40., y], [100., y], [100., y+20.], [40., y+20.]])
    topology = reconstruct_random_topology(vertices, tolerance_mm=1.)
    assert topology.graph_connected and len(topology.interfaces) == 5
    assert len([i for i in topology.interfaces if 'left' in (i.piece_a, i.piece_b)]) == 3


def test_each_branch_of_a_folded_seam_has_its_own_clearance():
    topology = reconstruct_random_topology(NOTCH, tolerance_mm=3.)
    deltas = np.array([[0., 0.], [0., 10.]])
    polygons = [Polygon(v+d) for v, d in zip(NOTCH.values(), deltas)]
    metrics, _ = _interface_metrics(polygons, deltas, topology, {'a': 0, 'b': 1}, 5.)
    gaps = [m.nominal_normal_gap_mm for m in metrics]
    assert len(gaps) == 2 and abs(gaps[0]-gaps[1]) > 1.5


def test_four_piece_notch_opens_every_interface_with_zero_overlap():
    p = [47.773996576612056, 5.097809704793219]
    vertices = {
        '1': np.array([[0.,0.], [100.,0.], [100.,26.165385690875155], p, [0.,60.]]),
        '2': np.array([[100.,26.165385690875155], [100.,60.], [64.22777078592682,60.], p]),
        '3': np.array([[64.22777078592682,60.], [38.88061868169651,60.], p]),
        '4': np.array([[38.88061868169651,60.], [0.,60.], p]),
    }
    topology = reconstruct_random_topology(vertices, tolerance_mm=3.)
    assert topology.graph_connected and len(topology.interfaces) == 4
    result = plan_random_clearance(vertices, topology, target_gap_mm=5., minimum_gap_mm=3.,
        maximum_gap_mm=6., maximum_vertex_distance_mm=20., measurement_error_radius_mm=0.,
        machine_position_margin_mm=0., overlap_tolerance_mm2=1e-6)
    assert result.feasible
    assert len(result.interface_metrics) == 4
    assert all(3.-1e-7 <= m.nominal_normal_gap_mm <= 6.+1e-7 for m in result.interface_metrics)
    polygons = [Polygon(v) for v in result.planned_vertices]
    assert all(p.intersection(q).area <= 1e-6 for i, p in enumerate(polygons) for q in polygons[:i])


def test_constrained_repair_moves_a_notch_out_of_an_invalid_gap():
    topology = reconstruct_random_topology(NOTCH, tolerance_mm=3.)
    initial = np.array([[-1., -1.], [1., 1.]])
    deltas = _constrained_interface_solution(list(NOTCH.values()), topology, {'a': 0, 'b': 1},
        initial, 5., 3., 6., 20., 1e-6)
    polygons = [Polygon(v+d) for v,d in zip(NOTCH.values(), deltas)]
    assert polygons[0].intersection(polygons[1]).area <= 1e-6
    assert 3. <= polygons[0].distance(polygons[1]) <= 6.
    metrics, endpoints = _interface_metrics(polygons, deltas, topology, {'a': 0, 'b': 1}, 5.)
    assert endpoints <= 20.
    assert all(3. <= m.nominal_normal_gap_mm <= 6. for m in metrics)
