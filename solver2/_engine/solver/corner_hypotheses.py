"""Supplement edge-chain search with measured rectangle-corner constraints.

This covers four fragments that each contain an outer corner. Every hypothesis
still uses rigid transforms of the measured polygons and the common final gates.
Other cuts continue through the general edge-chain search.
"""
from itertools import permutations, product
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from shapely import union_all
from shapely.geometry import Polygon, box

from ..transforms import rotation_matrix, transform_points
from .partition_search import PartitionSearchResult


def rectangle_corner_candidates(pieces, maximum_results=12, *, deadline=None):
    if len(pieces) != 4:
        return []
    choices = []
    area = sum(piece.area for piece in pieces)
    for piece in pieces:
        vertices = piece.vertices
        edges = np.roll(vertices,-1,axis=0)-vertices
        lengths = np.linalg.norm(edges,axis=1)
        angles = np.degrees(np.arccos(np.clip(
            np.sum(-np.roll(edges,1,axis=0)*edges,axis=1)/(np.roll(lengths,1)*lengths),-1,1)))
        options = [(i,float(np.degrees(np.arctan2(edges[i,1],edges[i,0]))-(90-angles[i])/2))
                   for i in range(len(vertices)) if abs(angles[i]-90) <= 6]
        if not options:
            return []
        choices.append(options)
    corners = np.array([[0,0],[1,0],[1,1],[0,1]],dtype=float)
    coarse = []
    # A rectangle offers four measured right-angle vertices, but opposite
    # vertices often produce the identical anchored polygon. Reuse geometric
    # scoring for those products, while retaining different material rotations
    # (important if the caller later compares printed textures).
    anchored_options = {}
    for index, (piece, options) in enumerate(zip(pieces, choices)):
        for corner in range(4):
            anchored = []
            for vertex, phi in options:
                angle = corner * 90 - phi
                local = (piece.vertices - piece.vertices[vertex]) @ rotation_matrix(angle).T
                rounded = np.round(local, 8)
                signature = min(tuple(map(tuple, np.roll(rounded, shift, axis=0)))
                                for shift in range(len(rounded)))
                anchored.append((vertex, phi, angle, local, signature))
            anchored_options[index, corner] = anchored
    coarse_cache = {}

    def objective(wh, local, assignment):
        polygons = [Polygon(v+corners[c]*wh) for v,c in zip(local,assignment)]
        union = union_all(polygons)
        rectangle = box(0,0,*wh)
        return (3*(area-union.area)+3*union.difference(rectangle).area
                +rectangle.difference(union).area)/max(area,1.)

    for assignment in permutations(range(4)):
        # Global 180-degree rotations preserve rectangle dimensions and all
        # contacts. Keep one of each equivalent assignment pair. Adjacent
        # corners remain distinct because width/height limits are asymmetric.
        if assignment[0] >= 2:
            continue
        if deadline is not None and perf_counter() >= deadline:
            break
        for selected in product(*(anchored_options[i, corner] for i, corner in enumerate(assignment))):
            if deadline is not None and perf_counter() >= deadline:
                break
            picks = [(row[0], row[1]) for row in selected]
            angles = [row[2] for row in selected]
            local = [row[3] for row in selected]
            geometry_key = assignment, tuple(row[4] for row in selected)
            if geometry_key not in coarse_cache:
                best = None
                for width, margin in product(np.linspace(90,120,9),(0.,.015,.03,.05)):
                    height = area*(1+margin)/width
                    if not 50 <= height <= 90:
                        continue
                    wh = np.array([width,height])
                    score = objective(wh,local,assignment)
                    if best is None or score < best[0]:
                        best = score,wh
                coarse_cache[geometry_key] = best
            best = coarse_cache[geometry_key]
            if best is not None:
                coarse.append((best[0],best[1],assignment,picks,angles,local,geometry_key))
    coarse.sort(key=lambda row:row[0])
    results = []
    fit_cache = {}
    for _score, initial, assignment, picks, angles, local, geometry_key in coarse[:maximum_results]:
        if deadline is not None and perf_counter() >= deadline:
            break
        if geometry_key not in fit_cache:
            fit_cache[geometry_key] = minimize(objective,initial,args=(local,assignment),method='Powell',
                           bounds=((90,120),(50,90)),options=dict(maxiter=35,xtol=.01,ftol=1e-5))
        fit = fit_cache[geometry_key]
        poses = {}
        for piece,corner,(vertex,_phi),angle in zip(pieces,assignment,picks,angles):
            translation = corners[corner]*fit.x-transform_points([piece.vertices[vertex]],angle)[0]
            poses[piece.piece_id] = angle,translation,transform_points(piece.vertices,angle,translation)
        results.append(PartitionSearchResult(poses,(),float(fit.fun),'measured_outer_corner_constraints'))
    results.sort(key=lambda r:r.cost)
    return results
