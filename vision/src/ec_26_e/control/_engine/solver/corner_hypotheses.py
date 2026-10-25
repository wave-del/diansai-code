"""Supplement edge-chain search with measured rectangle-corner constraints.

This covers four fragments that each contain an outer corner. Every hypothesis
still uses rigid transforms of the measured polygons and the common final gates.
Other cuts continue through the general edge-chain search.
"""
from itertools import permutations, product

import numpy as np
from scipy.optimize import minimize
from shapely import union_all
from shapely.geometry import Polygon, box

from ..transforms import rotation_matrix, transform_points
from .partition_search import PartitionSearchResult


def rectangle_corner_candidates(pieces, maximum_results=12):
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

    def objective(wh, local, assignment):
        polygons = [Polygon(v+corners[c]*wh) for v,c in zip(local,assignment)]
        union = union_all(polygons)
        rectangle = box(0,0,*wh)
        return (3*(area-union.area)+3*union.difference(rectangle).area
                +rectangle.difference(union).area)/max(area,1.)

    for assignment in permutations(range(4)):
        for picks in product(*choices):
            angles = [corner*90-phi for corner,(_vertex,phi) in zip(assignment,picks)]
            local = [(piece.vertices-piece.vertices[vertex])@rotation_matrix(angle).T
                     for piece,(vertex,_phi),angle in zip(pieces,picks,angles)]
            best = None
            for width, margin in product(np.linspace(90,120,9),(0.,.015,.03,.05)):
                height = area*(1+margin)/width
                if not 50 <= height <= 90:
                    continue
                wh = np.array([width,height])
                score = objective(wh,local,assignment)
                if best is None or score < best[0]:
                    best = score,wh
            if best is not None:
                coarse.append((best[0],best[1],assignment,picks,angles,local))
    coarse.sort(key=lambda row:row[0])
    results = []
    for _score, initial, assignment, picks, angles, local in coarse[:maximum_results]:
        fit = minimize(objective,initial,args=(local,assignment),method='Powell',
                       bounds=((90,120),(50,90)),options=dict(maxiter=35,xtol=.01,ftol=1e-5))
        poses = {}
        for piece,corner,(vertex,_phi),angle in zip(pieces,assignment,picks,angles):
            translation = corners[corner]*fit.x-transform_points([piece.vertices[vertex]],angle)[0]
            poses[piece.piece_id] = angle,translation,transform_points(piece.vertices,angle,translation)
        results.append(PartitionSearchResult(poses,(),float(fit.fun),'measured_outer_corner_constraints'))
    results.sort(key=lambda r:r.cost)
    return results
