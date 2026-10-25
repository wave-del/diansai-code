"""搜索树分支共享的不可变增量几何状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry

from .geometry_cache import CachedPoseGeometry, pairwise_overlap_cached


@dataclass(frozen=True)
class SearchState:
    placed_pose_ids: tuple[int, ...]
    pose_geometries: tuple[tuple[int, CachedPoseGeometry], ...]
    current_union: BaseGeometry
    current_bounds: tuple[float, float, float, float]
    current_hull: BaseGeometry
    accumulated_overlap: float
    accumulated_area: float
    occupied_intervals: tuple[tuple[Any, tuple[tuple[float, float], ...]], ...] = ()
    confirmed_interfaces: tuple[int, ...] = ()
    pending_interfaces: tuple[int, ...] = ()
    interface_graph: tuple[tuple[int, int], ...] = ()
    current_score_lower_bound: float = 0.0

    @classmethod
    def initial(cls, pose_id: int, geometry: CachedPoseGeometry) -> "SearchState":
        return cls(
            placed_pose_ids=(pose_id,),
            pose_geometries=((pose_id, geometry),),
            current_union=geometry.polygon,
            current_bounds=geometry.bounds,
            current_hull=geometry.convex_hull,
            accumulated_overlap=0.0,
            accumulated_area=geometry.area,
        )

    def add_pose(
        self,
        pose_id: int,
        geometry: CachedPoseGeometry,
        *,
        occupied_intervals: tuple | None = None,
        confirmed_interface: int | None = None,
    ) -> "SearchState":
        existing = tuple(item[1] for item in self.pose_geometries)
        overlap = pairwise_overlap_cached(geometry, existing)
        union = self.current_union.union(geometry.polygon)
        min_x = min(self.current_bounds[0], geometry.min_x)
        min_y = min(self.current_bounds[1], geometry.min_y)
        max_x = max(self.current_bounds[2], geometry.max_x)
        max_y = max(self.current_bounds[3], geometry.max_y)
        interfaces = self.confirmed_interfaces
        if confirmed_interface is not None:
            interfaces = interfaces + (int(confirmed_interface),)
        return SearchState(
            placed_pose_ids=self.placed_pose_ids + (int(pose_id),),
            pose_geometries=self.pose_geometries + ((int(pose_id), geometry),),
            current_union=union,
            current_bounds=(min_x, min_y, max_x, max_y),
            current_hull=union.convex_hull,
            accumulated_overlap=self.accumulated_overlap + overlap,
            accumulated_area=self.accumulated_area + geometry.area,
            occupied_intervals=(
                self.occupied_intervals
                if occupied_intervals is None else occupied_intervals
            ),
            confirmed_interfaces=interfaces,
            pending_interfaces=self.pending_interfaces,
            interface_graph=self.interface_graph,
            current_score_lower_bound=self.current_score_lower_bound + overlap,
        )
