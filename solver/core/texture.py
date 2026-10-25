"""扑克牌图案的现场图像消歧与复原预览。

该模块只比较同一张经过 A4 透视校正的现场图像中，候选拼缝两侧的像素连续性；
不包含任何完整扑克牌模板、答案图或针对单张牌的先验。
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from .geometry import transform_point, transform_polygon
from .models import Candidate, PuzzlePiece


class SceneTextureRanker:
    """使用一张竖向、校正后的现场图，对几何候选进行花纹排序。"""

    def __init__(
        self,
        image_path: str | Path,
        pieces: list[PuzzlePiece],
        *,
        paper_width_mm: float = 207.0,
        paper_height_mm: float = 292.0,
    ) -> None:
        started = time.perf_counter()
        self.image_path = Path(image_path)
        raw = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"无法读取花纹检测图片：{self.image_path}")
        if paper_width_mm <= 0.0 or paper_height_mm <= 0.0:
            raise ValueError("花纹参考坐标必须为正")
        self.image = raw
        self.lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.GaussianBlur(
            cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0.6
        )
        self.grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        self.grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        self.scale_x = raw.shape[1] / float(paper_width_mm)
        self.scale_y = raw.shape[0] / float(paper_height_mm)
        self.pieces = {piece.id: piece for piece in pieces}
        self.inverse_poses: dict[int, np.ndarray] = {}
        self.preprocess_ms = (time.perf_counter() - started) * 1000.0
        self.scored_candidates = 0
        self.scoring_ms = 0.0

    def score_candidate(self, candidate: Candidate) -> float:
        """把纹理指标附加到 candidate，并返回连续性分数。"""
        started = time.perf_counter()
        shapes = {
            item_id: transform_polygon(self.pieces[item_id].polygon, pose)
            for item_id, pose in candidate.poses.items()
        }
        self.inverse_poses = {
            item_id: np.linalg.inv(pose) for item_id, pose in candidate.poses.items()
        }
        seam_results: list[tuple[float, float, int, float]] = []
        seam_count = 0
        adjacency = {key:set() for key in shapes}
        for first_id, second_id, first_points, second_points, tangent, first_normal, second_normal in self._edge_interfaces(shapes):
            result = self._score_interface(
                first_id, second_id, first_points, second_points, tangent,
                first_normal, second_normal, shapes, candidate,
            )
            if result is not None:
                seam_results.append(result)
                seam_count += 1
                adjacency[first_id].add(second_id)
                adjacency[second_id].add(first_id)
        if seam_results:
            scores = np.asarray([item[0] for item in seam_results], dtype=float)
            weights = np.asarray([item[1] for item in seam_results], dtype=float)
            mean = float(np.average(scores, weights=np.maximum(weights, 1e-6)))
            # A strong mean must not hide one mismatched shared edge. Short
            # subsegments remain diagnostic: observation-sized contour error
            # can spoil a few pixels without invalidating an entire real seam.
            min_seam = float(np.min(scores))
            min_segment = float(np.min([item[3] for item in seam_results]))
            texture_score = 0.50 * mean + 0.50 * min_seam
            reference_continuity = texture_score
            samples = sum(item[2] for item in seam_results)
            information = float(np.clip(np.sum(weights) / max(samples, 1), 0.0, 1.0))
        else:
            texture_score, information, samples = 0.5, 0.0, 0
            min_seam = 0.0
            min_segment = 0.0
            reference_continuity = 0.5
        evidence = texture_score * information
        reached = set()
        pending = [next(iter(shapes))] if shapes else []
        while pending:
            key = pending.pop()
            if key not in reached:
                reached.add(key)
                pending.extend(adjacency[key]-reached)
        reference_evidence = reference_continuity * information
        geometry_score = float(candidate.score)
        candidate.score = geometry_score + 80.0 * evidence
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        candidate.metrics.update({
            "geometry_score": geometry_score,
            "texture_continuity": texture_score,
            "texture_graph_connected": float(bool(shapes) and len(reached)==len(shapes)),
            "texture_mean_seam": mean if seam_results else 0.0,
            "texture_min_seam": min_seam,
            "texture_min_segment": min_segment,
            "texture_information": information,
            "texture_confidence": information,
            "texture_evidence": evidence,
            "texture_reference_continuity": reference_continuity,
            "texture_reference_evidence": reference_evidence,
            "texture_adjustment": 80.0 * evidence,
            "texture_seam_count": float(seam_count),
            "texture_sample_count": float(samples),
            "texture_score_ms": elapsed_ms,
        })
        self.scored_candidates += 1
        self.scoring_ms += elapsed_ms
        return texture_score

    def summary(self) -> str:
        return (
            f"花纹预处理 {self.preprocess_ms:.1f} ms；评分候选 {self.scored_candidates} 个，"
            f"累计 {self.scoring_ms:.1f} ms"
        )

    def _edge_interfaces(self, shapes):
        """Match opposing measured edges, preserving each side's own boundary.

        Sampling a shared mathematical line lost most of a real seam when
        measured contours differed by a fraction of a millimeter. These paired
        edge coordinates tolerate observation error without moving the plan.
        """
        for first_id, second_id in itertools.combinations(sorted(shapes),2):
            first = np.asarray(shapes[first_id].exterior.coords[:-1],dtype=float)
            second = np.asarray(shapes[second_id].exterior.coords[:-1],dtype=float)
            for ai in range(len(first)):
                a0,a1 = first[ai],first[(ai+1)%len(first)]
                ua = a1-a0
                la = float(np.linalg.norm(ua))
                if la < 1e-6:
                    continue
                ua = ua/la
                for bi in range(len(second)):
                    b0,b1 = second[bi],second[(bi+1)%len(second)]
                    ub = b1-b0
                    lb = float(np.linalg.norm(ub))
                    if lb < 1e-6:
                        continue
                    ub = ub/lb
                    dot = float(ua@ub)
                    if dot > -np.cos(np.radians(8.)):
                        continue
                    lo,hi = max(0.,float((b1-a0)@ua)),min(la,float((b0-a0)@ua))
                    if hi-lo < 8.:
                        continue
                    along = np.linspace(lo+.08*(hi-lo),hi-.08*(hi-lo),max(12,min(96,int((hi-lo)*2))))
                    ac = a0+along[:,None]*ua
                    bc = b0+(((ac-b0)@ua)/dot)[:,None]*ub
                    if float(np.max(np.linalg.norm(ac-bc,axis=1))) > 3.:
                        continue
                    na,nb = np.array([-ua[1],ua[0]]),np.array([-ub[1],ub[0]])
                    if not shapes[first_id].contains(Point(ac.mean(axis=0)+na*.2)):
                        na = -na
                    if not shapes[second_id].contains(Point(bc.mean(axis=0)+nb*.2)):
                        nb = -nb
                    if na@nb > -.5:
                        continue
                    yield first_id,second_id,ac,bc,ua,na,nb

    def _score_interface(self, first_id, second_id, ac, bc, tangent, na, nb, shapes, candidate):
        from shapely import contains_xy
        best = None
        for da,db,shift in itertools.product((.15,.45,.8),(.15,.45,.8),(-1.,0.,1.)):
            left = ac+na*da
            right = bc+nb*db+tangent*shift
            valid = (contains_xy(shapes[first_id],left[:,0],left[:,1])
                     & contains_xy(shapes[second_id],right[:,0],right[:,1]))
            result = self._score_point_pairs(
                first_id,second_id,left[valid].tolist(),right[valid].tolist(),candidate,
            )
            if result is None:
                continue
            penalty = .012*abs(shift)+.01*abs(da-db)
            adjusted = (result[0]-penalty,result[1],result[2],result[3]-penalty)
            if best is None or adjusted[0] > best[0]:
                best = adjusted
        return best

    @staticmethod
    def _line_parts(geometry: object) -> list[LineString]:
        if getattr(geometry, "geom_type", "") == "LineString":
            return [geometry] if geometry.length >= 2.0 else []
        result: list[LineString] = []
        for child in getattr(geometry, "geoms", []):
            result.extend(SceneTextureRanker._line_parts(child))
        return result

    def _shared_lines(self, first: Polygon, second: Polygon) -> list[LineString]:
        exact = self._line_parts(first.boundary.intersection(second.boundary))
        if exact:
            return exact
        return self._line_parts(first.boundary.intersection(second.boundary.buffer(0.25)))

    def _score_line(
        self, first_id: int, second_id: int, first_shape: Polygon,
        second_shape: Polygon, first_inside: object, second_inside: object,
        line: LineString, candidate: Candidate,
    ) -> tuple[float, float, int, float] | None:
        coords = list(line.coords)
        if len(coords) < 2 or line.length < 2.0:
            return None
        start, end = np.asarray(coords[0], float), np.asarray(coords[-1], float)
        tangent = end - start
        length = float(np.linalg.norm(tangent))
        if length < 1e-6:
            return None
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        middle = np.asarray(line.interpolate(0.5, normalized=True).coords[0], float)
        probe = 0.7
        first_normal = normal if first_inside.contains(Point(middle + normal * probe)) else -normal
        second_normal = normal if second_inside.contains(Point(middle + normal * probe)) else -normal
        if float(np.dot(first_normal, second_normal)) > -.5:
            return None
        count = min(32, max(8, int(math.ceil(line.length / 2.0))))
        # 相机透视、YOLO 四角与真实切缝会有亚毫米至数毫米误差。沿候选
        # 公共边做小范围切向配准，允许正确花纹在视觉上稍有错位仍被识别；
        # 这不改变几何/电机目标，只影响花纹候选的比较分数。
        best: tuple[float, float, int, float] | None = None
        for tangent_shift in (-1.0, 0.0, 1.0):
            first_points: list[np.ndarray] = []
            second_points: list[np.ndarray] = []
            for fraction in np.linspace(.08, .92, count):
                center = np.asarray(
                    line.interpolate(float(fraction), normalized=True).coords[0], float
                )
                for offset in (.45, .85, 1.25):
                    left = center + first_normal * offset
                    right = center + tangent * tangent_shift + second_normal * offset
                    if (
                        first_inside.contains(Point(left))
                        and second_inside.contains(Point(right))
                    ):
                        first_points.append(left)
                        second_points.append(right)
            comparison = self._score_point_pairs(
                first_id, second_id, first_points, second_points, candidate
            )
            if comparison is None:
                continue
            # 大位移可能把重复花色“硬凑”在一起，轻微扣分而非完全禁止。
            penalty = 0.015 * abs(tangent_shift)
            adjusted = (
                comparison[0] - penalty,
                comparison[1], comparison[2], comparison[3] - penalty,
            )
            if best is None or (adjusted[0], adjusted[3]) > (best[0], best[3]):
                best = adjusted
        return best

    def _score_point_pairs(
        self,
        first_id: int,
        second_id: int,
        first_points: list[np.ndarray],
        second_points: list[np.ndarray],
        candidate: Candidate,
    ) -> tuple[float, float, int, float] | None:
        """比较一对已按候选公共边采样的图像带。"""
        if len(first_points) < 6:
            return None
        first_pixels = self._solution_to_pixels(first_id, first_points)
        second_pixels = self._solution_to_pixels(second_id, second_points)
        valid = self._valid_pixels(first_pixels) & self._valid_pixels(second_pixels)
        if int(np.count_nonzero(valid)) < 6:
            return None
        first_pixels, second_pixels = first_pixels[valid], second_pixels[valid]
        lab_first, lab_second = self._sample(self.lab, first_pixels), self._sample(self.lab, second_pixels)
        gx_first = self._sample(self.grad_x, first_pixels).reshape(-1)
        gy_first = self._sample(self.grad_y, first_pixels).reshape(-1)
        gx_second = self._sample(self.grad_x, second_pixels).reshape(-1)
        gy_second = self._sample(self.grad_y, second_pixels).reshape(-1)
        color_score = np.exp(-np.linalg.norm(lab_first - lab_second, axis=1) / 32.0)
        first_grad = np.column_stack((gx_first / self.scale_x, gy_first / self.scale_y)) @ candidate.poses[first_id][:2, :2].T
        second_grad = np.column_stack((gx_second / self.scale_x, gy_second / self.scale_y)) @ candidate.poses[second_id][:2, :2].T
        first_mag, second_mag = np.linalg.norm(first_grad, axis=1), np.linalg.norm(second_grad, axis=1)
        cosine = np.sum(first_grad * second_grad, axis=1) / np.maximum(first_mag * second_mag, 1e-6)
        direction = np.clip((cosine + 1.0) * .5, 0.0, 1.0)
        magnitude = 1.0 - np.abs(first_mag - second_mag) / np.maximum(first_mag + second_mag, 1e-6)
        gradient = .65 * direction + .35 * np.clip(magnitude, 0.0, 1.0)
        gradient[(first_mag + second_mag) < 10.0] = .5
        l_min = np.minimum(lab_first[:, 0], lab_second[:, 0])
        ink = np.clip((242.0 - l_min) / 150.0, 0.0, 1.0)
        chroma = np.clip(np.maximum(np.linalg.norm(lab_first[:, 1:] - 128.0, axis=1), np.linalg.norm(lab_second[:, 1:] - 128.0, axis=1)) / 55.0, 0.0, 1.0)
        gradient_info = np.clip(np.maximum(first_mag, second_mag) / 90.0, 0.0, 1.0)
        information = np.maximum(np.maximum(ink, chroma), gradient_info)
        information = np.maximum(information, .015)
        point_score = .58 * color_score + .42 * gradient
        seam_score = float(np.average(point_score, weights=information))
        # 一段边上只要出现明显错位，就不应由另一段碰巧相同的黑色图案抵消。
        # 四个等长小段也适用于横/竖一刀、田字和不等长矩形切分。
        segments = np.array_split(np.arange(len(point_score)), min(4, len(point_score)))
        segment_scores = [
            float(np.average(point_score[index], weights=information[index]))
            for index in segments if len(index)
        ]
        return (
            seam_score,
            float(np.mean(information) * len(point_score)),
            len(point_score),
            min(segment_scores),
        )

    def _solution_to_pixels(self, piece_id: int, points: list[np.ndarray]) -> np.ndarray:
        inverse = self.inverse_poses[piece_id]
        return np.asarray([
            (transform_point(inverse, point)[0] * self.scale_x,
             transform_point(inverse, point)[1] * self.scale_y)
            for point in points
        ], dtype=np.float32)

    def _valid_pixels(self, pixels: np.ndarray) -> np.ndarray:
        height, width = self.image.shape[:2]
        return ((pixels[:, 0] >= 1) & (pixels[:, 0] < width - 1)
                & (pixels[:, 1] >= 1) & (pixels[:, 1] < height - 1))

    @staticmethod
    def _sample(array: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        sampled = cv2.remap(array, pixels[:, 0].astype(np.float32).reshape(-1, 1),
                            pixels[:, 1].astype(np.float32).reshape(-1, 1),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return sampled.reshape(-1, 1 if array.ndim == 2 else array.shape[2])

    def render_reconstruction(self, candidate: Candidate, output_path: str | Path) -> Path:
        """将现场中每片的真实像素按候选位姿重投影，供上位机直观核验。"""
        shapes = [transform_polygon(self.pieces[item_id].polygon, pose)
                  for item_id, pose in candidate.poses.items()]
        points = np.vstack([np.asarray(shape.exterior.coords[:-1], float) for shape in shapes])
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        margin = 8.0
        scale = max(2.0, min(self.scale_x, self.scale_y))
        width = max(32, int(math.ceil((max_x - min_x + 2 * margin) * scale)))
        height = max(32, int(math.ceil((max_y - min_y + 2 * margin) * scale)))
        canvas = np.full((height, width, 3), (215, 215, 215), dtype=np.uint8)
        raw_to_pixel = np.array([[self.scale_x, 0, 0], [0, self.scale_y, 0], [0, 0, 1]], float)
        solution_to_canvas = np.array([[scale, 0, (-min_x + margin) * scale], [0, scale, (-min_y + margin) * scale], [0, 0, 1]], float)
        for item_id, pose in candidate.poses.items():
            source_poly = self.pieces[item_id].polygon
            coords = np.asarray(source_poly.exterior.coords[:-1], float)
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            pixels = np.round(coords * np.array([self.scale_x, self.scale_y])).astype(np.int32)
            cv2.fillPoly(mask, [pixels], 255)
            matrix = solution_to_canvas @ pose @ np.linalg.inv(raw_to_pixel)
            warped = cv2.warpPerspective(self.image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
            warped_mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_CONSTANT)
            canvas[warped_mask > 0] = warped[warped_mask > 0]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), canvas):
            raise ValueError(f"无法保存花纹复原图：{output}")
        return output

    def render_planned_reconstruction(
        self,
        records: list[dict[str, object]],
        output_path: str | Path,
        *,
        status_text: str | None = None,
    ) -> Path:
        """按最终 ``MovePlan`` 目标位置拼接现场裁片，而非绘制几何示意。

        ``records`` 在 clearance 后传入，所以预览中的方向、位置和拼缝间隔与
        机械臂实际将要摆出的结果一致。每块的 alpha mask 来自拍摄图中识别到的
        原始多边形，蓝纸背景不会被带入复原图。
        """
        if not records:
            raise ValueError("没有可供花纹复原的拼图块")
        target_points = np.vstack([
            np.asarray(record["target"], dtype=float) for record in records
        ])
        min_x, min_y = np.min(target_points, axis=0)
        max_x, max_y = np.max(target_points, axis=0)
        # 深色背景和每片的青色描边仅用于上位机核验：白色扑克牌裁片在
        # 白底上会让真实存在的 3--6 mm clearance 看起来像零缝。目标坐标、
        # 花纹像素和 alpha mask 完全不变。
        margin = 10.0
        header_px = 28
        scale = max(2.0, min(self.scale_x, self.scale_y))
        width = max(32, int(math.ceil((max_x - min_x + 2 * margin) * scale)))
        height = max(
            32,
            header_px + int(math.ceil((max_y - min_y + 2 * margin) * scale)),
        )
        canvas = np.full((height, width, 3), (35, 43, 55), dtype=np.uint8)
        raw_to_pixel = np.array(
            [[self.scale_x, 0, 0], [0, self.scale_y, 0], [0, 0, 1]], dtype=float
        )
        target_to_canvas = np.array(
            [[scale, 0, (-min_x + margin) * scale],
             [0, scale, header_px + (-min_y + margin) * scale], [0, 0, 1]], dtype=float
        )
        for record in records:
            observed = np.asarray(record["observed"], dtype=np.float32)
            target = np.asarray(record["target"], dtype=np.float32)
            if len(observed) < 3 or len(target) != len(observed):
                raise ValueError("花纹复原输入顶点不足或不匹配")
            # 求源毫米坐标到最终目标毫米坐标的仿射变换。拼图顶点来自同一
            # 刚体位姿，前三个不共线顶点即可精确确定；若遇到冗余共线轮廓，
            # 则自动寻找可用三角形。
            affine: np.ndarray | None = None
            for first, second, third in itertools.combinations(range(len(observed)), 3):
                source_triangle = observed[[first, second, third]]
                if abs(float(cv2.contourArea(source_triangle))) < 1e-4:
                    continue
                affine = cv2.getAffineTransform(
                    source_triangle, target[[first, second, third]]
                )
                break
            if affine is None:
                raise ValueError("花纹复原无法从共线顶点建立仿射变换")
            source_mm_to_target = np.vstack((affine, [0.0, 0.0, 1.0]))
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            source_pixels = np.round(
                observed * np.array([self.scale_x, self.scale_y], dtype=np.float32)
            ).astype(np.int32)
            cv2.fillPoly(mask, [source_pixels], 255)
            matrix = target_to_canvas @ source_mm_to_target @ np.linalg.inv(raw_to_pixel)
            warped = cv2.warpPerspective(
                self.image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_mask = cv2.warpPerspective(
                mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            canvas[warped_mask > 0] = warped[warped_mask > 0]
            target_canvas = cv2.perspectiveTransform(
                target.reshape(1, -1, 2), target_to_canvas.astype(np.float32)
            ).reshape(-1, 2)
            cv2.polylines(
                canvas,
                [np.round(target_canvas).astype(np.int32)],
                True,
                (255, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        if status_text:
            cv2.putText(
                canvas,
                status_text,
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), canvas):
            raise ValueError(f"无法保存实拍花纹复原图：{output}")
        return output
