"""Synthetic workspace tests and polygon defect reproducers; no device access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vision/src'),str(ROOT/'tests/a4_robust'),str(ROOT/'handover-puzzle')]
from ec_26_e.vision import Resize,ResizeConfig
from ec_26_e.vision import VisionConfig, SegInstance
from ec_26_e.vision.refine import (
    _choose_polygon, _consistent_foreground_component,
    _foreground_connected_to_workspace_edge, _recover_local_foreground, refine_instance,
    _retain_feasible_refinement, _polygon_geometry_is_acceptable,
)
from ec_26_e.vision.foreground import _workspace_mask
from ec_26_e.vision.geometry import polygon_iou,polygon_edge_lengths,polygon_corner_depths
from seed import _seed_from_chroma

TRUTH=np.asarray([[176,78],[974,69],[989,638],[176,646]],np.float32)


def scene(scale=1.,gradient=False):
    h,w=round(720*scale),round(1280*scale)
    corners=TRUTH*scale
    image=np.full((h,w,3),(130,175,205),np.uint8)
    paper=np.zeros((h,w),np.uint8);cv2.fillPoly(paper,[np.rint(corners).astype(np.int32)],255)
    color=np.full((h,w,3),(180,100,35),np.float32)
    if gradient:
        blend=np.linspace(.60,0,w,dtype=np.float32)[None,:,None]
        color=color*(1-blend)+255*blend
    image[paper>0]=color.astype(np.uint8)[paper>0]
    return image,corners


def config(corners,shape):
    h,w=shape[:2]
    # Deliberately imperfect reference must not be returned as an observation.
    shifted=corners+np.asarray((12.,8.))*h/720
    relative=shifted/np.asarray((w-1,h-1))
    return ResizeConfig(target_width=764,target_height=540,
            reference_corners_relative=tuple(map(tuple,relative)),require_observed_boundary=True)


class WorkspaceTests(unittest.TestCase):
    def test_curved_observed_edge_reports_fine_stage_without_warping(self):
        image,corners=scene()
        image[:]=(130,175,205)
        fractions=np.linspace(0,1,120)
        bottom=corners[2]+fractions[:,None]*(corners[3]-corners[2])
        bottom[:,1]+=16*np.sin(np.pi*fractions)
        polygon=np.vstack((corners[:2],bottom))
        cv2.fillPoly(image,[np.rint(polygon).astype(np.int32)],(180,100,35))
        with patch('cv2.warpPerspective',side_effect=AssertionError('Rejected edge must not be warped')):
            with self.assertRaisesRegex(ValueError,r'原图精定位失败.*第 2 条边.*残差过大'):
                Resize(config(corners,image.shape)).resize(image)

    def test_missing_observed_boundary_reports_stage_without_warping(self):
        image=np.full((720,1280,3),(30,30,30),np.uint8)
        with patch('cv2.warpPerspective',side_effect=AssertionError('Missing boundary must not be warped')):
            with self.assertRaisesRegex(ValueError,r'粗定位失败.*第 0 条边'):
                Resize(config(TRUTH,image.shape)).resize(image)

    def test_wood_background_uses_observed_edges_at_multiple_resolutions(self):
        for scale in (1.,1.5,2.):
            with self.subTest(scale=scale):
                image,corners=scene(scale)
                result=Resize(config(corners,image.shape)).resize(image)
                source=np.asarray(result.source_corners_px)
                np.testing.assert_allclose(source,corners,atol=2*scale)
                # Independently known homography: inspect interior points as well as corners.
                target=np.asarray([[0,0],[763,0],[763,539],[0,539]],np.float32)
                truth_h=cv2.getPerspectiveTransform(corners,target)
                points=np.asarray([[[300,200],[600,300],[850,500]]],np.float32)*scale
                expected=cv2.perspectiveTransform(points,truth_h)[0]
                actual=cv2.perspectiveTransform(points,result.homography)[0]
                mm=(actual-expected)*np.asarray((292/763,207/539))
                self.assertLess(float(np.linalg.norm(mm,axis=1).max()),2.)

    def test_exposure_split_hsv_is_recovered_without_new_color_thresholds(self):
        image,corners=scene(gradient=True)
        seed=_seed_from_chroma(image)
        self.assertIsNotNone(seed)
        self.assertEqual(seed.mode,'chroma-four')
        result=Resize(config(seed.corners,image.shape)).resize(image)
        np.testing.assert_allclose(result.source_corners_px,corners,atol=3.)

    def test_uniform_background_cannot_return_reference_as_success(self):
        for color in ((30,30,30),(180,180,180),(180,100,35)):
            with self.subTest(color=color):
                image=np.full((720,1280,3),color,np.uint8)
                with self.assertRaises(ValueError):Resize(config(TRUTH,image.shape)).resize(image)

    def test_clipped_workspace_is_rejected_without_inventing_missing_edge(self):
        image,corners=scene()
        outside=corners.copy();outside[1:3,0]+=500
        image[:]=(130,175,205)
        cv2.fillPoly(image,[outside.astype(np.int32)],(180,100,35))
        reference=outside.copy();reference[:,0]=np.minimum(reference[:,0],1260)
        with self.assertRaises(ValueError):Resize(config(reference-12,image.shape)).resize(image)


class PolygonTests(unittest.TestCase):
    def test_invalid_full_intersection_keeps_supported_feasible_progress(self):
        initial=np.asarray([[20,20],[130,20],[130,110],[20,45]],np.float32)
        fitted=np.asarray([[20,20],[130,20],[130,110],[20,28]],np.float32)
        foreground=np.zeros((150,160),np.uint8)
        cv2.fillPoly(foreground,[fitted.astype(np.int32)],255)
        contour=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0][0]
        cfg=VisionConfig(short_edge_merge_mm=6,minimum_physical_edge_mm=6)
        self.assertTrue(_polygon_geometry_is_acceptable(initial,cfg,.383))
        self.assertFalse(_polygon_geometry_is_acceptable(fitted,cfg,.383))
        result=_retain_feasible_refinement(initial,fitted,contour,cfg,.383)
        self.assertEqual(len(result),len(initial))
        self.assertTrue(_polygon_geometry_is_acceptable(result,cfg,.383))
        self.assertFalse(np.array_equal(result,initial))
        self.assertGreater(polygon_iou(contour,result),polygon_iou(contour,initial))

    def test_feasible_refinement_cannot_abandon_observed_foreground(self):
        initial=np.asarray([[20,20],[130,20],[130,110],[20,45]],np.float32)
        fitted=initial+np.asarray((200,0),np.float32)
        foreground=np.zeros((150,160),np.uint8)
        cv2.fillPoly(foreground,[initial.astype(np.int32)],255)
        contour=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0][0]
        cfg=VisionConfig(short_edge_merge_mm=6,minimum_physical_edge_mm=6)
        result=_retain_feasible_refinement(initial,fitted,contour,cfg,.383)
        np.testing.assert_array_equal(result,initial)

    def test_local_recovery_is_independent_of_previous_opencv_random_state(self):
        generator=np.random.default_rng(37)
        image=np.full((160,220,3),(180,90,30),np.int16)
        image[35:125,50:170]=(165,155,145)
        image=np.clip(image+generator.integers(-28,29,image.shape),0,255).astype(np.uint8)
        coarse=np.zeros((160,220),np.uint8);coarse[38:122,53:167]=255
        core=cv2.erode(coarse,np.ones((7,7),np.uint8))
        search=cv2.dilate(coarse,np.ones((21,21),np.uint8))
        results=[]
        for seed in (7,139,2026):
            cv2.setRNGSeed(seed)
            results.append(_recover_local_foreground(image,coarse,core,search))
        self.assertGreater(cv2.countNonZero(results[0]),0)
        np.testing.assert_array_equal(results[0],results[1])
        np.testing.assert_array_equal(results[0],results[2])

    def test_flooded_global_foreground_uses_existing_local_recovery(self):
        image=np.full((540,764,3),(180,90,30),np.uint8)
        image[100:230,120:250]=(240,240,240)
        coarse=np.zeros((540,764),np.uint8);coarse[102:228,122:248]=255
        polluted=_workspace_mask((764,540))
        cfg=VisionConfig(short_edge_merge_mm=6,minimum_physical_edge_mm=6)
        piece,diagnostic=refine_instance(image,SegInstance((120.,100.,250.,230.),.95,0,coarse),0,cfg,polluted)
        self.assertIsNotNone(piece)
        self.assertEqual(len(piece.corners_px),4)
        self.assertEqual(diagnostic.boundary_source,'local_grabcut')
        self.assertTrue(diagnostic.fallback_used)
        self.assertLess(abs(piece.center_px[0]-184.5),1.)
        self.assertLess(abs(piece.center_px[1]-164.5),1.)

    def test_workspace_connection_requires_an_interior_core(self):
        flooded=_workspace_mask((764,540));core=np.zeros_like(flooded)
        core[100:150,100:150]=255
        self.assertTrue(_foreground_connected_to_workspace_edge(flooded,core))
        core[4:30,100:150]=255
        self.assertFalse(_foreground_connected_to_workspace_edge(flooded,core))

    def test_consistent_component_preserves_corner_beyond_local_band(self):
        foreground=np.zeros((200,240),np.uint8)
        points=np.asarray([[20,35],[180,30],[120,180],[25,100]],np.int32)
        cv2.fillPoly(foreground,[points],255)
        coarse=foreground.copy();coarse[161:]=0
        core=cv2.erode(coarse,np.ones((7,7),np.uint8))
        result=_consistent_foreground_component(foreground,coarse,core,.85)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result,foreground)
        # Full component evidence is independent of which corner the model lost.
        result_poly,_=_choose_polygon(max(cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea),VisionConfig(short_edge_merge_mm=6,minimum_physical_edge_mm=6),.383)
        self.assertEqual(len(result_poly),4)

    def test_attached_background_cannot_be_recovered_as_instance(self):
        coarse=np.zeros((200,240),np.uint8);coarse[50:150,60:160]=255
        foreground=coarse.copy();foreground[:,130:]=255
        core=cv2.erode(coarse,np.ones((7,7),np.uint8))
        self.assertIsNone(_consistent_foreground_component(foreground,coarse,core,.85))

    def test_multiple_weak_bevel_vertices_keep_a_feasible_polygon(self):
        # Minimal reproducer of the observed bevel defect, not a holdout case.
        # Depth-only deletion loses a feasible physical interpretation.
        points=np.asarray([[19,23],[21,31],[131,102],[188,35],[32,18]],np.int32)
        contour=np.zeros((140,220),np.uint8)
        cv2.fillPoly(contour,[points],255)
        contours,_=cv2.findContours(contour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cfg=VisionConfig(short_edge_merge_mm=6,minimum_physical_edge_mm=6)
        result,_=_choose_polygon(contours[0],cfg,.383)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(polygon_iou(contours[0],result),cfg.minimum_polygon_iou)
        self.assertGreater(float(polygon_edge_lengths(result).min())*.383,6)
        self.assertGreaterEqual(float(polygon_corner_depths(result).min())*.383,4)

    def test_iou_counts_fitted_polygon_outside_original_contour(self):
        small=np.asarray([[10,10],[110,10],[110,110],[10,110]],np.int32)
        large=np.asarray([[-90,-90],[210,-90],[210,210],[-90,210]],np.float32)
        self.assertLess(polygon_iou(small,large),.2)
        self.assertAlmostEqual(polygon_iou(small,large),polygon_iou(large.astype(np.int32),small.astype(np.float32)))


if __name__=='__main__':unittest.main()
