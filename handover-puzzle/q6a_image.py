#!/usr/bin/env python3
"""File-only Q6A HTP validation and optional solver; never opens peripherals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'vision/src'),str(ROOT/'tests/a4_robust'),str(ROOT)]
import cv2
import numpy as np
from ec_26_e.vision import Resize, ResizeConfig, VisionConfig, VisionDetector
from ec_26_e.vision.qnn import QnnYolo11SegInferencer


def rectify_image(image, params_name='new'):
    """Use existing formal HSV candidates and their selection order, no best-effort corners."""
    import test_a4_detection as a4
    from task1_yolo_flow import _formal_a4_candidates
    from seed import seed_a4_robust
    valid, diagnostics = [], []
    for name,args in _formal_a4_candidates(dict(a4.PARAM_PRESETS[params_name])):
        try:
            seed = seed_a4_robust(image,args)
            valid.append((name,args,seed))
            diagnostics.append({'candidate':name,'status':'ok','mode':seed.mode})
        except ValueError as error:
            diagnostics.append({'candidate':name,'status':'failed','error':str(error)})
    if not valid:
        raise ValueError('所有正式 A4 候选均失败：'+json.dumps(diagnostics,ensure_ascii=False))
    name,args,seed = max(valid,key=lambda item:(sum(bool(x) for x in item[2].reliability),
                          -float(item[2].aspect_error),float(item[2].area_ratio)))
    h,w=image.shape[:2]
    relative=seed.corners/np.asarray((w-1,h-1))
    result=Resize(ResizeConfig(target_width=764,target_height=540,
                  reference_corners_relative=tuple(map(tuple,relative)),
                  require_observed_boundary=True)).resize(image)
    return result,{'selected_candidate':name,'seed_mode':seed.mode,'candidates':diagnostics}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,type=Path,help='本地真实图片')
    parser.add_argument('--rectified',action='store_true',help='图片已经过 A4 校正')
    parser.add_argument('--full-frame',action='store_true',help='只输出全图 NPU 实例，不作毫米几何或求解')
    parser.add_argument('--solve',action='store_true',help='将精修结果送入随机几何求解器，仅保存候选')
    parser.add_argument('--timeout',type=float,default=20,help='求解预算秒数')
    parser.add_argument('--params',choices=('current','new'),default='new')
    parser.add_argument('--source-region',choices=('left','all'),default='left',help='默认沿用左侧来源区；all 只用于离线诊断')
    parser.add_argument('--output',type=Path,default=ROOT/'host/outputs/q6a-images'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    args=parser.parse_args(argv)
    if args.rectified and args.full_frame: parser.error('--rectified 与 --full-frame 互斥')
    if args.full_frame and args.solve: parser.error('全图实例没有 A4 毫米坐标，不能求解')
    if args.timeout<=0: parser.error('--timeout 必须为正数')
    source=args.image.resolve()
    if not source.is_file() or source.is_relative_to('/dev'): parser.error('必须指定普通图片文件')
    args.output.mkdir(parents=True,exist_ok=True)
    report={'source':str(source),'backend':'QNN HTP','cpu_fallback':False,'camera_opened':False,
            'motion_accepted':False,'source_region':args.source_region,'status':'failed'}
    started=time.perf_counter()
    try:
        image=cv2.imread(str(source))
        if image is None: raise ValueError('无法读取图片')
        if not args.rectified and not args.full_frame:
            resized,diagnostics=rectify_image(image,args.params)
            image=resized.image_bgr;report['workspace']={**resized.to_dict(),**diagnostics}
        config=VisionConfig(short_edge_merge_mm=6,minimum_physical_edge_mm=6)
        with QnnYolo11SegInferencer(config) as inferencer:
            if args.full_frame:
                from ec_26_e.vision.detector import _suppress_contained_duplicates
                instances,removed=_suppress_contained_duplicates(inferencer.infer(image))
                report['instances']=[{'box_px':list(s.box_px),'score':s.score,'class_id':s.class_id,
                                      'mask_file':f'mask_{i:02d}.png'} for i,s in enumerate(instances)]
                for i,s in enumerate(instances):cv2.imwrite(str(args.output/f'mask_{i:02d}.png'),s.mask*255)
                report['contained_duplicates_removed']=removed
                report['timing']=inferencer.last_timing.to_dict()
            else:
                detector=VisionDetector(config,inferencer=inferencer)
                result=detector.detect(image)
                report['vision']=result.to_dict();report['diagnostics']=detector.last_diagnostics.to_dict()
                pieces=tuple(result.pieces)
                if args.source_region=='left':
                    from source_region_filter import filter_left_source_pieces
                    filtered=filter_left_source_pieces(pieces,max_x_ratio=.52)
                    pieces=filtered.accepted
                    report['source_rejected']=len(filtered.rejected)
                report['selected_pieces']=len(pieces)
                for piece in pieces:
                    cv2.polylines(image,[np.rint(piece.corners_px).astype(np.int32)],True,(0,0,255),2)
                if args.solve:
                    if not 2<=len(pieces)<=4: raise ValueError('求解需要 2～4 个通过精修及来源区检查的实例')
                    # Reject incomplete geometry instead of silently solving a partial set.
                    d=detector.last_diagnostics
                    if d.refined_count!=d.yolo_count-d.contained_duplicate_count:
                        raise ValueError('部分 NPU 实例未通过几何精修，拒绝对缺片结果求解')
                    from ec_26_e.control import PuzzlePiece
                    from task1_yolo_flow import _portrait_vertices
                    from random_white_solver_adapter import solve_random_white
                    observed=tuple(PuzzlePiece(str(i),_portrait_vertices(tuple(p.corners_relative),'clockwise',207,292))
                                   for i,p in enumerate(pieces))
                    report['solver']=solve_random_white(observed,args.output/'solver',timeout_s=args.timeout)
                    report['solver']['executable']=False
                    report['solver']['machine_executable']=False
                    report['solver']['offline_only']=True
                    report['solver']['execution_blocked_reason']='本地单图验证，未通过连续帧和运动验收'
                    (args.output/'solver/solution.json').write_text(json.dumps(report['solver'],ensure_ascii=False,indent=2)+'\n')
                    report['solver_note']='单张照片几何候选，未通过连续帧/运动验收，禁止直接作为执行命令'
        cv2.imwrite(str(args.output/'result.png'),image)
        report['status']='ok'
    except Exception as error:
        report['error']=str(error)
    report['wall_seconds']=time.perf_counter()-started
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'report':str(args.output/'report.json'),
                      'error':report.get('error')},ensure_ascii=False))
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
