"""Collapse observation-sized pose changes, while preserving different assemblies."""
import numpy as np
from ..transforms import estimate_rigid, transform_points


def same_assembly(first, second, tolerance_mm=2.0):
    if not getattr(first, 'records', None) or not getattr(second, 'records', None):
        return first.candidate_id == second.candidate_id
    a = {str(r['piece_id']):np.asarray(r['target']) for r in first.records}
    b = {str(r['piece_id']):np.asarray(r['target']) for r in second.records}
    if set(a) != set(b):
        return False
    root = min(a)
    if a[root].shape != b[root].shape:
        return False
    angle, translation, _ = estimate_rigid(a[root], b[root])
    for key in a:
        if a[key].shape != b[key].shape:
            return False
        distances = np.linalg.norm(transform_points(a[key],angle,translation)-b[key],axis=1)
        if float(np.sqrt(np.mean(distances**2))) > tolerance_mm:
            return False
    return True


def distinct_safe_variants(variants):
    result = []
    for variant in sorted((v for v in variants if v.safe and v.machine_executable),key=lambda v:v.quality_key):
        if not any(same_assembly(variant,old) for old in result):
            result.append(variant)
    return result
