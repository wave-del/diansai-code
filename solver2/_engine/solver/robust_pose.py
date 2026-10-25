"""固定模板使用的鲁棒二维刚体位姿估计。"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares

from ..transforms import estimate_rigid, normalize_angle, transform_points


def weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray, float]:
    src, dst = np.asarray(source, float), np.asarray(target, float)
    w = np.clip(np.asarray(weights, float), 1e-6, None)
    w /= w.sum()
    src_c, dst_c = (src * w[:, None]).sum(axis=0), (dst * w[:, None]).sum(axis=0)
    h = (src - src_c).T @ ((dst - dst_c) * w[:, None])
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    angle = normalize_angle(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))
    translation = dst_c - src_c @ rotation.T
    errors = np.linalg.norm(src @ rotation.T + translation - dst, axis=1)
    return angle, translation, float(np.sqrt(np.average(errors**2, weights=w)))


def huber_irls_kabsch(
    source: np.ndarray, target: np.ndarray, base_weights: np.ndarray | None = None, delta: float = 2.0
) -> tuple[float, np.ndarray, float]:
    weights = np.ones(len(source)) if base_weights is None else np.asarray(base_weights, float).copy()
    angle, translation, rms = weighted_kabsch(source, target, weights)
    for _ in range(20):
        errors = np.linalg.norm(transform_points(source, angle, translation) - target, axis=1)
        robust = np.where(errors <= delta, 1.0, delta / np.maximum(errors, 1e-9))
        new_angle, new_translation, new_rms = weighted_kabsch(source, target, weights * robust)
        if abs(normalize_angle(new_angle - angle)) < 1e-7 and np.linalg.norm(new_translation - translation) < 1e-7:
            break
        angle, translation, rms = new_angle, new_translation, new_rms
    return angle, translation, rms


def ransac_se2(
    source: np.ndarray, target: np.ndarray, threshold_mm: float = 3.0
) -> tuple[float, np.ndarray, float]:
    if len(source) < 2:
        return estimate_rigid(source, target)
    best: tuple[int, float, float, np.ndarray] | None = None
    for first in range(len(source)):
        for second in range(first + 1, len(source)):
            angle, translation, _ = estimate_rigid(source[[first, second]], target[[first, second]])
            errors = np.linalg.norm(transform_points(source, angle, translation) - target, axis=1)
            inliers = errors <= threshold_mm
            score = (int(inliers.sum()), -float(np.median(errors)), angle, translation)
            if best is None or score[:2] > best[:2]:
                best = score
    assert best is not None
    angle, translation = best[2], best[3]
    errors = np.linalg.norm(transform_points(source, angle, translation) - target, axis=1)
    inliers = errors <= threshold_mm
    if inliers.sum() >= 2:
        angle, translation, _ = estimate_rigid(source[inliers], target[inliers])
    final = np.linalg.norm(transform_points(source, angle, translation) - target, axis=1)
    return angle, translation, float(np.sqrt(np.mean(final**2)))


def soft_l1_pose(
    source: np.ndarray, target: np.ndarray, initial: tuple[float, np.ndarray, float], weights: np.ndarray
) -> tuple[float, np.ndarray, float]:
    angle0, translation0, _ = initial
    sqrt_weights = np.sqrt(weights[:, None])
    target_edges = np.roll(target, -1, axis=0) - target
    edge_scale = 0.35 / np.maximum(np.linalg.norm(target_edges, axis=1), 1e-6)

    def residual(params: np.ndarray) -> np.ndarray:
        fitted = transform_points(source, params[0], params[1:3])
        vertex = (fitted - target) * sqrt_weights
        observed_edges = np.roll(fitted, -1, axis=0) - fitted
        cross = observed_edges[:, 0] * target_edges[:, 1] - observed_edges[:, 1] * target_edges[:, 0]
        return np.concatenate((vertex.ravel(), edge_scale * cross))

    def jacobian(params: np.ndarray) -> np.ndarray:
        # 求解器的角度单位是度；解析 SE(2) 导数避免每轮三次有限差分。
        radians = math.radians(float(params[0]))
        sine, cosine = math.sin(radians), math.cos(radians)
        derivative_rotation = np.asarray([[-sine, -cosine], [cosine, -sine]]) * (math.pi / 180.0)
        derivative = source @ derivative_rotation.T
        jac = np.zeros((3 * len(source), 3), dtype=float)
        jac[:2 * len(source), 0] = (derivative * sqrt_weights).ravel()
        jac[:2 * len(source):2, 1] = sqrt_weights[:, 0]
        jac[1:2 * len(source):2, 2] = sqrt_weights[:, 0]
        edge_derivative = np.roll(derivative, -1, axis=0) - derivative
        jac[2 * len(source):, 0] = edge_scale * (
            edge_derivative[:, 0] * target_edges[:, 1]
            - edge_derivative[:, 1] * target_edges[:, 0]
        )
        return jac

    result = least_squares(
        residual, np.r_[angle0, translation0], jac=jacobian,
        loss="soft_l1", f_scale=1.5, max_nfev=500
    )
    angle = normalize_angle(float(result.x[0]))
    translation = result.x[1:3]
    errors = np.linalg.norm(transform_points(source, angle, translation) - target, axis=1)
    return angle, translation, float(np.sqrt(np.mean(errors**2)))


def automatic_vertex_weights(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    n = len(source)
    weights = np.ones(n) if confidence is None else np.clip(np.asarray(confidence, float), 0.05, 1.0)
    lengths = np.linalg.norm(np.roll(source, -1, axis=0) - source, axis=1)
    scale = max(float(np.median(lengths)), 1e-6)
    for index in range(n):
        local = min(lengths[index - 1], lengths[index]) / scale
        weights[index] *= float(np.clip(local, 0.35, 1.5))
        a = source[index - 1] - source[index]
        b = source[(index + 1) % n] - source[index]
        cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
        angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        if angle < 30.0:
            weights[index] *= 0.5
    source_ratios = lengths / max(lengths.sum(), 1e-9)
    target_lengths = np.linalg.norm(np.roll(target, -1, axis=0) - target, axis=1)
    target_ratios = target_lengths / max(target_lengths.sum(), 1e-9)
    mismatch = np.abs(source_ratios - target_ratios)
    weights *= np.exp(-4.0 * (mismatch + np.roll(mismatch, 1)))
    return np.clip(weights, 0.03, None)


def compare_pose_estimators(
    source: np.ndarray, target: np.ndarray, confidence: np.ndarray | None = None,
    *, initial_fits=None, full_diagnostics: bool = True,
) -> tuple[float, np.ndarray, float, str, dict[str, float]]:
    ordinary = estimate_rigid(source, target) if initial_fits is None else initial_fits[0].pose()
    ordinary_errors = np.linalg.norm(
        transform_points(source, ordinary[0], ordinary[1]) - target, axis=1
    )
    median_error = max(float(np.median(ordinary_errors)), 1e-9)
    tail_margin = float(ordinary_errors.max()) - 2.5 * median_error
    if initial_fits is not None and abs(tail_margin) < 0.01:
        return compare_pose_estimators(source, target, confidence, full_diagnostics=full_diagnostics)
    # The existing selection rule always chooses ordinary Kabsch without a
    # heavy tail. Production need not run four discarded estimators solely to
    # populate diagnostics; detailed/offline calls retain their complete report.
    if not full_diagnostics and tail_margin <= 0.0:
        rms = float(np.sqrt(np.mean(ordinary_errors**2)))
        return ordinary[0], ordinary[1], rms, "kabsch", {"kabsch": rms}
    weights = automatic_vertex_weights(source, target, confidence)
    methods = {
        "kabsch": ordinary,
        "weighted_kabsch": (weighted_kabsch(source, target, weights)
                            if initial_fits is None else initial_fits[1].pose()),
        "huber_irls": huber_irls_kabsch(source, target, weights),
        "ransac_se2": ransac_se2(source, target),
    }
    methods["soft_l1_edge_joint"] = soft_l1_pose(source, target, methods["huber_irls"], weights)
    diagnostics: dict[str, float] = {}
    scores: list[float] = []
    best = None
    for name, value in methods.items():
        angle, translation, _ = value
        errors = np.linalg.norm(transform_points(source, angle, translation) - target, axis=1)
        score = float(np.median(errors) + 0.35 * np.sqrt(np.average(errors**2, weights=weights)))
        diagnostics[name] = float(np.sqrt(np.mean(errors**2)))
        scores.append(score)
        if best is None or score < best[0]:
            best = (score, angle, translation, diagnostics[name], name)
    assert best is not None
    # 无明显离群点时，普通 Kabsch 是独立同分布顶点噪声下的最小方差估计；
    # 只有残差呈明显重尾时才切换到鲁棒估计。
    if initial_fits is not None:
        # Float32 noise must not decide a near tie or change the heavy-tail branch.
        ordered = sorted(scores)
        if tail_margin > 0.0 and ordered[1] - ordered[0] < 0.01:
            return compare_pose_estimators(source, target, confidence, full_diagnostics=full_diagnostics)
    if tail_margin <= 0.0:
        return ordinary[0], ordinary[1], diagnostics["kabsch"], "kabsch", diagnostics
    return best[1], best[2], best[3], best[4], diagnostics
