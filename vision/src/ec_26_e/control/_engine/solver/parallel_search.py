"""Windows spawn 安全的顶层锚定分支并行搜索。

每个 worker 拥有独立搜索树、visited/completed/diagnostics，且不写文件。
主进程执行稳定合并；任何 worker 异常都会回退到单进程可靠路径。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing
from typing import Any

from ..runtime_config import RuntimeConfig
from .observation_uncertainty import ObservationUncertaintyProfile
from .preprocess import ProcessedPiece
from .search import RandomCandidate, solve_random


def _candidate_key(candidate: RandomCandidate) -> tuple:
    return (
        float(candidate.total),
        float(candidate.geometry_score),
        tuple(sorted(item.interface_id for item in candidate.interfaces)),
        tuple(sorted(
            (
                pose.piece_id,
                round(float(pose.angle), 10),
                round(float(pose.translation[0]), 10),
                round(float(pose.translation[1]), 10),
            )
            for pose in candidate.poses
        )),
    )


def _anchor_worker(payload: tuple) -> tuple[list[RandomCandidate], dict[str, Any]]:
    (
        pieces, noise_mm, top_n, appearance_edge_matches,
        allowed_adjacency, config_mapping, anchor_index, uncertainty_profile,
    ) = payload
    config = RuntimeConfig.from_mapping(config_mapping)
    config = replace(config, parallel_mode="off", worker_count=1)
    diagnostics: dict[str, Any] = {}
    candidates = solve_random(
        pieces,
        noise_mm,
        top_n,
        appearance_edge_matches,
        allowed_adjacency,
        diagnostics,
        config,
        None,
        anchor_index=anchor_index,
        uncertainty_profile=uncertainty_profile,
    )
    return candidates, diagnostics


def solve_random_top_level_processes(
    pieces: list[ProcessedPiece],
    noise_mm: float,
    top_n: int,
    appearance_edge_matches: list[dict] | None,
    allowed_adjacency: set[frozenset[str]] | None,
    diagnostics: dict[str, Any],
    config: RuntimeConfig,
    *,
    uncertainty_profile: ObservationUncertaintyProfile | None = None,
) -> list[RandomCandidate]:
    workers = min(config.worker_count, len(pieces))
    if workers <= 1:
        return solve_random(
            pieces, noise_mm, top_n, appearance_edge_matches,
            allowed_adjacency, diagnostics, config,
            uncertainty_profile=uncertainty_profile,
        )
    payloads = [
        (
            pieces, noise_mm, top_n, appearance_edge_matches,
            allowed_adjacency, config.as_dict(), anchor,
            uncertainty_profile,
        )
        for anchor in range(len(pieces))
    ]
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            results = list(executor.map(_anchor_worker, payloads))
    except Exception as exc:
        diagnostics["parallel_fallback"] = True
        diagnostics["parallel_failure"] = repr(exc)
        return solve_random(
            pieces, noise_mm, top_n, appearance_edge_matches,
            allowed_adjacency, diagnostics,
            replace(config, parallel_mode="off", worker_count=1),
            uncertainty_profile=uncertainty_profile,
        )
    merged = sorted(
        (candidate for candidates, _diag in results for candidate in candidates),
        key=_candidate_key,
    )
    unique: list[RandomCandidate] = []
    seen: set[tuple] = set()
    for candidate in merged:
        key = _candidate_key(candidate)[2:]
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= top_n:
            break
    diagnostics.update({
        "parallel_mode": "top_level_processes",
        "worker_count": workers,
        "parallel_fallback": False,
        "anchor_branch_count": len(results),
        "search_nodes": sum(
            int(item.get("search_nodes", 0)) for _candidates, item in results
        ),
        "generated_states": sum(
            int(item.get("generated_states", 0)) for _candidates, item in results
        ),
        "expanded_states": sum(
            int(item.get("expanded_states", 0)) for _candidates, item in results
        ),
        "candidate_count": len(unique),
        "branch_diagnostics": [
            {
                key: value for key, value in item.items()
                if key != "interface_candidates"
            }
            for _candidates, item in results
        ],
    })
    return unique
