#!/usr/bin/env python3
"""Deterministic geometry stress test for the standalone puzzle solver.

It deliberately generates only cuts that satisfy the contest's basic geometric
limits.  This is a regression/coverage tool, not a replacement for real-camera
contour tests.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import PuzzleDocument
from core.solver import RectanglePuzzleSolver


def trusted(result) -> bool:
    return any(
        candidate.metrics["coverage"] >= 0.99
        and candidate.metrics["perimeter_ratio"] >= 0.99
        and candidate.metrics["overlap_mm2"] <= 0.80
        and candidate.metrics["frame_corners"] >= 4.0
        and candidate.metrics["outer_piece_count"] >= len(candidate.poses)
        for candidate in result.candidates
    )


def leftmost_piece(document: PuzzleDocument) -> int:
    return min(document.pieces, key=lambda piece_id: document.pieces[piece_id].centroid.x)


def build_straight(rng: random.Random) -> PuzzleDocument:
    document = PuzzleDocument(100.0, 60.0)
    x = rng.uniform(30.0, 70.0)
    ok, message = document.cut(0, (x, -10.0), (x, 70.0))
    assert ok, message
    return document


def build_v_cut(rng: random.Random) -> PuzzleDocument:
    document = PuzzleDocument(100.0, 60.0)
    bend_x = rng.uniform(42.0, 58.0)
    bend_y = rng.choice([rng.uniform(12.0, 20.0), rng.uniform(40.0, 48.0)])
    ok, message = document.cut_polyline(0, [(-10.0, 30.0), (bend_x, bend_y), (110.0, 30.0)])
    assert ok, message
    return document


def build_t_cut(rng: random.Random) -> PuzzleDocument:
    document = PuzzleDocument(100.0, 60.0)
    x = rng.uniform(35.0, 65.0)
    y = rng.uniform(22.0, 38.0)
    ok, message = document.cut(0, (x, -10.0), (x, 70.0))
    assert ok, message
    left_id = leftmost_piece(document)
    ok, message = document.cut(left_id, (-10.0, y), (110.0, y))
    assert ok, message
    return document


def build_four_grid(rng: random.Random) -> PuzzleDocument:
    document = PuzzleDocument(100.0, 60.0)
    x = rng.uniform(38.0, 62.0)
    y = rng.uniform(23.0, 37.0)
    ok, message = document.cut(0, (x, -10.0), (x, 70.0))
    assert ok, message
    for piece_id in sorted(document.pieces, key=lambda item: document.pieces[item].centroid.x):
        ok, message = document.cut(piece_id, (-10.0, y), (110.0, y))
        assert ok, message
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated white-puzzle solver stress cases.")
    parser.add_argument("--count", type=int, default=12, help="number of cases (default: 12)")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--timeout", type=float, default=20.0, help="per-case solve limit in seconds")
    parser.add_argument("--jitter", type=float, default=0.0, help="observed vertex jitter in mm")
    parser.add_argument("--four-piece", action="store_true", help="include slower four-rectangle grid cases")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    builders = [("straight", build_straight), ("v-cut", build_v_cut), ("t-cut", build_t_cut)]
    if args.four_piece:
        builders.append(("four-grid", build_four_grid))

    passed = 0
    timings: list[float] = []
    started = time.monotonic()
    for index in range(args.count):
        name, builder = builders[index % len(builders)]
        try:
            document = builder(rng)
        except AssertionError as exc:
            print(f"SKIP {index + 1:02d} {name}: {exc}")
            continue
        document.scatter(rng.randrange(1_000_000_000), args.jitter)
        result = RectanglePuzzleSolver().solve(document.solver_pieces(), args.timeout)
        good = trusted(result)
        timings.append(result.elapsed_s)
        best = result.candidates[0].metrics["coverage"] if result.candidates else 0.0
        print(
            f"{'PASS' if good else 'FAIL'} {index + 1:02d} {name:<10} "
            f"pieces={len(document.pieces)} status={result.status} "
            f"time={result.elapsed_s:.2f}s coverage={best:.5f}"
        )
        passed += int(good)

    print(
        f"SUMMARY {passed}/{len(timings)} passed; "
        f"slowest={max(timings, default=0.0):.2f}s; total={time.monotonic() - started:.2f}s"
    )
    if passed != len(timings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
