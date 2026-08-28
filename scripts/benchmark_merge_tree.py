#!/usr/bin/env python3
"""Benchmark the exact NumPy merge-tree forward against per-query bottleneck search."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathrel.labels import merge_tree_bottleneck_scores  # noqa: E402


def dijkstra_score(scores: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> float:
    height, width = scores.shape
    best = np.full_like(scores, -np.inf, dtype=np.float64)
    best[start] = scores[start]
    queue: list[tuple[float, int, int]] = [(-float(scores[start]), *start)]
    while queue:
        negative_capacity, row, col = heapq.heappop(queue)
        capacity = -negative_capacity
        if capacity < best[row, col]:
            continue
        if (row, col) == goal:
            return float(capacity)
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            candidate = min(capacity, float(scores[n_row, n_col]))
            if candidate > best[n_row, n_col]:
                best[n_row, n_col] = candidate
                heapq.heappush(queue, (-candidate, n_row, n_col))
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--queries", type=int, nargs="+", default=[8, 64, 512])
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "merge_tree_benchmark.json")
    args = parser.parse_args()
    if args.size < 2 or any(count < 1 for count in args.queries):
        raise ValueError("size must be >=2 and query counts must be positive")
    rng = np.random.default_rng(args.seed)
    scores = rng.random((args.size, args.size))
    maximum_queries = max(args.queries)
    starts = [tuple(point) for point in rng.integers(0, args.size, (maximum_queries, 2))]
    goals = [tuple(point) for point in rng.integers(0, args.size, (maximum_queries, 2))]

    rows = []
    for count in args.queries:
        begin = time.perf_counter()
        reference = np.asarray([dijkstra_score(scores, start, goal) for start, goal in zip(starts[:count], goals[:count])])
        dijkstra_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        merge_tree = merge_tree_bottleneck_scores(scores, starts[:count], goals[:count])
        merge_tree_seconds = time.perf_counter() - begin
        np.testing.assert_allclose(merge_tree, reference, rtol=0.0, atol=1e-12)
        rows.append(
            {
                "queries": count,
                "dijkstra_seconds": dijkstra_seconds,
                "merge_tree_seconds": merge_tree_seconds,
                "speedup": dijkstra_seconds / merge_tree_seconds,
                "max_absolute_error": float(np.max(np.abs(merge_tree - reference))),
            }
        )
    report = {
        "seed": args.seed,
        "map_shape": [args.size, args.size],
        "semantics": "exact four-neighbor maximum-minimum node-score path",
        "implementation": "NumPy reference; not a CUDA/differentiable training kernel",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
