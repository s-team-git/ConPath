#!/usr/bin/env python3
"""Benchmark the exact batched merge-tree forward against query-count growth.

This is a CPU contract benchmark, not a paper result. It deliberately uses synthetic node scores
and reports the same map/query shapes that a future CUDA exact-forward operator must support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pathrel.labels import batched_merge_tree_bottleneck_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--queries", type=int, nargs="+", default=[32, 128, 512, 2048])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch <= 0 or args.samples <= 0 or args.height <= 0 or args.width <= 0:
        raise SystemExit("batch, samples, height, and width must be positive")
    if not args.queries or any(query_count <= 0 for query_count in args.queries):
        raise SystemExit("queries must contain positive counts")

    rng = np.random.default_rng(args.seed)
    scores = rng.random((args.batch, args.samples, args.height, args.width))
    records: list[dict[str, object]] = []
    for query_count in args.queries:
        starts = rng.integers(0, [args.height, args.width], size=(args.batch, query_count, 2), dtype=np.int64)
        goals = rng.integers(0, [args.height, args.width], size=(args.batch, query_count, 2), dtype=np.int64)
        start_time = time.perf_counter()
        values = batched_merge_tree_bottleneck_scores(scores, starts, goals)
        elapsed = time.perf_counter() - start_time
        records.append(
            {
                "queries": query_count,
                "seconds": elapsed,
                "queries_per_second": args.batch * args.samples * query_count / elapsed,
                "output_shape": list(values.shape),
                "finite": bool(np.all(np.isfinite(values))),
            }
        )

    result = {
        "schema_version": 1,
        "kind": "batched_merge_tree_cpu_benchmark",
        "paper_result": False,
        "seed": args.seed,
        "batch": args.batch,
        "samples": args.samples,
        "map_shape": [args.height, args.width],
        "records": records,
        "claim_boundary": "Synthetic CPU contract benchmark; not a public-data or paper result.",
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
