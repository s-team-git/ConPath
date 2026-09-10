#!/usr/bin/env python3
"""Freeze validation-only examples and reference witnesses before model inference."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pathrel.formal_data import RADII_CELLS, load_formal_data, sha256, stable_rank

DATA = ROOT / "results/flatlands_external_formal_protocol_v1/data_eligible_v1"
NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def bfs_path(safe, start, goal):
    """A deterministic shortest grid path, or None. Fixed up/left/right/down order."""
    safe = np.asarray(safe, dtype=bool)
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    if not safe[start] or not safe[goal]:
        return None
    height, width = safe.shape
    start_flat, goal_flat = start[0] * width + start[1], goal[0] * width + goal[1]
    previous = np.full(height * width, -1, dtype=np.int32)
    previous[start_flat] = start_flat
    queue = deque([start_flat])
    while queue and previous[goal_flat] == -1:
        current = queue.popleft()
        row, col = divmod(current, width)
        for dy, dx in NEIGHBOURS:
            nr, nc = row + dy, col + dx
            if 0 <= nr < height and 0 <= nc < width:
                neighbour = nr * width + nc
                if safe[nr, nc] and previous[neighbour] == -1:
                    previous[neighbour] = current
                    queue.append(neighbour)
    if previous[goal_flat] == -1:
        return None
    current, path = goal_flat, []
    while current != start_flat:
        path.append(list(divmod(current, width)))
        current = int(previous[current])
    path.append(list(start))
    return list(reversed(path))


def two_internal_vertex_disjoint_paths(safe, start, goal):
    first = bfs_path(safe, start, goal)
    if first is None:
        return None
    remainder = np.asarray(safe, dtype=bool).copy()
    for row, col in first[1:-1]:
        remainder[row, col] = False
    second = bfs_path(remainder, start, goal)
    if second is None:
        return None
    first_internal, second_internal = set(map(tuple, first[1:-1])), set(map(tuple, second[1:-1]))
    if first_internal & second_internal or first == second:
        return None
    return first, second


def clearance(world):
    distances = ndimage.distance_transform_edt(np.pad(np.asarray(world, bool), 1))[1:-1, 1:-1]
    result = np.ceil(distances).astype(np.int64) - 1
    result[~world] = -1
    return result


def explicit_disk_safe(world, radius):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = x * x + y * y <= radius * radius
    return ndimage.binary_erosion(world, structure=disk, border_value=0)


def path_is_valid(path, safe, start, goal):
    if not path or path[0] != list(start) or path[-1] != list(goal):
        return False
    if len(set(map(tuple, path))) != len(path):
        return False
    if not all(safe[tuple(point)] for point in path):
        return False
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))


def describe_case(sample, query_index, category):
    row = sample.row
    case = {"case_category": category, "global_id": row["global_id"], "parent_group": row["parent_group"],
            "scene_id": row["scene_id"], "source": row["source_dataset"],
            "physical_archive_split": row["archive_split"], "candidate_split": row["candidate_split"],
            "validation_only": True, "radii_cells": list(RADII_CELLS),
            "reference_packet_sha256": sha256(DATA / "packets" / (row["global_id"] + ".npz"))}
    if query_index is None:
        case.update(query_id=None, candidate_index=None, start_rc=None, goal_rc=None,
                    reference_events=None, no_eligible_query=True)
    else:
        candidate = int(sample.candidate_indices[query_index])
        case.update(query_id=f"{row['global_id']}:q{candidate:03d}", candidate_index=candidate,
                    start_rc=sample.starts[query_index].tolist(), goal_rc=sample.goals[query_index].tolist(),
                    reference_events=sample.targets[query_index].tolist(), no_eligible_query=False)
    return case


def select_reference_cases(samples):
    output, search_counts = {}, {}
    clearance_cache = {}
    labels = {
        "unreachable": "参考图中自由端点之间不连通（半径0格）",
        "narrow_bottleneck": "参考图内部瓶颈；较大半径的起终点自身均可容纳",
        "multiple_alternatives": "参考图上两条内部节点不相交的栅格路径",
    }
    for category in labels:
        candidates = [(sample, q) for sample in samples for q in range(len(sample.starts))]
        candidates.sort(key=lambda item: (item[0].row["source_dataset"],
                        stable_rank("reference_figure", category, item[0].row["global_id"], int(item[0].candidate_indices[item[1]]))))
        selected, used_parents, examined = [], set(), 0
        for sample, q in candidates:
            if sample.scene_key in used_parents:
                continue
            examined += 1
            start, goal = tuple(sample.starts[q]), tuple(sample.goals[q])
            targets = sample.targets[q]
            accepted, witness = False, {}
            if category == "unreachable":
                accepted = not targets[0] and sample.target[start] and sample.target[goal]
            elif category == "narrow_bottleneck" and targets[0]:
                gid = sample.row["global_id"]
                if gid not in clearance_cache:
                    clearance_cache[gid] = clearance(sample.target)
                clear = clearance_cache[gid]
                for index, radius in enumerate(RADII_CELLS[1:], start=1):
                    if not targets[index] and clear[start] >= radius and clear[goal] >= radius:
                        accepted = True
                        witness = {"bottleneck_failed_radius_cells": radius, "reachable_radius_cells": 0,
                                   "start_clearance_cells": int(clear[start]), "goal_clearance_cells": int(clear[goal])}
                        break
            elif category == "multiple_alternatives":
                gid = sample.row["global_id"]
                if gid not in clearance_cache:
                    clearance_cache[gid] = clearance(sample.target)
                for radius, index in ((10, 1), (0, 0)):
                    if not targets[index]:
                        continue
                    paths = two_internal_vertex_disjoint_paths(clearance_cache[gid] >= radius, start, goal)
                    if paths is not None:
                        accepted = True
                        witness = {"witness_radius_cells": radius, "reference_path_a_rc": paths[0],
                                   "reference_path_b_rc": paths[1], "shared_internal_vertices": 0,
                                   "meaning": "Distinct free centre-cell paths; not necessarily different semantic corridors or disjoint robot footprints",
                                   "search_limit": "This deterministic first-path exclusion algorithm is sufficient but not exhaustive; failure does not prove only one path exists"}
                        break
            if accepted:
                case = describe_case(sample, q, category)
                case.update(witness)
                case["label_zh"] = labels[category]
                case["selection_rule"] = "source name then fixed input-identity SHA256 order; first two qualifying distinct parents; reference geometry only, no model outputs"
                selected.append(case)
                used_parents.add(sample.scene_key)
                if len(selected) == 2:
                    break
        output[category] = {"requested_cases": 2, "available_cases": len(selected), "cases": selected,
                            "status": "available" if len(selected) == 2 else "partly_unavailable" if selected else "unavailable"}
        search_counts[category] = {"candidate_query_count": len(candidates), "queries_examined": examined,
                                   "selected_distinct_parents": len(used_parents)}
    return output, search_counts


def verify_cases(payload, samples):
    by_id = {sample.row["global_id"]: sample for sample in samples}
    cases = payload["general_cases"] + payload["multi_radius_cases"]
    cases += [case for group in payload["reference_categories"].values() for case in group["cases"]]
    receipts, safe_cache = [], {}
    for case in cases:
        sample = by_id[case["global_id"]]
        if case["no_eligible_query"]:
            if len(sample.starts):
                raise ValueError("A nonempty query list was called empty")
            receipts.append({"global_id": case["global_id"], "case_category": case["case_category"], "no_query_verified": True})
            continue
        q = sample.candidate_indices.tolist().index(case["candidate_index"])
        start, goal = tuple(case["start_rc"]), tuple(case["goal_rc"])
        if list(start) != sample.starts[q].tolist() or list(goal) != sample.goals[q].tolist():
            raise ValueError("Figure query changed from frozen input list")
        if not sample.observation[0][start] or not sample.hidden[goal] or not sample.valid[goal]:
            raise ValueError("Figure query is not input-eligible")
        exact_events = []
        for radius in RADII_CELLS:
            key = (case["global_id"], radius)
            if key not in safe_cache:
                safe_cache[key] = explicit_disk_safe(sample.target, radius)
            exact_events.append(bfs_path(safe_cache[key], start, goal) is not None)
        if exact_events != case["reference_events"]:
            raise ValueError("Explicit disk+BFS oracle disagrees with cached label")
        evidence = {"global_id": case["global_id"], "query_id": case["query_id"], "case_category": case["case_category"],
                    "query_matches_frozen_input": True, "explicit_disk_four_neighbour_events": exact_events,
                    "cached_label_agrees": True}
        if case["case_category"] == "unreachable":
            assert not exact_events[0] and sample.target[start] and sample.target[goal]
            evidence["both_target_endpoints_free"] = True
        elif case["case_category"] == "narrow_bottleneck":
            radius = case["bottleneck_failed_radius_cells"]
            safe = safe_cache[(case["global_id"], radius)]
            assert exact_events[0] and not exact_events[list(RADII_CELLS).index(radius)] and safe[start] and safe[goal]
            evidence["both_endpoints_fit_failed_radius"] = True
        elif case["case_category"] == "multiple_alternatives":
            safe = safe_cache[(case["global_id"], case["witness_radius_cells"])]
            a, b = case["reference_path_a_rc"], case["reference_path_b_rc"]
            assert path_is_valid(a, safe, start, goal) and path_is_valid(b, safe, start, goal)
            overlap = set(map(tuple, a[1:-1])) & set(map(tuple, b[1:-1]))
            assert not overlap and a != b
            evidence.update(path_a_vertices=len(a), path_b_vertices=len(b), shared_internal_vertices=0,
                            both_paths_valid_four_neighbour_reference_witnesses=True)
        receipts.append(evidence)
    return {"passed": True, "case_entries_checked": len(cases), "checks": receipts,
            "geometry_oracle": "independent explicit integer-disk binary erosion plus four-neighbour BFS; no trained model involved",
            "new_png_or_test_reads": 0}


def main():
    output = DATA / "figure_cases_v1.json"
    receipt_path = DATA / "figure_geometry_verification_v1.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Frozen figure selection/receipt already exists")
    samples = load_formal_data(DATA, "validation")
    by_id = {sample.row["global_id"]: sample for sample in samples}
    original = json.loads((DATA / "primary_figure_selection_before_images.json").read_text())
    general = []
    for original_case in original["cases"]:
        sample = by_id[original_case["global_id"]]
        case = describe_case(sample, 0 if len(sample.starts) else None, "general")
        case["selection_rule"] = "Original pre-image metadata-only two cases per source, unchanged; first input-hash-ranked query, or explicit none"
        case["label_zh"] = "固定验证场景"
        general.append(case)
    categories, searches = select_reference_cases(samples)
    multi = [{**case, "case_category": "multi_radius", "label_zh": "同一固定查询的0/10/20格footprint对照",
              "selection_rule": "first two nonempty-query general cases in the original fixed list"}
             for case in general if not case["no_eligible_query"]][:2]
    payload = {"id": "flatlands_external_formal_figure_cases_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "validation_only": True, "frozen_before_formal_model_training_and_inference": True,
               "data_seal_sha256": sha256(DATA / "seal.json"),
               "original_primary_selection_sha256": sha256(DATA / "primary_figure_selection_before_images.json"),
               "frozen_queries_sha256": sha256(DATA / "queries.csv"),
               "general_cases": general, "reference_categories": categories, "multi_radius_cases": multi,
               "reference_category_search": searches,
               "source_hashes": {str(Path(__file__).relative_to(ROOT)): sha256(__file__),
                                 "src/pathrel/formal_data.py": sha256(ROOT / "src/pathrel/formal_data.py")},
               "interpretation": "General figures are input-blind. Special cases are reference-geometry strata, selected before any formal model outputs. They are examples, not an unbiased estimate of category frequency or model superiority.",
               "display_requirement": "Every rendered panel must state parent/global scene, query, radius in cells, method, seed and validation-only; no winner/SOTA language"}
    receipt = verify_cases(payload, samples)
    receipt.update(created_utc=datetime.now(timezone.utc).isoformat(), data_seal_sha256=payload["data_seal_sha256"],
                   source_hashes=payload["source_hashes"])
    with output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    receipt["figure_cases_sha256"] = sha256(output)
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": receipt["passed"], "figure_cases_sha256": sha256(output),
                      "receipt_sha256": sha256(receipt_path), "general_cases": len(general),
                      "categories": {key: {"available": value["available_cases"], "global_ids": [c["global_id"] for c in value["cases"]]}
                                     for key, value in categories.items()}, "multi_radius_cases": len(multi)}), flush=True)


if __name__ == "__main__":
    main()
