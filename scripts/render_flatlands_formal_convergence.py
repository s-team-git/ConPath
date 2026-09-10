#!/usr/bin/env python3
"""Side-by-side staged-convergence evidence from frozen saved validation worlds.

Rows are always initialization / 1000 updates / 5000 updates. Map methods show
observed / reference / every actual world: four for stochastic methods and
LaMa's true ensemble, one for the deterministic control. Direct-query models
show observed / reference / their saved event score, with map output N/A.
The script never infers, trains, duplicates worlds or approves a visual gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pathrel.formal_checkpoint import atomic_json
from pathrel.formal_data import load_formal_data, sha256
from pathrel.formal_inference import LABELS, METHODS, RADII
from pathrel.formal_metrics import exact_world_events
from scripts.render_flatlands_formal_report import (SEEDS, configure_plotting, discover_evaluations,
                                                   layout_map_grid, load_saved_prediction, map_legend,
                                                   map_panel, save_figure, frozen_figure_artifacts)


STAGE_STEPS = (("untrained", 0), ("smoke", 1000), ("pilot", 5000))
STAGE_LABELS = {"untrained": "随机初始化 · 0步", "smoke": "稳定性阶段 · 1000步", "pilot": "收敛阶段 · 5000步"}


def output_contract(method):
    if method not in METHODS:
        raise ValueError("Unregistered convergence method")
    k = None if method == "direct_query" else 1 if method == "deterministic" else 4
    return {"actual_world_count": k, "common_world_budget": 4,
            "event_score_semantics": "direct_query_sigmoid" if k is None else "raw_world_event_frequency",
            "column_layout": ["observed", "reference"] + (["direct_event_score"] if k is None else [f"world_{i}" for i in range(k)]),
            "world_order": None if k is None else "one actual deterministic world; never replicated" if k == 1
                           else "exact saved order; LaMa independent members 0..3, other models samples 0..3"}


def frozen_cases(manifest):
    """Deduplicate frozen queries while preserving all category/radius groups."""
    entries = [("general", c) for c in manifest["general_cases"]]
    for category in ("unreachable", "narrow_bottleneck", "multiple_alternatives"):
        entries.extend((category, c) for c in manifest["reference_categories"][category]["cases"])
    entries.extend(("multi_radius", c) for c in manifest["multi_radius_cases"])
    selected = {}
    for group, case in entries:
        key = (case["global_id"], case.get("candidate_index"))
        if case.get("validation_only") is not True or case.get("candidate_split") != "validation" or case.get("physical_archive_split") != "train":
            raise ValueError("Convergence pictures require frozen physical-train validation cases")
        if key not in selected:
            selected[key] = {"case": dict(case), "groups": [], "radii_to_render": []}
        else:
            original = selected[key]["case"]
            for field in ("query_id", "start_rc", "goal_rc", "reference_events", "parent_group"):
                if original.get(field) != case.get(field):
                    raise ValueError("Duplicate frozen query has inconsistent geometry")
        item = selected[key]
        if group not in item["groups"]:
            item["groups"].append(group)
        radii = list(RADII) if group == "multi_radius" else [case.get("witness_radius_cells", case.get("bottleneck_failed_radius_cells", 10))]
        for radius in radii:
            if radius not in RADII:
                raise ValueError("Only frozen grid-cell radii may be rendered")
            if radius not in item["radii_to_render"]:
                item["radii_to_render"].append(radius)
    for item in selected.values():
        item["radii_to_render"].sort()
    # Dict insertion order is the original frozen manifest order, not score order.
    return list(selected.values())


def stage_records(records, method, seed):
    selected = {}
    for stage, step in STAGE_STEPS:
        candidates = [r for r in records if r["method"] == method and r["seed"] == seed and tuple(r["group"]) == (stage, step)]
        if len(candidates) > 1:
            raise ValueError("Discovery did not resolve repeated attempts unambiguously")
        selected[stage] = candidates[0] if candidates else None
    return selected


def match_query(sample, case):
    if sample.row["global_id"] != case["global_id"] or sample.row["parent_group"] != case["parent_group"]:
        raise ValueError("Frozen scene identity does not match the cached packet")
    if case.get("no_eligible_query"):
        if len(sample.candidate_indices):
            raise ValueError("A frozen no-query case unexpectedly contains queries")
        return None
    matches = np.flatnonzero(sample.candidate_indices == case["candidate_index"])
    if len(matches) != 1:
        raise ValueError("Frozen candidate query is missing or duplicated")
    query = int(matches[0])
    if not np.array_equal(sample.starts[query], case["start_rc"]) or not np.array_equal(sample.goals[query], case["goal_rc"]):
        raise ValueError("Frozen start/goal geometry changed")
    if case["query_id"] != f"{case['global_id']}:q{int(case['candidate_index']):03d}":
        raise ValueError("Frozen query identifier is inconsistent")
    return query


def checked_worlds(saved, sample, case, *, expected_k=4):
    """Validate exact world events without changing any world's saved order."""
    query = match_query(sample, case)
    worlds = np.asarray(saved["worlds"])
    if expected_k not in (1, 4) or worlds.shape != (expected_k, *sample.valid.shape) or not np.isin(worlds, (0, 1)).all():
        raise ValueError("Each stage must contain exactly the registered actual binary world count")
    if worlds[:, ~sample.valid].any() or np.any(worlds[:, ~sample.hidden] != sample.observation[0, ~sample.hidden]):
        raise ValueError("Saved worlds violate observed/support constraints")
    if query is None:
        actual = np.empty((expected_k, 0, len(RADII)), dtype=bool)
        reference = np.empty((0, len(RADII)), dtype=bool)
    else:
        actual = exact_world_events(worlds, sample.starts[query:query+1], sample.goals[query:query+1], RADII)
        reference = exact_world_events(sample.target[None], sample.starts[query:query+1], sample.goals[query:query+1], RADII)[0]
        if not np.array_equal(actual[:, 0], saved["world_events"][:, query]) or not np.array_equal(actual[:, 0].mean(0), saved["event_scores"][query]):
            raise ValueError("Saved event frequency does not match exact per-world connectivity")
        if not np.array_equal(reference[0], sample.targets[query]) or not np.array_equal(reference[0], case["reference_events"]):
            raise ValueError("Reference footprint event changed")
    hashes = [hashlib.sha256(np.ascontiguousarray(world, dtype=np.uint8).tobytes()).hexdigest() for world in worlds]
    return {"worlds": worlds, "events": actual[:, 0] if query is not None else None,
            "reference_events": reference[0] if query is not None else None,
            "world_order_sha256": hashes, "query_array_index": query,
            "event_scores": saved["event_scores"][query].copy() if query is not None else None,
            "actual_world_count": expected_k, "event_score_semantics": "raw_world_event_frequency"}


def checked_prediction(saved, sample, case, method):
    contract = output_contract(method)
    query = match_query(sample, case)
    scores = np.asarray(saved["event_scores"])
    if scores.shape != (len(sample.candidate_indices), len(RADII)) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Saved event scores must be finite, in [0,1], and match the complete frozen query/radius array")
    if contract["actual_world_count"] is not None:
        k = contract["actual_world_count"]
        if np.asarray(saved["world_events"]).shape != (k, len(sample.candidate_indices), len(RADII)):
            raise ValueError("World event array differs from the actual K/query/radius contract")
        return checked_worlds(saved, sample, case, expected_k=k)
    if any(key in saved for key in ("worlds", "world_events", "continuous_score")):
        raise ValueError("Direct-query predictions must not invent map/world artifacts")
    reference = None
    if query is not None:
        reference = exact_world_events(sample.target[None], sample.starts[query:query+1], sample.goals[query:query+1], RADII)[0, 0]
        if not np.array_equal(reference, sample.targets[query]) or not np.array_equal(reference, case["reference_events"]):
            raise ValueError("Reference footprint event changed")
    return {"worlds": None, "events": None, "reference_events": reference, "world_order_sha256": None,
            "query_array_index": query, "event_scores": scores[query].copy() if query is not None else None,
            "actual_world_count": None, "event_score_semantics": "direct_query_sigmoid"}


def direct_score_panel(ax, sample, case, checked, seed, radius, stage):
    """A numeric score panel, explicitly not a synthesized map or vote count."""
    scene = sample.row.get("scene_id", sample.row["parent_group"])
    ax.set_facecolor("#f2f5f3"); ax.set_box_aspect(1)
    ax.set_title(f"{LABELS['direct_query']}\nscene={scene}\n{case['global_id']} · {case.get('query_id') or '无合格查询'}\n"
                 f"r={radius} 格 · seed={seed}\nvalidation-only\n{STAGE_LABELS[stage]}", fontsize=7.4, pad=7)
    if checked is None:
        score, status = "待完成", "本阶段尚无可展示的直接查询分数"
    elif checked["event_scores"] is None:
        score, status = "N/A", "冻结案例没有合格查询；不补造分数"
    else:
        index = RADII.index(radius)
        score = f"{float(checked['event_scores'][index]):.6f}"
        status = "固定查询参考：" + ("可达" if bool(checked["reference_events"][index]) else "不可达")
    ax.text(.5, .77, "地图输出：N/A（该方法不生成地图）", ha="center", transform=ax.transAxes, fontsize=9)
    ax.text(.5, .60, "网络原始事件分数", ha="center", transform=ax.transAxes, fontsize=10)
    ax.text(.5, .43, score, ha="center", transform=ax.transAxes, fontsize=25, color="#137d6e")
    ax.text(.5, .25, status, ha="center", transform=ax.transAxes, fontsize=9)
    ax.text(.5, .12, "0–1连续分数；没有世界样本或可达票数", ha="center", transform=ax.transAxes, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def completeness(selected, case_count, rendered_case_count):
    missing = [stage for stage, _ in STAGE_STEPS if selected[stage] is None]
    return {"complete": not missing and case_count == rendered_case_count,
            "missing_stages": missing, "unique_frozen_queries": case_count,
            "rendered_unique_queries": rendered_case_count,
            "visual_gate_passed": None,
            "visual_gate_status": "Requires an independent visual review; rendering never approves convergence"}


def render_case(item, sample, selected, method, seed, output, inventory):
    case = item["case"]
    contract = output_contract(method)
    k = contract["actual_world_count"]
    query = match_query(sample, case)
    stage_predictions, stage_sources = {}, {}
    for stage, _ in STAGE_STEPS:
        record = selected[stage]
        if record is None:
            stage_predictions[stage] = None
            stage_sources[stage] = {"status": "pending", "metrics_sha256": None, "prediction_sha256": None, "world_order_sha256": None,
                                    "actual_world_count": None, "expected_world_count": k, "events_by_world_and_radius": None,
                                    "event_scores_by_radius": None, "reference_events_by_radius": None,
                                    "event_score_semantics": contract["event_score_semantics"]}
            continue
        saved = load_saved_prediction(record, sample)
        checked = checked_prediction(saved, sample, case, method)
        stage_predictions[stage] = checked
        path = record["path"] / "validation/predictions" / f"{case['global_id']}.npz"
        stage_sources[stage] = {"status": "complete", "metrics_sha256": record["metrics_sha256"],
                                "prediction_path": str(path.relative_to(ROOT)), "prediction_sha256": sha256(path),
                                "world_order_sha256": checked["world_order_sha256"],
                                "events_by_world_and_radius": checked["events"].tolist() if checked["events"] is not None else None,
                                "event_scores_by_radius": checked["event_scores"].tolist() if checked["event_scores"] is not None else None,
                                "reference_events_by_radius": checked["reference_events"].tolist() if checked["reference_events"] is not None else None,
                                "actual_world_count": checked["actual_world_count"], "event_score_semantics": checked["event_score_semantics"],
                                "query_array_index": query,
                                "model_provenance": record["report"]["checkpoint_provenance"]}
    for radius in item["radii_to_render"]:
        ri = RADII.index(radius)
        columns = len(contract["column_layout"])
        fig, axes = plt.subplots(3, columns, figsize=(22 if columns == 6 else 13, 18.2))
        for row, (stage, step) in enumerate(STAGE_STEPS):
            checked = stage_predictions[stage]
            reference = None if query is None else bool(sample.targets[query, ri])
            row_label = STAGE_LABELS[stage]
            map_panel(axes[row, 0], sample, case, method, None, seed, radius, state="observed", note=row_label + "\n共同输入，所有阶段相同")
            map_panel(axes[row, 1], sample, case, method, sample.target, seed, radius, state="reference",
                      note=row_label + "\n参考：" + ("无合格查询" if reference is None else "可达" if reference else "不可达"))
            if k is None:
                direct_score_panel(axes[row, 2], sample, case, checked, seed, radius, stage)
            for wi in range(k or 0):
                world = checked["worlds"][wi] if checked is not None else None
                identity = f"独立成员 {wi}（第{wi+1}/4个）" if method == "lama" else "唯一确定性地图 · 实际K=1" if k == 1 else f"样本 {wi+1}/4"
                if checked is None:
                    outcome = "本阶段待完成；没有模型数据"
                elif query is None:
                    outcome = "无合格查询；仅检查地图"
                else:
                    event = bool(checked["events"][wi, ri]); successes = int(checked["events"][:, ri].sum())
                    outcome = f"本图：{'可达' if event else '不可达'}；本组可达 {successes}/{k}"
                map_panel(axes[row, wi+2], sample, case, method, world, seed, radius,
                          note=row_label + "\n" + identity + "\n" + outcome)
                if world is not None and sample.hidden.any() and (~sample.hidden).any():
                    axes[row, wi+2].contour(sample.hidden.astype(float), levels=[.5], colors=["#5c6c64"], linewidths=.4, alpha=.65)
        output_description = "地图输出N/A；仅展示固定查询的真实网络分数" if k is None else f"按保存顺序展示全部{k}张实际世界；不复制补足K4"
        fig.suptitle(f"{LABELS[method]} · 阶段对照：0 / 1000 / 5000步\n"
                     f"冻结查询 {case.get('query_id') or '无合格查询'} · 半径={radius}格 · 外层seed={seed} · validation-only\n"
                     "每阶段：同一输入 / 同一参考 / " + output_description + "；本图不批准视觉门槛",
                     fontsize=12, y=1.01)
        map_legend(fig)
        footer = ("直接查询模型没有地图或世界采样；原始事件分数是网络sigmoid输出，不是四张地图的可达比例。" if k is None else
                  "模型图中的细灰线标记未知区域边界。可达次数逐张使用精确footprint连通性计算；不是逐格free比例。")
        fig.text(.5, -.075, footer, ha="center", fontsize=9)
        layout = layout_map_grid(fig, axes)
        name = f"{case['global_id']}_q{case.get('candidate_index', 'none')}_radius_{radius}"
        save_figure(fig, output / "figures", name, inventory,
                    f"{case.get('query_id')} · r={radius}格 · 0/1000/5000步 · {output_description} · validation-only")
        inventory[-1].update(radius_cells=radius, groups=item["groups"], layout_audit=layout,
                              global_id=case["global_id"], query_id=case.get("query_id"), stage_sources=stage_sources,
                              actual_world_count=k, column_layout=contract["column_layout"], event_score_semantics=contract["event_score_semantics"])
    return {"global_id": case["global_id"], "query_id": case.get("query_id"), "candidate_index": case.get("candidate_index"),
            "groups": item["groups"], "radii_rendered_cells": item["radii_to_render"], "stage_sources": stage_sources}


def main():
    source = Path(__file__).read_text()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/flatlands_external_formal_protocol_v1/protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("Refusing a nonempty convergence output directory")
    protocol = json.loads(args.protocol.read_text()); protocol_hash = sha256(args.protocol)
    contract = output_contract(args.method)
    if protocol["methods"][args.method]["actual_K"] != contract["actual_world_count"]:
        raise ValueError("Registered method's actual K differs from the frozen protocol")
    result_root, data_root = ROOT / protocol["output_root"], ROOT / protocol["data_root"]
    if (protocol["test_lock"].get("final_test_locked") is not True or protocol["test_lock"].get("location_6_locked") is not True
            or sha256(data_root / "seal.json") != protocol["data_seal_sha256"]):
        raise ValueError("Test lock or eligible-data seal mismatch")
    frozen, audit, cases_path, audit_path = frozen_figure_artifacts(protocol)
    cases = frozen_cases(frozen)
    records, rejected = discover_evaluations(result_root, protocol_hash, protocol)
    selected = stage_records(records, args.method, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=False)
    (args.output_dir / "renderer_source.py").write_text(source)
    samples = {s.row["global_id"]: s for s in load_formal_data(data_root, "validation")}
    configure_plotting()
    inventory, rendered = [], []
    for item in cases:
        rendered.append(render_case(item, samples[item["case"]["global_id"]], selected, args.method, args.seed, args.output_dir, inventory))
    manifest = {"schema_version": 2, "method": args.method, "label": LABELS[args.method], "outer_seed": args.seed,
                **contract,
                "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
                "figure_cases_sha256": sha256(cases_path), "figure_geometry_audit_sha256": sha256(audit_path),
                "data_seal_sha256": protocol["data_seal_sha256"], "renderer_sha256": sha256(args.output_dir / "renderer_source.py"),
                "dependency_source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                    (ROOT / "scripts/render_flatlands_formal_report.py", ROOT / "src/pathrel/formal_metrics.py", ROOT / "src/pathrel/formal_data.py")},
                **completeness(selected, len(cases), len(rendered)),
                "case_order": "frozen manifest order, deduplicated by observation and query; never score-sorted",
                "stage_metrics_sha256": {stage: selected[stage]["metrics_sha256"] if selected[stage] is not None else None for stage, _ in STAGE_STEPS},
                "stage_order": [stage for stage, _ in STAGE_STEPS], "cases": rendered, "figures": inventory,
                "rejected_evaluations": rejected, "validation_only": True,
                "new_physical_test_images_opened": 0, "raw_archive_opened": False,
                "final_test_locked": True, "location_6_locked": True,
                "review_record_to_be_written_separately": str(result_root / "full_gates" / args.method / "visual.json")}
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("method", "outer_seed", "complete", "missing_stages", "unique_frozen_queries", "visual_gate_passed")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
