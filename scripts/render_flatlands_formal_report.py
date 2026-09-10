#!/usr/bin/env python3
"""Immutable Chinese validation diagnostic reports from saved formal predictions.

No inference, test access, result-dependent image selection, or silent mixing of
training stages. Missing outputs stay missing. Top-level report pointers may be
refreshed only inside the protocol-owned result directory; every old report is
retained under reports/<UTC timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathrel.formal_checkpoint import atomic_json
from pathrel.formal_data import load_formal_data, sha256
from pathrel.formal_inference import LABELS, METHODS, RADII
from pathrel.formal_metrics import exact_world_events


SEEDS = (20260831, 20260901, 20260902)
STAGES = {"untrained": 0, "smoke": 1, "pilot": 2, "formal": 3}
STAGE_ZH = {"untrained": "随机初始化诊断", "smoke": "1000步稳定性诊断", "pilot": "5000步收敛诊断", "formal": "完整训练验证（仍须最终审计）"}
NAMES = {"conpath": "ConPath", "conpath_original": "ConPath 原采样器", "independent": "独立单元补全",
         "no_reach": "ConPath 去可达性损失", "deterministic": "确定性补全对照", "direct_query": "直接查询对照",
         "lama": "LaMa BEV adaptation", "flow": "FM+XAttn\nliterature reimplementation"}
COLORS = dict(zip(METHODS, ("#b56a34", "#557ba5", "#137d6e", "#82a59c", "#8d67a8", "#ac4f62", "#72807e", "#b6a145")))
METRICS = ("map_brier", "map_nll", "event_brier", "event_nll", "event_ece", "false_safe_at_0_8", "coverage_at_0_8")
PALETTE = np.array([[54, 70, 84], [101, 181, 157], [220, 228, 236], [247, 248, 246]], dtype=np.uint8)


def group_key(report):
    stage = report.get("stage")
    if stage not in STAGES:
        raise ValueError("Only prespecified evaluation stages can be compared")
    steps = {p["completed_steps"] for p in report["checkpoint_provenance"]}
    if len(steps) != 1:
        raise ValueError("Ensemble members used different checkpoint steps")
    if stage != "formal" and next(iter(steps)) != {"untrained": 0, "smoke": 1000, "pilot": 5000}[stage]:
        raise ValueError("Diagnostic stage has the wrong checkpoint step")
    return stage, next(iter(steps)) if stage != "formal" else "selected"


def validate_evaluation_metadata(report, receipt, protocol):
    """Cheap fixed-protocol identity checks; not an independent numerical audit."""
    method, seed = report["method"], report["outer_seed"]
    expected_members = protocol["methods"][method]["members"]
    provenance = report["checkpoint_provenance"]
    if (len(provenance) != expected_members or [p["member_index"] for p in provenance] != list(range(expected_members))
            or [p["initialization_seed"] for p in provenance] != protocol["member_seeds"][method][str(seed)]):
        raise ValueError("Member indices or initialization seeds differ from the frozen protocol")
    if receipt.get("completed_steps") != [p["completed_steps"] for p in provenance]:
        raise ValueError("Completion receipt checkpoint steps differ from provenance")
    if report["stage"] != "untrained" and len({p.get("checkpoint") for p in provenance}) != expected_members:
        raise ValueError("Independent members reuse the same checkpoint")
    expected_k = protocol["methods"][method]["actual_K"]
    for split, report_split in report["splits"].items():
        if split not in ("calibration", "validation"):
            raise ValueError("Only locked development splits are allowed")
        expected = {row["global_id"]: row for row in protocol["scenes"][split]}
        cases = report_split["cases"]
        ids = [row["global_id"] for row in cases]
        if len(ids) != len(expected) or set(ids) != set(expected):
            raise ValueError("Evaluation is missing, duplicating, or replacing fixed split cases")
        if report_split["efficiency"]["case_count"] != len(cases):
            raise ValueError("Evaluation case count disagrees with actual records")
        for row in cases:
            fixed = expected[row["global_id"]]
            cost = row["efficiency"]
            if (row["parent_group"] != fixed["parent_group"] or row["source"] != fixed["source"]
                    or cost["actual_world_count"] != expected_k or cost["actual_members"] != expected_members
                    or cost["common_budget_K"] != 4 or cost.get("stage_one_member_diagnostic") is not False):
                raise ValueError("Case source/parent or actual ensemble budget differs from protocol")
        overall = report_split["event_raw"]["scene_weighted"]["overall"]
        if overall["events"] != len(RADII) * sum(row["queries"] for row in cases):
            raise ValueError("Event total differs from frozen-radius case query records")


def independent_stage_receipt(result_root, report, metrics_hash, complete_hash, protocol_hash):
    for path in sorted((Path(result_root) / "audits" / report["method"] / str(report["outer_seed"])).rglob("verification.json")):
        value = json.loads(path.read_text())
        if (value.get("passed") is True and value.get("protocol_sha256") == protocol_hash
                and value.get("metrics_sha256") == metrics_hash and value.get("complete_sha256") == complete_hash
                and value.get("method") == report["method"] and value.get("seed") == report["outer_seed"]
                and value.get("stage") == report["stage"] and value.get("new_raw_archive_or_test_images_opened") == 0):
            return {"passed": True, "path": str(path), "sha256": sha256(path)}
    return {"passed": False, "path": None, "sha256": None}


def discover_evaluations(result_root, protocol_hash, protocol):
    """Use only complete, hash-matching, locked validation results."""
    result_root = Path(result_root)
    accepted, rejected = [], []
    for root in (result_root / "stages", result_root / "evaluations"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("complete.json")):
            try:
                receipt = json.loads(path.read_text())
                metrics_path = path.parent / "metrics.json"
                digest = sha256(metrics_path)
                report = json.loads(metrics_path.read_text())
                if receipt.get("passed") is not True or receipt.get("metrics_sha256") != digest:
                    raise ValueError("Incomplete or hash-mismatched evaluation")
                if report.get("protocol_sha256") != protocol_hash:
                    raise ValueError("Different formal protocol")
                if "validation" not in report.get("splits", {}):
                    continue  # calibration-only checkpoint selection is not a validation row
                if report.get("new_physical_test_images_opened") != 0 or report.get("final_test_locked") is not True or report.get("location_6_locked") is not True:
                    raise ValueError("Missing test-lock evidence")
                if report.get("observed_and_support_constraints_passed") is not True:
                    raise ValueError("Constraint audit failed")
                if report.get("method") not in METHODS or report.get("outer_seed") not in SEEDS:
                    raise ValueError("Unregistered method/seed")
                if (receipt.get("method") != report["method"] or receipt.get("seed") != report["outer_seed"]
                        or receipt.get("stage") != report.get("stage")):
                    raise ValueError("Completion identity does not match metrics")
                expected_k = None if report["method"] == "direct_query" else 1 if report["method"] == "deterministic" else 4
                if any(c["efficiency"]["actual_world_count"] != expected_k for c in report["splits"]["validation"]["cases"]):
                    raise ValueError("Actual world budget differs from the common comparison")
                if report["method"] == "lama" and len(report["checkpoint_provenance"]) != 4:
                    raise ValueError("Single-member LaMa is an engineering diagnostic, not the four-member comparison")
                validate_evaluation_metadata(report, receipt, protocol)
                group = group_key(report)
                accepted.append({"path": path.parent, "metrics_sha256": digest, "report": report,
                                 "group": group, "method": report["method"], "seed": report["outer_seed"],
                                 "independent_stage_audit": independent_stage_receipt(result_root, report, digest, sha256(path), protocol_hash),
                                 "completion_mtime_ns": path.stat().st_mtime_ns})
            except (ValueError, KeyError, OSError, TypeError) as error:
                rejected.append({"path": str(path), "reason": str(error)})
    # Multiple attempts are not repeated seeds. Pick the last complete attempt
    # by its completion receipt timestamp, never by numerical prediction quality.
    unique = {}
    for item in sorted(accepted, key=lambda r: (r["completion_mtime_ns"], str(r["path"]))):
        unique[(item["group"], item["method"], item["seed"])] = item
    return list(unique.values()), rejected


def paper_eligibility(records, full_plan, final_audit, protocol_hash, full_plan_hash=None, *, result_root=None, method=None):
    """Full training plus three repeats is necessary, but not the final audit."""
    reasons = []
    if not full_plan or full_plan.get("protocol_sha256") != protocol_hash or full_plan.get("pilot_gate_passed") is not True or full_plan.get("full_steps", 0) <= 5000:
        reasons.append("缺少通过阶段验证后冻结的完整训练计划")
    actual_seeds = [r["seed"] for r in records if r["report"].get("stage") == "formal"]
    if sorted(actual_seeds) != list(SEEDS):
        reasons.append("尚未完成三个指定随机种子的完整训练验证")
    formal_steps = [p["completed_steps"] for r in records if r["report"].get("stage") == "formal" for p in r["report"]["checkpoint_provenance"]]
    if any(step < 10000 for step in formal_steps):
        reasons.append("所选检查点早于10000步，按冻结规则仅作诊断；不替换成更晚的较差检查点")
    if full_plan and any(step > full_plan.get("full_steps", 0) for step in formal_steps):
        reasons.append("所选检查点超出冻结完整训练预算")
    expected_digests = sorted(r["metrics_sha256"] for r in records)
    audit = final_audit or {}
    fields = ("passed", "test_lock_passed", "checkpoint_selection_passed", "implementation_checks_passed", "convergence_review_passed")
    if (any(audit.get(k) is not True for k in fields) or audit.get("protocol_sha256") != protocol_hash
            or sorted(audit.get("fully_trained_outer_seeds", [])) != list(SEEDS)
            or sorted(audit.get("evaluation_metrics_sha256", [])) != expected_digests
            or not full_plan_hash or audit.get("full_plan_sha256") != full_plan_hash):
        reasons.append("尚无绑定完整训练计划和三次评估哈希的最终独立审计")
    try:
        attempt = (ROOT / audit["audit_attempt"]).resolve()
        verification = (ROOT / audit["verification"]).resolve()
        if (result_root is None or not attempt.is_relative_to(Path(result_root).resolve()) or verification != attempt / "verification.json"
                or sha256(verification) != audit["verification_sha256"]):
            raise ValueError("Final verification path/hash mismatch")
        checked = json.loads(verification.read_text())
        if (checked != {k: v for k, v in audit.items() if k not in ("audit_attempt", "verification", "verification_sha256")}
                or checked.get("passed") is not True or checked.get("main_table_eligible") is not True
                or checked.get("method") != method or checked.get("new_raw_archive_or_test_images_opened") != 0):
            raise ValueError("Final verification content/identity mismatch")
    except (KeyError, ValueError, OSError, TypeError):
        reasons.append("最终资格未绑定实际通过的独立审计verification文件；手写布尔标记不能授予主表资格")
    return {"eligible": not reasons, "reasons": reasons}


def summarize_group(records, methods, *, result_root, protocol_hash):
    """Explicit missingness; short budgets never become paper main results."""
    result = {}
    for method in methods:
        rows = sorted((r for r in records if r["method"] == method), key=lambda r: r["seed"])
        metrics = {}
        for key in METRICS:
            values = [r["report"]["summary"]["validation"].get(key) for r in rows]
            values = [float(v) for v in values if v is not None]
            metrics[key] = {"mean": float(np.mean(values)) if values else None,
                            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "n": len(values), "values": values}
        plan_file = Path(result_root) / "full_plans" / f"{method}.json"
        audit_file = Path(result_root) / "final_audits" / f"{method}.json"
        plan = json.loads(plan_file.read_text()) if plan_file.exists() else None
        audit = json.loads(audit_file.read_text()) if audit_file.exists() else None
        gate = paper_eligibility(rows, plan, audit, protocol_hash, sha256(plan_file) if plan_file.exists() else None,
                                 result_root=result_root, method=method)
        result[method] = {"label": LABELS[method], "completed_outer_seeds": [r["seed"] for r in rows],
                          "missing_outer_seeds": [seed for seed in SEEDS if seed not in {r["seed"] for r in rows}],
                          "status_zh": "本阶段完整验证尚未齐备（见运行快照）" if not rows else "通过正式核验" if gate["eligible"] else "验证完成；独立数值审计已通过；仍为诊断" if all(r.get("independent_stage_audit", {}).get("passed") for r in rows) else "验证完成；独立数值审计未全部通过；仍为诊断",
                          "metrics": metrics, "paper_gate": gate,
                          "runs": [{"seed": r["seed"], "metrics_sha256": r["metrics_sha256"],
                                    "selected_step": r["report"]["checkpoint_provenance"][0]["completed_steps"],
                                    "full_steps": plan.get("full_steps") if plan else None,
                                    "selection_before_10000_diagnostic": r["report"].get("stage") == "formal" and r["report"]["checkpoint_provenance"][0]["completed_steps"] < 10000,
                                    "independent_stage_audit": r.get("independent_stage_audit", {"passed": False}),
                                    "validation_metrics": validation_summary(r["report"]["splits"]["validation"]),
                                    "checkpoint_and_training_cost_provenance": r["report"]["checkpoint_provenance"]} for r in rows]}
    return result


def validation_summary(validation):
    """Preserve every saved aggregate/stratum; omit only the large case list.

    Solver counts are derived from actual per-case receipts, never from the
    nominal protocol. They remain distinct from network batch forwards and
    examples (which include classifier-free guidance branches).
    """
    summary = {key: value for key, value in validation.items() if key != "cases"}
    cost = dict(validation.get("efficiency", {}))
    cases = [row["efficiency"] for row in validation.get("cases", [])]
    count = cost.get("case_count")
    for field in ("actual_batch_forward_calls", "actual_model_input_examples"):
        cost[field + "_total"] = cost.get(field)
        cost[field + "_mean_per_case"] = cost[field] / count if count and cost.get(field) is not None else None
    nfe = [c.get("velocity_evaluations_per_sample") for c in cases]
    worlds = [c.get("actual_world_count") for c in cases]
    all_velocity_receipts = bool(cases) and all(v is not None for v in nfe) and all(v is not None for v in worlds)
    nfe_total = sum(v * k for v, k in zip(nfe, worlds)) if all_velocity_receipts else None
    cost["velocity_evaluations_all_worlds_total"] = nfe_total
    cost["velocity_evaluations_all_worlds_mean_per_case"] = nfe_total / len(cases) if all_velocity_receipts else None
    cost["velocity_evaluations_mean_per_actual_world"] = nfe_total / sum(worlds) if all_velocity_receipts and sum(worlds) else None
    cost["solver_names_recorded"] = sorted({c["solver"] for c in cases if c.get("solver") is not None})
    cost["solver_steps_per_sample_recorded"] = sorted({c["steps_per_sample"] for c in cases if c.get("steps_per_sample") is not None})
    cost["count_scope"] = "all saved validation cases; total fields are sums; mean_per_case divides by case_count, not queries; velocity NFE excludes CFG branch multiplicity, input examples include it; null means unavailable or inapplicable"
    summary["efficiency"] = cost
    return summary


def summary_csv_rows(grouped, chosen, chosen_group):
    rows = []
    for method, value in grouped.items():
        for outer_seed in SEEDS:
            record = next((r for r in chosen if r["method"] == method and r["seed"] == outer_seed), None)
            metrics = record["report"]["summary"]["validation"] if record else {}
            provenance = record["report"]["checkpoint_provenance"] if record else []
            training_costs = [p.get("training_seconds") for p in provenance]
            parameters = [p.get("model_parameters") for p in provenance]
            cost = validation_summary(record["report"]["splits"]["validation"])["efficiency"] if record else {}
            def peak(field):
                values = [p.get(field) for p in provenance if p.get(field) is not None]
                return max(values) if values else None
            rows.append({"method": LABELS[method], "seed": outer_seed, "stage": chosen_group[0], "checkpoint": chosen_group[1],
                         "status": ("formal_audited" if value["paper_gate"]["eligible"] else "diagnostic_complete") if record else "pending",
                         "paper_main_eligible": value["paper_gate"]["eligible"],
                         "selected_step": provenance[0]["completed_steps"] if provenance else None,
                         "full_steps": next((run["full_steps"] for run in value["runs"] if run["seed"] == outer_seed), None),
                         "selection_before_10000_diagnostic": bool(record and chosen_group[0] == "formal" and provenance[0]["completed_steps"] < 10000),
                         "common_max_world_budget": 4,
                         "actual_world_count": (None if method == "direct_query" else 1 if method == "deterministic" else 4) if record else None,
                         "actual_members": len(provenance) if record else None,
                         "total_model_parameters": sum(parameters) if parameters and all(p is not None for p in parameters) else None,
                         "training_seconds_all_members": sum(training_costs) if training_costs and all(t is not None for t in training_costs) else None,
                         "training_seconds_is_lower_bound": any(p.get("training_wall_time_is_lower_bound", True) for p in provenance) if provenance else None,
                         "training_cost_captured_utc_by_member": json.dumps([p.get("training_cost_captured_utc") for p in provenance]) if provenance else None,
                         "training_peak_allocated_bytes_max_member_session": peak("training_peak_allocated_bytes"),
                         "training_peak_reserved_bytes_max_member_session": peak("training_peak_reserved_bytes"),
                         "generation_seconds_per_actual_bundle": cost.get("generation_seconds_mean"),
                         "full_update_seconds": cost.get("full_update_seconds_mean"),
                         "validation_case_count": cost.get("case_count"),
                         "evaluation_peak_allocated_bytes": cost.get("peak_allocated_bytes"),
                         "evaluation_peak_reserved_bytes": cost.get("peak_reserved_bytes"),
                         **{field: cost.get(field) for field in ("actual_batch_forward_calls_total", "actual_batch_forward_calls_mean_per_case",
                             "actual_model_input_examples_total", "actual_model_input_examples_mean_per_case",
                             "velocity_evaluations_all_worlds_total", "velocity_evaluations_all_worlds_mean_per_case",
                             "velocity_evaluations_mean_per_actual_world")},
                         "solver_steps_per_sample_recorded": json.dumps(cost["solver_steps_per_sample_recorded"]) if cost else None,
                         **{key: metrics.get(key) for key in (*METRICS, "observed_evidence_violation_count", "valid_support_violation_count")},
                         "metrics_sha256": record["metrics_sha256"] if record else None, "validation_only": True})
    return rows


def status_file_snapshot(path):
    with Path(path).open("rb") as handle:
        content = handle.read(); info = os.fstat(handle.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
            "file_modified_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
            "value": json.loads(content)}


def runtime_snapshot(result_root, protocol_hash):
    """Capture status-file evidence once; no polling or inferred live state."""
    result_root = Path(result_root)
    snapshot = {"captured_utc": datetime.now(timezone.utc).isoformat(), "queue": None, "methods": {}, "full_methods": {}, "failures": [],
                "interpretation": "status file snapshot, not a live-process assertion; incomplete stages are not evaluation results"}
    queue = result_root / "queue.json"
    if queue.exists():
        snapshot["queue"] = status_file_snapshot(queue)
    for path in sorted((result_root / "stage_status").glob("*.json")):
        item = status_file_snapshot(path); status = item["value"]
        if status.get("protocol_sha256") != protocol_hash:
            item["ignored_reason"] = "different protocol"
        elif status.get("current_action") == "training" and status.get("current_member") is not None:
            progress = result_root / "runs" / status["method"] / str(status["seed"]) / f"member_{status['current_member']}" / "progress.json"
            if progress.exists():
                item["training_progress"] = status_file_snapshot(progress)
        snapshot["methods"][status.get("method", path.stem)] = item
    for path in sorted((result_root / "full_runs").glob("*/status.json")):
        item = status_file_snapshot(path); status = item["value"]
        if status.get("protocol_sha256") != protocol_hash:
            item["ignored_reason"] = "different protocol"
        snapshot["full_methods"][status.get("method", path.parent.name)] = item
    for path in sorted((result_root / "runs").glob("*/*/member_*/failure.json")):
        item = status_file_snapshot(path); failure = item["value"]
        item["method"], item["seed"], item["member"] = path.parents[2].name, path.parents[1].name, path.parent.name
        item["historical_failure_may_have_resumed"] = True
        for field in ("last_valid_checkpoint", "loss_curve"):
            relative = failure.get(field)
            if relative:
                target = (path.parent / relative).resolve()
                if not target.is_relative_to(result_root.resolve()):
                    raise ValueError("Failure artifact escapes the owned formal result directory")
                item[field + "_path"] = str(target)
                item[field + "_exists"] = target.exists()
        if (path.parent / "time_summary.json").exists():
            item["training_time_summary"] = status_file_snapshot(path.parent / "time_summary.json")
        snapshot["failures"].append(item)
    return snapshot


def runtime_markdown(snapshot):
    if snapshot is None:
        return []
    lines = ["", "## 执行状态快照", "", f"抓取时间：{snapshot['captured_utc']}。这是状态文件快照；下方阶段成绩表仅列完整评估，不能当作实时训练状态。"]
    if snapshot["queue"] is not None:
        queue = snapshot["queue"]
        lines.append(f"队列记录：`{json.dumps(queue['value'], ensure_ascii=False)}`；来源 `{queue['path']}`，文件更新时间 {queue['file_modified_utc']}。")
    actions = {"training": "训练", "calibration_and_validation": "校准集与验证集评估", "calibration_only": "仅校准集检查点选优",
               "selected_validation_only": "所选检查点的验证集评估", None: "无当前动作"}
    lines.append("")
    for method, item in snapshot["methods"].items():
        if item.get("ignored_reason"):
            lines.append(f"- {method}：忽略不匹配协议的状态文件。")
            continue
        value = item["value"]; progress = item.get("training_progress", {}).get("value", {})
        member = f"；成员 {value['current_member']}" if value.get("current_action") == "training" and value.get("current_member") is not None else ""
        completed = f"；已记录完整更新 {progress['step']} 步（非检查点恢复边界）" if "step" in progress else ""
        historical = "阶段训练历史记录（另有完整训练状态）" if method in snapshot.get("full_methods", {}) else "阶段训练记录"
        lines.append(f"- {LABELS.get(method, method)} · {historical}：状态 `{value.get('status')}`；阶段 {STAGE_ZH.get(value.get('current_stage'), value.get('current_stage'))}；动作 **{actions.get(value.get('current_action'), value.get('current_action'))}**{member}{completed}。来源 `{item['path']}`，最后更新时间 {item['file_modified_utc']}。")
    for method, item in snapshot.get("full_methods", {}).items():
        if item.get("ignored_reason"):
            continue
        value = item["value"]
        lines.append(f"- {LABELS.get(method, method)} · **完整训练状态**：`{value.get('status')}`；seed={value.get('current_seed')}；计划执行至step={value.get('current_step')}；成员={value.get('current_member')}；动作 **{actions.get(value.get('current_action'), value.get('current_action'))}**。来源 `{item['path']}`，文件更新时间 {item['file_modified_utc']}。")
    if snapshot["failures"]:
        lines += ["", "### 保留的失败记录", "", "这些是历史失败快照，可能已经恢复；不据此把当前任务一律标为失败。", ""]
        for item in snapshot["failures"]:
            value = item["value"]
            lines.append(f"- {item['method']} / seed={item['seed']} / {item['member']}：原因 `{value.get('error', value.get('reason'))}`；类别 `{value.get('category')}`；最后有效检查点 `{item.get('last_valid_checkpoint_path')}`；最后持久步数 {value.get('last_durable_step')}；损失日志 `{item.get('loss_curve_path')}`；本进程耗时 {value.get('wall_seconds_this_session')} 秒；峰值 allocated/reserved={value.get('peak_allocated_bytes')}/{value.get('peak_reserved_bytes')} 字节。来源 `{item['path']}`，更新时间 {item['file_modified_utc']}。")
    return lines + [""]


def risk_curve_points(curve):
    """Keep full score ties and undefined empty acceptance; do not interpolate."""
    points = [(float(r["coverage"]), float(r["false_safe"])) for r in curve if r["false_safe"] is not None]
    if any(not (0 <= c <= 1 and 0 <= r <= 1) for c, r in points):
        raise ValueError("Invalid saved false-safe/coverage curve")
    if any(a[0] > b[0] for a, b in zip(points, points[1:])):
        raise ValueError("Saved coverage curve is not ordered")
    return np.array(points, dtype=float).reshape(-1, 2)


def read_loss_journal(path):
    """Capture the complete-line prefix of the actual log, without step filtering."""
    with Path(path).open("rb") as handle:
        content = handle.read(); info = os.fstat(handle.fileno())
    boundary = content.rfind(b"\n") + 1
    prefix, tail = content[:boundary], content[boundary:]
    records = [json.loads(line) for line in prefix.splitlines() if line.strip()]
    steps = [int(r["step"]) for r in records]
    if steps and (steps[0] != 1 or any(b != a+1 for a, b in zip(steps, steps[1:]))):
        raise ValueError("Current training journal is not a contiguous complete-update sequence")
    for row in records:
        if not all(np.isfinite(float(v)) for v in row["losses"].values()):
            raise ValueError("Nonfinite stored loss; cannot silently omit it from a curve")
    return records, {"path": str(path), "captured_complete_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                     "captured_complete_prefix_bytes": len(prefix),
                     "file_modified_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
                     "complete_updates": len(records), "uncommitted_trailing_bytes": len(tail),
                     "plotted_step_range": [steps[0], steps[-1]] if steps else None,
                     "selection": "all complete lines of the current canonical journal; no loss-based range selection"}


def loss_durability_snapshot(journal, records):
    token = records[-1].get("session_id") if records else None
    if token is None:
        return {"last_durable_step_at_receipt": None, "receipt": None,
                "meaning": "No matching session receipt; plotted complete updates do not imply resumable checkpoints"}
    if Path(token).name != token:
        raise ValueError("Session token must be a basename")
    for suffix in ("finished", "heartbeat"):
        path = Path(journal).parent / "sessions" / f"{token}.{suffix}.json"
        if path.exists():
            receipt = status_file_snapshot(path)
            return {"last_durable_step_at_receipt": receipt["value"].get("last_durable_step"), "receipt": receipt,
                    "meaning": "Executed trajectory and separately captured checkpoint durability; complete updates after durable step can be replayed on resume"}
    return {"last_durable_step_at_receipt": None, "receipt": None,
            "meaning": "Matching session receipt not yet written; durability unknown"}


def trailing_loss_mean(steps, values, window=100):
    steps, values = np.asarray(steps), np.asarray(values, dtype=float)
    if steps.shape != values.shape or steps.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid aligned raw loss sequence")
    if window < 1 or int(window) != window:
        raise ValueError("Moving-average window must be a positive fixed integer")
    if len(steps) < window:
        return np.array([], dtype=steps.dtype), np.array([], dtype=float)
    return steps[window-1:], np.convolve(values, np.ones(window)/window, mode="valid")


def render_training_losses(result_root, folder, inventory, training_scope):
    """Raw losses and a fixed trailing 100-update mean; never a quality gate."""
    sources = []
    for method in ("lama", "flow"):
        run_root = Path(result_root) / "runs" / method
        available_seeds = [seed for seed in SEEDS if (run_root / str(seed)).exists()] or [SEEDS[0]]
        for seed in available_seeds:
            members = range(4) if method == "lama" else range(1)
            logs, missing = {}, []
            for member in members:
                path = run_root / str(seed) / f"member_{member}" / "progress.jsonl"
                if path.exists():
                    values, provenance = read_loss_journal(path)
                    provenance.update(method=method, seed=seed, member=member, durability=loss_durability_snapshot(path, values))
                    sources.append(provenance)
                    if values:
                        logs[member] = values
                        archive = folder.parent / "loss_plot_data"; archive.mkdir(exist_ok=True)
                        target = archive / f"{method}_{seed}_member{member}.npz"
                        loss_names = sorted(values[0]["losses"])
                        np.savez_compressed(target, steps=np.array([r["step"] for r in values]),
                                            loss_names=np.array(loss_names),
                                            losses=np.array([[r["losses"][key] for key in loss_names] for r in values]))
                        provenance["archived_plot_values_path"] = str(target.relative_to(folder.parent))
                        provenance["archived_plot_values_sha256"] = sha256(target)
                    else:
                        missing.append(member)
                else:
                    missing.append(member)
            if method == "lama":
                fig, grid = plt.subplots(2, 2, figsize=(12, 8))
                panels = (("reconstruction", "未知有效区域 L1"), ("adversarial", "生成器 hinge 对抗项"),
                          ("feature_matching", "多尺度特征 MSE"), ("discriminator", "判别器 hinge 损失"))
            else:
                fig, grid = plt.subplots(1, 1, figsize=(10, 4.5))
                panels = (("velocity_mse", "未知有效区域速度 MSE"),)
            for ax, (key, label) in zip(np.asarray(grid).reshape(-1), panels):
                for member, records in logs.items():
                    steps = np.array([r["step"] for r in records]); values = np.array([r["losses"][key] for r in records])
                    color = ("#137d6e", "#557ba5", "#b56a34", "#8d67a8")[member]
                    ax.plot(steps, values, color=color, linewidth=.65, alpha=.22, label=f"成员{member} 原始记录" if len(steps) < 100 else None)
                    mx, my = trailing_loss_mean(steps, values)
                    if len(mx):
                        ax.plot(mx, my, color=color, linewidth=1.5, label=f"成员{member} · 后向100步均线")
                if not logs:
                    ax.text(.5, .5, "尚无完整训练更新记录", ha="center", transform=ax.transAxes, color="#71837a")
                elif missing:
                    ax.text(.02, .97, "尚无更新记录：成员 " + ",".join(map(str, missing)), ha="left", va="top", transform=ax.transAxes, fontsize=8, color="#71837a")
                ax.set(xlabel="真实完整优化更新步数", ylabel=label, title=label)
                ax.grid(alpha=.15)
                if logs:
                    ax.legend(fontsize=7, frameon=False)
            fig.suptitle(f"{LABELS[method]} · seed={seed} · 训练日志快照（非validation成绩）\n"
                         f"scene=固定{training_scope['parents']}个训练父场景；配套query/radius=冻结训练协议/0、10、20格", fontsize=11)
            durable = "；".join(f"成员{s['member']}恢复边界={s['durability']['last_durable_step_at_receipt'] if s['durability']['last_durable_step_at_receipt'] is not None else '未知'}步"
                              for s in sources if s["method"] == method and s["seed"] == seed)
            fig.text(.5, -.03, "浅线=全部原始损失，实线=固定后向100步均值；前99步无均线，不挑区间，不据损失曲线判优。\n"
                     "完整更新轨迹不等于已持久检查点，恢复时可能回放；独立时刻的session回执：" + (durable or "暂无"), ha="center", fontsize=7.5)
            fig.tight_layout()
            save_figure(fig, folder, f"11_training_losses_{method}_{seed}", inventory,
                        "实际完整更新日志的训练诊断，非泛化指标；对抗与判别器损失不要求单调下降")
    return sources


def configure_plotting():
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#d8e1dc", "axes.labelcolor": "#415852",
                         "text.color": "#203832", "xtick.color": "#526961", "ytick.color": "#526961",
                         "svg.fonttype": "path", "pdf.fonttype": 42, "savefig.facecolor": "white"})


def frozen_figure_artifacts(protocol, *, project_root=ROOT):
    spec = protocol["figures"]
    paths = [Path(project_root) / spec["cases"], Path(project_root) / spec["geometry_verification"]]
    if sha256(paths[0]) != spec["sha256"] or sha256(paths[1]) != spec["geometry_verification_sha256"]:
        raise ValueError("Figure selection/geometry differs from the protocol's frozen hashes")
    manifest, audit = [json.loads(path.read_text()) for path in paths]
    if (audit.get("passed") is not True or audit.get("figure_cases_sha256") != spec["sha256"]
            or audit.get("data_seal_sha256") != protocol["data_seal_sha256"]
            or manifest.get("data_seal_sha256") != protocol["data_seal_sha256"]
            or manifest.get("validation_only") is not True
            or manifest.get("frozen_before_formal_model_training_and_inference") is not True):
        raise ValueError("Frozen figure scope or independent geometry verification mismatch")
    return manifest, audit, paths[0], paths[1]


def checkpoint_note(record):
    if record is None:
        return "本阶段检查点待完成"
    step = record["report"]["checkpoint_provenance"][0]["completed_steps"]
    budget = record.get("full_steps")
    return f"所选step={step}" + (f"；完整训练预算={budget}步" if budget is not None else "；完整预算未冻结" if record["report"]["stage"] == "formal" else "")


def budget_annotation(grouped):
    if not any(run.get("selected_step") is not None for value in grouped.values() for run in value["runs"]):
        return ""
    return "\n".join(LABELS[method] + "：" + "；".join(f"seed={run['seed']} 选{run['selected_step']}/完整{run['full_steps'] if run['full_steps'] is not None else '未冻结'}步" for run in value["runs"])
                     for method, value in grouped.items() if value["runs"])


def save_figure(fig, folder, name, inventory, description):
    paths = []
    for suffix in ("svg", "png", "pdf"):
        path = folder / f"{name}.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append({"path": str(path.relative_to(folder.parent)), "sha256": sha256(path)})
    plt.close(fig)
    inventory.append({"name": name, "description_zh": description, "files": paths})


def map_colors(sample, world=None):
    states = np.where(sample.observation[0] > .5, 1, np.where(sample.hidden, 2, 0)) if world is None else np.asarray(world, dtype=np.uint8)
    pixels = PALETTE[states].copy()
    rr, cc = np.indices(sample.valid.shape)
    invalid = ~sample.valid
    pixels[invalid] = np.where((((rr[invalid] // 8) + (cc[invalid] // 8)) % 2)[:, None],
                               np.array([232, 236, 238]), np.array([247, 248, 246]))
    return pixels


def map_panel(ax, sample, case, method, world, seed, radius, *, state="model", note=""):
    name = {"observed": "已观测输入", "reference": "真实参考"}.get(state, NAMES.get(method, method))
    if state == "model" and world is None:
        placeholder = np.zeros((*sample.valid.shape, 3), dtype=np.uint8)
        placeholder[:] = (241, 244, 242)
        ax.imshow(placeholder, interpolation="nearest")
        ax.text(.5, .55, "尚未生成模型输出", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#64756e")
        ax.text(.5, .4, "缺失项，不代表性能为零", transform=ax.transAxes, ha="center", fontsize=8, color="#64756e")
    else:
        ax.imshow(map_colors(sample, None if state == "observed" else world), interpolation="nearest")
        if not case.get("no_eligible_query", False):
            for point, marker, color, letter in ((case["start_rc"], "o", "#276bb0", "S"), (case["goal_rc"], "D", "#b96b27", "G")):
                row, col = point
                ax.scatter(col, row, c=color, marker=marker, s=32, edgecolors="white", linewidths=.7)
                ax.annotate(letter, (col, row), xytext=(4, -9), textcoords="offset points", fontsize=8, color=color,
                            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": .4})
                if radius > 0:
                    ax.add_patch(Circle((col, row), radius, facecolor="none", edgecolor=color, linewidth=.8, alpha=.8))
    qid = case.get("query_id") or "无合格查询"
    scene = sample.row.get("scene_id", sample.row["parent_group"])
    # Every panel carries the actual scene/query/radius/method/seed/scope.
    ax.set_title(f"{name}\nscene={scene}\n{sample.row['global_id']} · {qid}\nr={radius} 格 · seed={seed}\nvalidation-only" + ("\n" + note if note else ""), fontsize=7.4, pad=7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def map_legend(fig):
    fig.legend(handles=[Patch(color="#65b59d", label="可通行"), Patch(color="#364654", label="阻挡"),
                        Patch(color="#dce4ec", label="未知输入"), Patch(color="#f7f8f6", hatch="..", label="有效范围外（封闭）")],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5, -.02))
    fig.text(.5, -.05, "S 蓝圆：起点；G 棕菱形：目标；空心圆：机器人半径（格）。端点之间不画未经验证的连线。", ha="center", fontsize=8)


def layout_map_grid(fig, axes):
    """Reserve measured title space between rows; reject overlapping output."""
    fig.tight_layout(rect=[0, .03, 1, .98], h_pad=4.0, w_pad=1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gaps = []
    for row in range(1, len(axes)):
        previous_bottom = min(ax.get_window_extent(renderer).y0 for ax in axes[row-1])
        for ax in axes[row]:
            gap = previous_bottom - ax.title.get_window_extent(renderer).y1
            gaps.append(float(gap))
            if gap < 5:
                raise ValueError("Map grid title intrudes into the preceding image row")
    return {"passed": True, "minimum_row_title_gap_canvas_pixels": min(gaps), "canvas_dpi": fig.dpi,
            "font_size_reduced": False}


def load_saved_prediction(record, sample):
    if record is None:
        return None
    cases = {c["global_id"]: c for c in record["report"]["splits"]["validation"]["cases"]}
    gid = sample.row["global_id"]
    if gid not in cases:
        raise ValueError("Frozen figure case missing from evaluated validation")
    path = record["path"] / "validation" / "predictions" / f"{gid}.npz"
    if sha256(path) != cases[gid]["predictions_sha256"]:
        raise ValueError("Saved world artifact changed")
    with np.load(path, allow_pickle=False) as packet:
        for key, expected in (("candidate_indices", sample.candidate_indices), ("starts", sample.starts), ("goals", sample.goals), ("radii_cells", RADII)):
            if not np.array_equal(packet[key], expected):
                raise ValueError("Figure input/query contract drift")
        saved = {key: packet[key].copy() for key in packet.files}
    if record["method"] != "direct_query":
        k = 1 if record["method"] == "deterministic" else 4
        worlds = saved["worlds"]
        if worlds.shape != (k, *sample.valid.shape) or not np.isin(worlds, (0, 1)).all():
            raise ValueError("Saved world arrays differ from the actual binary K budget")
        if worlds[:, ~sample.valid].any() or np.any(worlds[:, ~sample.hidden] != sample.observation[0, ~sample.hidden]):
            raise ValueError("Saved worlds violate observed evidence or valid support")
        events = exact_world_events(worlds, sample.starts, sample.goals, RADII)
        if not np.array_equal(events, saved["world_events"]) or not np.array_equal(events.mean(0), saved["event_scores"]):
            raise ValueError("Displayed events differ from exact saved-world connectivity")
    return saved


def case_comparison(case, samples, records, seed, folder, inventory, name, title):
    sample = samples[case["global_id"]]
    radius = case.get("witness_radius_cells", case.get("bottleneck_failed_radius_cells", 10))
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.7))
    map_panel(axes[0], sample, case, "observed", None, seed, radius, state="observed")
    reference_note = "无合格查询" if case.get("no_eligible_query") else "参考事件：" + ("可达" if case["reference_events"][RADII.index(radius)] else "不可达")
    map_panel(axes[1], sample, case, "reference", sample.target, seed, radius, state="reference", note=reference_note)
    for ax, method in zip(axes[2:], ("conpath", "lama", "flow")):
        saved = load_saved_prediction(records.get(method), sample)
        world = saved["worlds"][0] if saved is not None and "worlds" in saved else None
        note = ("实际第1张；不挑最佳样本" if world is not None else "等待四世界模型输出") + "\n" + checkpoint_note(records.get(method))
        if saved is not None and not case.get("no_eligible_query"):
            qi = int(np.flatnonzero(sample.candidate_indices == case["candidate_index"])[0]); ri = RADII.index(radius)
            note += f"\n4世界事件频率={saved['event_scores'][qi, ri]:.2f}"
        map_panel(ax, sample, case, method, world, seed, radius, note=note)
    fig.suptitle(title, y=1.04, fontsize=12)
    map_legend(fig); fig.tight_layout()
    save_figure(fig, folder, name, inventory, title)


def render_map_figures(manifest, samples, records, seed, group_title, folder, inventory):
    general = manifest["general_cases"]
    for index, case in enumerate(general):
        case_comparison(case, samples, records, seed, folder, inventory, f"01_comparison_{index:02d}",
                        f"固定案例 {index+1} · {group_title} · 同一输入和参考；validation-only")
    if general:
        case = general[0]; sample = samples[case["global_id"]]
        fig, axes = plt.subplots(3, 4, figsize=(13, 14.5))
        for row, method in enumerate(("conpath", "lama", "flow")):
            saved = load_saved_prediction(records.get(method), sample)
            for col, ax in enumerate(axes[row]):
                world = saved["worlds"][col] if saved is not None and "worlds" in saved and col < len(saved["worlds"]) else None
                map_panel(ax, sample, case, method, world, seed, 10, note=(f"实际样本 {col+1}/4" if world is not None else f"样本位置 {col+1}/4 · 待运行") + "\n" + checkpoint_note(records.get(method)))
        fig.suptitle(f"每方法四个样本位置；缺失处明确占位\n{group_title} · validation-only", y=1.01)
        map_legend(fig); layout_audit = layout_map_grid(fig, axes)
        save_figure(fig, folder, "02_multiple_worlds", inventory, "固定案例的四张实际世界；LaMa为四个独立成员，不复制单张图")
        inventory[-1]["layout_audit"] = layout_audit
    category_names = {"narrow_bottleneck": ("07_bottleneck", "窄瓶颈：参考地图定义；并非按模型优劣挑图"),
                      "unreachable": ("08_unreachable", "不可达：参考地图定义；全部方法使用同一自然负例"),
                      "multiple_alternatives": ("09_alternatives", "多替代路径：参考图存在两条内部节点不相交路径")}
    for category, (prefix, title) in category_names.items():
        entries = manifest["reference_categories"][category]["cases"]
        if not entries:
            fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
            ax.text(.5, .5, f"{title}\n固定验证集中暂无符合冻结几何规则的案例\n未根据模型表现替换；validation-only", ha="center", va="center", transform=ax.transAxes)
            save_figure(fig, folder, prefix + "_unavailable", inventory, title + "；无符合条件案例")
        for index, case in enumerate(entries):
            case_comparison(case, samples, records, seed, folder, inventory, f"{prefix}_{index:02d}", title + " · " + group_title)
            if category == "multiple_alternatives":
                sample = samples[case["global_id"]]
                fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
                for ax, name, color in zip(axes, ("a", "b"), ("#a13fb2", "#2468bb")):
                    route = np.asarray(case[f"reference_path_{name}_rc"])
                    map_panel(ax, sample, case, "reference", sample.target, seed, case["witness_radius_cells"], state="reference",
                              note=f"参考几何路径 {name.upper()}；非模型预测")
                    ax.plot(route[:, 1], route[:, 0], color=color, linewidth=1.6, label=f"经核验的参考路径 {name.upper()}")
                    ax.legend(loc="lower right", fontsize=7, framealpha=.9)
                fig.suptitle("两条实际参考路线，除起终点外不共用格点；不声称语义走廊独立\nvalidation-only · 在模型推理前按参考几何冻结", fontsize=11, y=1.02)
                map_legend(fig); fig.tight_layout()
                save_figure(fig, folder, f"{prefix}_reference_routes_{index:02d}", inventory,
                            "有明确几何依据的参考路线A/B；逐格四邻域、footprint有效，两条路线内部节点不相交")
    for index, case in enumerate(manifest["multi_radius_cases"]):
        sample = samples[case["global_id"]]
        fig, axes = plt.subplots(3, 4, figsize=(13, 14.5))
        for row, radius in enumerate(RADII):
            map_panel(axes[row, 0], sample, case, "reference", sample.target, seed, radius, state="reference",
                      note="参考事件：" + ("可达" if case["reference_events"][row] else "不可达"))
            for col, method in enumerate(("conpath", "lama", "flow"), 1):
                saved = load_saved_prediction(records.get(method), sample)
                world = saved["worlds"][0] if saved is not None and "worlds" in saved else None
                note = "尚无事件输出"
                if saved is not None:
                    qi = int(np.flatnonzero(sample.candidate_indices == case["candidate_index"])[0])
                    note = f"4世界事件频率：{saved['event_scores'][qi, row]:.2f}"
                map_panel(axes[row, col], sample, case, method, world, seed, radius, note=note + "\n" + checkpoint_note(records.get(method)))
        fig.suptitle(f"同一地图与查询 · 半径0/10/20格 · {group_title} · validation-only", y=1.01)
        map_legend(fig); layout_audit = layout_map_grid(fig, axes)
        save_figure(fig, folder, f"10_radius_{index:02d}", inventory, "半径改变真实footprint事件，不把grid cell擅自换成米")
        inventory[-1]["layout_audit"] = layout_audit


def metric_bar(ax, grouped, metric, label):
    for index, (method, report) in enumerate(grouped.items()):
        stats = report["metrics"][metric]
        if method == "direct_query" and metric in ("map_brier", "map_nll"):
            ax.text(.01, index, "无地图输出（不适用）", va="center", color="#889890", fontsize=8)
        elif stats["mean"] is None:
            ax.text(.01, index, "尚无本阶段结果", va="center", color="#889890", fontsize=8)
        else:
            ax.errorbar(stats["mean"], index, xerr=stats["sd"] if stats["sd"] is not None else None,
                        fmt="o", markersize=5, capsize=3, color=COLORS[method])
            ax.annotate(f"{stats['mean']:.4f}  (n={stats['n']})", (stats["mean"], index),
                        xytext=(7, 5), textcoords="offset points", fontsize=8)
    ax.set_yticks(range(len(grouped)), [NAMES[m] for m in grouped], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel(label); ax.grid(axis="x", alpha=.18)
    available = [row["metrics"][metric] for row in grouped.values()
                 if row["metrics"][metric]["mean"] is not None]
    # Explicit bounds leave room for markers and annotations even when two
    # scores almost coincide. set_xlim(left=0) freezes the old right bound,
    # so a subsequent margins() call cannot repair its clipped edge marker.
    left = min([0.] + [s["mean"] - (s["sd"] or 0.) for s in available])
    right = max([s["mean"] + (s["sd"] or 0.) for s in available], default=1.)
    span = max(right - left, .01)
    ax.set_xlim(left - .05 * span if left < 0 else 0, right + .28 * span)


def metric_scope_annotation(data_scope, seeds, *, map_metric=False):
    seed_text = ", ".join(map(str, sorted(set(seeds)))) if seeds else "尚无该阶段完整评估"
    prefix = "地图指标不依赖半径；配套" if map_metric else ""
    return (f"scene=固定{data_scope['parents']}父场景（地图）/{data_scope['parents_with_queries']}父场景（事件）；"
            f"{prefix}query=冻结全部{data_scope['queries']}对端点 × r=0/10/20格\n"
            f"本图实际纳入seed={seed_text}；各方法完成数见n及汇总记录；validation-only")


def render_metric_figures(grouped, selected_records, title, seed, folder, inventory, data_scope, *, budget_note=""):
    def finish(name, description):
        if budget_note:
            fig.text(.01, -.27, "实际检查点与完整预算（不要求各方法所选步数相同）\n" + budget_note,
                     ha="left", va="top", fontsize=6.5)
        save_figure(fig, folder, name, inventory, description)
    actual_seeds = sorted({s for v in grouped.values() for s in v['completed_outer_seeds']})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, label in zip(axes, ("map_brier", "map_nll"), ("未知有效单元 Brier ↓", "未知有效单元 NLL ↓")):
        metric_bar(ax, grouped, key, label)
    fig.suptitle("地图逐格free频率指标 · " + title + " · validation-only")
    fig.text(.5, -.03, "按独立地点加权；n为已完成该阶段评估的种子数。误差棒为种子标准差，不是置信区间。缺失不是零。", ha="center", fontsize=8)
    fig.text(.5, -.11, metric_scope_annotation(data_scope, actual_seeds, map_metric=True), ha="center", fontsize=8)
    fig.tight_layout(); finish("03_hidden_map_metrics", "未知有效区域逐格频率Brier/NLL，和事件频率明确分开")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, label in zip(axes, ("event_raw", "event_calibrated_diagnostic"), ("原始事件分数", "仅校准集拟合的Platt诊断")):
        ax.plot([0, 1], [0, 1], color="#a8b5ae", linestyle="--", linewidth=1, label="理想参考线")
        for method, record in selected_records.items():
            bins = record["report"]["splits"]["validation"][key]["scene_weighted"]["overall"]["reliability"]
            nonempty = [b for b in bins if b["events"]]
            ax.plot([b["predicted"] for b in nonempty], [b["observed"] for b in nonempty], "o-", markersize=4,
                    linewidth=1.1, color=COLORS[method], label=NAMES[method])
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="事件分数", ylabel="实际可达比例", title=label)
        if not selected_records:
            ax.text(.5, .4, "尚无本阶段完整评估", ha="center")
        ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.15)
    fig.suptitle(f"事件可靠性图 · {title} · seed={seed}\nscene/query/radius=全部冻结验证清单 · validation-only")
    fig.text(.5, -.075, metric_scope_annotation(data_scope, [seed] if selected_records else []), ha="center", fontsize=8)
    fig.tight_layout(); finish("04_event_reliability", "固定10bins，原始事件分数与校准诊断分栏；地图方法为世界事件频率，直接查询为网络分数；空bin不伪造值")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, metric, label in zip(axes, ("event_brier", "event_nll", "event_ece"), ("事件 Brier ↓", "事件 NLL ↓", "事件 ECE ↓")):
        metric_bar(ax, grouped, metric, label)
    fig.suptitle("事件指标 · " + title + " · validation-only；全部固定场景/查询/半径")
    fig.text(.5, -.025, "误差棒为已完成该阶段评估的种子标准差（SD），不是置信区间；单一种子不画误差棒。", ha="center", fontsize=8)
    fig.text(.5, -.075, metric_scope_annotation(data_scope, actual_seeds), ha="center", fontsize=8)
    fig.tight_layout(); finish("05_event_metrics", "事件Brier/NLL/ECE，scene-weighted，仅汇合同阶段真实重复；完整阶段保留各自校准集所选步数")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, label in zip(axes, ("event_raw", "event_calibrated_diagnostic"), ("原始事件分数", "校准集映射诊断")):
        for method, record in selected_records.items():
            values = record["report"]["splits"]["validation"][key]["scene_weighted"]["overall"]
            points = risk_curve_points(values["false_safe_coverage_curve"])
            if len(points):
                ax.plot(points[:, 0], points[:, 1], "o-", markersize=3, color=COLORS[method], label=NAMES[method])
        if not selected_records:
            ax.text(.5, .5, "尚无本阶段完整评估", ha="center")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="接受覆盖率", ylabel="接受查询中不可达比例（误判风险）", title=label)
        if selected_records:
            ax.legend(fontsize=6, frameon=False)
        ax.grid(alpha=.15)
    fig.suptitle(f"误判风险与覆盖率 · {title} · seed={seed}\nscene/query/radius=全部冻结验证清单 · validation-only")
    fig.text(.5, -.03, "相同分数整体纳入，空接受集风险未定义。仅实际K4地图方法：频率≥0.8只含4/4；确定性K1；直接查询为连续网络分数。", ha="center", fontsize=7.5)
    fig.text(.5, -.12, metric_scope_annotation(data_scope, [seed] if selected_records else []), ha="center", fontsize=8)
    fig.tight_layout(); finish("06_false_safe_coverage", "保留原始分数阈值点；不将直接查询网络分数叫世界频率，不用平滑或标签排序制造优势")


def _format(value):
    return "—" if value is None else f"{value:.5f}"


def render_report_text(grouped, selected_group, records, rejected, protocol_hash, inventory, runtime=None):
    stage, step = selected_group
    lines = ["# ConPath 外部基线统一验证进度", "", f"当前展示：{STAGE_ZH[stage]}；检查点：{step}。成绩均为 validation-only；训练损失另列工程日志。",
             "", "**未训练、1000步和5000步结果仅用于工程与收敛诊断。没有完整三次重复、充分训练及最终审计时，不能进入论文主表。**",
             "", "| 方法 | 已完成种子 | 地图Brier | 事件Brier | 事件NLL | 事件ECE | 状态 |", "|---|---|---:|---:|---:|---:|---|"]
    # The state snapshot is separate from the immutable stage comparison table.
    lines[6:6] = runtime_markdown(runtime)
    for method, value in grouped.items():
        metrics = value["metrics"]
        lines.append("| " + " | ".join([LABELS[method], ", ".join(map(str, value["completed_outer_seeds"])) or "无",
                      *[_format(metrics[k]["mean"]) for k in ("map_brier", "event_brier", "event_nll", "event_ece")], value["status_zh"]]) + " |")
    lines += ["", "LaMa正式比较要求每个外层种子四个真实独立训练成员；FM与ConPath各生成四张世界。确定性和直接查询对照保留实际输出数，不复制为四张。",
              "地图指标为真实二值世界逐格free频率的误差。事件分数先在每张世界上进行离散圆盘侵蚀与四邻域连通性判断，再统计可达次数；二者不能混淆。",
              "Platt映射只在校准集拟合，原始频率与映射后诊断分栏；不保证校准。半径只用grid cell，不换算成米。",
              "", "## 六个论文问题", "",
              "1. **LaMa和FM充分训练后能否完成未知地图？** 当前汇总只展示实际完成阶段；完整训练和收敛复核尚未共同通过时无法正式判断。退化比例、参考基率、未知区域误差与固定图片用于诊断。",
              "2. **地图质量是否转化为可达性校准？** 分别报告地图和事件指标、可靠性图及风险覆盖曲线；只有相同协议充分训练后的三次重复才能支持结论。",
              "3. **ConPath是否优于普通补全加连通性？** 目前不作正式优越性判断；不兼容的外部论文3D指标没有混入。若外部基线领先，应保留该结果。",
              "4. **优势是否来自空间相关性和reachability loss？** 需要同协议独立单元与去可达性损失消融的完整三次重复；单个改进模型结果不能归因。",
              "5. **哪些结果可以进入论文主表？** 只有通过每方法完整计划、三个指定种子、统一验证与最终测试锁定审计的行；其余全部为诊断或待完成。",
              "6. **进入真实室内还是修改研究问题？** 在外部矩阵完成前暂不扩展研究方向；先依据充分训练后的任务校准结果判断假设是否成立。",
              "", "## 冻结图片与审计", "",
              "普通案例按预先冻结的输入无关hash选取。瓶颈、不可达和两条独立路径案例按参考地图几何规则挑选，选择在模型结果之前冻结；特殊案例不是无条件随机样本。",
              "普通模型对比图展示固定第1个样本，多样本图展示全部4张。起点和终点只标S/G，不画没有依据的连线。旧报告保留在reports时间戳目录中。",
              f"协议SHA-256：`{protocol_hash}`。完整可用评估数：{len(records)}；拒绝纳入的异常记录数：{len(rejected)}。",
              "本轮仅使用封存的physical-train开发缓存；location_6与最终测试继续锁定。该声明不抹去历史实验已披露的测试访问记录。", ""]
    for item in inventory:
        lines.append(f"- [{item['name']}](figures/{item['name']}.png)：{item['description_zh']}")
    return "\n".join(lines) + "\n"


def atomic_bytes(path, content):
    temporary = path.with_name("." + path.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path, text):
    atomic_bytes(path, text.encode("utf-8"))


def main():
    renderer_source = Path(__file__).read_text()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/flatlands_external_formal_protocol_v1/protocol.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()); protocol_hash = sha256(args.protocol)
    result_root, data_root = ROOT / protocol["output_root"], ROOT / protocol["data_root"]
    ownership = json.loads((result_root / "ownership.json").read_text())
    if ownership.get("protocol_sha256") != protocol_hash:
        raise ValueError("Report writes require the matching owned formal result root")
    if sha256(data_root / "seal.json") != protocol["data_seal_sha256"]:
        raise ValueError("Frozen eligible data seal changed")
    records, rejected = discover_evaluations(result_root, protocol_hash, protocol)
    runtime = runtime_snapshot(result_root, protocol_hash)
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    figure_manifest, geometry_audit, figure_manifest_path, geometry_audit_path = frozen_figure_artifacts(protocol)
    for record in records:
        plan_file = result_root / "full_plans" / f"{record['method']}.json"
        record["full_steps"] = json.loads(plan_file.read_text()).get("full_steps") if plan_file.exists() else None
    groups = sorted({r["group"] for r in records}, key=lambda g: (STAGES[g[0]], -1 if g[1] == "selected" else g[1]))
    chosen_group = groups[-1] if groups else ("untrained", 0)
    chosen = [r for r in records if r["group"] == chosen_group]
    grouped = summarize_group(chosen, protocol["methods"], result_root=result_root, protocol_hash=protocol_hash)
    seed = next((s for s in SEEDS if any(r["seed"] == s for r in chosen)), SEEDS[0])
    selected_records = {r["method"]: r for r in chosen if r["seed"] == seed}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot = result_root / "reports" / stamp
    figures = snapshot / "figures"; figures.mkdir(parents=True, exist_ok=False)
    atomic_text(snapshot / "renderer_source.py", renderer_source)
    configure_plotting()
    inventory = []
    title = f"{STAGE_ZH[chosen_group[0]]} · checkpoint={chosen_group[1]}"
    render_metric_figures(grouped, selected_records, title, seed, figures, inventory, data_audit["summary"]["validation"],
                          budget_note=budget_annotation(grouped) if chosen_group[0] == "formal" else "")
    loss_sources = render_training_losses(result_root, figures, inventory, data_audit["summary"]["train"])
    samples = {s.row["global_id"]: s for s in load_formal_data(data_root, "validation")}
    render_map_figures(figure_manifest, samples, selected_records, seed, title, figures, inventory)
    summary = {"protocol_sha256": protocol_hash, "selected_group": list(chosen_group), "selected_figure_seed": seed,
               "methods": grouped, "all_completed_groups": [list(g) for g in groups], "figures": inventory,
               "rejected_evaluations": rejected, "new_physical_test_images_opened": 0,
               "runtime_status_snapshot": runtime, "training_loss_sources": loss_sources,
               "final_test_locked": True, "location_6_locked": True, "validation_only": True,
               "comparison_selection": "latest completed stage/budget, then first prespecified available seed; never best score",
               "evaluation_records": [{"method": r["method"], "seed": r["seed"], "group": list(r["group"]),
                                        "path": str(r["path"].relative_to(ROOT)), "metrics_sha256": r["metrics_sha256"],
                                        "validation_metrics": validation_summary(r["report"]["splits"]["validation"])} for r in records]}
    atomic_json(snapshot / "metrics.json", summary)
    rows = summary_csv_rows(grouped, chosen, chosen_group)
    with (snapshot / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_text(snapshot / "report.md", render_report_text(grouped, chosen_group, records, rejected, protocol_hash, inventory, runtime))
    reproducibility = {"created_utc": stamp, "protocol_sha256": protocol_hash, "data_seal_sha256": sha256(data_root / "seal.json"),
                       "renderer_sha256": sha256(snapshot / "renderer_source.py"), "figure_manifest_sha256": sha256(figure_manifest_path),
                       "figure_geometry_audit_sha256": sha256(geometry_audit_path), "python": platform.python_version(),
                       "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                       "evaluation_records": [{key: value for key, value in row.items() if key != "validation_metrics"} for row in summary["evaluation_records"]], "figures": inventory,
                       "runtime_status_snapshot": runtime, "training_loss_sources": loss_sources,
                       "report_files": {name: sha256(snapshot / name) for name in ("metrics.json", "summary.csv", "report.md")}}
    atomic_json(snapshot / "reproducibility.json", reproducibility)
    # Refresh only owned summaries. Training checkpoints and immutable reports
    # are never changed or removed. A non-symlink figures directory is protected.
    current_figures = result_root / "figures"
    if current_figures.exists() and not current_figures.is_symlink():
        if current_figures.is_dir() and not any(current_figures.iterdir()):
            # Protocol initialization may reserve an empty owned output folder.
            # rmdir cannot remove any files, and fails if another writer adds one.
            current_figures.rmdir()
        else:
            raise ValueError("Refusing to replace a nonempty existing figures directory")
    if current_figures.is_symlink() and not current_figures.resolve().is_relative_to(result_root / "reports"):
        raise ValueError("Existing figures symlink does not belong to generated reports")
    temporary_link = result_root / (".figures.tmp." + str(os.getpid()))
    temporary_link.symlink_to(figures.relative_to(result_root), target_is_directory=True)
    os.replace(temporary_link, current_figures)
    for name in ("metrics.json", "summary.csv", "report.md", "reproducibility.json"):
        atomic_bytes(result_root / name, (snapshot / name).read_bytes())
    atomic_json(result_root / "current_report.json", {"snapshot": str(snapshot.relative_to(result_root)),
                                                     "reproducibility_sha256": sha256(snapshot / "reproducibility.json")})
    print(json.dumps({"report_snapshot": str(snapshot), "figures": len(inventory), "completed_evaluations": len(records),
                      "selected_stage": list(chosen_group), "paper_eligible_methods": [m for m, v in grouped.items() if v["paper_gate"]["eligible"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
