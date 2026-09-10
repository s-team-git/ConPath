#!/usr/bin/env python3
"""Build a separate Chinese validation-diagnostic page from audited snapshots.

No model, dataset packet, raw archive, checkpoint, training log or GPU access.
No report rendering or publication. The homepage and old assets are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260831, 20260901, 20260902)
STAGES = {"untrained": "随机初始化诊断 · 0步", "smoke": "稳定性诊断 · 1000步",
          "pilot": "收敛诊断 · 5000步", "formal": "完整训练后的开发验证"}
EXPLANATIONS = {"lama": "普通地图补全；每个外层种子训练四个独立成员", "flow": "按论文方法重新实现的流匹配补全；一个模型生成四张世界",
                "conpath": "联合空间随机地图与机器人尺寸感知可达性", "conpath_original": "ConPath 原类别采样器对照",
                "independent": "独立单元采样对照", "no_reach": "去除可达性损失的对照",
                "deterministic": "只输出一张地图的确定性对照", "direct_query": "直接输出查询分数；没有地图输出"}
METRICS = ("map_brier", "map_nll", "event_brier", "event_nll", "event_ece", "false_safe_at_0_8", "coverage_at_0_8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path, expected=None):
    content = Path(path).read_bytes()
    checksum = digest(content)
    require(expected is None or checksum == expected, "Content hash mismatch: " + str(path))
    return json.loads(content), checksum


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Artifact escapes its recorded root")
    return path


def figure_order(manifest):
    """Use every frozen case in manifest order, followed by all metric panels."""
    result = []
    def add(name, section, case=None, display_radii_cells=None):
        # Match the report renderer's panel rules, not the case's full radius
        # registration. A single-radius picture must never look like three.
        if display_radii_cells is None:
            display_radii_cells = [case.get("witness_radius_cells", case.get("bottleneck_failed_radius_cells", 10))] if case else [0, 10, 20]
        require(all(radius in (0, 10, 20) for radius in display_radii_cells), "Unregistered displayed radius")
        result.append({"name": name, "section": section, "case": case, "display_radii_cells": display_radii_cells})
    for index, case in enumerate(manifest["general_cases"]):
        add(f"01_comparison_{index:02d}", "固定案例完整图库", case)
    if manifest["general_cases"]:
        add("02_multiple_worlds", "四张真实世界", manifest["general_cases"][0], [10])
    for category, prefix, section in (("narrow_bottleneck", "07_bottleneck", "窄瓶颈"),
                                       ("unreachable", "08_unreachable", "不可达查询"),
                                       ("multiple_alternatives", "09_alternatives", "多替代路径")):
        cases = manifest["reference_categories"][category]["cases"]
        if not cases:
            add(prefix + "_unavailable", section)
        for index, case in enumerate(cases):
            add(f"{prefix}_{index:02d}", section, case)
            if category == "multiple_alternatives":
                add(f"{prefix}_reference_routes_{index:02d}", section, case, [case["witness_radius_cells"]])
    for index, case in enumerate(manifest["multi_radius_cases"]):
        add(f"10_radius_{index:02d}", "机器人半径的影响", case, [0, 10, 20])
    for name, section in (("03_hidden_map_metrics", "未知地图误差"), ("04_event_reliability", "可靠性图"),
                          ("05_event_metrics", "可达性误差"), ("06_false_safe_coverage", "误判风险与覆盖率")):
        add(name, section)
    return result


def evaluation_identity(report):
    steps = {p["completed_steps"] for p in report["checkpoint_provenance"]}
    require(len(steps) == 1, "An ensemble must use one common selected step")
    stage = report["stage"]
    require(stage in STAGES, "Unregistered evaluation stage")
    step = next(iter(steps))
    if stage != "formal":
        require(step == {"untrained": 0, "smoke": 1000, "pilot": 5000}[stage], "Wrong diagnostic step")
    return [stage, "selected" if stage == "formal" else step], step


def audited_run(run, method, selected_group, record, audit, protocol, protocol_hash, project_root):
    """Return only numbers bound to a passed independent evaluation audit."""
    require(audit["value"].get("passed") is True, "Independent stage audit did not pass")
    receipt = audit["value"]
    expected = {"protocol_sha256": protocol_hash, "data_seal_sha256": protocol["data_seal_sha256"],
                "method": method, "seed": run["seed"], "stage": selected_group[0],
                "metrics_sha256": run["metrics_sha256"], "real_world_budget": protocol["methods"][method]["actual_K"],
                "new_raw_archive_or_test_images_opened": 0, "gpu_inference_performed": False}
    require(all(receipt.get(key) == value for key, value in expected.items()), "Stage audit identity, K, or lock mismatch")
    evaluation = confined(project_root, record["path"])
    require(Path(receipt["evaluation_dir"]).resolve() == evaluation, "Audit references a different evaluation directory")
    original, _ = read_json(evaluation / "metrics.json", run["metrics_sha256"])
    complete, _ = read_json(evaluation / "complete.json", receipt.get("complete_sha256"))
    require(receipt.get("complete_sha256") is not None, "Stage audit lacks completion hash")
    require(complete.get("passed") is True and complete.get("metrics_sha256") == run["metrics_sha256"], "Evaluation is incomplete")
    for source in (original, complete):
        require(source.get("method") == method and source.get("stage") == selected_group[0], "Completion method/stage mismatch")
    require(original.get("outer_seed") == run["seed"] and complete.get("seed") == run["seed"], "Completion seed mismatch")
    group, step = evaluation_identity(original)
    require(group == selected_group == record["group"], "Cannot mix stages or diagnostic budgets")
    require(original.get("protocol_sha256") == protocol_hash and original.get("final_test_locked") is True
            and original.get("location_6_locked") is True and original.get("new_physical_test_images_opened") == 0
            and original.get("observed_and_support_constraints_passed") is True, "Evaluation scope/constraint mismatch")
    require(set(original["splits"]) == {"calibration", "validation"}, "A complete development evaluation needs both splits")
    validation = original["splits"]["validation"]
    for key in ("map", "event_raw", "event_calibrated_diagnostic", "by_source", "equal_source_macro"):
        require(run["validation_metrics"].get(key) == validation.get(key), "Snapshot metric strata differ from the audited source")
    values = {key: original["summary"]["validation"].get(key) for key in METRICS}
    require(all(value is None or isinstance(value, (int, float)) and math.isfinite(value) for value in values.values()), "Nonfinite metric")
    return {"seed": run["seed"], "selected_step": step, "full_steps": run.get("full_steps"), "metrics": values,
            "metrics_sha256": run["metrics_sha256"], "audit_sha256": audit["sha256"],
            "audit_created_utc": receipt.get("created_utc"), "actual_K": expected["real_world_budget"],
            "validation_metrics": {key: value for key, value in validation.items() if key != "cases"},
            "checkpoint_and_training_cost_provenance": run.get("checkpoint_and_training_cost_provenance", [])}


def prepare_site(snapshot, protocol_path, visual_audit_path, stage_audit_paths, *, project_root=ROOT):
    project_root = Path(project_root).resolve()
    protocol, protocol_hash = read_json(protocol_path)
    require(protocol["seeds"] == list(SEEDS) and protocol["sampling"]["K"] == 4
            and protocol["query"]["radii_cells"] == [0, 10, 20], "Unregistered comparison protocol")
    require(protocol["test_lock"].get("final_test_locked") is True and protocol["test_lock"].get("location_6_locked") is True, "Test lock absent")
    result_root = confined(project_root, protocol["output_root"])
    snapshot = Path(snapshot).resolve()
    require(snapshot.parent == result_root / "reports", "An explicit immutable reports/<timestamp> snapshot is required")
    reproducibility, repro_hash = read_json(snapshot / "reproducibility.json")
    require(reproducibility.get("protocol_sha256") == protocol_hash and reproducibility.get("data_seal_sha256") == protocol["data_seal_sha256"], "Snapshot protocol mismatch")
    report, report_hash = read_json(snapshot / "metrics.json", reproducibility["report_files"]["metrics.json"])
    for name, checksum in reproducibility["report_files"].items():
        require(digest(confined(snapshot, name).read_bytes()) == checksum, "Immutable report file changed")
    require(report.get("validation_only") is True and report.get("final_test_locked") is True and report.get("location_6_locked") is True
            and report.get("new_physical_test_images_opened") == 0 and report.get("protocol_sha256") == protocol_hash, "Report is outside locked validation scope")
    group = report["selected_group"]
    require(group[0] in STAGES, "Unregistered report stage")
    require(report["figures"] == reproducibility["figures"], "Figure inventories disagree")
    require(digest((snapshot / "renderer_source.py").read_bytes()) == reproducibility["renderer_sha256"], "Archived renderer changed")
    manifest, manifest_hash = read_json(confined(project_root, protocol["figures"]["cases"]), protocol["figures"]["sha256"])
    geometry, geometry_hash = read_json(confined(project_root, protocol["figures"]["geometry_verification"]), protocol["figures"]["geometry_verification_sha256"])
    require(manifest_hash == reproducibility["figure_manifest_sha256"]
            and manifest.get("frozen_before_formal_model_training_and_inference") is True
            and manifest.get("validation_only") is True and manifest.get("data_seal_sha256") == protocol["data_seal_sha256"], "Figure list was not frozen before model outputs")
    require(geometry.get("passed") is True and geometry.get("figure_cases_sha256") == manifest_hash
            and geometry.get("data_seal_sha256") == protocol["data_seal_sha256"]
            and geometry_hash == reproducibility["figure_geometry_audit_sha256"], "Frozen reference geometry audit mismatch")
    visual, visual_hash = read_json(visual_audit_path)
    require(visual.get("passed") is True and visual.get("snapshot") == str(snapshot.relative_to(result_root))
            and visual.get("reproducibility_sha256") == repro_hash and visual.get("new_physical_test_images_opened") == 0,
            "A passed visual receipt bound to this snapshot reproducibility hash is required")
    audit_by_hash = {}
    for path in stage_audit_paths:
        value, checksum = read_json(path)
        if value.get("passed") is True and value.get("protocol_sha256") == protocol_hash:
            audit_by_hash[value.get("metrics_sha256")] = {"value": value, "sha256": checksum}
    records = {(r["method"], r["seed"], r["metrics_sha256"]): r for r in report["evaluation_records"]}
    methods, withheld, selected_count = {}, [], 0
    for method, recipe in protocol["methods"].items():
        require(method in EXPLANATIONS, "Unknown method needs an explicit Chinese explanation")
        runs = report["methods"][method]["runs"]
        require(len({r["seed"] for r in runs}) == len(runs) and all(r["seed"] in SEEDS for r in runs), "Repeated or unregistered seed")
        verified = []
        for run in runs:
            selected_count += 1
            audit = audit_by_hash.get(run["metrics_sha256"])
            if audit is None:
                withheld.append({"method": method, "seed": run["seed"], "reason": "完整结果尚无匹配的独立审计，不发布数值或复合图"})
                continue
            key = (method, run["seed"], run["metrics_sha256"])
            require(key in records, "Snapshot run lacks an original evaluation reference")
            verified.append(audited_run(run, method, group, records[key], audit, protocol, protocol_hash, project_root))
        stats = {}
        for key in METRICS:
            values = [r["metrics"][key] for r in verified if r["metrics"][key] is not None]
            stats[key] = {"mean": statistics.mean(values) if values else None, "sd": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        methods[method] = {"label": recipe["label"], "explanation_zh": EXPLANATIONS[method], "actual_K": recipe["actual_K"],
                           "verified_seeds": [r["seed"] for r in verified], "pending_seeds": [s for s in SEEDS if s not in {r["seed"] for r in verified}],
                           "runs": verified, "metrics": stats, "paper_main_or_superiority_authorized": False}
    allow_figures = selected_count > 0 and not withheld
    inventory = {item["name"]: item for item in report["figures"]}
    require(len(inventory) == len(report["figures"]), "Repeated figure name")
    assets, figures = [], []
    for expected in figure_order(manifest):
        require(expected["name"] in inventory, "Fixed gallery is incomplete: " + expected["name"])
        item = inventory[expected["name"]]
        if not allow_figures:
            continue
        files = []
        for recorded in item["files"]:
            path = confined(snapshot, recorded["path"])
            require(path.suffix in (".png", ".svg", ".pdf") and path.stem == expected["name"], "Unexpected figure artifact")
            require(digest(path.read_bytes()) == recorded["sha256"], "Figure asset hash changed")
            relative = "figures/" + path.name
            assets.append({"source": path, "relative": relative, "sha256": recorded["sha256"]})
            files.append({"path": relative, "sha256": recorded["sha256"]})
        require({Path(f["path"]).suffix for f in files} == {".png", ".svg", ".pdf"}, "Each frozen figure needs PNG/SVG/PDF")
        case = expected["case"]
        case_public = {key: case.get(key) for key in ("global_id", "scene_id", "query_id", "radii_cells", "label_zh")} if case else None
        figures.append({"name": expected["name"], "section": expected["section"], "case": case_public,
                        "display_radii_cells": expected["display_radii_cells"],
                        "description_zh": item["description_zh"], "files": files})
    return {"schema_version": 1, "validation_only": True, "paper_main_or_superiority_authorized": False,
            "snapshot": snapshot.name, "created_utc": datetime.now(timezone.utc).isoformat(), "snapshot_created_utc": reproducibility["created_utc"],
            "protocol_sha256": protocol_hash, "report_metrics_sha256": report_hash, "reproducibility_sha256": repro_hash,
            "visual_audit_sha256": visual_hash, "figure_manifest_sha256": manifest_hash,
            "selected_group": group, "stage_zh": STAGES[group[0]], "figure_seed": report["selected_figure_seed"],
            "registered_seeds": list(SEEDS), "radii_cells": [0, 10, 20], "common_K": 4,
            "methods": methods, "figures": figures, "withheld": withheld,
            "figure_status": "已核验完整固定图库" if allow_figures else "同阶段输出尚未全部通过独立审计，复合图暂缓发布",
            "final_test_locked": True, "location_6_locked": True, "new_physical_test_images_opened": 0,
            "training_loss_images_published": False, "homepage_entry_proposal": {"href": "formal.html", "label": "外部基线统一验证（开发诊断）"}}, assets


CSS = """:root{color-scheme:light;--ink:#203d36;--muted:#60766e;--line:#dbe5df;--paper:#f7f8f5;--teal:#147566}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 system-ui,-apple-system,'Noto Sans CJK SC','Microsoft Yahei',sans-serif}a{color:var(--teal);text-underline-offset:4px}header,main,footer{width:min(1180px,calc(100% - 40px));margin:auto}header{display:flex;justify-content:space-between;align-items:center;padding:24px 0;border-bottom:1px solid var(--line)}nav{display:flex;flex-wrap:wrap;gap:18px;font-size:14px}.brand{white-space:nowrap;font-weight:750;color:var(--ink);text-decoration:none}.hero{padding:64px 0 40px;max-width:920px}.eyebrow{letter-spacing:.1em;color:var(--teal);font-size:13px;font-weight:650}h1{font-size:clamp(30px,5vw,52px);letter-spacing:-.035em;line-height:1.25;margin:16px 0 20px}h2{font-size:26px;letter-spacing:-.02em;margin:0 0 12px}p{margin:10px 0}.lead{font-size:19px;color:var(--muted)}.badge{display:inline-block;border:1px solid #c4dcd0;background:#edf5ef;border-radius:30px;padding:5px 13px;font-size:13px}.notice{border-left:3px solid #b88046;background:#f3efe5;padding:16px 20px;margin:20px 0;color:#66583f}.section{padding:38px 0;border-top:1px solid var(--line)}.small{font-size:13px;color:var(--muted)}.table-scroll,.image-scroll{overflow:auto;-webkit-overflow-scrolling:touch}table{width:100%;border-collapse:collapse;text-align:left;font-size:14px;min-width:930px}caption{text-align:left;padding:10px 0;color:var(--muted)}th,td{padding:15px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}thead th{font-weight:600;color:var(--muted);white-space:nowrap}tbody th{font-weight:600;min-width:230px}td{white-space:nowrap}.method-en{display:block;font-size:13px;font-weight:500;color:var(--muted)}.pending{color:#8b8070}.how{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}.how article{border-top:2px solid #bfd4c7;padding-top:13px}.how h3{margin:5px 0;font-size:17px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:13px}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:3px;vertical-align:-1px}.free{background:#65b59d}.blocked{background:#364654}.unknown{background:#dce4ec}.support{background:repeating-conic-gradient(#e8ecee 0 25%,#f7f8f6 0 50%) 0/8px 8px}.start{color:#276bb0}.goal{color:#b96b27}details{background:#fff;border:1px solid var(--line);border-radius:12px;margin:13px 0;overflow:hidden}summary{padding:18px 20px;cursor:pointer;font-weight:550}summary span{display:block;color:var(--muted);font-size:12px;font-weight:400;margin:5px 0 0 18px}.figure-body{padding:0 18px 18px}.image-scroll{border:1px solid #eef1ee;border-radius:7px;background:white}.image-scroll img{display:block;width:100%;min-width:900px;height:auto}.downloads{font-size:13px;display:flex;gap:15px;padding-top:8px}.provenance{overflow-wrap:anywhere}code{font-size:12px}footer{padding:35px 0 50px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}:focus-visible{outline:3px solid #dcad6f;outline-offset:4px}@media(max-width:650px){header,main,footer{width:calc(100% - 28px)}header{flex-direction:column;align-items:flex-start;gap:12px}nav{width:100%;gap:18px}.hero{padding:40px 0 28px}.how{grid-template-columns:1fr;gap:16px}.section{padding:28px 0}.lead{font-size:17px}summary{padding:14px}.figure-body{padding:0 10px 12px}}"""


def esc(value):
    return html.escape(str(value), quote=True)


def render_html(data, version_url):
    rows = []
    for method, row in data["methods"].items():
        cells = []
        for metric in METRICS:
            stats = row["metrics"][metric]
            if method == "direct_query" and metric.startswith("map_"):
                label = "不适用"
            elif stats["mean"] is None:
                label = "—"
            else:
                label = f'{stats["mean"]:.4f}' + (f' ± {stats["sd"]:.4f}' if stats["sd"] is not None else "")
            cells.append("<td>" + label + "</td>")
        status = f'{len(row["verified_seeds"])}/3 已核验' if row["verified_seeds"] else "待完成 / 待核验"
        names = ", ".join(map(str, row["verified_seeds"])) or "本阶段尚无可发布结果"
        rows.append(f'<tr><th scope="row">{esc(EXPLANATIONS[method])}<span class="method-en">{esc(row["label"])}</span></th><td title="{esc(names)}">{esc(status)}<span class="method-en">{esc(names)}</span></td>' + ''.join(cells) + '</tr>')
    gallery = []
    previous = None
    for figure in data["figures"]:
        if figure["section"] != previous:
            gallery.append('<h3>' + esc(figure["section"]) + '</h3>')
            previous = figure["section"]
        png = next(file["path"] for file in figure["files"] if file["path"].endswith('.png'))
        case = figure["case"]
        displayed_radius = "/".join(map(str, figure["display_radii_cells"]))
        detail = (f'scene={case["scene_id"]} · query={case["query_id"]} · r={displayed_radius} 格' if case
                  else '全部冻结验证场景、查询与半径；具体集合和实际种子见原图注记')
        detail += f' · seed={data["figure_seed"]} · validation-only'
        download = ' '.join(f'<a href="{esc(version_url + "/" + file["path"])}" target="_blank" rel="noopener">{Path(file["path"]).suffix[1:].upper()} 原图 ↗</a>' for file in figure["files"])
        gallery.append(f'<details><summary>{esc(figure["description_zh"])}<span>{esc(detail)}</span></summary><div class="figure-body"><div class="image-scroll" tabindex="0" aria-label="可横向滚动的完整原图"><img src="{esc(version_url + "/" + png)}" loading="lazy" decoding="async" alt="{esc(figure["description_zh"] + "；" + detail)}"></div><div class="downloads">{download}</div></div></details>')
    verified_methods = sum(bool(row["runs"]) for row in data["methods"].values())
    verified_runs = sum(len(row["runs"]) for row in data["methods"].values())
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="ConPath外部基线统一开发验证：相同地图输入、固定查询与机器人半径，真实结果独立核验后展示。"><title>外部基线开发验证 · ConPath</title><link rel="icon" href="favicon.svg"><link rel="stylesheet" href="{esc(version_url)}/formal.css"></head>
<body><header><a class="brand" href="index.html">ConPath / 外部验证</a><nav aria-label="页内导航"><a href="#results">阶段数据</a><a href="#gallery">效果图库</a><a href="#reading">如何读图</a></nav></header>
<main><section class="hero"><p class="eyebrow">相同输入 · 固定查询 · 独立核验</p><h1>外部基线，现在验证到哪一步？</h1><span class="badge">{esc(data["stage_zh"])} · validation-only</span><p class="lead">本页展示同一阶段已经完成并通过独立核验的结果：{verified_methods}种方法、{verified_runs}组种子评估。缺失处表示工作尚未完成或尚未核验。</p><div class="notice"><strong>这是开发验证诊断，没有授予论文主表或正式优越性结论。</strong><p>随机初始化、1000步和5000步不能代替充分训练。正式矩阵还要求每种方法完成三个固定种子、校准集选优与最终审计；下表的“已核验”仅指当前阶段。</p></div><p class="small">快照：{esc(data["snapshot_created_utc"])}。这是已完成证据的固定快照，不是实时训练看板；不同阶段不会拼成同一比较。</p></section>
<section class="section" id="results"><h2>同一阶段，地图与可达性分开看</h2><p>所有方法使用相同场景、隐藏区域、有效范围、起终点与半径。指标按父级场景等权；多种子时显示均值 ± 标准差，只有一个种子时不伪造标准差。</p><div class="table-scroll" tabindex="0"><table><caption>{esc(data["stage_zh"])}；三个指定种子为20260831、20260901、20260902。横向滑动查看完整表。</caption><thead><tr><th scope="col">方法与中文解释</th><th scope="col">本阶段种子</th><th scope="col">地图 Brier ↓</th><th scope="col">地图 NLL ↓</th><th scope="col">事件 Brier ↓</th><th scope="col">事件 NLL ↓</th><th scope="col">事件 ECE ↓</th><th scope="col">0.8阈值误判率 ↓</th><th scope="col">0.8阈值覆盖率</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="small">误判率、覆盖率使用0–1比例，0.13即13%。 “—”不是零分；没有任何查询被接受时，误判率未定义。直接查询模型不产生地图，地图指标不适用。所有可发布分层数据与成本保存在<a href="{esc(version_url)}/diagnostics.json">本快照 JSON</a>。效率来自共享GPU环境，记录实际生成与完整更新时间、显存和计算次数；不能据此当作独占设备上的速度排名。</p></section>
<section class="section" id="reading"><h2>先认识颜色，再看地图</h2><div class="legend"><span><i class="swatch free"></i>可通行</span><span><i class="swatch blocked"></i>阻挡</span><span><i class="swatch unknown"></i>输入中的未知区域</span><span><i class="swatch support"></i>棋盘纹：支持范围外，计算中封闭</span><span class="start">S 蓝圆：起点</span><span class="goal">G 棕菱形：目标</span></div><div class="how"><article><h3>四张地图，不是四次复制</h3><p>共同外部预算为K=4。LaMa使用四个独立训练成员；FM和ConPath各生成四张真实世界。确定性对照实际K=1，直接查询对照没有世界输出。</p></article><article><h3>逐张判断，才得到事件频率</h3><p>每张世界先做机器人尺寸对应的精确连通性判断，再统计“几张可达”。逐格可通行比例不是路径概率。对实际K=4的地图方法，原始世界事件频率达到0.8只能是4/4通过。直接查询输出连续网络分数，确定性对照仅一张地图。</p></article><article><h3>圆圈表示半径，不是路线</h3><p>空心圈表示半径，固定为0、10、20个网格，不换算为米。绿色格子连通不保证带半径的机器人能通过：窄通道或端点放不下圆盘时仍不可达。S/G之间没有凭空连线；只有“参考路径A/B”图绘制经过几何核验的参考路线。</p></article></div><p class="small">校准后的曲线仅使用校准集拟合统一映射，与原始事件频率分栏展示，不承诺完美校准。LaMa BEV adaptation是本项目BEV适配；FM+XAttn literature reimplementation是文献重新实现，均不是原作者官方checkpoint，也未混入不兼容的3D论文指标。</p></section>
<section class="section" id="gallery"><h2>按冻结顺序浏览完整图库</h2><p>普通案例按模型输出之前冻结的清单排序；瓶颈、不可达与多路径案例按事先冻结的参考几何规则选择。全部列出，未按模型优劣挑图；训练损失图不在本页发布。</p><p class="small">{esc(data["figure_status"])}。点击展开；手机可横向滑动完整原图，也可打开PNG、SVG或PDF。</p>{''.join(gallery)}</section>
<section class="section provenance"><h2>范围与来源</h2><p>本轮使用已封存的physical-train开发划分。location_6与最终测试继续锁定，没有读取新的测试图像。这不抹去历史访问记录：<a href="research.html#baseline-review">旧实验与数据隔离更正</a> · <a href="data/flatlands_read_scope_erratum.json">历史读取范围更正记录</a>。</p><p>协议SHA-256：<code>{esc(data["protocol_sha256"])}</code><br>不可变报告：<code>{esc(data["snapshot"])}</code><br>报告复现回执SHA-256：<code>{esc(data["reproducibility_sha256"])}</code></p><p class="small">本页不会根据当前分数挑选检查点，也不会把早期更优的合法校准集检查点替换成较晚检查点。所有正式资格判断仍由完整训练计划和三次重复的最终审计决定。</p></section></main>
<footer>ConPath · 统一外部基线开发验证。<a href="index.html">返回已有模型展示</a>。该首页包含此前独立阶段的历史结果，不能与本页数值直接拼表。</footer></body></html>'''


def atomic_bytes(path, content):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def write_site(data, assets, site_root):
    """Versioned immutable assets, then atomic page refresh; no homepage edits."""
    site_root = Path(site_root)
    site_root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    data = dict(data, builder_sha256=digest(source))
    encoded = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    version = re.sub(r"[^A-Za-z0-9_.-]", "_", data["snapshot"]) + "_" + digest(encoded)[:12]
    version_url = "assets/formal/" + version
    bundle = site_root / version_url
    require(not bundle.exists(), "Refusing to overwrite an existing versioned site bundle")
    bundle.mkdir(parents=True)
    inventory = []
    for asset in assets:
        content = asset["source"].read_bytes()
        require(digest(content) == asset["sha256"], "Source figure changed before copy")
        target = confined(bundle, asset["relative"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        inventory.append({"path": asset["relative"], "sha256": digest(content)})
    (bundle / "diagnostics.json").write_bytes(encoded)
    (bundle / "formal.css").write_text(CSS)
    (bundle / "builder_source.py").write_bytes(source)
    page = render_html(data, version_url).encode()
    (bundle / "formal.html").write_bytes(page)
    receipt = {"schema_version": 1, "validation_only": True, "website_published": False,
               "homepage_modified": False, "version_url": version_url, "data_sha256": digest(encoded),
               "page_sha256": digest(page), "stylesheet_sha256": digest(CSS.encode()),
               "source_snapshot": data["snapshot"], "source_reproducibility_sha256": data["reproducibility_sha256"],
               "builder_sha256": digest(source), "assets": inventory,
               "homepage_entry_proposal": data["homepage_entry_proposal"]}
    receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode()
    (bundle / "build_receipt.json").write_bytes(receipt_bytes)
    atomic_bytes(site_root / "formal.html", page)
    (site_root / "data").mkdir(exist_ok=True)
    atomic_bytes(site_root / "data/formal_current.json", receipt_bytes)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-snapshot", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/flatlands_external_formal_protocol_v1/protocol.json")
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--stage-audit", type=Path, action="append", required=True)
    parser.add_argument("--site-root", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    data, assets = prepare_site(args.report_snapshot, args.protocol, args.visual_audit, args.stage_audit)
    receipt = write_site(data, assets, args.site_root)
    print(json.dumps({key: receipt[key] for key in ("version_url", "page_sha256", "website_published", "homepage_modified")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
