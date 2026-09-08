#!/usr/bin/env python3
"""Render real, preselected native training-sanity outputs with Chinese legends."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("results/cogniplan_native_profile_v2")
    report = json.loads((root / "report.json").read_text())
    assert report["native_interface_passed"] and not report["test_assets_opened"]
    for name, expected in report["outputs"].items():
        assert sha(root / name) == expected
    data = np.load(root / "native_outputs.npz")
    rows = list(csv.DictReader((root / "native_samples.csv").open()))
    out = Path("site/assets/zh/cogniplan-native")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 12})
    colors = ["#263447", "#c8eee2", "#dce1e8"]
    cmap = ListedColormap(colors)
    votes = LinearSegmentedColormap.from_list("vote", ["#f5f4eb", "#197f70"])
    layouts = {"room": "房间", "tunnel": "隧道", "outdoor": "户外"}
    cases, assets = [], []
    for layout, zh in layouts.items():
        row = next(r for r in rows if r["layout"] == layout)
        idx = int(row["index"])
        partial, target, worlds = (data[f"{key}_{idx}"] for key in ("partial", "target", "worlds"))
        observed = np.where(partial == 127, 2, partial == 255)
        panels = {"observed": (observed, "① 模型看到的地图"),
                  "vote": (worlds.mean(0), "② 四张补全的可通行比例"),
                  "reference": (target, "③ 完整参考 · 仅核对用")}
        panels.update({f"world{k}": (world, name) for k, (world, name) in enumerate(zip(worlds,
                       ("固定条件一：均衡", "固定条件二：房间", "固定条件三：隧道", "固定条件四：户外")))})
        paths = {}
        for key, (array, title) in panels.items():
            fig, ax = plt.subplots(figsize=(4.6, 5.55), dpi=130)
            fig.subplots_adjust(left=.04, right=.96, bottom=.255, top=.84)
            ax.set_title(title, fontsize=14, pad=10)
            image = ax.imshow(array, cmap=votes if key == "vote" else cmap, vmin=0, vmax=1 if key == "vote" else 2,
                              interpolation="nearest")
            ax.set_axis_off()
            fig.text(.5, .95, f"{zh} · {row['mother_id']}", ha="center", fontsize=12, color="#5b6573")
            if key == "vote":
                cax = fig.add_axes([.15, .185, .7, .026])
                fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, .25, .5, .75, 1])
                fig.text(.5, .102, "0 = 四张均阻挡；1 = 四张均可通行", ha="center", fontsize=10)
            else:
                handles = [Patch(facecolor=colors[i], label=label) for i, label in
                           ((1, "可通行"), (0, "阻挡"), (2, "未知")) if i != 2 or key == "observed"]
                fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .11), ncol=len(handles), frameon=False, fontsize=11)
            fig.text(.5, .045, "官方权重 · 原训练地图接口检查\n不代表留出验证或 ConPath 对比成绩", ha="center", fontsize=10, color="#5b6573")
            path = out / f"{layout}-{key}.png"
            fig.savefig(path, facecolor="white"); plt.close(fig)
            paths[key] = path.relative_to("site").as_posix()
            assets.append({"path": paths[key], "sha256": sha(path), "source_index": idx, "source_array": key})
        cases.append({"layout": layout, "layout_zh": zh, "member": row["member"], "mother_id": row["mother_id"],
                      "index": idx, "panels": paths})
    # All 32 fixed outcomes, including imperfect predictions, for visual audit.
    for page in range(4):
        fig, axes = plt.subplots(8, 3, figsize=(9, 20), dpi=90)
        for position, row in enumerate(rows[page * 8:(page + 1) * 8]):
            idx = page * 8 + position
            partial = data[f"partial_{idx}"]
            arrays = (np.where(partial == 127, 2, partial == 255), data[f"worlds_{idx}"].mean(0), data[f"target_{idx}"])
            for col, array in enumerate(arrays):
                axes[position, col].imshow(array, cmap=votes if col == 1 else cmap, vmin=0, vmax=1 if col == 1 else 2)
                axes[position, col].set_title(f"{row['mother_id']} · {('观测', '四图比例', '参考')[col]}", fontsize=10)
                axes[position, col].set_axis_off()
        fig.tight_layout()
        fig.savefig(root / f"contact_sheet_{page}.png"); plt.close(fig)
    snapshot = {"schema_version": 1, "kind": "native_training_sanity_only", "test_evaluated": False,
                "formal_comparison": False, "report_sha256": sha(root / "report.json"),
                "raw_predictions_sha256": sha(root / "native_outputs.npz"),
                "selection": "First frozen hash-ranked profile mother in each layout; chosen without inspecting predictions or scores.",
                "cases": cases, "assets": assets, "samples": 32, "worlds": 128,
                "training_phases": report["training_phases"],
                "training_hours_per_seed_extrapolation": report["original_recipe_training_only_hours_per_seed_estimate"],
                "training_hours_three_seeds_extrapolation": report["original_recipe_training_only_hours_three_seeds_serial_estimate"],
                "renderer_sha256": sha(Path(__file__)),
                "provenance": "marmotlab/CogniPlan@444fab8d5d3b8d2b004d83865088a9d707d2e3c8; gen_00500000.pt"}
    (Path("site/data") / "cogniplan_native_sanity_zh.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"assets": len(assets), "public_cases": len(cases), "audit_cases": 32}))


if __name__ == "__main__":
    main()
