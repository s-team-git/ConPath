#!/usr/bin/env python3
"""Independently replay the external adapter and verify the frozen group split."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from pathrel.cogniplan import configure_reproducible_cuda, load_native, predict_four


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path("results/external_publication_20260908")
    out.mkdir(parents=True, exist_ok=True)
    protocol = Path("results/external_protocol_v1")
    profile = Path("results/cogniplan_native_profile_v2")
    checks = []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise ValueError(name)
    for root in (protocol, profile):
        report = json.loads((root / "report.json").read_text())
        for name, expected in report["outputs"].items():
            check(f"{root}/{name} hash", sha(root / name) == expected)
        for name, expected in report["inputs"].items():
            check(f"{name} input hash", sha(Path(name)) == expected)
        check(f"{root} test lock", report["test_assets_opened"] is False)
    mothers = list(csv.DictReader((protocol / "cogniplan_mothers.csv").open()))
    views = list(csv.DictReader((protocol / "cogniplan_observations.csv").open()))
    by_id = {r["mother_id"]: r for r in mothers}
    check("all views inherit mother split", all(by_id[r["mother_id"]]["split"] == r["split"] for r in views))
    check("3000 distinct mother IDs", len(by_id) == len(mothers) == 3000)
    for a, b in (("train", "calibration"), ("train", "validation"), ("calibration", "validation")):
        groups_a = {r["d4_group_sha256"] for r in mothers if r["split"] == a}
        groups_b = {r["d4_group_sha256"] for r in mothers if r["split"] == b}
        check(f"{a}/{b} D4 group disjoint", not groups_a & groups_b)
    fixed = list(csv.DictReader((protocol / "cogniplan_profile32.csv").open()))
    check("32 distinct train-only profile mothers", len({r["mother_id"] for r in fixed}) == 32 and
          all(by_id[r["mother_id"]]["split"] == "train" for r in fixed))
    torch.set_num_threads(4)
    configure_reproducible_cuda()
    model = load_native()["Generator"]({"input_dim": 1, "ngf": 16}, True).cuda()
    archive = Path("data/raw/cogniplan/paper_model_exploration/checkpoints_wgan_inpainting.tar.gz")
    with tarfile.open(archive) as tar:
        raw = tar.extractfile("wgan_inpainting/gen_00500000.pt").read()
    check("public checkpoint hash", hashlib.sha256(raw).hexdigest() ==
          json.loads((profile / "report.json").read_text())["public_checkpoint_sha256"])
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True), strict=True)
    saved = np.load(profile / "native_outputs.npz")
    drift = []
    for i in range(32):
        raw, worlds = predict_four(model, saved[f"partial_{i}"])
        difference = float(np.max(np.abs(raw - saved[f"raw_{i}"])))
        drift.append(difference)
        check(f"sample {i}: adapter raw output exact replay", difference == 0)
        check(f"sample {i}: adapter binary worlds exact", np.array_equal(worlds, saved[f"worlds_{i}"]))
    snapshot = json.loads(Path("site/data/cogniplan_native_sanity_zh.json").read_text())
    check("public sanity is not formal comparison", not snapshot["formal_comparison"] and not snapshot["test_evaluated"])
    check("public report source hash", snapshot["report_sha256"] == sha(profile / "report.json"))
    check("renderer source hash", snapshot["renderer_sha256"] == sha(Path("scripts/render_cogniplan_native_zh.py")))
    for asset in snapshot["assets"]:
        check(asset["path"] + " asset hash", sha(Path("site") / asset["path"]) == asset["sha256"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": all(c["passed"] for c in checks),
              "checks": checks, "check_count": len(checks), "raw_max_absolute_drift": max(drift),
              "binary_worlds_replayed": 128, "test_assets_opened": False,
              "script_sha256": sha(Path(__file__))}
    (out / "native_replay_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
