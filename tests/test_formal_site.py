import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_flatlands_formal_site import (SEEDS, METRICS, CSS, digest, figure_order, prepare_site, render_html, write_site)


class FormalSiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results/formal"
        self.snapshot = self.results / "reports/fixed-snapshot"
        self.snapshot.mkdir(parents=True)
        self.protocol_path = self.root / "protocol.json"
        self.visual_path = self.root / "visual.json"
        self.stage_audit = self.root / "stage_audit.json"
        self.case = {"global_id": "obs_2", "scene_id": "scene2", "query_id": "obs_2:q001", "radii_cells": [0, 10, 20]}
        self.manifest = {"validation_only": True, "data_seal_sha256": "seal", "frozen_before_formal_model_training_and_inference": True,
                         "general_cases": [self.case], "multi_radius_cases": [],
                         "reference_categories": {key: {"cases": []} for key in ("narrow_bottleneck", "unreachable", "multiple_alternatives")}}
        self.write(self.root / "cases.json", self.manifest)
        self.geometry = {"passed": True, "figure_cases_sha256": self.sha(self.root / "cases.json"), "data_seal_sha256": "seal"}
        self.write(self.root / "geometry.json", self.geometry)
        self.protocol = {"seeds": list(SEEDS), "sampling": {"K": 4}, "query": {"radii_cells": [0, 10, 20]},
                         "test_lock": {"final_test_locked": True, "location_6_locked": True}, "output_root": "results/formal", "data_seal_sha256": "seal",
                         "figures": {"cases": "cases.json", "sha256": self.sha(self.root / "cases.json"), "geometry_verification": "geometry.json",
                                     "geometry_verification_sha256": self.sha(self.root / "geometry.json")},
                         "methods": {"lama": {"label": "LaMa BEV adaptation", "actual_K": 4},
                                     "flow": {"label": "FM+XAttn literature reimplementation", "actual_K": 4}}}
        self.write(self.protocol_path, self.protocol)
        self.ph = self.sha(self.protocol_path)
        validation = {"map": {"scene_weighted": {"sample_vote_cell_brier": .2}},
                      "event_raw": {"scene_weighted": {"overall": {"event_brier": .3}}},
                      "event_calibrated_diagnostic": {"pooled": {"overall": {"event_brier": .25}}},
                      "by_source": {"source": {"event_raw": {"scene_weighted": {"overall": {"event_brier": .3}}}}},
                      "equal_source_macro": {"event_brier": .3}, "efficiency": {"case_count": 191}}
        self.original = {"method": "lama", "outer_seed": SEEDS[0], "stage": "smoke", "protocol_sha256": self.ph,
                         "checkpoint_provenance": [{"completed_steps": 1000} for _ in range(4)], "final_test_locked": True,
                         "location_6_locked": True, "new_physical_test_images_opened": 0, "observed_and_support_constraints_passed": True,
                         "splits": {"calibration": {}, "validation": validation}, "summary": {"validation": {key: .314159 for key in METRICS}}}
        self.evaluation = self.results / "stages/lama/20260831/smoke/attempt_1"
        self.evaluation.mkdir(parents=True)
        self.write(self.evaluation / "metrics.json", self.original)
        self.complete = {"passed": True, "metrics_sha256": self.sha(self.evaluation / "metrics.json"), "method": "lama", "seed": SEEDS[0], "stage": "smoke"}
        self.write(self.evaluation / "complete.json", self.complete)
        self.audit = {"passed": True, "protocol_sha256": self.ph, "data_seal_sha256": "seal", "method": "lama", "seed": SEEDS[0], "stage": "smoke",
                      "metrics_sha256": self.complete["metrics_sha256"], "real_world_budget": 4, "new_raw_archive_or_test_images_opened": 0,
                      "gpu_inference_performed": False, "evaluation_dir": str(self.evaluation), "complete_sha256": self.sha(self.evaluation / "complete.json")}
        self.write(self.stage_audit, self.audit)
        inventory = []
        for item in figure_order(self.manifest):
            files = []
            for extension in ("png", "svg", "pdf"):
                path = self.snapshot / "figures" / (item["name"] + "." + extension)
                path.parent.mkdir(exist_ok=True)
                path.write_bytes((item["name"] + "synthetic-asset").encode())
                files.append({"path": str(path.relative_to(self.snapshot)), "sha256": self.sha(path)})
            inventory.append({"name": item["name"], "description_zh": "固定验证图 " + item["name"], "files": files})
        inventory.append({"name": "11_training_losses_lama", "description_zh": "private-loss", "files": [{"path": "training-log-that-must-not-open.png", "sha256": "not-read"}]})
        run = {"seed": SEEDS[0], "metrics_sha256": self.complete["metrics_sha256"], "validation_metrics": copy.deepcopy(validation), "full_steps": None}
        self.report = {"selected_group": ["smoke", 1000], "selected_figure_seed": SEEDS[0], "protocol_sha256": self.ph,
                       "validation_only": True, "final_test_locked": True, "location_6_locked": True, "new_physical_test_images_opened": 0,
                       "methods": {"lama": {"runs": [run]}, "flow": {"runs": []}}, "figures": inventory,
                       "evaluation_records": [{"method": "lama", "seed": SEEDS[0], "metrics_sha256": self.complete["metrics_sha256"],
                                               "group": ["smoke", 1000], "path": str(self.evaluation.relative_to(self.root))}]}
        (self.snapshot / "renderer_source.py").write_text("synthetic renderer fixture")
        self.refresh_snapshot()

    def write(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

    def sha(self, path):
        return digest(path.read_bytes())

    def refresh_snapshot(self):
        self.write(self.snapshot / "metrics.json", self.report)
        self.repro = {"created_utc": "fixture-time", "protocol_sha256": self.ph, "data_seal_sha256": "seal", "figures": self.report["figures"],
                      "renderer_sha256": self.sha(self.snapshot / "renderer_source.py"), "figure_manifest_sha256": self.sha(self.root / "cases.json"),
                      "figure_geometry_audit_sha256": self.sha(self.root / "geometry.json"), "report_files": {"metrics.json": self.sha(self.snapshot / "metrics.json")}}
        self.write(self.snapshot / "reproducibility.json", self.repro)
        self.visual = {"passed": True, "snapshot": "reports/fixed-snapshot", "reproducibility_sha256": self.sha(self.snapshot / "reproducibility.json"), "new_physical_test_images_opened": 0}
        self.write(self.visual_path, self.visual)

    def prepare(self, audits=None):
        return prepare_site(self.snapshot, self.protocol_path, self.visual_path, [self.stage_audit] if audits is None else audits, project_root=self.root)

    def test_complete_audited_stage_keeps_missing_methods_and_fixed_gallery(self):
        data, assets = self.prepare()
        self.assertEqual(data["selected_group"], ["smoke", 1000])
        self.assertEqual(data["methods"]["lama"]["metrics"]["event_brier"]["mean"], .314159)
        self.assertIsNone(data["methods"]["flow"]["metrics"]["event_brier"]["mean"])
        self.assertEqual([f["name"] for f in data["figures"]], [f["name"] for f in figure_order(self.manifest)])
        self.assertEqual(len(assets), 3 * len(data["figures"]))
        self.assertFalse(data["paper_main_or_superiority_authorized"])

    def test_figure_radii_follow_actual_panels_instead_of_full_registration(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["reference_categories"]["narrow_bottleneck"]["cases"] = [dict(self.case, bottleneck_failed_radius_cells=20)]
        manifest["reference_categories"]["unreachable"]["cases"] = [self.case]
        manifest["reference_categories"]["multiple_alternatives"]["cases"] = [dict(self.case, witness_radius_cells=0)]
        manifest["multi_radius_cases"] = [self.case]
        radii = {entry["name"]: entry["display_radii_cells"] for entry in figure_order(manifest)}
        for name in ("01_comparison_00", "02_multiple_worlds", "08_unreachable_00"):
            self.assertEqual(radii[name], [10])
        self.assertEqual(radii["07_bottleneck_00"], [20])
        self.assertEqual(radii["09_alternatives_00"], [0])
        self.assertEqual(radii["09_alternatives_reference_routes_00"], [0])
        self.assertEqual(radii["10_radius_00"], [0, 10, 20])
        data, _ = self.prepare()
        page = render_html(data, "assets/formal/test")
        self.assertIn("query=obs_2:q001 · r=10 格", page)
        self.assertNotIn("r=[0, 10, 20]", page)

    def test_mobile_header_and_ratio_units_are_explicit(self):
        data, _ = self.prepare()
        page = render_html(data, "assets/formal/test")
        self.assertIn("误判率、覆盖率使用0–1比例，0.13即13%", page)
        self.assertIn(".brand{white-space:nowrap;", CSS)
        mobile = CSS.split("@media(max-width:650px)")[1]
        self.assertIn("header{flex-direction:column;", mobile)
        self.assertIn("nav{width:100%;", mobile)

    def test_unaudited_numbers_and_composite_images_are_withheld_without_zero_fill(self):
        data, assets = self.prepare(audits=[])
        self.assertIsNone(data["methods"]["lama"]["metrics"]["event_brier"]["mean"])
        self.assertEqual(data["methods"]["lama"]["runs"], [])
        self.assertEqual(assets, [])
        self.assertEqual(data["figures"], [])
        self.assertNotIn("0.314159", json.dumps(data))
        self.assertNotIn("0.3142", render_html(data, "assets/formal/test"))

    def test_failed_audit_is_not_result_authorization(self):
        self.audit["passed"] = False; self.write(self.stage_audit, self.audit)
        data, assets = self.prepare()
        self.assertEqual(data["methods"]["lama"]["verified_seeds"], [])
        self.assertEqual(assets, [])

    def test_earlier_stage_cannot_fill_the_current_comparison(self):
        self.report["evaluation_records"][0]["group"] = ["untrained", 0]
        self.refresh_snapshot()
        with self.assertRaisesRegex(ValueError, "mix stages"):
            self.prepare()

    def test_replaced_figure_manifest_is_rejected_even_with_new_report_and_visual_hashes(self):
        self.manifest["general_cases"] = list(reversed([self.case, dict(self.case, global_id="favorable-new")]))
        self.write(self.root / "cases.json", self.manifest)
        self.refresh_snapshot()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepare()

    def test_visual_receipt_must_bind_exact_reproducibility_content(self):
        self.visual.pop("reproducibility_sha256"); self.write(self.visual_path, self.visual)
        with self.assertRaisesRegex(ValueError, "visual receipt"):
            self.prepare()

    def test_changed_map_asset_is_rejected(self):
        path = self.snapshot / self.report["figures"][0]["files"][0]["path"]
        path.write_bytes(b"different model image")
        with self.assertRaisesRegex(ValueError, "asset hash"):
            self.prepare()

    def test_training_loss_assets_are_never_opened_or_published(self):
        data, assets = self.prepare()
        self.assertFalse(any("training" in str(a["source"]) for a in assets))
        self.assertFalse(any(f["name"].startswith("11_") for f in data["figures"]))
        self.assertNotIn("private-loss", render_html(data, "assets/formal/test"))

    def test_complete_fixed_gallery_cannot_be_reduced_to_selected_pretty_cases(self):
        self.report["figures"].pop(0); self.refresh_snapshot()
        with self.assertRaisesRegex(ValueError, "gallery is incomplete"):
            self.prepare()

    def test_site_build_preserves_homepage_and_refuses_existing_version_assets(self):
        data, assets = self.prepare()
        site = self.root / "site"; site.mkdir()
        (site / "index.html").write_bytes(b"original homepage")
        receipt = write_site(data, assets, site)
        self.assertEqual((site / "index.html").read_bytes(), b"original homepage")
        self.assertEqual(self.sha(site / "formal.html"), receipt["page_sha256"])
        self.assertFalse(receipt["website_published"])
        self.assertNotIn("<script", (site / "formal.html").read_text())
        self.assertIn("横向", (site / "formal.html").read_text())
        self.assertIn("圆盘", (site / "formal.html").read_text())
        with self.assertRaisesRegex(ValueError, "overwrite"):
            write_site(data, assets, site)

    def test_report_scope_and_escaped_artifact_paths_are_rejected(self):
        self.report["validation_only"] = False; self.refresh_snapshot()
        with self.assertRaisesRegex(ValueError, "locked validation"):
            self.prepare()
        self.report["validation_only"] = True
        self.report["figures"][0]["files"][0]["path"] = "../../../../outside.png"; self.refresh_snapshot()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
