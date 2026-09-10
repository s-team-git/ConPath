import json
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from pathrel.formal_data import sha256
from scripts.render_flatlands_formal_report import (SEEDS, discover_evaluations, group_key,
                                                    paper_eligibility, risk_curve_points, summarize_group, atomic_bytes,
                                                    metric_scope_annotation, read_loss_journal, trailing_loss_mean,
                                                    runtime_snapshot, runtime_markdown, validation_summary,
                                                    summary_csv_rows, render_training_losses, configure_plotting,
                                                    frozen_figure_artifacts, validate_evaluation_metadata, loss_durability_snapshot,
                                                    render_metric_figures, budget_annotation, load_saved_prediction)


class FormalReportTests(unittest.TestCase):
    def record(self, seed=20260831, stage="pilot", value=.2):
        report = {"stage": stage, "method": "flow", "outer_seed": seed,
                  "checkpoint_provenance": [{"completed_steps": 10000 if stage == "formal" else 5000}],
                  "protocol_sha256": "a" * 64, "new_physical_test_images_opened": 0,
                  "final_test_locked": True, "location_6_locked": True,
                  "observed_and_support_constraints_passed": True,
                  "splits": {"validation": {"cases": [{"efficiency": {"actual_world_count": 4}}]}},
                  "summary": {"validation": {"map_brier": value, "map_nll": value,
                               "event_brier": value, "event_nll": value, "event_ece": value,
                               "false_safe_at_0_8": None, "coverage_at_0_8": 0.}}}
        return {"method": "flow", "seed": seed, "metrics_sha256": str(seed), "report": report, "group": (stage, 5000)}

    def test_missing_methods_have_no_fabricated_zeros(self):
        with tempfile.TemporaryDirectory() as root:
            summary = summarize_group([self.record()], ["lama", "flow"], result_root=root, protocol_hash="a" * 64)
        self.assertIsNone(summary["lama"]["metrics"]["event_brier"]["mean"])
        self.assertEqual(summary["lama"]["metrics"]["event_brier"]["n"], 0)
        self.assertEqual(summary["flow"]["metrics"]["event_brier"]["mean"], .2)
        self.assertIsNone(summary["flow"]["metrics"]["event_brier"]["sd"])
        self.assertIsNone(summary["flow"]["metrics"]["false_safe_at_0_8"]["mean"])
        self.assertFalse(summary["flow"]["paper_gate"]["eligible"])

    def test_three_pilots_never_qualify_for_main_table(self):
        records = [self.record(seed=s) for s in SEEDS]
        plan = {"protocol_sha256": "a" * 64, "pilot_gate_passed": True, "full_steps": 10000}
        gate = paper_eligibility(records, plan, None, "a" * 64, "b" * 64)
        self.assertFalse(gate["eligible"])
        self.assertIn("尚未完成三个指定随机种子的完整训练验证", gate["reasons"])

    def test_complete_formal_repeats_still_require_bound_final_audit(self):
        records = [self.record(seed=s, stage="formal") for s in SEEDS]
        plan = {"protocol_sha256": "a" * 64, "pilot_gate_passed": True, "full_steps": 10000}
        self.assertFalse(paper_eligibility(records, plan, {}, "a" * 64, "b" * 64)["eligible"])
        audit = dict.fromkeys(("passed", "test_lock_passed", "checkpoint_selection_passed", "implementation_checks_passed", "convergence_review_passed"), True)
        audit.update(protocol_sha256="a" * 64, fully_trained_outer_seeds=list(SEEDS),
                     evaluation_metrics_sha256=[str(s) for s in SEEDS], full_plan_sha256="b" * 64,
                     method="flow", main_table_eligible=True, new_raw_archive_or_test_images_opened=0)
        self.assertFalse(paper_eligibility(records, plan, audit, "a" * 64, "b" * 64)["eligible"])
        with tempfile.TemporaryDirectory() as root:
            attempt = Path(root) / "audit_attempts/final/flow/attempt_1"; attempt.mkdir(parents=True)
            verification = attempt / "verification.json"; verification.write_text(json.dumps(audit))
            audit.update(audit_attempt=str(attempt), verification=str(verification), verification_sha256=sha256(verification))
            self.assertTrue(paper_eligibility(records, plan, audit, "a" * 64, "b" * 64, result_root=root, method="flow")["eligible"])
            verification.write_text('{}')
            self.assertFalse(paper_eligibility(records, plan, audit, "a" * 64, "b" * 64, result_root=root, method="flow")["eligible"])

    def test_different_formal_selected_steps_share_one_group(self):
        a, b = self.record(stage="formal"), self.record(seed=20260901, stage="formal")
        a["report"]["checkpoint_provenance"][0]["completed_steps"] = 10000
        b["report"]["checkpoint_provenance"][0]["completed_steps"] = 12500
        self.assertEqual(group_key(a["report"]), group_key(b["report"]))
        a["report"]["checkpoint_provenance"][0]["completed_steps"] = 5000
        gate = paper_eligibility([a], {"protocol_sha256": "a" * 64, "pilot_gate_passed": True, "full_steps": 15000}, None, "a" * 64)
        self.assertTrue(any("10000" in reason for reason in gate["reasons"]))

    def test_stage_and_budget_are_not_silently_mixed(self):
        record = self.record()
        self.assertEqual(group_key(record["report"]), ("pilot", 5000))
        record["report"]["checkpoint_provenance"].append({"completed_steps": 1000})
        with self.assertRaises(ValueError):
            group_key(record["report"])

    def test_risk_curve_does_not_create_empty_acceptance_zero(self):
        points = risk_curve_points([{"coverage": 0., "false_safe": None}, {"coverage": .5, "false_safe": .2}, {"coverage": 1., "false_safe": .4}])
        np.testing.assert_allclose(points, [[.5, .2], [1., .4]])
        with self.assertRaises(ValueError):
            risk_curve_points([{"coverage": .8, "false_safe": .2}, {"coverage": .4, "false_safe": .2}])

    def test_only_hash_matching_complete_evaluations_are_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output = root / "stages/flow/20260831/pilot/attempt_0"; output.mkdir(parents=True)
            report = self.record()["report"]
            report["checkpoint_provenance"][0].update(member_index=0, initialization_seed=20260831, checkpoint="model.pt")
            case = report["splits"]["validation"]["cases"][0]
            case.update(global_id="obs_a", parent_group="parent", source="source", queries=2)
            case["efficiency"].update(actual_members=1, common_budget_K=4, stage_one_member_diagnostic=False)
            report["splits"]["validation"].update(efficiency={"case_count": 1}, event_raw={"scene_weighted": {"overall": {"events": 6}}})
            protocol = {"methods": {"flow": {"members": 1, "actual_K": 4}}, "member_seeds": {"flow": {"20260831": [20260831]}},
                        "scenes": {"validation": [{"global_id": "obs_a", "parent_group": "parent", "source": "source"}]}}
            (output / "metrics.json").write_text(json.dumps(report))
            complete = {"passed": True, "metrics_sha256": sha256(output / "metrics.json"), "method": "flow", "seed": 20260831, "stage": "pilot", "completed_steps": [5000]}
            (output / "complete.json").write_text(json.dumps(complete))
            records, rejected = discover_evaluations(root, "a" * 64, protocol)
            self.assertEqual(len(records), 1); self.assertEqual(len(rejected), 0)
            self.assertFalse(records[0]["independent_stage_audit"]["passed"])
            for change in (lambda r: r["checkpoint_provenance"][0].update(member_index=2),
                           lambda r: r["checkpoint_provenance"][0].update(initialization_seed=123),
                           lambda r: r["splits"]["validation"]["cases"].clear(),
                           lambda r: r["splits"]["validation"]["cases"][0]["efficiency"].update(actual_world_count=1)):
                invalid = copy.deepcopy(report); change(invalid)
                with self.assertRaises(ValueError):
                    validate_evaluation_metadata(invalid, complete, protocol)
            complete["metrics_sha256"] = "bad"
            (output / "complete.json").write_text(json.dumps(complete))
            records, rejected = discover_evaluations(root, "a" * 64, protocol)
            self.assertEqual(len(records), 0); self.assertEqual(len(rejected), 1)

    def test_top_level_refresh_preserves_csv_bytes_and_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "summary.csv"
            original = b"method,value\r\nflow,\r\n"
            atomic_bytes(path, original)
            self.assertEqual(path.read_bytes(), original)

    def test_scope_annotations_show_actual_seeds_and_scene_query_radius_counts(self):
        note = metric_scope_annotation({"parents": 191, "parents_with_queries": 190, "queries": 2460}, [20260831], map_metric=True)
        for value in ("191", "190", "2460", "0/10/20", "20260831", "不依赖半径", "validation-only"):
            self.assertIn(value, note)

    def test_loss_curves_keep_full_complete_prefix_and_fixed_100_step_mean(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "progress.jsonl"
            path.write_bytes(b''.join((json.dumps({"step": i, "losses": {"velocity_mse": float(i)}})+'\n').encode() for i in range(1, 121)) + b'{"step":121')
            records, receipt = read_loss_journal(path)
            self.assertEqual(receipt["complete_updates"], 120)
            self.assertGreater(receipt["uncommitted_trailing_bytes"], 0)
            x, y = trailing_loss_mean([r["step"] for r in records], [r["losses"]["velocity_mse"] for r in records])
            self.assertEqual(x[0], 100); self.assertEqual(x[-1], 120)
            self.assertEqual(len(y), 21); self.assertAlmostEqual(y[0], 50.5)

    def test_runtime_status_distinguishes_active_evaluation_from_missing_results(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); (root / "stage_status").mkdir()
            (root / "queue.json").write_text(json.dumps({"phase": "staged_only", "methods": ["flow"]}))
            (root / "stage_status/flow.json").write_text(json.dumps({"method": "flow", "seed": 20260831, "status": "running", "protocol_sha256": "a" * 64,
                                                                    "current_stage": "untrained", "current_action": "calibration_and_validation"}))
            snapshot = runtime_snapshot(root, "a" * 64)
            text = '\n'.join(runtime_markdown(snapshot))
            self.assertIn("校准集与验证集评估", text)
            self.assertIn("状态文件快照", text)
            self.assertIsNotNone(snapshot["queue"]["file_modified_utc"])
            (root / "full_runs/flow").mkdir(parents=True)
            (root / "full_runs/flow/status.json").write_text(json.dumps({"method": "flow", "protocol_sha256": "a" * 64,
                "status": "running", "current_action": "calibration_only", "current_seed": 20260901, "current_step": 15000}))
            updated = '\n'.join(runtime_markdown(runtime_snapshot(root, "a" * 64)))
            self.assertIn("阶段训练历史记录", updated)
            self.assertIn("完整训练状态", updated)
            self.assertIn("15000", updated)

    def test_changed_figure_and_geometry_cannot_rebind_each_other_outside_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = {"data_seal_sha256": "seal", "validation_only": True, "frozen_before_formal_model_training_and_inference": True}
            (root / "cases.json").write_text(json.dumps(manifest))
            geometry = {"passed": True, "figure_cases_sha256": sha256(root / "cases.json"), "data_seal_sha256": "seal"}
            (root / "geometry.json").write_text(json.dumps(geometry))
            protocol = {"data_seal_sha256": "seal", "figures": {"cases": "cases.json", "sha256": sha256(root / "cases.json"),
                        "geometry_verification": "geometry.json", "geometry_verification_sha256": sha256(root / "geometry.json")}}
            self.assertEqual(frozen_figure_artifacts(protocol, project_root=root)[0], manifest)
            manifest["favorable_new_case"] = True; (root / "cases.json").write_text(json.dumps(manifest))
            geometry["figure_cases_sha256"] = sha256(root / "cases.json"); (root / "geometry.json").write_text(json.dumps(geometry))
            with self.assertRaisesRegex(ValueError, "protocol"):
                frozen_figure_artifacts(protocol, project_root=root)

    def test_executed_loss_prefix_is_not_mislabeled_as_durable_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); (root / "sessions").mkdir()
            (root / "sessions/run1.heartbeat.json").write_text(json.dumps({"last_durable_step": 100, "last_complete_step": 125}))
            result = loss_durability_snapshot(root / "progress.jsonl", [{"step": 125, "session_id": "run1"}])
            self.assertEqual(result["last_durable_step_at_receipt"], 100)
            self.assertIn("replayed", result["meaning"])

    def test_raw_score_titles_do_not_call_direct_query_outputs_world_frequencies(self):
        configure_plotting()
        with tempfile.TemporaryDirectory() as root:
            grouped = summarize_group([], ["direct_query"], result_root=root, protocol_hash="a" * 64)
            captured = []
            def capture(fig, folder, name, inventory, description):
                captured.append((name, [ax.get_title() for ax in fig.axes], [t.get_text() for t in fig.texts])); plt.close(fig)
            with patch("scripts.render_flatlands_formal_report.save_figure", capture):
                render_metric_figures(grouped, {}, "diagnostic", SEEDS[0], Path(root), [], {"parents": 191, "parents_with_queries": 190, "queries": 2460})
            for name, titles, notes in captured:
                if name in ("04_event_reliability", "06_false_safe_coverage"):
                    self.assertEqual(titles[0], "原始事件分数")
                if name == "06_false_safe_coverage":
                    self.assertTrue(any("仅实际K4" in text and "直接查询" in text for text in notes))
                if name == "05_event_metrics":
                    self.assertTrue(any("不是置信区间" in text for text in notes))

    def test_one_saved_world_cannot_be_displayed_as_a_four_world_bundle(self):
        sample = SimpleNamespace(row={"global_id": "obs_a"}, candidate_indices=np.array([], dtype=int),
                                 starts=np.empty((0, 2), dtype=int), goals=np.empty((0, 2), dtype=int),
                                 valid=np.ones((5, 5), dtype=bool), hidden=np.zeros((5, 5), dtype=bool),
                                 observation=np.stack([np.ones((5, 5)), np.zeros((5, 5)), np.zeros((5, 5))]))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); (root / "validation/predictions").mkdir(parents=True)
            path = root / "validation/predictions/obs_a.npz"
            saved = dict(candidate_indices=sample.candidate_indices, starts=sample.starts, goals=sample.goals, radii_cells=[0, 10, 20],
                         worlds=np.ones((4, 5, 5)), world_events=np.empty((4, 0, 3), dtype=bool), event_scores=np.empty((0, 3)))
            np.savez_compressed(path, **saved)
            case = {"global_id": "obs_a", "predictions_sha256": sha256(path)}
            record = {"method": "flow", "path": root, "report": {"splits": {"validation": {"cases": [case]}}}}
            self.assertEqual(load_saved_prediction(record, sample)["worlds"].shape[0], 4)
            saved["worlds"] = saved["worlds"][:1]; np.savez_compressed(path, **saved)
            case["predictions_sha256"] = sha256(path)
            with self.assertRaisesRegex(ValueError, "actual binary K"):
                load_saved_prediction(record, sample)

    def test_validation_export_preserves_strata_and_distinguishes_nfe_forward_cost(self):
        validation = {"map": {"pooled": {"map_brier": .2}}, "event_raw": {"scene_weighted": {"by_radius_and_truth": {"10": {"unreachable": .4}}}},
                      "event_calibrated_diagnostic": {"pooled": {"by_truth": {"reachable": .3}}},
                      "by_source": {"source_a": {"map": {"scene_weighted": .4}}}, "equal_source_macro": {"raw": .5},
                      "cases": [{"efficiency": {"solver": "Heun", "steps_per_sample": 25, "actual_world_count": 4, "velocity_evaluations_per_sample": 50}} for _ in range(2)],
                      "efficiency": {"case_count": 2, "actual_batch_forward_calls": 100, "actual_model_input_examples": 800}}
        exported = validation_summary(validation)
        self.assertNotIn("cases", exported)
        for field in ("map", "event_raw", "event_calibrated_diagnostic", "by_source", "equal_source_macro"):
            self.assertEqual(exported[field], validation[field])
        cost = exported["efficiency"]
        self.assertEqual(cost["actual_batch_forward_calls_total"], 100)
        self.assertEqual(cost["actual_batch_forward_calls_mean_per_case"], 50)
        self.assertEqual(cost["actual_model_input_examples_mean_per_case"], 400)
        self.assertEqual(cost["velocity_evaluations_all_worlds_total"], 400)
        self.assertEqual(cost["velocity_evaluations_all_worlds_mean_per_case"], 200)
        self.assertEqual(cost["velocity_evaluations_mean_per_actual_world"], 50)
        self.assertNotIn("velocity_evaluations_all_worlds_total", validation["efficiency"])
        self.assertIsNone(validation_summary({"cases": [{"efficiency": {"velocity_evaluations_per_sample": None}}]})["efficiency"]["velocity_evaluations_all_worlds_total"])

    def test_csv_exports_violation_and_peak_counts_without_filling_missing_rows(self):
        record = self.record()
        record["report"]["summary"]["validation"].update(observed_evidence_violation_count=0, valid_support_violation_count=0)
        record["report"]["checkpoint_provenance"][0].update(training_peak_allocated_bytes=123, training_peak_reserved_bytes=456)
        record["report"]["splits"]["validation"]["efficiency"] = {"case_count": 1, "peak_allocated_bytes": 789, "peak_reserved_bytes": 999,
                                                                     "actual_batch_forward_calls": 50, "actual_model_input_examples": 400}
        with tempfile.TemporaryDirectory() as root:
            grouped = summarize_group([record], ["lama", "flow"], result_root=root, protocol_hash="a" * 64)
            rows = summary_csv_rows(grouped, [record], ("pilot", 5000))
        flow = next(row for row in rows if row["method"].startswith("FM") and row["seed"] == 20260831)
        lama = rows[0]
        self.assertEqual(flow["observed_evidence_violation_count"], 0)
        self.assertEqual(flow["training_peak_allocated_bytes_max_member_session"], 123)
        self.assertEqual(flow["evaluation_peak_allocated_bytes"], 789)
        self.assertEqual(flow["actual_batch_forward_calls_mean_per_case"], 50)
        for field in ("observed_evidence_violation_count", "evaluation_peak_allocated_bytes", "actual_batch_forward_calls_total"):
            self.assertIsNone(lama[field])

    def test_loss_renderer_keeps_all_members_and_archives_plotted_values_without_image_writes(self):
        configure_plotting()
        rendered = []
        def capture(fig, folder, name, inventory, description):
            rendered.append((name, [len(ax.lines) for ax in fig.axes]))
            plt.close(fig)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); figures = root / "report/figures"; figures.mkdir(parents=True)
            for method, members in (("lama", range(4)), ("flow", range(1))):
                for member in members:
                    journal = root / "runs" / method / "20260831" / f"member_{member}" / "progress.jsonl"
                    journal.parent.mkdir(parents=True)
                    keys = ("reconstruction", "adversarial", "feature_matching", "discriminator") if method == "lama" else ("velocity_mse",)
                    journal.write_text(''.join(json.dumps({"step": step, "losses": {key: float(step + member) for key in keys}}) + '\n' for step in range(1, 121)))
            with patch("scripts.render_flatlands_formal_report.save_figure", capture):
                sources = render_training_losses(root, figures, [], {"parents": 985})
            self.assertEqual(rendered[0][1], [8, 8, 8, 8])
            self.assertEqual(rendered[1][1], [2])
            self.assertEqual(len(sources), 5)
            for source in sources:
                path = figures.parent / source["archived_plot_values_path"]
                self.assertEqual(sha256(path), source["archived_plot_values_sha256"])
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["steps"], np.arange(1, 121))


if __name__ == "__main__":
    unittest.main()
