import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pathrel.formal_metrics import exact_world_events
from scripts.evaluate_flatlands_formal import evaluate_split, load_registered_models, main, update_source_macro, training_cost_receipt


class FormalEvaluatorTests(unittest.TestCase):
    def sample(self, gid, has_query=True):
        valid = np.ones((7, 7), dtype=bool); valid[0] = False
        observed = np.zeros_like(valid); observed[3, 1] = True
        hidden = valid & ~observed
        target = valid.copy()
        starts = np.array([[3, 1]]) if has_query else np.empty((0, 2), dtype=np.int64)
        goals = np.array([[3, 5]]) if has_query else np.empty((0, 2), dtype=np.int64)
        return SimpleNamespace(row={"global_id": gid, "parent_group": gid, "source_dataset": "synthetic"},
                               observation=np.stack([observed, np.zeros_like(valid), hidden]).astype(np.float32),
                               valid=valid, hidden=hidden, target=target, starts=starts, goals=goals,
                               candidate_indices=np.array([3]) if has_query else np.empty(0, dtype=np.int64),
                               targets=exact_world_events(target[None], starts, goals)[0])

    def fake_prediction(self, _models, _method, sample, _seed, **unused):
        worlds = np.repeat(sample.valid[None], 4, axis=0)
        events = exact_world_events(worlds, sample.starts, sample.goals)
        return {"worlds": worlds, "continuous_score": worlds.astype(float), "world_events": events,
                "continuous_score_semantics": "synthetic completion score", "event_scores": events.mean(0),
                "event_score_semantics": "raw_world_event_frequency", "sampling_seed": 3, "input_key_sha256": "f" * 64,
                "efficiency": {"generation_seconds": .1, "full_update_seconds": .2, "actual_batch_forward_calls": 1,
                               "actual_model_input_examples": 1, "peak_allocated_bytes": None, "peak_reserved_bytes": None}}

    def test_zero_query_case_kept_for_maps_and_label_free_artifacts(self):
        samples = [self.sample("a"), self.sample("b", False)]
        with tempfile.TemporaryDirectory() as root, patch("scripts.evaluate_flatlands_formal.sample_models", self.fake_prediction):
            folder = Path(root) / "validation"
            report, flat = evaluate_split({}, "conpath", samples, 3, folder)
            self.assertEqual(report["map"]["cases"], 2)
            self.assertEqual(report["by_source"]["synthetic"]["map"]["cases"], 2)
            self.assertEqual(report["equal_source_macro"]["event_raw"]["event_brier"]["mean"], 0.)
            self.assertEqual(report["event_raw"]["events"], 3)
            self.assertEqual(report["event_raw"]["scene_weighted"]["overall"]["event_brier"], 0.)
            with np.load(folder / "predictions/a.npz", allow_pickle=False) as artifact:
                self.assertFalse({"target", "targets", "reference"} & set(artifact.files))
                self.assertEqual(artifact["worlds"].shape[0], 4)
            self.assertEqual(len(flat["query_keys"]), 3)
            self.assertEqual(report["map"]["collapse_diagnostic"]["generated_nearly_all_free_fraction"], 1.)
            self.assertEqual(report["map"]["collapse_diagnostic"]["reference_nearly_all_free_case_fraction"], 1.)
            with self.assertRaises(FileExistsError):
                evaluate_split({}, "conpath", samples, 3, folder)

    def test_random_models_use_frozen_member_initializations(self):
        seeds = [21, 22, 23, 24]
        protocol = {"member_seeds": {"lama": {"20260831": seeds}}, "lama_member_seeds": {"20260831": seeds}}
        def factory(*unused):
            return {"model": torch.nn.Linear(1, 1), "discriminator": torch.nn.Linear(1, 1)}
        with patch("scripts.evaluate_flatlands_formal.make_models", factory):
            a, provenance = load_registered_models("lama", 20260831, [], protocol, "a" * 64, "cpu")
            b, _ = load_registered_models("lama", 20260831, [], protocol, "a" * 64, "cpu")
        self.assertEqual([p["initialization_seed"] for p in provenance], seeds)
        for left, right in zip(a, b):
            torch.testing.assert_close(left["model"].weight, right["model"].weight)
            self.assertNotIn("discriminator", left)
        self.assertGreater(len({m["model"].weight.item() for m in a}), 1)

    def test_main_serializes_both_splits_and_calibration_provenance(self):
        provenance = [{"member_index": 0, "initialization_seed": 4, "checkpoint": None, "completed_steps": 0}]
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new_evaluation"
            argv = ["evaluate", "--protocol-dir", root, "--data-dir", root, "--output-dir", str(output),
                    "--method", "conpath", "--seed", "20260831", "--stage", "untrained", "--device", "cpu"]
            with patch("sys.argv", argv), patch("scripts.evaluate_flatlands_formal.verify_protocol", return_value=({}, "a" * 64)), \
                 patch("scripts.evaluate_flatlands_formal.load_registered_models", return_value=({}, provenance)), \
                 patch("scripts.evaluate_flatlands_formal.load_formal_data", side_effect=lambda _, split: [self.sample(split)]), \
                 patch("scripts.evaluate_flatlands_formal.sample_models", self.fake_prediction):
                main()
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(set(report["summary"]), {"calibration", "validation"})
            self.assertTrue(report["observed_and_support_constraints_passed"])
            self.assertFalse(report["main_table_eligible"])
            self.assertEqual(report["summary"]["validation"]["event_brier"], 0.)
            fitted = json.loads((output / "calibration_platt.json").read_text())
            self.assertEqual(fitted["fit_split"], "calibration")
            self.assertEqual(fitted["prediction_model_identity"][0]["completed_steps"], 0)
            self.assertTrue((output / "complete.json").exists())
            self.assertEqual(report["splits"]["validation"]["equal_source_macro"]["event_calibrated_diagnostic"]["event_brier"]["contributing_sources"], 1)

    def test_equal_source_macro_does_not_zero_fill_undefined_risk(self):
        keys = ("event_brier", "event_nll", "event_ece", "false_safe_at_confidence", "coverage_at_confidence")
        a = dict.fromkeys(keys, .2); a["false_safe_at_confidence"] = None
        b = dict.fromkeys(keys, .6)
        report = {"by_source": {"small": {"event_raw": {"scene_weighted": {"overall": a}}, "map": None},
                                "large": {"event_raw": {"scene_weighted": {"overall": b}}, "map": None}}}
        update_source_macro(report)
        macro = report["equal_source_macro"]["event_raw"]
        self.assertEqual(macro["event_brier"]["mean"], .4)
        self.assertEqual(macro["false_safe_at_confidence"]["mean"], .6)
        self.assertEqual(macro["false_safe_at_confidence"]["contributing_sources"], 1)

    def test_training_cost_uses_all_actual_sessions_not_early_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); (root / "sessions").mkdir()
            (root / "time_summary.json").write_text(json.dumps({"training_wall_seconds": 90., "wall_time_is_lower_bound": True, "sessions_without_final_receipt": ["b"]}))
            (root / "sessions/a.finished.json").write_text(json.dumps({"finished_utc": "2026-09-09T12:00:00Z", "peak_allocated_bytes": 100, "peak_reserved_bytes": 120}))
            (root / "sessions/b.heartbeat.json").write_text(json.dumps({"peak_allocated_bytes": 80, "peak_reserved_bytes": 150}))
            receipt = training_cost_receipt(root, {"training_wall_seconds": 10., "optimizer_update_seconds": 8., "peak_allocated_bytes": 40, "peak_reserved_bytes": 60})
            self.assertEqual(receipt["training_seconds"], 90.)
            self.assertEqual(receipt["checkpoint_snapshot_cost"]["training_wall_seconds"], 10.)
            self.assertTrue(receipt["training_wall_time_is_lower_bound"])
            self.assertEqual(receipt["training_peak_allocated_bytes"], 100)
            self.assertEqual(receipt["training_peak_reserved_bytes"], 150)
            self.assertEqual(len(receipt["training_resource_receipts"]), 2)


if __name__ == "__main__":
    unittest.main()
