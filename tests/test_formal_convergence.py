from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from pathrel.formal_metrics import exact_world_events
from scripts.render_flatlands_formal_convergence import (checked_worlds, completeness, frozen_cases,
                                                        main, match_query, stage_records, render_case, configure_plotting,
                                                        checked_prediction, output_contract)


class FormalConvergenceTests(unittest.TestCase):
    def case(self):
        return {"global_id": "obs_000001", "parent_group": "synthetic:a", "query_id": "obs_000001:q007",
                "candidate_index": 7, "start_rc": [4, 1], "goal_rc": [4, 7], "reference_events": [True, False, False],
                "no_eligible_query": False, "validation_only": True, "candidate_split": "validation", "physical_archive_split": "train"}

    def sample(self):
        valid = np.ones((9, 9), dtype=bool); valid[0] = False
        free = np.zeros_like(valid); free[4, 1] = True
        return SimpleNamespace(row={"global_id": "obs_000001", "parent_group": "synthetic:a"},
                               valid=valid, hidden=valid & ~free, target=valid.copy(),
                               observation=np.stack([free, np.zeros_like(free), valid & ~free]).astype(float),
                               starts=np.array([[4, 1]]), goals=np.array([[4, 7]]), candidate_indices=np.array([7]),
                               targets=np.array([[True, False, False]]))

    def saved(self, sample):
        worlds = np.repeat(sample.valid[None], 4, axis=0)
        worlds[0, 2, 2] = False
        worlds[1] = sample.observation[0] > .5
        worlds[3, :, 4] = False
        events = exact_world_events(worlds, sample.starts, sample.goals)
        return {"worlds": worlds, "world_events": events, "event_scores": events.mean(0)}

    def test_saved_worlds_are_never_sorted_by_success_or_score(self):
        sample = self.sample(); saved = self.saved(sample)
        original = saved["worlds"].copy()
        checked = checked_worlds(saved, sample, self.case())
        np.testing.assert_array_equal(checked["worlds"], original)
        self.assertEqual(checked["events"][:, 0].tolist(), [True, False, True, False])
        self.assertEqual(len(set(checked["world_order_sha256"])), 4)

    def test_query_identity_and_geometry_must_match(self):
        sample = self.sample(); case = self.case()
        self.assertEqual(match_query(sample, case), 0)
        for bad in ({**case, "candidate_index": 8}, {**case, "goal_rc": [4, 6]}, {**case, "query_id": "wrong"}):
            with self.assertRaises(ValueError):
                match_query(sample, bad)

    def test_saved_event_vote_tampering_is_rejected(self):
        sample = self.sample(); saved = self.saved(sample)
        saved["event_scores"][0, 0] = 1.
        with self.assertRaises(ValueError):
            checked_worlds(saved, sample, self.case())

    def test_registered_map_controls_keep_four_actual_worlds_in_original_order(self):
        sample = self.sample(); saved = self.saved(sample)
        for method in ("conpath", "conpath_original", "independent", "no_reach", "lama", "flow"):
            with self.subTest(method=method):
                checked = checked_prediction(saved, sample, self.case(), method)
                self.assertEqual(checked["actual_world_count"], 4)
                self.assertEqual(checked["event_score_semantics"], "raw_world_event_frequency")
                np.testing.assert_array_equal(checked["worlds"], saved["worlds"])
                self.assertEqual(len(output_contract(method)["column_layout"]), 6)

    def test_deterministic_control_has_one_world_and_cannot_be_replicated_to_four(self):
        sample = self.sample(); four = self.saved(sample)
        single = {"worlds": four["worlds"][:1], "world_events": four["world_events"][:1], "event_scores": four["world_events"][:1].mean(0)}
        checked = checked_prediction(single, sample, self.case(), "deterministic")
        self.assertEqual(checked["actual_world_count"], 1)
        self.assertEqual(len(checked["world_order_sha256"]), 1)
        self.assertEqual(output_contract("deterministic")["column_layout"], ["observed", "reference", "world_0"])
        with self.assertRaises(ValueError):
            checked_prediction(four, sample, self.case(), "deterministic")

    def test_direct_query_retains_continuous_nonmonotone_scores_without_worlds(self):
        sample = self.sample(); scores = np.array([[.731234, .891234, .221234]])
        checked = checked_prediction({"event_scores": scores}, sample, self.case(), "direct_query")
        np.testing.assert_array_equal(checked["event_scores"], scores[0])
        self.assertEqual(checked["reference_events"].tolist(), [True, False, False])
        for field in ("worlds", "events", "world_order_sha256", "actual_world_count"):
            self.assertIsNone(checked[field])
        self.assertEqual(checked["event_score_semantics"], "direct_query_sigmoid")
        for field in ("worlds", "world_events", "continuous_score"):
            with self.assertRaisesRegex(ValueError, "invent"):
                checked_prediction({"event_scores": scores, field: np.zeros((1, 9, 9))}, sample, self.case(), "direct_query")
        for bad in (np.array([[np.nan, .2, .3]]), np.array([[.1, 1.1, .3]]), np.ones((2, 3))):
            with self.assertRaises(ValueError):
                checked_prediction({"event_scores": bad}, sample, self.case(), "direct_query")

    def test_direct_query_no_eligible_query_has_no_fabricated_score(self):
        sample = self.sample()
        sample.candidate_indices = np.empty(0, dtype=int)
        sample.starts = sample.goals = np.empty((0, 2), dtype=int)
        sample.targets = np.empty((0, 3), dtype=bool)
        case = {**self.case(), "no_eligible_query": True, "candidate_index": None, "query_id": None}
        checked = checked_prediction({"event_scores": np.empty((0, 3))}, sample, case, "direct_query")
        self.assertIsNone(checked["query_array_index"])
        self.assertIsNone(checked["event_scores"])
        self.assertIsNone(checked["reference_events"])

    def test_missing_stage_is_not_complete_or_visually_approved(self):
        result = completeness({"untrained": {}, "smoke": {}, "pilot": None}, 16, 16)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_stages"], ["pilot"])
        self.assertIsNone(result["visual_gate_passed"])
        complete = completeness(dict.fromkeys(("untrained", "smoke", "pilot"), {}), 16, 16)
        self.assertTrue(complete["complete"])
        self.assertIsNone(complete["visual_gate_passed"])

    def test_deduplication_keeps_radius_and_category_membership(self):
        case = self.case()
        manifest = {"general_cases": [case], "multi_radius_cases": [case],
                    "reference_categories": {"unreachable": {"cases": []}, "narrow_bottleneck": {"cases": [{**case, "bottleneck_failed_radius_cells": 20}]},
                                             "multiple_alternatives": {"cases": []}}}
        result = frozen_cases(manifest)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["groups"], ["general", "narrow_bottleneck", "multi_radius"])
        self.assertEqual(result[0]["radii_to_render"], [0, 10, 20])

    def test_stage_selection_cannot_use_wrong_step_or_better_score(self):
        records = [{"method": "flow", "seed": 20260831, "group": ("smoke", 1000), "score": .9},
                   {"method": "flow", "seed": 20260831, "group": ("smoke", 5000), "score": 0.}]
        result = stage_records(records, "flow", 20260831)
        self.assertEqual(result["smoke"]["score"], .9)
        self.assertIsNone(result["pilot"])

    def test_nonempty_output_rejected_before_reading_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); (root / "keep.txt").write_text("existing")
            for method in ("lama", "flow", "conpath", "conpath_original", "independent", "no_reach", "deterministic", "direct_query"):
                with self.subTest(method=method), patch("sys.argv", ["render", "--method", method, "--protocol", "/missing/protocol.json", "--output-dir", str(root)]):
                    with self.assertRaisesRegex(ValueError, "nonempty"):
                        main()
            self.assertEqual((root / "keep.txt").read_text(), "existing")

    def test_three_by_six_layout_preserves_all_world_labels_without_writing_images(self):
        configure_plotting()
        sample, case = self.sample(), self.case()
        saved = self.saved(sample)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); selected = {}
            for stage in ("untrained", "smoke", "pilot"):
                folder = root / stage; (folder / "validation/predictions").mkdir(parents=True)
                np.savez_compressed(folder / "validation/predictions/obs_000001.npz", **saved)
                selected[stage] = {"path": folder, "metrics_sha256": stage,
                                   "report": {"checkpoint_provenance": [{"member_index": i} for i in range(4)]}}
            inventory = []
            def fake_save(fig, folder, name, items, description):
                self.assertEqual(len(fig.axes), 18)
                titles = [ax.title.get_text() for ax in fig.axes]
                self.assertEqual(sum("本组可达 2/4" in t for t in titles), 12)
                items.append({"name": name}); plt.close(fig)
            with patch("scripts.render_flatlands_formal_convergence.ROOT", root), \
                 patch("scripts.render_flatlands_formal_convergence.load_saved_prediction", return_value=saved), \
                 patch("scripts.render_flatlands_formal_convergence.save_figure", fake_save):
                result = render_case({"case": case, "groups": ["general"], "radii_to_render": [0]}, sample, selected,
                                     "lama", 20260831, root, inventory)
            self.assertTrue(inventory[0]["layout_audit"]["passed"])
            self.assertGreaterEqual(inventory[0]["layout_audit"]["minimum_row_title_gap_canvas_pixels"], 5.)
            self.assertEqual(result["radii_rendered_cells"], [0])

    def test_single_map_and_direct_score_layouts_use_three_columns_and_bound_sources(self):
        configure_plotting(); sample, case = self.sample(), self.case()
        original = self.saved(sample)
        for method in ("deterministic", "direct_query"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                saved = {"event_scores": np.array([[.731234, .891234, .221234]])} if method == "direct_query" else {
                    "worlds": original["worlds"][:1], "world_events": original["world_events"][:1], "event_scores": original["world_events"][:1].mean(0)}
                selected = {}
                for stage in ("untrained", "smoke", "pilot"):
                    folder = root / stage; (folder / "validation/predictions").mkdir(parents=True)
                    np.savez_compressed(folder / "validation/predictions/obs_000001.npz", **saved)
                    selected[stage] = {"path": folder, "metrics_sha256": stage,
                                       "report": {"checkpoint_provenance": [{"member_index": 0}]}}
                inventory = []
                def capture(fig, folder, name, items, description):
                    self.assertEqual(len(fig.axes), 9)
                    titles = [ax.title.get_text() for ax in fig.axes]
                    if method == "direct_query":
                        self.assertEqual(sum(any(t.get_text() == "0.731234" for t in ax.texts) for ax in fig.axes), 3)
                        self.assertFalse(any("本组可达" in title for title in titles))
                        self.assertEqual(sum(any("地图输出：N/A" in t.get_text() for t in ax.texts) for ax in fig.axes), 3)
                    else:
                        self.assertEqual(sum("唯一确定性地图" in title and "本组可达 1/1" in title for title in titles), 3)
                        self.assertFalse(any("样本 2/4" in title for title in titles))
                    items.append({"name": name}); plt.close(fig)
                with patch("scripts.render_flatlands_formal_convergence.ROOT", root), \
                     patch("scripts.render_flatlands_formal_convergence.load_saved_prediction", return_value=saved), \
                     patch("scripts.render_flatlands_formal_convergence.save_figure", capture):
                    result = render_case({"case": case, "groups": ["general"], "radii_to_render": [0]}, sample, selected,
                                         method, 20260831, root, inventory)
                self.assertTrue(inventory[0]["layout_audit"]["passed"])
                self.assertEqual(len(inventory[0]["column_layout"]), 3)
                for stage, source in result["stage_sources"].items():
                    np.testing.assert_array_equal(source["event_scores_by_radius"], saved["event_scores"][0])
                    self.assertEqual(source["reference_events_by_radius"], [True, False, False])
                    self.assertEqual(source["metrics_sha256"], stage)
                    self.assertEqual(len(source["prediction_sha256"]), 64)
                    if method == "direct_query":
                        self.assertIsNone(source["events_by_world_and_radius"])
                        self.assertIsNone(source["world_order_sha256"])
                    else:
                        self.assertEqual(len(source["events_by_world_and_radius"]), 1)
                        self.assertEqual(len(source["world_order_sha256"]), 1)


if __name__ == "__main__":
    unittest.main()
