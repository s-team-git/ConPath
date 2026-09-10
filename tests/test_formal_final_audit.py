"""Synthetic CPU checks only: no real experiment, GPU or original data reads."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pathrel.formal_data import FormalSample, sha256
from scripts import audit_flatlands_formal_final as audit


class FinalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patch = mock.patch.object(audit, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def save(self, relative, value):
        target = self.root / relative
        audit.write_new(target, value)
        return target

    def session(self, token, finish=None, heartbeat=None, failure=None):
        self.save(f"run/sessions/{token}.started.json", dict(session_id=token))
        for kind, elapsed in (("finished", finish), ("heartbeat", heartbeat), ("failure", failure)):
            if elapsed is not None:
                key = "wall_seconds_this_session" if kind == "failure" else "elapsed_seconds"
                self.save(f"run/sessions/{token}.{kind}.json", {"session_id": token, key: elapsed})

    def test_global_minimum_tie_set_does_not_chain_near_ties(self):
        candidates = [dict(step=1000, event_brier=.1, map_nll=5.),
                      dict(step=5000, event_brier=.100009, map_nll=2.),
                      dict(step=7500, event_brier=.100018, map_nll=1.)]
        self.assertEqual(audit.choose_checkpoint(candidates)["step"], 5000)
        self.assertEqual(audit.choose_checkpoint(candidates[::-1])["step"], 5000)
        candidates[0]["map_nll"] = 2.
        self.assertEqual(audit.choose_checkpoint(candidates)["step"], 1000)

    def test_selection_rejects_nan_boolean_and_duplicate_steps(self):
        for value in (float("nan"), float("inf"), True, -.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.choose_checkpoint([dict(step=1000, event_brier=value, map_nll=.1)])
        row = dict(step=1000, event_brier=.1, map_nll=.2)
        with self.assertRaises(ValueError):
            audit.choose_checkpoint([row, row])

    def test_legal_failure_then_resume_counts_time_once_and_hashes_failure(self):
        self.session("failed", finish=12., heartbeat=10., failure=11.)
        self.session("resumed", finish=7., heartbeat=6.)
        cost, files = audit.session_cost(self.root / "run")
        self.assertEqual(cost, dict(training_wall_seconds=19., wall_time_is_lower_bound=False,
                                   sessions_without_final_receipt=[]))
        self.assertEqual(len(files), 7)
        self.assertIn("run/sessions/failed.failure.json", [f["path"] for f in files])

    def test_missing_finish_is_visible_lower_bound_including_start_only(self):
        self.session("a", heartbeat=3.)
        self.session("b")
        cost, _ = audit.session_cost(self.root / "run")
        self.assertEqual(cost["training_wall_seconds"], 3.)
        self.assertEqual(cost["sessions_without_final_receipt"], ["a", "b"])
        self.assertTrue(cost["wall_time_is_lower_bound"])

    def test_orphan_failure_or_nan_or_bool_heartbeat_are_rejected(self):
        self.session("a", finish=4.)
        orphan = self.save("run/sessions/orphan.failure.json", {"session_id": "orphan", "wall_seconds_this_session": 2.})
        with self.assertRaisesRegex(ValueError, "Orphan"):
            audit.session_cost(self.root / "run")
        orphan.unlink()
        heartbeat = self.root / "run/sessions/a.heartbeat.json"
        for value in (float("inf"), True, -1., 5.):
            heartbeat.write_text(json.dumps({"session_id": "a", "elapsed_seconds": value}))
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.session_cost(self.root / "run")

    def test_artifacts_reject_overwrite_and_project_escape(self):
        target = self.save("receipt.json", {"passed": False})
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            audit.write_new(target, {"passed": True})
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaises(ValueError):
            audit.path("../outside.json")
        with self.assertRaises(ValueError):
            audit.write_new(self.root / "invalid.json", {"bad": float("nan")})
        self.assertFalse((self.root / "invalid.json").exists())
        self.assertFalse(list(self.root.glob(".audit-*")))

    def sample(self, gid="obs_000001", queries=True):
        valid = np.ones((25, 25), bool); valid[:, -1] = False
        hidden = valid.copy(); hidden[12, 10] = False
        target = valid.copy()
        free = valid & ~hidden
        observation = np.stack([free, valid & ~hidden & ~free, hidden]).astype(np.float32)
        starts = np.array([[12, 10]], int) if queries else np.empty((0, 2), int)
        goals = np.array([[12, 15]], int) if queries else np.empty((0, 2), int)
        targets = audit.independent.explicit_disk_events(target[None], starts, goals)[0]
        return FormalSample(dict(global_id=gid, parent_group=gid, source_dataset="SyntheticFixture"),
                            observation, valid, target, hidden, starts, goals, targets,
                            np.array([8], int) if queries else np.empty(0, int))

    def test_cached_channels_known_truth_and_eligibility_are_independently_checked(self):
        sample = self.sample()
        audit.check_cached_inputs(sample)
        sample.observation[1, 12, 10] = 1
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit.check_cached_inputs(sample)
        sample = self.sample(); sample.target[12, 10] = False
        with self.assertRaisesRegex(ValueError, "contradiction"):
            audit.check_cached_inputs(sample)
        sample = self.sample(); sample.goals[0] = sample.starts[0]
        with self.assertRaisesRegex(ValueError, "eligibility"):
            audit.check_cached_inputs(sample)

    def test_candidate_scores_use_worlds_and_include_no_query_map(self):
        samples = [self.sample(), self.sample("obs_000002", queries=False)]
        cases, maps = [], []
        for sample in samples:
            worlds = np.repeat(sample.target[None], 4, axis=0)
            if len(sample.starts) == 0:
                worlds[:, sample.hidden] = False
            events = audit.independent.explicit_disk_events(worlds, sample.starts, sample.goals)
            file = self.root / "eval/calibration/predictions" / (sample.row["global_id"] + ".npz")
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, global_id=np.array(sample.row["global_id"]), candidate_indices=sample.candidate_indices,
                starts=sample.starts, goals=sample.goals, radii_cells=np.array([0, 10, 20]),
                worlds=worlds, world_events=events, event_scores=events.mean(0), continuous_score=worlds.astype(float))
            value = audit.independent.independent_map_case(sample, worlds, worlds.astype(float)); maps.append(value)
            cases.append(dict(global_id=sample.row["global_id"], predictions_sha256=sha256(file), map=value))
        report = audit.independent.independent_map_report(maps, [s.scene_key for s in samples])
        expected = dict(event_brier=0., map_brier=report["scene_weighted"]["sample_vote_cell_brier"],
                        map_nll=report["scene_weighted"]["sample_vote_cell_nll"], collapse_diagnostic=report["collapse_diagnostic"])
        metrics = dict(method="lama", splits={"calibration": {"cases": cases}}, summary={"calibration": expected})
        computed = audit.candidate_scores(self.root / "eval", "calibration", samples, metrics)
        self.assertEqual(computed["map_brier"], .5)
        self.assertEqual(computed["event_brier"], 0.)
        metrics["summary"]["calibration"] = {**expected, "event_brier": .125}
        with self.assertRaises(ValueError):
            audit.candidate_scores(self.root / "eval", "calibration", samples, metrics)

    def checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
        model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        recipe = dict(protocol_sha256="p", method="flow", seed=20260831, member=0)
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        run = self.save("run/run.json", dict(schema_version=1, recipe=recipe, recipe_sha256=digest))
        journal = b'{"step": 1,"update_seconds":0.5}\n'
        (run.parent / "progress.jsonl").write_bytes(journal)
        generator = torch.Generator(device="cpu")
        state = dict(schema_version=1, completed_steps=1, models={"model": model.state_dict()}, optimizers={"model": optimizer.state_dict()},
            schedulers={"model": dict(base_lrs=[.0001], pilot_steps=1, full_steps=None, minimum_ratio=.1, completed_steps=1)},
            model_training_modes={"model": {"": True}}, parameter_requires_grad={"model": {k: True for k, _ in model.named_parameters()}},
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                     torch_cuda=[torch.zeros(16, dtype=torch.uint8)], cuda_initialized=True),
            generators={"views": {"device": "cpu", "state": generator.get_state()},
                        "noise": {"device": "cuda", "state": torch.zeros(16, dtype=torch.uint8)}},
            sampler=dict(size=2, batch_size=1, permutation=torch.tensor([1, 0]), cursor=1, epoch=0,
                         generator_state=generator.get_state()), extra=dict(full_plan_sha256=None,
                            training_wall_seconds=1., optimizer_update_seconds=.5, peak_allocated_bytes=0, peak_reserved_bytes=0))
        envelope = dict(schema_version=1, recipe_sha256=digest, progress_index=1, journal_sha256=hashlib.sha256(journal).hexdigest(), state=state)
        file = self.root / "run/checkpoint.pt"; torch.save(envelope, file)
        member = dict(checkpoint="run/checkpoint.pt", checkpoint_sha256=sha256(file), member_index=0,
                      initialization_seed=101, completed_steps=1, run_receipt="run/run.json", recipe_sha256=digest, model_parameters=3,
                      checkpoint_snapshot_cost=dict(training_wall_seconds=1., optimizer_update_seconds=.5,
                                                    peak_allocated_bytes=0, peak_reserved_bytes=0))
        protocol = dict(member_seeds={"flow": {"20260831": [101]}}, partition={"train": 2},
                        methods={"flow": dict(batch_size=1, learning_rate=.0001, weight_decay=.01, betas=[.9, .999])}, staged={"pilot_steps": 1})
        return file, envelope, member, protocol

    def test_real_cpu_checkpoint_checks_full_state_and_journal_prefix(self):
        file, envelope, member, protocol = self.checkpoint()
        args = (member, "flow", 20260831, 0, 1, protocol, "p", "f")
        expected = audit.recovery_identity(envelope["state"]["sampler"], envelope["state"]["generators"]["views"]["state"])
        self.assertTrue(audit.check_checkpoint(*args, full_steps=3, expected_recovery=expected, deep=True)["tensor_finiteness_checked"])
        (file.parent / "progress.jsonl").write_text('{"step":1,"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "journal hash"):
            audit.check_checkpoint(*args, full_steps=3, expected_recovery=expected)

    def test_empty_optimizer_or_nonfinite_model_is_rejected(self):
        file, envelope, member, protocol = self.checkpoint()
        original = copy.deepcopy(envelope)
        expected = audit.recovery_identity(envelope["state"]["sampler"], envelope["state"]["generators"]["views"]["state"])
        for mutation in ("optimizer", "nonfinite"):
            envelope = copy.deepcopy(original)
            if mutation == "optimizer":
                envelope["state"]["optimizers"]["model"]["state"] = {}
            else:
                envelope["state"]["models"]["model"]["weight"].fill_(float("nan"))
            torch.save(envelope, file); member["checkpoint_sha256"] = sha256(file)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.check_checkpoint(member, "flow", 20260831, 0, 1, protocol, "p", "f", full_steps=3, expected_recovery=expected, deep=True)

    def test_valid_but_wrong_recovery_rng_and_finite_malformed_moments_are_rejected(self):
        file, original, member, protocol = self.checkpoint()
        expected = audit.recovery_identity(original["state"]["sampler"], original["state"]["generators"]["views"]["state"])
        for mutation in ("cursor", "view_rng", "sampler_rng", "permutation_dtype", "moment_shape", "gradient_flag", "schema", "update_cost"):
            envelope = copy.deepcopy(original); state = envelope["state"]
            receipt = copy.deepcopy(member)
            if mutation == "cursor":
                state["sampler"]["cursor"] = 0
            elif mutation == "view_rng":
                state["generators"]["views"]["state"] = torch.Generator().manual_seed(17).get_state()
            elif mutation == "sampler_rng":
                state["sampler"]["generator_state"] = torch.Generator().manual_seed(19).get_state()
            elif mutation == "permutation_dtype":
                state["sampler"]["permutation"] = state["sampler"]["permutation"].float()
            elif mutation == "moment_shape":
                first = next(iter(state["optimizers"]["model"]["state"].values()))
                first["exp_avg"] = torch.zeros(999)
            elif mutation == "gradient_flag":
                state["parameter_requires_grad"]["model"]["weight"] = False
            elif mutation == "schema":
                envelope["schema_version"] = 2
            else:
                state["extra"]["optimizer_update_seconds"] = .9
                receipt["checkpoint_snapshot_cost"]["optimizer_update_seconds"] = .9
            torch.save(envelope, file); receipt["checkpoint_sha256"] = sha256(file)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.check_checkpoint(receipt, "flow", 20260831, 0, 1, protocol, "p", "f",
                    full_steps=3, expected_recovery=expected, deep=True)

    def training_receipt(self):
        self.session("done", finish=12., heartbeat=10.)
        totals, files = audit.session_cost(self.root / "run")
        summary = self.save("run/time_summary.json", totals)
        reconciliation = self.save("run/time_reconciliation/current.json", dict(totals=totals, session_files=files,
            previous_summary_archives=[], current_time_summary="run/time_summary.json", current_time_summary_sha256=sha256(summary)))
        actual = dict(member=0, completed_steps=10000, progress_sha256="progress", time_summary_sha256=sha256(summary),
                      actual_training_cost=totals, peak_allocated_bytes=42, peak_reserved_bytes=84, session_files=files)
        member = dict(member=0, updates=10000, progress_sha256="progress", all_update_losses_finite=True,
            time_summary="run/time_summary.json", time_summary_sha256=sha256(summary), actual_training_time=totals,
            peak_allocated_bytes=42, peak_reserved_bytes=84,
            time_reconciliation="run/time_reconciliation/current.json", time_reconciliation_sha256=sha256(reconciliation))
        return dict(passed=True, errors=[], members=[member], training_seconds_sum_across_members=12., time_is_lower_bound=False), [actual]

    def test_training_reconciliation_refuses_stale_summary_and_omitted_failure(self):
        stored, records = self.training_receipt()
        audit.check_training_reconciliation(stored, records)
        tampered = copy.deepcopy(stored); tampered["training_seconds_sum_across_members"] = 10.
        with self.assertRaises(ValueError):
            audit.check_training_reconciliation(tampered, records)
        (self.root / "run/time_summary.json").write_text('{"training_wall_seconds": 8}')
        with self.assertRaisesRegex(ValueError, "summary changed"):
            audit.check_training_reconciliation(stored, records)

    def convergence_fixture(self, values=(.1, .1, .1), selected_step=10000, visual_good=True):
        training, records = self.training_receipt()
        candidates = [dict(step=step, metrics_sha256=str(step), event_brier=value, map_brier=value, collapse_diagnostic={"diagnostic": 1})
                      for step, value in zip((5000, 7500, 10000), values)]
        selection = dict(best_step=selected_step, selected_metrics_sha256=str(selected_step), short_selected_checkpoint=selected_step < 10000)
        trends = {}
        for key in ("map_brier", "event_brier"):
            changes = [(values[0]-values[1])/max(abs(values[0]), 1e-12),
                       (values[1]-values[2])/max(abs(values[1]), 1e-12),
                       (values[0]-values[2])/max(abs(values[0]), 1e-12)]
            trends[key] = dict(values=list(values), relative_reductions=changes, stable_below_two_percent=all(abs(v) < .02 for v in changes))
        numeric = all(t["stable_below_two_percent"] for t in trends.values())
        visual = self.save("visual.json", dict(method="flow", seed=20260831, protocol_sha256="p", full_plan_sha256="f",
            selected_calibration_metrics_sha256=str(selected_step), last_three_calibration_metrics_sha256=[str(c["step"]) for c in candidates],
            passed=visual_good, no_collapse=visual_good))
        receipt = dict(protocol_sha256="p", full_plan_sha256="f", method="flow", seed=20260831,
            full_steps=10000, selected_step=selected_step, last_three_calibration_steps=[c["step"] for c in candidates],
            last_three_calibration_metrics_sha256=[c["metrics_sha256"] for c in candidates],
            numerical_convergence_passed=numeric, trends=trends, collapse_diagnostics=[c["collapse_diagnostic"] for c in candidates],
            visual_review="visual.json", visual_review_sha256=sha256(visual), sufficient_training=numeric and visual_good and selected_step >= 10000,
            short_selected_checkpoint=selected_step < 10000, main_table_eligible=False, final_test_locked=True, training_audit=training)
        file = self.save("convergence/body.json", receipt)
        pointer = self.save("convergence.json", dict(receipt="convergence/body.json", sha256=sha256(file)))
        args = (selection, candidates, pointer, "flow", 20260831, {"staged": {"allowed_full_steps": [10000, 20000]}}, "p", "f", records)
        return args, receipt, file

    def test_convergence_still_improving_cannot_become_passed_by_boolean(self):
        args, receipt, file = self.convergence_fixture(values=(.12, .11, .1))
        self.assertFalse(audit.check_convergence(*args)["passed"])
        receipt["sufficient_training"] = True
        file.write_text(json.dumps(receipt))
        args[2].write_text(json.dumps(dict(receipt="convergence/body.json", sha256=sha256(file))))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            audit.check_convergence(*args)

    def test_early_selected_checkpoint_remains_diagnostic_despite_full_budget(self):
        args, _, _ = self.convergence_fixture(selected_step=5000)
        result = audit.check_convergence(*args)
        self.assertTrue(result["numerical_convergence_passed"])
        self.assertTrue(result["selected_checkpoint_is_short"])
        self.assertFalse(result["passed"])

    def test_full_visual_review_cannot_be_swapped_or_point_to_other_weights(self):
        args, receipt, file = self.convergence_fixture()
        self.assertTrue(audit.check_convergence(*args)["passed"])
        visual_file = self.root / "visual.json"
        visual = json.loads(visual_file.read_text()); visual["selected_calibration_metrics_sha256"] = "other"
        visual_file.write_text(json.dumps(visual))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit.check_convergence(*args)
        receipt["visual_review_sha256"] = sha256(visual_file); file.write_text(json.dumps(receipt))
        args[2].write_text(json.dumps(dict(receipt="convergence/body.json", sha256=sha256(file))))
        with self.assertRaisesRegex(ValueError, "different selected"):
            audit.check_convergence(*args)

    def test_pilot_visual_requires_actual_stage_hashes_and_complete_fixed_cases(self):
        cases = dict(general_cases=[dict(global_id="obs_000001", query_id="q1", case_category="general")],
                     multi_radius_cases=[], reference_categories={})
        self.save("cases.json", cases); self.save("render.json", {"fixture": True})
        protocol = {"figures": {"cases": "cases.json", "sha256": "figure"}}
        stages = ("untrained", "smoke", "pilot")
        numeric = {"stage_metrics": {s: {"sha256": s} for s in stages}}
        metrics = {s: {"splits": {"validation": {"cases": [{"global_id": "obs_000001", "predictions_sha256": s+"npz"}]}}} for s in stages}
        visual = dict(passed=True, figure_cases_sha256="figure", reviewed_stages=list(stages),
            pilot_quality_passed=True, pilot_no_persistent_collapse=True,
            stage_metrics_sha256={s: s for s in stages}, render_manifest=dict(path="render.json", sha256=sha256(self.root / "render.json")),
            case_reviews=[dict(stage=s, case_category="general", global_id="obs_000001", query_id="q1",
                reviewed=True, no_unexplained_collapse=True, persistent_collapse_detected=s != "pilot",
                predictions_sha256=s+"npz", notes_zh="仅合成测试：前两阶段允许已记录退化") for s in stages])
        with mock.patch.object(audit, "check_convergence_render"):
            audit.check_pilot_visual(visual, numeric, protocol, metrics)
            tampered = copy.deepcopy(visual); tampered["stage_metrics_sha256"]["pilot"] = "other"
            with self.assertRaisesRegex(ValueError, "other stage"):
                audit.check_pilot_visual(tampered, numeric, protocol, metrics)
            tampered = copy.deepcopy(visual); tampered["case_reviews"][-1]["persistent_collapse_detected"] = True
            with self.assertRaisesRegex(ValueError, "retained visually"):
                audit.check_pilot_visual(tampered, numeric, protocol, metrics)
            visual["case_reviews"].pop()
            with self.assertRaisesRegex(ValueError, "omitted"):
                audit.check_pilot_visual(visual, numeric, protocol, metrics)

    def render_fixture(self, method="flow"):
        actual_k = None if method == "direct_query" else 1 if method == "deterministic" else 4
        semantics = "direct_query_sigmoid" if actual_k is None else "raw_world_event_frequency"
        self.save("cases.json", dict(general_cases=[dict(global_id="obs_000001", query_id="q1", candidate_index=1,
                                                       reference_events=[False, False, False], case_category="general")],
                                    multi_radius_cases=[], reference_categories={}))
        protocol = dict(figures=dict(cases="cases.json", sha256="figure", geometry_verification_sha256="geometry"),
                        data_seal_sha256="data", source_hashes={})
        numeric = {"stage_metrics": {s: {"sha256": s} for s in ("untrained", "smoke", "pilot")}}
        metrics, sources = {}, {}
        for stage in numeric["stage_metrics"]:
            source = self.root / f"{stage}/source.npz"; source.parent.mkdir()
            if actual_k is None:
                scores = np.array([[.25, .5, .75]], float)
                world_hashes, world_events = None, None
                np.savez_compressed(source, candidate_indices=np.array([1]), event_scores=scores)
            else:
                worlds = np.zeros((actual_k, 2, 2), dtype=np.uint8); events = np.zeros((actual_k, 1, 3), bool)
                scores = events.mean(0)
                world_hashes = [hashlib.sha256(w.tobytes()).hexdigest() for w in worlds]
                world_events = events[:, 0].tolist()
                np.savez_compressed(source, worlds=worlds, world_events=events, candidate_indices=np.array([1]), event_scores=scores)
            metrics[stage] = dict(method=method, outer_seed=20260831, protocol_sha256="p", checkpoint_provenance=[],
                splits={"validation": {"cases": [dict(global_id="obs_000001", predictions_sha256=sha256(source))]}})
            sources[stage] = dict(status="complete", metrics_sha256=stage, prediction_sha256=sha256(source),
                                 actual_world_count=actual_k, event_score_semantics=semantics,
                                 prediction_path=f"{stage}/source.npz", model_provenance=[],
                                 world_order_sha256=world_hashes, query_array_index=0, events_by_world_and_radius=world_events,
                                 event_scores_by_radius=scores[0].tolist(), reference_events_by_radius=[False, False, False])
        renderer = self.root / "render/renderer_source.py"; renderer.parent.mkdir()
        renderer.write_text("# synthetic source snapshot only")
        files = []
        for suffix in ("svg", "png", "pdf"):
            file = self.root / f"render/fixture.{suffix}"; file.write_text("synthetic export fixture only")
            files.append(dict(path=file.name, sha256=sha256(file)))
        case = dict(global_id="obs_000001", query_id="q1", stage_sources=sources)
        manifest = dict(schema_version=2, method=method, actual_world_count=actual_k, common_world_budget=4,
            event_score_semantics=semantics, world_order=None if actual_k is None else "exact saved order",
            column_layout=["observed", "reference"] + (["direct_event_score"] if actual_k is None else [f"world_{i}" for i in range(actual_k)]),
            stage_metrics_sha256={stage: stage for stage in numeric["stage_metrics"]},
            outer_seed=20260831, protocol_sha256="p", figure_cases_sha256="figure",
            figure_geometry_audit_sha256="geometry", data_seal_sha256="data", complete=True, missing_stages=[],
            visual_gate_passed=None, validation_only=True, new_physical_test_images_opened=0,
            raw_archive_opened=False, final_test_locked=True, location_6_locked=True,
            stage_order=["untrained", "smoke", "pilot"], renderer_sha256=sha256(renderer), dependency_source_hashes={},
            unique_frozen_queries=1, rendered_unique_queries=1, cases=[case], figures=[dict(case, radius_cells=10, files=files)])
        file = self.save("render/manifest.json", manifest)
        reference = dict(path="render/manifest.json", sha256=sha256(file))
        return reference, numeric, protocol, metrics, manifest, file

    def test_actual_convergence_manifest_rejects_missing_stage_or_changed_export(self):
        reference, numeric, protocol, metrics, manifest, file = self.render_fixture()
        audit.check_convergence_render(reference, numeric, protocol, metrics)
        (self.root / "render/fixture.png").write_text("changed")
        with self.assertRaisesRegex(ValueError, "image export hash"):
            audit.check_convergence_render(reference, numeric, protocol, metrics)
        manifest["complete"] = False; manifest["missing_stages"] = ["pilot"]
        file.write_text(json.dumps(manifest)); reference["sha256"] = sha256(file)
        with self.assertRaises(ValueError):
            audit.check_convergence_render(reference, numeric, protocol, metrics)

    def test_deterministic_convergence_requires_one_real_world(self):
        reference, numeric, protocol, metrics, manifest, file = self.render_fixture("deterministic")
        audit.check_convergence_render(reference, numeric, protocol, metrics)
        self.assertEqual(len(manifest["cases"][0]["stage_sources"]["pilot"]["world_order_sha256"]), 1)
        manifest["actual_world_count"] = 4
        file.write_text(json.dumps(manifest)); reference["sha256"] = sha256(file)
        with self.assertRaises(ValueError):
            audit.check_convergence_render(reference, numeric, protocol, metrics)

    def test_direct_convergence_retains_raw_scores_and_rejects_fabricated_maps(self):
        reference, numeric, protocol, metrics, manifest, file = self.render_fixture("direct_query")
        audit.check_convergence_render(reference, numeric, protocol, metrics)
        self.assertEqual(manifest["cases"][0]["stage_sources"]["pilot"]["event_scores_by_radius"], [.25, .5, .75])
        manifest["cases"][0]["stage_sources"]["pilot"]["world_order_sha256"] = ["fabricated"]
        file.write_text(json.dumps(manifest)); reference["sha256"] = sha256(file)
        with self.assertRaisesRegex(ValueError, "fabricated map"):
            audit.check_convergence_render(reference, numeric, protocol, metrics)

    def test_direct_convergence_cannot_be_relabelled_as_world_frequency(self):
        reference, numeric, protocol, metrics, manifest, file = self.render_fixture("direct_query")
        manifest["event_score_semantics"] = "raw_world_event_frequency"
        file.write_text(json.dumps(manifest)); reference["sha256"] = sha256(file)
        with self.assertRaises(ValueError):
            audit.check_convergence_render(reference, numeric, protocol, metrics)

    def test_one_seed_cannot_publish_even_with_all_boolean_gates_true(self):
        report = dict(method="flow", passed=True, test_lock_passed=True, checkpoint_selection_passed=True,
                      implementation_checks_passed=True, convergence_review_passed=True,
                      fully_trained_outer_seeds=[20260831], evaluation_metrics_sha256=["one"])
        attempt = self.root / "attempt"
        self.save("attempt/verification.json", report)
        with self.assertRaisesRegex(ValueError, "all three"):
            audit.publish_final_receipt(report, attempt, self.root / "output")
        self.assertFalse((self.root / "output/final_audits/flow.json").exists())

    def test_training_journal_replays_parent_view_rng_and_refuses_validation_id(self):
        config = dict(batch_size=2, learning_rate=.0001)
        scenes = [dict(global_id=f"obs_{i}{v}", parent_group=str(i)) for i in range(2) for v in range(2)]
        protocol = dict(methods={"flow": config}, scenes={"train": scenes},
                        partition=dict(train=2, train_views_per_parent=2), staged=dict(pilot_steps=1, smoke_steps=1),
                        training_runtime=dict(checkpoint_interval=1))
        recipe = dict(protocol_sha256="p", method="flow", seed=20260831, member=0, config=config,
                      test_assets_opened=False, validation_loaded_by_training_process=False)
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.save("run/run.json", dict(recipe=recipe, recipe_sha256=digest, schema_version=1,
            data_workers=0, prefetch=False, checkpoint_interval=1, maximum_journaled_updates_replayed_after_unsafe_interruption=1))
        self.session("a", finish=3., heartbeat=2.)
        for file in (self.root / "run/sessions").glob("*.json"):
            value = json.loads(file.read_text()); value.update(method="flow", seed=20260831, member=0, protocol_sha256="p", test_assets_opened=False)
            file.write_text(json.dumps(value))
        self.save("run/time_summary.json", dict(training_wall_seconds=3., wall_time_is_lower_bound=False, sessions_without_final_receipt=[]))
        parent_rng = torch.Generator().manual_seed(20260831); views_rng = torch.Generator().manual_seed(20260833)
        rows = []
        for step in (1, 2, 3):
            ids = torch.randperm(2, generator=parent_rng).tolist(); views = torch.randint(2, (2,), generator=views_rng).tolist()
            lr = .0001 * (.1 + .9 * .5 * (1 + np.cos(np.pi * max(0, (step - 1) / 2))))
            rows.append(dict(step=step, losses={"velocity_mse": 1.}, update_seconds=.1, global_ids=[f"obs_{i}{v}" for i, v in zip(ids, views)],
                parent_indices=ids, view_indices=views, learning_rates={"model": lr}, session_id="a", peak_allocated_bytes=0, peak_reserved_bytes=0))
        progress = self.root / "run/progress.jsonl"; progress.write_text("".join(json.dumps(r)+"\n" for r in rows))
        checkpoint = self.root / "run/step_00000003.pt"; checkpoint.write_bytes(b"synthetic boundary checksum fixture")
        (self.root / "run/latest.pt").write_bytes(checkpoint.read_bytes())
        self.save("run/boundary_00000003.json", dict(method="flow", seed=20260831, member=0, completed_steps=3,
            protocol_sha256="p", checkpoint_sha256=sha256(checkpoint), test_assets_opened=False))
        args = (self.root / "run", "flow", 20260831, 0, 3, "p", "f", protocol)
        self.assertEqual(audit.check_training(*args)[0]["completed_steps"], 3)
        rows[0]["global_ids"][0] = "validation_observation"
        progress.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "escaped frozen training"):
            audit.check_training(*args)

    def test_efficiency_rejects_infinite_times_and_wrong_aggregate(self):
        cost = dict(actual_world_count=4, common_budget_K=4, actual_members=4, stage_one_member_diagnostic=False,
                    generation_seconds=1., full_update_seconds=2., actual_batch_forward_calls=4, actual_model_input_examples=4,
                    peak_allocated_bytes=42, peak_reserved_bytes=84)
        aggregate = dict(case_count=1, includes_first_case_cold_execution=True, generation_seconds_mean=1., generation_seconds_median=1.,
            full_update_seconds_mean=2., full_update_seconds_median=2., actual_batch_forward_calls=4, actual_model_input_examples=4,
            peak_allocated_bytes=42, peak_reserved_bytes=84)
        report = dict(cases=[dict(efficiency=cost)], efficiency=aggregate)
        audit.check_efficiency(report, "lama")
        report["efficiency"]["actual_batch_forward_calls"] = 1
        with self.assertRaises(ValueError):
            audit.check_efficiency(report, "lama")
        report["cases"][0]["efficiency"]["generation_seconds"] = float("inf")
        report["cases"][0]["efficiency"]["full_update_seconds"] = float("inf")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            audit.check_efficiency(report, "lama")

    def test_fm_solver_steps_guidance_and_real_forward_counts_are_bound(self):
        cost = dict(actual_world_count=4, common_budget_K=4, actual_members=1, stage_one_member_diagnostic=False,
            generation_seconds=1., full_update_seconds=2., actual_batch_forward_calls=50, actual_model_input_examples=400,
            peak_allocated_bytes=42, peak_reserved_bytes=84, velocity_evaluations_per_sample=50, solver="Heun",
            steps_per_sample=25, samples_per_observation=4, cfg_scale_s=2., cfg_branches=2, batched_model_calls=50,
            sample_equivalent_forwards=400, condition_cache_used=False)
        aggregate = dict(case_count=1, includes_first_case_cold_execution=True, generation_seconds_mean=1., generation_seconds_median=1.,
            full_update_seconds_mean=2., full_update_seconds_median=2., actual_batch_forward_calls=50, actual_model_input_examples=400,
            peak_allocated_bytes=42, peak_reserved_bytes=84)
        report = dict(cases=[dict(efficiency=cost)], efficiency=aggregate)
        audit.check_efficiency(report, "flow")
        cost["steps_per_sample"] = 24
        with self.assertRaises(ValueError):
            audit.check_efficiency(report, "flow")


if __name__ == "__main__":
    unittest.main()
