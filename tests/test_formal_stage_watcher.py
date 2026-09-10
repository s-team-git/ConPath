"""CPU-only watcher fixtures; no real experiment audits, renders or GPU calls."""
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("formal_stage_watcher_test", ROOT / "scripts/watch_flatlands_formal_stages.py")
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class Fixture:
    def __init__(self, root):
        self.root, self.out = root, root / "out"
        self.out.mkdir()
        self.calls = []
        for name in (*watch.EXTRA_SOURCES, "src/frozen.py"):
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("CPU source fixture: " + name)
        for name in ("data/seal.json", "data/figures.json", "data/geometry.json"):
            watch.atomic_json(root / name, {"fixture": name})
        self.protocol = dict(output_root="out", data_root="data", data_seal_sha256=watch.sha(root / "data/seal.json"),
            staged=dict(seed=20260831, smoke_steps=1000, pilot_steps=5000),
            methods={"lama": {"members": 4}, "flow": {"members": 1}},
            test_lock=dict(final_test_locked=True, location_6_locked=True), source_hashes={"src/frozen.py": watch.sha(root / "src/frozen.py")},
            figures=dict(cases="data/figures.json", sha256=watch.sha(root / "data/figures.json"),
                         geometry_verification="data/geometry.json", geometry_verification_sha256=watch.sha(root / "data/geometry.json")))
        self.file = root / "protocol.json"
        watch.atomic_json(self.file, self.protocol)
        self.digest = watch.sha(self.file)
        watch.atomic_json(self.out / "ownership.json", {"protocol_sha256": self.digest})

    def stage(self, method="flow", stage="smoke", attempt=1):
        folder = self.out / "stages" / method / "20260831" / stage / f"attempt_{attempt}"
        steps = {"untrained": 0, "smoke": 1000, "pilot": 5000}[stage]
        count = self.protocol["methods"][method]["members"]
        metrics = dict(method=method, outer_seed=20260831, stage=stage, protocol_sha256=self.digest,
            splits={"calibration": {}, "validation": {}}, checkpoint_provenance=[{"completed_steps": steps}] * count,
            final_test_locked=True, location_6_locked=True, new_physical_test_images_opened=0,
            observed_and_support_constraints_passed=True, main_table_eligible=False)
        watch.atomic_json(folder / "metrics.json", metrics)
        watch.atomic_json(folder / "complete.json", dict(passed=True, method=method, seed=20260831, stage=stage,
            completed_steps=[steps] * count, metrics_sha256=watch.sha(folder / "metrics.json")))
        return dict(key=f"{method}/20260831/{stage}", method=method, seed=20260831, stage=stage, step=steps,
            evaluation=watch.relative(folder), metrics_sha256=watch.sha(folder / "metrics.json"),
            complete_sha256=watch.sha(folder / "complete.json"))

    def audit(self, row, attempt=1, passed=True):
        file = self.out / "audits" / row["method"] / str(row["seed"]) / row["stage"] / f"attempt_{attempt}" / "verification.json"
        watch.atomic_json(file, dict(passed=passed, protocol_sha256=self.digest, data_seal_sha256=self.protocol["data_seal_sha256"],
            method=row["method"], seed=row["seed"], stage=row["stage"], metrics_sha256=row["metrics_sha256"],
            complete_sha256=row["complete_sha256"], auditor_sha256=watch.sha(self.root / watch.AUDITOR),
            gpu_inference_performed=False, new_raw_archive_or_test_images_opened=0))
        return dict(row, audit=dict(path=watch.relative(file), sha256=watch.sha(file), auditor_sha256=watch.sha(self.root / watch.AUDITOR)))

    def report(self, rows, pointer=True):
        snapshot = self.out / "reports" / f"snapshot_{len(list((self.out / 'reports').glob('*'))) + 1}"
        snapshot.mkdir(parents=True)
        (snapshot / "renderer_source.py").write_bytes((self.root / watch.RENDERER).read_bytes())
        group = max(([r["stage"], r["step"]] for r in rows), key=lambda g: g[1])
        figure = snapshot / "figures/fixed.png"
        figure.parent.mkdir()
        figure.write_bytes(b"synthetic archived bytes, never a real render")
        inventory = [{"files": [{"path": "figures/fixed.png", "sha256": watch.sha(figure)}]}]
        methods = {m: {"runs": []} for m in self.protocol["methods"]}
        for row in rows:
            if [row["stage"], row["step"]] == group:
                methods[row["method"]]["runs"].append(dict(seed=row["seed"], independent_stage_audit=dict(row["audit"], passed=True)))
        records = [dict(method=r["method"], seed=r["seed"], group=[r["stage"], r["step"]], metrics_sha256=r["metrics_sha256"],
                        path=r["evaluation"]) for r in rows]
        summary = dict(protocol_sha256=self.digest, validation_only=True, new_physical_test_images_opened=0,
            final_test_locked=True, location_6_locked=True, selected_group=group, methods=methods,
            evaluation_records=records, figures=inventory)
        watch.atomic_json(snapshot / "metrics.json", summary)
        (snapshot / "summary.csv").write_text("synthetic,fixture\n")
        (snapshot / "report.md").write_text("Synthetic CPU fixture only\n")
        repro = dict(protocol_sha256=self.digest, data_seal_sha256=self.protocol["data_seal_sha256"],
            renderer_sha256=watch.sha(snapshot / "renderer_source.py"), figure_manifest_sha256=self.protocol["figures"]["sha256"],
            figure_geometry_audit_sha256=self.protocol["figures"]["geometry_verification_sha256"], figures=inventory,
            evaluation_records=records,
            report_files={name: watch.sha(snapshot / name) for name in ("metrics.json", "summary.csv", "report.md")})
        watch.atomic_json(snapshot / "reproducibility.json", repro)
        if pointer:
            watch.atomic_json(self.out / "current_report.json", dict(snapshot=str(snapshot.relative_to(self.out)),
                reproducibility_sha256=watch.sha(snapshot / "reproducibility.json")))
        return snapshot

    def child(self, script, arguments, logfile):
        self.calls.append(script)
        rows = watch.evaluations(self.out, self.protocol, self.digest)
        if script == watch.AUDITOR:
            folder = arguments[arguments.index("--evaluation-dir") + 1]
            row = next(r for r in rows if watch.project_path(r["evaluation"]) == folder)
            self.audit(row)
        elif script == watch.RENDERER:
            self.report([dict(row, audit=watch.passed_audit(self.out, row, self.digest, self.protocol["data_seal_sha256"])) for row in rows])
        else:
            raise AssertionError("GPU/publication command must never be invoked")


class StageWatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(watch, "ROOT", self.root))
        self.stack.enter_context(mock.patch.object(watch, "STOP_REQUESTED", False))
        self.fixture = Fixture(self.root)
        self.ctx = watch.StageWatcher(self.fixture.file)

    def tearDown(self):
        self.ctx.close()
        self.stack.close()
        self.temp.cleanup()

    def test_three_existing_audits_and_complete_report_are_reused_without_children(self):
        rows = [self.fixture.audit(self.fixture.stage(m, s)) for m, s in
                (("lama", "untrained"), ("flow", "untrained"), ("lama", "smoke"))]
        self.fixture.report(rows)
        with mock.patch.object(self.ctx, "run_cpu") as child:
            for _ in range(2):
                self.assertEqual(self.ctx.cycle()["audited_stages"], 3)
            child.assert_not_called()
        self.assertEqual(len(self.ctx.state["reports"]), 1)

    def test_new_stage_audited_and_reported_once_across_restart(self):
        self.fixture.stage()
        with mock.patch.object(self.ctx, "run_cpu", side_effect=self.fixture.child):
            self.assertEqual(self.ctx.cycle()["status"], "caught_up")
            self.ctx.cycle()
        self.assertEqual(self.fixture.calls, [watch.AUDITOR, watch.RENDERER])
        self.ctx.close(); self.ctx = watch.StageWatcher(self.fixture.file)
        with mock.patch.object(self.ctx, "run_cpu") as child:
            self.ctx.cycle(); child.assert_not_called()

    def test_failed_audit_is_kept_without_retry_and_manual_success_unblocks(self):
        row = self.fixture.stage()
        def fail(*unused):
            self.fixture.audit(row, passed=False)
            raise RuntimeError("synthetic failed audit")
        with mock.patch.object(self.ctx, "run_cpu", side_effect=fail) as child:
            self.assertEqual(self.ctx.cycle()["status"], "blocked_manual_review")
            before = len(json.dumps(self.ctx.state))
            self.ctx.cycle(); self.ctx.cycle()
            self.assertEqual(child.call_count, 1)
            self.assertLess(len(json.dumps(self.ctx.state)), before + 200)  # No recursive previous-state growth.
        self.fixture.audit(row, attempt=2)
        with mock.patch.object(self.ctx, "run_cpu", side_effect=self.fixture.child):
            self.assertEqual(self.ctx.cycle()["status"], "caught_up")
        self.assertEqual(self.fixture.calls, [watch.RENDERER])

    def test_claimed_or_partial_attempt_is_never_automatically_retried(self):
        row = self.fixture.stage()
        self.ctx.state["stages"][row["key"]] = dict(row, status="started")
        self.ctx.save()
        with mock.patch.object(self.ctx, "run_cpu") as child:
            self.assertEqual(self.ctx.cycle()["status"], "blocked_manual_review")
            child.assert_not_called()

    def test_partial_audit_json_blocks_only_that_stage_and_other_audits_continue(self):
        row = self.fixture.stage()
        partial = self.fixture.out / "audits/flow/20260831/smoke/attempt_1/verification.json"
        partial.parent.mkdir(parents=True)
        partial.write_text('{"passed":')
        self.fixture.stage("lama")
        with mock.patch.object(self.ctx, "run_cpu", side_effect=self.fixture.child):
            result = self.ctx.cycle()
            self.ctx.cycle()
        self.assertEqual(result["blocked_stages"], [row["key"]])
        self.assertEqual(result["audited_stages"], 1)
        self.assertEqual(self.fixture.calls, [watch.AUDITOR])
        self.assertEqual(partial.read_text(), '{"passed":')

    def test_failed_render_not_retried_but_finished_snapshot_after_crash_reused(self):
        row = self.fixture.audit(self.fixture.stage())
        with mock.patch.object(self.ctx, "run_cpu", side_effect=RuntimeError("render interrupted")) as child:
            with self.assertRaises(RuntimeError):
                self.ctx.cycle()
            self.assertEqual(self.ctx.cycle()["status"], "blocked_manual_review")
            self.assertEqual(child.call_count, 1)
        self.fixture.report([row], pointer=False)
        with mock.patch.object(self.ctx, "run_cpu") as child:
            self.assertEqual(self.ctx.cycle()["status"], "caught_up")
            child.assert_not_called()

    def test_wrong_audit_metrics_hash_is_not_reused_or_retried(self):
        row = self.fixture.stage()
        self.fixture.audit(dict(row, metrics_sha256="another-evaluation"))
        with mock.patch.object(self.ctx, "run_cpu") as child:
            self.assertEqual(self.ctx.cycle()["status"], "blocked_manual_review")
            child.assert_not_called()

    def test_duplicate_complete_identity_or_changed_metrics_fails_closed(self):
        self.fixture.stage()
        self.fixture.stage(attempt=2)
        with self.assertRaisesRegex(ValueError, "Multiple completed"):
            self.ctx.cycle()

    def test_report_image_hash_change_is_not_silently_rerendered(self):
        row = self.fixture.audit(self.fixture.stage())
        snapshot = self.fixture.report([row])
        (snapshot / "figures/fixed.png").write_bytes(b"changed")
        with mock.patch.object(self.ctx, "run_cpu") as child:
            with self.assertRaisesRegex(ValueError, "figure changed"):
                self.ctx.cycle()
            child.assert_not_called()

    def test_report_cannot_omit_a_displayed_method_while_claiming_full_input_coverage(self):
        row = self.fixture.audit(self.fixture.stage())
        snapshot = self.fixture.report([row])
        summary = watch.read(snapshot / "metrics.json")
        summary["methods"]["flow"]["runs"] = []
        watch.atomic_json(snapshot / "metrics.json", summary)
        repro = watch.read(snapshot / "reproducibility.json")
        repro["report_files"]["metrics.json"] = watch.sha(snapshot / "metrics.json")
        watch.atomic_json(snapshot / "reproducibility.json", repro)
        with self.assertRaisesRegex(ValueError, "omitted or duplicated"):
            watch.report_receipt(snapshot, [row], self.fixture.protocol, self.fixture.digest, self.ctx.source_hashes[watch.RENDERER])

    def test_protocol_source_and_frozen_figure_changes_stop_before_children(self):
        for file in (self.root / watch.RENDERER, self.root / "src/frozen.py", self.root / "data/figures.json", self.fixture.file):
            before = file.read_bytes()
            file.write_bytes(before + b" ")
            with self.subTest(file=file), mock.patch.object(self.ctx, "run_cpu") as child:
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.ctx.cycle()
                child.assert_not_called()
            file.write_bytes(before)

    def test_own_lock_prevents_second_watcher_and_stop_does_not_touch_gpu_queue(self):
        with self.assertRaises(BlockingIOError):
            watch.StageWatcher(self.fixture.file)
        self.fixture.stage()
        (self.fixture.out / "STOP").touch()
        with mock.patch.object(self.ctx, "run_cpu") as child, mock.patch.object(watch.os, "kill") as kill:
            self.assertEqual(self.ctx.cycle(), {"status": "paused"})
            child.assert_not_called(); kill.assert_not_called()

    def test_cpu_child_whitelist_empty_cuda_and_source_change_interrupt_only_own_child(self):
        class Child:
            pid = 42
            returncode = None
            def poll(self):
                return self.returncode
            def send_signal(self, signal):
                self.returncode = -signal
            def wait(self, timeout=None):
                return self.returncode
        process = Child()
        def create(*args, **kwargs):
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(args[0][1], str(self.root / watch.AUDITOR))
            (self.root / watch.AUDITOR).write_text("changed while CPU child is active")
            return process
        with mock.patch.object(watch.subprocess, "Popen", side_effect=create) as popen:
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.ctx.run_cpu(watch.AUDITOR, [], self.ctx.session / "child.log")
            self.assertEqual(process.returncode, -watch.signal.SIGINT)
            self.assertEqual(popen.call_count, 1)
        with self.assertRaisesRegex(ValueError, "Only CPU"):
            self.ctx.run_cpu("scripts/train_flatlands_formal.py", [], self.ctx.session / "forbidden.log")

    def test_watch_exits_after_queue_finished_without_polling_or_launching_child(self):
        with mock.patch.object(watch, "StageWatcher", return_value=self.ctx), \
             mock.patch.object(watch.sys, "argv", ["watcher", "--protocol", str(self.fixture.file), "--watch"]), \
             mock.patch.object(watch.signal, "signal"), mock.patch.object(self.ctx, "run_cpu") as child, \
             mock.patch.object(watch.time, "sleep") as sleep:
            watch.main()
            child.assert_not_called(); sleep.assert_not_called()
        self.assertTrue((self.ctx.session / "finished.json").exists())

    def test_new_completion_during_audit_is_drained_before_one_composite_report(self):
        self.fixture.stage()
        added = False
        def child(script, args, logfile):
            nonlocal added
            self.fixture.child(script, args, logfile)
            if script == watch.AUDITOR and not added:
                added = True
                self.fixture.stage("lama")
        with mock.patch.object(self.ctx, "run_cpu", side_effect=child):
            result = self.ctx.cycle()
        self.assertEqual(result["audited_stages"], 2)
        self.assertEqual(self.fixture.calls, [watch.AUDITOR, watch.AUDITOR, watch.RENDERER])

    def test_new_completion_during_render_is_new_input_not_blocked_retry(self):
        self.fixture.audit(self.fixture.stage())
        added = False
        def child(script, args, logfile):
            nonlocal added
            self.fixture.child(script, args, logfile)
            if script == watch.RENDERER and not added:
                added = True
                self.fixture.stage("lama")
        with mock.patch.object(self.ctx, "run_cpu", side_effect=child):
            result = self.ctx.cycle()
            self.ctx.cycle()
        self.assertEqual(result["status"], "caught_up")
        self.assertEqual(self.fixture.calls, [watch.RENDERER, watch.AUDITOR, watch.RENDERER])
        self.assertNotIn("blocked_manual_review", [j["status"] for j in self.ctx.state["reports"].values()])

    def test_final_completion_between_scan_and_queue_release_is_drained(self):
        def finished():
            self.fixture.stage()
            return False
        with mock.patch.object(watch, "StageWatcher", return_value=self.ctx), \
             mock.patch.object(watch.sys, "argv", ["watcher", "--protocol", str(self.fixture.file), "--watch"]), \
             mock.patch.object(watch.signal, "signal"), mock.patch.object(self.ctx, "queue_running", side_effect=finished), \
             mock.patch.object(self.ctx, "run_cpu", side_effect=self.fixture.child):
            watch.main()
        self.assertEqual(self.fixture.calls, [watch.AUDITOR, watch.RENDERER])


if __name__ == "__main__":
    unittest.main()
