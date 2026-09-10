#!/usr/bin/env python3
"""CPU-only stage audit/report handoff; no training, publication or visual gate.

Use --once for one scan or --watch for 30-second polling. The watcher owns only
postprocessing receipts and a separate lock. Existing passed audits are reused;
a failed/interrupted attempt requires a manually completed audit/report before
automatic processing can continue. No daemon is started by importing this file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = "scripts/audit_flatlands_formal_stage.py"
RENDERER = "scripts/render_flatlands_formal_report.py"
WATCHER = "scripts/watch_flatlands_formal_stages.py"
EXTRA_SOURCES = (AUDITOR, RENDERER, WATCHER, "scripts/evaluate_flatlands_support_clamped.py")
STOP_REQUESTED = False


def request_stop(*unused):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evaluation_set(rows):
    return {(r["key"], r["metrics_sha256"], r["complete_sha256"]) for r in rows}


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Artifact escapes the project")
    return path


def relative(path):
    return str(project_path(path).relative_to(ROOT.resolve()))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def evaluations(out, protocol, protocol_hash):
    """Only complete fixed first-seed development stages, never final/test data."""
    stages = {"untrained": 0, "smoke": protocol["staged"]["smoke_steps"], "pilot": protocol["staged"]["pilot_steps"]}
    found = {}
    for complete in sorted((out / "stages").glob("*/*/*/attempt_*/complete.json")):
        receipt = read(complete)
        file = complete.parent / "metrics.json"
        metrics = read(file)
        method, seed, stage = metrics["method"], metrics["outer_seed"], metrics["stage"]
        if (method not in protocol["methods"] or seed != protocol["staged"]["seed"] or stage not in stages
                or complete.parent.parent.parent.parent.name != method
                or complete.parent.parent.parent.name != str(seed) or complete.parent.parent.name != stage):
            raise ValueError("Unexpected stage directory/method/seed")
        count = protocol["methods"][method]["members"]
        if (receipt.get("passed") is not True or receipt.get("metrics_sha256") != sha(file)
                or receipt.get("method") != method or receipt.get("seed") != seed or receipt.get("stage") != stage
                or metrics.get("protocol_sha256") != protocol_hash
                or set(metrics.get("splits", {})) != {"calibration", "validation"}
                or receipt.get("completed_steps") != [stages[stage]] * count
                or [p["completed_steps"] for p in metrics["checkpoint_provenance"]] != [stages[stage]] * count
                or metrics.get("final_test_locked") is not True or metrics.get("location_6_locked") is not True
                or metrics.get("new_physical_test_images_opened") != 0
                or metrics.get("observed_and_support_constraints_passed") is not True
                or metrics.get("main_table_eligible") is not False):
            raise ValueError("Stage completion/protocol/budget/test-lock mismatch: " + str(complete))
        key = f"{method}/{seed}/{stage}"
        if key in found:
            raise ValueError("Multiple completed attempts for one stage; manual review required: " + key)
        found[key] = dict(key=key, method=method, seed=seed, stage=stage, step=stages[stage],
                          evaluation=relative(complete.parent), metrics_sha256=sha(file), complete_sha256=sha(complete))
    return [found[key] for key in sorted(found)]


def passed_audit(out, row, protocol_hash, data_hash):
    parent = out / "audits" / row["method"] / str(row["seed"])
    for file in sorted(parent.rglob("verification.json")):
        try:
            value = read(file)
        except (json.JSONDecodeError, OSError):
            continue  # Preserved partial attempt blocks this stage below.
        if (value.get("passed") is True and value.get("protocol_sha256") == protocol_hash
                and value.get("metrics_sha256") == row["metrics_sha256"]
                and value.get("complete_sha256") == row["complete_sha256"]
                and value.get("data_seal_sha256") == data_hash
                and value.get("method") == row["method"] and value.get("seed") == row["seed"]
                and value.get("stage") == row["stage"] and value.get("gpu_inference_performed") is False
                and value.get("new_raw_archive_or_test_images_opened") == 0):
            return dict(path=relative(file), sha256=sha(file), auditor_sha256=value["auditor_sha256"])
    return None


def report_receipt(snapshot, rows, protocol, protocol_hash, renderer_hash):
    """Verify immutable report coverage and all archived exports before reuse."""
    snapshot = project_path(snapshot)
    repro_file = snapshot / "reproducibility.json"
    if not repro_file.exists():
        return None
    repro = read(repro_file)
    if (repro.get("protocol_sha256") != protocol_hash or repro.get("data_seal_sha256") != protocol["data_seal_sha256"]
            or repro.get("renderer_sha256") != renderer_hash):
        return None
    if (repro.get("figure_manifest_sha256") != protocol["figures"]["sha256"]
            or repro.get("figure_geometry_audit_sha256") != protocol["figures"]["geometry_verification_sha256"]):
        raise ValueError("Report figure scope differs from the frozen protocol")
    if sha(snapshot / "renderer_source.py") != repro["renderer_sha256"]:
        raise ValueError("Archived report renderer changed")
    for name in ("metrics.json", "summary.csv", "report.md"):
        if sha(snapshot / name) != repro["report_files"][name]:
            raise ValueError("Archived report file changed")
    summary = read(snapshot / "metrics.json")
    expected = {(r["method"], r["seed"], r["stage"], r["step"], r["metrics_sha256"], r["evaluation"]) for r in rows}
    actual = {(r["method"], r["seed"], *r["group"], r["metrics_sha256"], r["path"]) for r in summary["evaluation_records"]}
    if actual != expected or len(summary["evaluation_records"]) != len(expected):
        return None
    if repro.get("evaluation_records") != [{k: v for k, v in row.items() if k != "validation_metrics"}
                                          for row in summary["evaluation_records"]]:
        raise ValueError("Report reproducibility evaluation list differs from its metrics")
    if (summary.get("protocol_sha256") != protocol_hash or summary.get("validation_only") is not True
            or summary.get("new_physical_test_images_opened") != 0 or summary.get("final_test_locked") is not True
            or summary.get("location_6_locked") is not True or summary["figures"] != repro["figures"]):
        raise ValueError("Report scope/figure provenance changed")
    # Only the newest common stage is displayed, while evaluation_records also
    # lists earlier stages. Bind every displayed row's actual independent audit.
    audits = {(r["method"], r["seed"]): r["audit"] for r in rows
              if [r["stage"], r["step"]] == summary["selected_group"]}
    displayed = [(method, run["seed"]) for method, detail in summary["methods"].items() for run in detail["runs"]]
    if len(displayed) != len(audits) or set(displayed) != set(audits):
        raise ValueError("Report omitted or duplicated an available displayed method/seed")
    for method, detail in summary["methods"].items():
        for run in detail["runs"]:
            expected_audit = audits.get((method, run["seed"]))
            actual_audit = run.get("independent_stage_audit", {})
            if (expected_audit is None or actual_audit.get("passed") is not True
                    or project_path(actual_audit["path"]) != project_path(expected_audit["path"])
                    or actual_audit.get("sha256") != expected_audit["sha256"]):
                return None
    for figure in repro["figures"]:
        for artifact in figure["files"]:
            target = (snapshot / artifact["path"]).resolve()
            if not target.is_relative_to(snapshot) or sha(target) != artifact["sha256"]:
                raise ValueError("Archived report figure changed")
    return dict(snapshot=relative(snapshot), reproducibility_sha256=sha(repro_file), renderer_sha256=renderer_hash)


class StageWatcher:
    def __init__(self, protocol_file):
        self.protocol_file = project_path(protocol_file)
        self.protocol_bytes = self.protocol_file.read_bytes()
        self.protocol = json.loads(self.protocol_bytes)
        self.protocol_hash = hashlib.sha256(self.protocol_bytes).hexdigest()
        self.out = project_path(self.protocol["output_root"])
        self.home = self.out / "postprocessing"
        if (read(self.out / "ownership.json")["protocol_sha256"] != self.protocol_hash
                or self.protocol["test_lock"].get("final_test_locked") is not True
                or self.protocol["test_lock"].get("location_6_locked") is not True
                or sha(project_path(self.protocol["data_root"]) / "seal.json") != self.protocol["data_seal_sha256"]):
            raise ValueError("Owned protocol/data/test-lock mismatch")
        self.source_hashes = dict(self.protocol["source_hashes"])
        self.asset_hashes = {
            relative(project_path(self.protocol["data_root"]) / "seal.json"): self.protocol["data_seal_sha256"],
            self.protocol["figures"]["cases"]: self.protocol["figures"]["sha256"],
            self.protocol["figures"]["geometry_verification"]: self.protocol["figures"]["geometry_verification_sha256"]}
        self.source_bytes = {name: project_path(name).read_bytes() for name in EXTRA_SOURCES}
        for name, content in self.source_bytes.items():
            digest = hashlib.sha256(content).hexdigest()
            if name in self.source_hashes and self.source_hashes[name] != digest:
                raise ValueError("Frozen postprocessing source already changed: " + name)
            self.source_hashes[name] = digest
        self.guard()
        self.home.mkdir(exist_ok=True)
        self.lock = (self.home / ".watcher.lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_file = self.home / "state.json"
            self.state = read(state_file) if state_file.exists() else {"protocol_sha256": self.protocol_hash, "stages": {}, "reports": {}}
            if self.state["protocol_sha256"] != self.protocol_hash:
                raise ValueError("Postprocessing state belongs to another protocol")
            self.session = self.home / "sessions" / uuid.uuid4().hex
            self.session.mkdir(parents=True, exist_ok=False)
            (self.session / "protocol.json").write_bytes(self.protocol_bytes)
            for name, content in self.source_bytes.items():
                (self.session / Path(name).name).write_bytes(content)
            atomic_json(self.session / "started.json", {"created_utc": now(), "pid": os.getpid(),
                "protocol_sha256": self.protocol_hash, "source_hashes": self.source_hashes,
                "frozen_asset_hashes": self.asset_hashes, "python": sys.version,
                "cuda_visible_devices_for_children": "", "maximum_cpu_children": 1,
                "gpu_training_or_inference_started": False, "publication_or_visual_gate_allowed": False})
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def guard(self):
        if sha(self.protocol_file) != self.protocol_hash:
            raise ValueError("Protocol changed while watching; stop and review before restarting")
        for name, expected in self.source_hashes.items():
            if sha(project_path(name)) != expected:
                raise ValueError("Source changed while watching; stop and review before restarting: " + name)
        for name, expected in self.asset_hashes.items():
            if sha(project_path(name)) != expected:
                raise ValueError("Frozen data/figure seal changed while watching: " + name)

    def stopped(self):
        return STOP_REQUESTED or (self.out / "STOP").exists()

    def save(self):
        atomic_json(self.home / "state.json", self.state)

    def run_cpu(self, script, arguments, log):
        if script not in (AUDITOR, RENDERER):
            raise ValueError("Only CPU stage audit and report rendering are allowed")
        self.guard()
        if self.stopped():
            raise InterruptedError("Stage postprocessing paused")
        command = [sys.executable, str(project_path(script)), *map(str, arguments)]
        with Path(log).open("x") as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src"),
                     "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
            atomic_json(self.home / "active.json", {"pid": os.getpid(), "child_pid": process.pid,
                "command": command, "source_sha256": self.source_hashes[script], "started_utc": now(), "cpu_only": True})
            try:
                while process.poll() is None:
                    self.guard()
                    if self.stopped():
                        raise InterruptedError("STOP received during CPU postprocessing")
                    time.sleep(.25)
                self.guard()
                if process.returncode:
                    raise RuntimeError(f"CPU child exited {process.returncode}; see {log}")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                atomic_json(self.home / "active.json", {"pid": os.getpid(), "child_pid": None, "finished_utc": now(), "cpu_only": True})

    def audit(self, row):
        if self.stopped():
            raise InterruptedError("Stage postprocessing paused before audit")
        previous = self.state["stages"].get(row["key"])
        if previous and previous["metrics_sha256"] != row["metrics_sha256"]:
            raise ValueError("A previously observed stage changed metrics; manual review required")
        passed = passed_audit(self.out, row, self.protocol_hash, self.protocol["data_seal_sha256"])
        if passed:
            self.state["stages"][row["key"]] = dict(row, status="passed", audit=passed)
            self.save()
            return dict(row, audit=passed)
        parent = self.out / "audits" / row["method"] / str(row["seed"]) / row["stage"]
        if previous or (parent.exists() and any(parent.iterdir())):
            self.state["stages"][row["key"]] = dict(previous or row, status="blocked_manual_review",
                note="Prior failed/interrupted audit exists; no automatic retry")
            self.save()
            return None
        attempt = parent / "attempt_1"
        job = dict(row, status="started", output=relative(attempt), started_utc=now(),
                   auditor_sha256=self.source_hashes[AUDITOR])
        self.state["stages"][row["key"]] = job
        self.save()  # Claim before creating any child; crash cannot duplicate work.
        try:
            self.run_cpu(AUDITOR, ["--evaluation-dir", project_path(row["evaluation"]), "--protocol", self.protocol_file,
                "--output-dir", attempt], self.session / (row["key"].replace("/", "_") + ".audit.log"))
            passed = passed_audit(self.out, row, self.protocol_hash, self.protocol["data_seal_sha256"])
            if passed is None:
                raise ValueError("Audit exited without a matching passed verification")
            self.state["stages"][row["key"]] = dict(row, status="passed", audit=passed)
            self.save()
            return dict(row, audit=passed)
        except BaseException as error:
            job.update(status="blocked_manual_review", error=repr(error), finished_utc=now())
            self.save()
            if isinstance(error, InterruptedError) or self.stopped():
                raise
            self.guard()
            return None

    def report(self, rows):
        signature = digest_json({"protocol": self.protocol_hash, "renderer": self.source_hashes[RENDERER], "stages": rows})
        previous = self.state["reports"].get(signature)
        candidates = []
        if previous and previous.get("report"):
            candidates.append(project_path(previous["report"]["snapshot"]))
        pointer = self.out / "current_report.json"
        if pointer.exists():
            value = read(pointer)
            snapshot = project_path(self.out / value["snapshot"])
            if sha(snapshot / "reproducibility.json") != value["reproducibility_sha256"]:
                raise ValueError("Current report pointer/hash changed")
            candidates.append(snapshot)
        # A completed immutable snapshot can survive a watcher/report crash
        # before current_report.json was refreshed; reuse it without rerendering.
        candidates += [file.parent for file in sorted((self.out / "reports").glob("*/reproducibility.json"), reverse=True)]
        for snapshot in dict.fromkeys(candidates):
            result = report_receipt(snapshot, rows, self.protocol, self.protocol_hash, self.source_hashes[RENDERER])
            if result:
                self.state["reports"][signature] = {"status": "passed", "report": result}
                self.save()
                return result
        if previous:
            return {"status": "blocked_manual_review", "reason": "Prior report attempt is incomplete; no automatic retry"}
        job = {"status": "started", "started_utc": now(), "input_stages": rows, "renderer_sha256": self.source_hashes[RENDERER]}
        self.state["reports"][signature] = job
        self.save()
        try:
            self.run_cpu(RENDERER, ["--protocol", self.protocol_file], self.session / (signature + ".report.log"))
            latest = evaluations(self.out, self.protocol, self.protocol_hash)
            if evaluation_set(latest) != evaluation_set(rows):
                # A stage can finish while the independent renderer discovers
                # its inputs. This is new work, not a failed rendering attempt.
                job.update(status="completed_with_new_stage_inputs", finished_utc=now())
                self.save()
                return {"status": "inputs_changed"}
            # Resolve the concrete completed output instead of trusting exit=0.
            return self.report(rows)
        except BaseException as error:
            job.update(status="blocked_manual_review", error=repr(error), finished_utc=now())
            self.save()
            raise

    def cycle(self):
        while True:
            self.guard()
            if self.stopped():
                return {"status": "paused"}
            rows = evaluations(self.out, self.protocol, self.protocol_hash)
            passed, blocked = [], []
            for row in rows:
                result = self.audit(row)
                (passed if result else blocked).append(result if result else row["key"])
            # Audit newly completed stages before starting a composite report.
            latest = evaluations(self.out, self.protocol, self.protocol_hash)
            if evaluation_set(latest) != evaluation_set(rows):
                continue
            report = self.report(passed) if passed and not blocked else None
            if report and report.get("status") == "inputs_changed":
                continue
            latest = evaluations(self.out, self.protocol, self.protocol_hash)
            if evaluation_set(latest) == evaluation_set(rows):
                break
        self.guard()
        result = dict(status="blocked_manual_review" if blocked or report and report.get("status") == "blocked_manual_review" else "caught_up",
                      completed_stages=len(rows), audited_stages=len(passed), blocked_stages=blocked, report=report)
        atomic_json(self.home / "status.json", dict(result, pid=os.getpid(), updated_utc=now(), protocol_sha256=self.protocol_hash))
        return result

    def queue_running(self):
        file = self.out / "queue.json"
        if not file.exists():
            return False
        queue = read(file)
        if queue.get("phase") != "staged_only" or not queue.get("pid"):
            return False
        try:
            os.kill(queue["pid"], 0)
            with (self.out / ".queue.lock").open("rb") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return False  # Stale PID receipt; no actual queue owns its lock.
                except BlockingIOError:
                    return True
        except (FileNotFoundError, ProcessLookupError):
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/flatlands_external_formal_protocol_v1/protocol.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    watcher = StageWatcher(args.protocol)
    result = {"status": "initializing"}
    try:
        while True:
            result = watcher.cycle()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if args.once or watcher.stopped():
                break
            if not watcher.queue_running():
                # Queue release follows its last completion write. Scan once
                # more after observing release so that terminal race is drained.
                result = watcher.cycle()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                break
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not watcher.stopped():
                watcher.guard()
                time.sleep(min(1., max(0., deadline - time.monotonic())))
    except BaseException as error:
        result = {"status": "paused" if watcher.stopped() or isinstance(error, InterruptedError) else "failed_manual_review", "error": repr(error)}
        raise
    finally:
        atomic_json(watcher.session / "finished.json", dict(result, finished_utc=now(), protocol_sha256=watcher.protocol_hash))
        watcher.close()
    if result["status"] == "blocked_manual_review":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
