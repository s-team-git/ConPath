from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pathrel.recovery import validate_recovery_state


class RecoveryStateTest(unittest.TestCase):
    def test_validates_required_artifact_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracked = root / "CONTINUATION.md"
            tracked.write_text("resume here\n", encoding="utf-8")
            artifact = root / "results" / "report.json"
            artifact.parent.mkdir()
            artifact.write_bytes(b"durable-result\n")
            state = {
                "schema_version": 1,
                "checkpoint_id": "fixture",
                "active_task": "test",
                "tracked_files": ["CONTINUATION.md"],
                "ignored_artifacts": [
                    {
                        "path": "results/report.json",
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
                "next_command": "continue",
            }
            state_path = root / "RECOVERY_STATE.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            report = validate_recovery_state(state_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["next_command"], "continue")

            artifact.write_bytes(b"changed\n")
            report = validate_recovery_state(state_path)
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["failures"]), 2)

    def test_rejects_paths_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "RECOVERY_STATE.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tracked_files": ["../outside"],
                        "ignored_artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes repository"):
                validate_recovery_state(state_path, check_hashes=False)


if __name__ == "__main__":
    unittest.main()
