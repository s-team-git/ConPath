"""Validation helpers for durable, versioned research recovery state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"recovery paths must be relative: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"recovery path escapes repository: {relative}")
    return resolved


def validate_recovery_state(
    state_path: Path,
    *,
    check_hashes: bool = True,
) -> dict[str, object]:
    """Check tracked recovery metadata and its required ignored artifacts read-only."""

    state_path = state_path.resolve()
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("RECOVERY_STATE.json must be an object with schema_version 1")

    root = state_path.parent
    checks: list[dict[str, object]] = []
    failures: list[str] = []

    for value in state.get("tracked_files", []):
        path = _workspace_path(root, value)
        passed = path.is_file()
        checks.append({"kind": "tracked_file", "path": str(value), "passed": passed})
        if not passed:
            failures.append(f"missing tracked file: {value}")

    artifacts = state.get("ignored_artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("ignored_artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or "path" not in artifact:
            raise ValueError("each ignored artifact must be an object with path")
        relative = str(artifact["path"])
        path = _workspace_path(root, relative)
        required = bool(artifact.get("required_for_resume", True))
        exists = path.is_file()
        record: dict[str, object] = {
            "kind": "ignored_artifact",
            "path": relative,
            "required": required,
            "exists": exists,
            "passed": True,
        }
        if not exists:
            record["passed"] = not required
            if required:
                failures.append(f"missing required artifact: {relative}")
            checks.append(record)
            continue

        actual_bytes = path.stat().st_size
        record["bytes"] = actual_bytes
        expected_bytes = artifact.get("bytes")
        if expected_bytes is not None:
            record["expected_bytes"] = int(expected_bytes)
            if actual_bytes != int(expected_bytes):
                record["passed"] = False
                failures.append(
                    f"size mismatch for {relative}: {actual_bytes} != {int(expected_bytes)}"
                )

        expected_sha256 = artifact.get("sha256")
        if check_hashes and expected_sha256 is not None:
            actual_sha256 = _sha256(path)
            record["sha256"] = actual_sha256
            record["expected_sha256"] = str(expected_sha256)
            if actual_sha256 != str(expected_sha256):
                record["passed"] = False
                failures.append(f"SHA-256 mismatch for {relative}")
        elif expected_sha256 is not None:
            record["sha256"] = "skipped"
            record["expected_sha256"] = str(expected_sha256)
        checks.append(record)

    return {
        "checkpoint_id": state.get("checkpoint_id"),
        "active_task": state.get("active_task"),
        "hashes_checked": check_hashes,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "next_command": state.get("next_command"),
    }
