#!/usr/bin/env python3
"""Verify durable hand-off state before resuming an interrupted research session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from pathrel.recovery import validate_recovery_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("RECOVERY_STATE.json"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check presence and byte counts without rehashing large artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_recovery_state(args.state, check_hashes=not args.quick)
    root = args.state.resolve().parent
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    report["git_status"] = status.stdout.rstrip().splitlines()
    checkpoint = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", args.state.name],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    report["recovery_state_commit"] = checkpoint.stdout.strip() or None
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
