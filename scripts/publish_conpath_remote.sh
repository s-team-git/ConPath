#!/usr/bin/env bash
set -euo pipefail

# The user explicitly requested replacing the CRANE repository contents.  Keep this operation
# opt-in and fail closed: credentials must already be configured (PAT credential helper or SSH),
# the working tree must be clean, and the expected remote must be selected.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" != "--confirm-replace" ]]; then
  echo "Refusing to replace remote contents without --confirm-replace." >&2
  echo "After configuring GitHub authentication, run: $0 --confirm-replace" >&2
  exit 2
fi

expected_remote="https://github.com/s-team-git/CRANE.git"
actual_remote="$(git remote get-url origin 2>/dev/null || true)"
case "$actual_remote" in
  "$expected_remote"|"git@github.com:s-team-git/CRANE.git"|"ssh://git@github.com/s-team-git/CRANE.git") ;;
  *)
  echo "origin is '$actual_remote', expected the CRANE HTTPS or SSH URL" >&2
  exit 2
  ;;
esac
if [[ -n "$(git status --short)" ]]; then
  echo "Working tree is not clean; commit or review changes before publishing." >&2
  git status --short >&2
  exit 2
fi

# Fetch first so --force-with-lease protects against replacing a branch that changed since the
# caller last inspected it.  This still replaces the selected main branch, as explicitly requested.
git fetch origin --prune
git push --force-with-lease --set-upstream origin main
echo "Published local ConPath main to $actual_remote (remote repository rename, if desired, is a GitHub setting)."
