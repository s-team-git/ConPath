#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_BYTES=2054773316
readonly EXPECTED_SHA256=e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f
readonly DOWNLOAD_URL='https://huggingface.co/datasets/Rudra1ssb/FlatLands/resolve/main/FlatLands_final_dataset.zip?download=true'

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
archive_dir="${repo_root}/data/raw/flatlands"
archive_path="${archive_dir}/FlatLands_final_dataset.zip"
partial_path="${archive_path}.part"

usage() {
  echo "Usage: $0 --check | --download" >&2
}

verify_archive() {
  local candidate="$1"
  local actual_bytes actual_sha256

  if [[ ! -f "${candidate}" ]]; then
    echo "missing: ${candidate}" >&2
    return 2
  fi
  actual_bytes="$(stat --format='%s' "${candidate}")"
  if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" ]]; then
    echo "size mismatch: ${candidate}" >&2
    echo "expected=${EXPECTED_BYTES} actual=${actual_bytes}" >&2
    return 3
  fi
  actual_sha256="$(sha256sum "${candidate}" | cut -d ' ' -f 1)"
  if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
    echo "SHA-256 mismatch: ${candidate}" >&2
    echo "expected=${EXPECTED_SHA256} actual=${actual_sha256}" >&2
    return 4
  fi
  echo "verified: ${candidate}"
  echo "bytes=${actual_bytes}"
  echo "sha256=${actual_sha256}"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 64
fi

case "$1" in
  --check)
    verify_archive "${archive_path}"
    ;;
  --download)
    mkdir -p "${archive_dir}"
    if [[ -f "${archive_path}" ]]; then
      verify_archive "${archive_path}"
      exit 0
    fi

    available_bytes="$(df --output=avail --block-size=1 "${archive_dir}" | tail -n 1 | tr -d ' ')"
    if (( available_bytes < EXPECTED_BYTES * 2 )); then
      echo "insufficient free space for a resumable archive download" >&2
      echo "required_at_least=$((EXPECTED_BYTES * 2)) available=${available_bytes}" >&2
      exit 5
    fi

    echo "downloading to resumable partial file: ${partial_path}"
    curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
      --output "${partial_path}" "${DOWNLOAD_URL}"
    verify_archive "${partial_path}"
    mv "${partial_path}" "${archive_path}"
    echo "finalized: ${archive_path}"
    ;;
  *)
    usage
    exit 64
    ;;
esac
