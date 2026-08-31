"""Read-only integrity and split auditing for the FlatLands release archive."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import stat
from typing import Callable, Iterable
from zipfile import ZipFile, ZipInfo


FLATLANDS_ARCHIVE_BYTES = 2_054_773_316
FLATLANDS_ARCHIVE_SHA256 = "e4f2e5c7c54f7ba62ea696fb103fb5d3794f30f5a2e63715773e59d6a9f1d26f"
FLATLANDS_SPLITS = ("train", "validation", "test")
FLATLANDS_SPLIT_COUNTS = {"train": 215_342, "validation": 26_890, "test": 28_343}
FLATLANDS_PACKET_FILES = (
    "observed_floor.png",
    "floor_map.png",
    "unobserved.png",
    "epistemic_mask.png",
    "metadata.json",
)

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class MetadataMember:
    split: str
    packet_directory: str
    member_name: str


@dataclass(frozen=True)
class FlatLandsArchiveIndex:
    report: dict[str, object]
    metadata_members: tuple[MetadataMember, ...]


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_parts(name: str) -> tuple[str, ...] | None:
    if not name or "\\" in name or name.startswith("/"):
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path.parts


def _packet_location(parts: tuple[str, ...]) -> tuple[str, str, str] | None:
    for index, part in enumerate(parts):
        if part not in FLATLANDS_SPLITS:
            continue
        if len(parts) != index + 3 or not parts[index + 1].startswith("obs_"):
            return None
        packet_directory = "/".join(parts[: index + 2])
        return part, packet_directory, parts[index + 2]
    return None


def build_archive_index(
    infos: Iterable[ZipInfo],
    *,
    progress: ProgressCallback | None = None,
) -> FlatLandsArchiveIndex:
    expected_bits = {name: 1 << index for index, name in enumerate(FLATLANDS_PACKET_FILES)}
    complete_mask = (1 << len(FLATLANDS_PACKET_FILES)) - 1
    packets: dict[tuple[str, str], list[object]] = {}
    seen_names: set[str] = set()
    duplicate_member_count = 0
    duplicate_members: list[str] = []
    unsafe_member_count = 0
    unsafe_members: list[str] = []
    symlink_member_count = 0
    symlink_members: list[str] = []
    encrypted_member_count = 0
    encrypted_members: list[str] = []
    unexpected_packet_member_count = 0
    unexpected_packet_members: list[str] = []
    unrelated_files = 0
    member_count = 0

    for member_count, info in enumerate(infos, start=1):
        name = info.filename
        if name in seen_names:
            duplicate_member_count += 1
            if len(duplicate_members) < 100:
                duplicate_members.append(name)
        seen_names.add(name)
        if info.is_dir():
            continue
        if stat.S_ISLNK(info.external_attr >> 16):
            symlink_member_count += 1
            if len(symlink_members) < 100:
                symlink_members.append(name)
        if info.flag_bits & 0x1:
            encrypted_member_count += 1
            if len(encrypted_members) < 100:
                encrypted_members.append(name)
        parts = _safe_member_parts(name)
        if parts is None:
            unsafe_member_count += 1
            if len(unsafe_members) < 100:
                unsafe_members.append(name)
            continue
        location = _packet_location(parts)
        if location is None:
            unrelated_files += 1
            continue
        split, packet_directory, filename = location
        key = (split, packet_directory)
        state = packets.setdefault(key, [0, None])
        bit = expected_bits.get(filename)
        if bit is None:
            unexpected_packet_member_count += 1
            if len(unexpected_packet_members) < 100:
                unexpected_packet_members.append(name)
            continue
        state[0] = int(state[0]) | bit
        if filename == "metadata.json":
            state[1] = name
        if progress is not None and member_count % 100_000 == 0:
            progress({"event": "index", "members": member_count, "packets": len(packets)})

    split_counts = Counter(split for split, _ in packets)
    incomplete_examples: list[dict[str, object]] = []
    incomplete_count = 0
    metadata_members: list[MetadataMember] = []
    for (split, packet_directory), (mask, metadata_name) in packets.items():
        missing = [
            filename
            for filename, bit in expected_bits.items()
            if not (int(mask) & bit)
        ]
        if int(mask) != complete_mask:
            incomplete_count += 1
            if len(incomplete_examples) < 100:
                incomplete_examples.append(
                    {"split": split, "packet": packet_directory, "missing": missing}
                )
        if metadata_name is not None:
            metadata_members.append(
                MetadataMember(split, packet_directory, str(metadata_name))
            )

    official_counts_match = all(
        split_counts[split] == expected for split, expected in FLATLANDS_SPLIT_COUNTS.items()
    )
    report: dict[str, object] = {
        "member_count": member_count,
        "unique_member_count": len(seen_names),
        "packet_count": len(packets),
        "split_packet_counts": {split: split_counts[split] for split in FLATLANDS_SPLITS},
        "expected_split_packet_counts": dict(FLATLANDS_SPLIT_COUNTS),
        "official_counts_match": official_counts_match,
        "metadata_member_count": len(metadata_members),
        "incomplete_packet_count": incomplete_count,
        "incomplete_packet_examples": incomplete_examples,
        "duplicate_member_count": duplicate_member_count,
        "duplicate_member_examples": duplicate_members,
        "unsafe_member_count": unsafe_member_count,
        "unsafe_member_examples": unsafe_members,
        "symlink_member_count": symlink_member_count,
        "symlink_member_examples": symlink_members,
        "encrypted_member_count": encrypted_member_count,
        "encrypted_member_examples": encrypted_members,
        "unexpected_packet_member_count": unexpected_packet_member_count,
        "unexpected_packet_member_examples": unexpected_packet_members,
        "unrelated_file_count": unrelated_files,
    }
    return FlatLandsArchiveIndex(
        report=report,
        metadata_members=tuple(sorted(metadata_members, key=lambda item: item.member_name)),
    )


def _metadata_identity(metadata: dict[str, object]) -> tuple[str | None, str | None]:
    scene = metadata.get("scene")
    if not isinstance(scene, dict):
        return None, None
    source = scene.get("dataset")
    if source is None:
        source = scene.get("source_dataset")
    scene_id = scene.get("scene_id")
    if scene_id is None:
        scene_id = scene.get("id")
    return (
        None if source is None else str(source),
        None if scene_id is None else str(scene_id),
    )


def _interesting_metadata_paths(metadata: dict[str, object]) -> list[str]:
    paths: list[str] = []

    def visit(value: object, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if any(token in lowered for token in ("resolution", "camera", "pixel", "grid")):
                    paths.append(path)
                visit(child, path)

    visit(metadata, "")
    return sorted(set(paths))


def audit_metadata(
    archive: ZipFile,
    members: tuple[MetadataMember, ...],
    *,
    limit: int = 0,
    seed: int = 20260830,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if limit < 0:
        raise ValueError("metadata limit must be non-negative")
    selected = list(members)
    if limit and limit < len(selected):
        selected = random.Random(seed).sample(selected, limit)
        selected.sort(key=lambda item: item.member_name)

    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scenes_by_split: dict[str, set[tuple[str, str]]] = defaultdict(set)
    missing_identity = 0
    malformed_count = 0
    malformed_examples: list[dict[str, str]] = []
    top_level_key_sets: Counter[tuple[str, ...]] = Counter()
    interesting_paths: set[str] = set()
    global_ids: set[str] = set()
    duplicate_global_ids = 0
    original_split_mismatches = 0

    for index, member in enumerate(selected, start=1):
        try:
            metadata = json.loads(archive.read(member.member_name))
            if not isinstance(metadata, dict):
                raise ValueError("metadata root is not an object")
        except Exception as error:  # corrupt audit input must be reported, not hidden
            malformed_count += 1
            if len(malformed_examples) < 100:
                malformed_examples.append(
                    {"member": member.member_name, "error": f"{type(error).__name__}: {error}"}
                )
            continue

        top_level_key_sets[tuple(sorted(str(key) for key in metadata))] += 1
        if len(interesting_paths) < 200:
            interesting_paths.update(_interesting_metadata_paths(metadata))
        source, scene_id = _metadata_identity(metadata)
        if source is None or scene_id is None:
            missing_identity += 1
        else:
            source_counts[member.split][source] += 1
            scenes_by_split[member.split].add((source, scene_id))

        provenance = metadata.get("provenance")
        if isinstance(provenance, dict):
            global_id = provenance.get("global_id")
            if global_id is not None:
                global_id_string = str(global_id)
                if global_id_string in global_ids:
                    duplicate_global_ids += 1
                global_ids.add(global_id_string)
            original_split = provenance.get("original_split")
            if original_split is not None and str(original_split) != member.split:
                original_split_mismatches += 1

        if progress is not None and index % 1_000 == 0:
            progress({"event": "metadata", "processed": index, "selected": len(selected)})

    overlap: dict[str, dict[str, object]] = {}
    for first_index, first in enumerate(FLATLANDS_SPLITS):
        for second in FLATLANDS_SPLITS[first_index + 1 :]:
            shared = scenes_by_split[first] & scenes_by_split[second]
            overlap[f"{first}__{second}"] = {
                "count": len(shared),
                "examples": [list(item) for item in sorted(shared)[:100]],
            }

    complete = len(selected) == len(members)
    scene_disjoint = complete and missing_identity == 0 and all(
        int(value["count"]) == 0 for value in overlap.values()
    )
    return {
        "available_metadata_members": len(members),
        "selected_metadata_members": len(selected),
        "complete_metadata_scan": complete,
        "seed": seed,
        "malformed_count": malformed_count,
        "malformed_examples": malformed_examples,
        "missing_scene_identity_count": missing_identity,
        "duplicate_global_id_count": duplicate_global_ids,
        "original_split_mismatch_count": original_split_mismatches,
        "scene_counts": {split: len(scenes_by_split[split]) for split in FLATLANDS_SPLITS},
        "source_observation_counts": {
            split: dict(sorted(source_counts[split].items())) for split in FLATLANDS_SPLITS
        },
        "scene_overlap": overlap,
        "scene_disjoint": scene_disjoint,
        "top_level_key_sets": [
            {"keys": list(keys), "count": count}
            for keys, count in top_level_key_sets.most_common(20)
        ],
        "interesting_metadata_paths": sorted(interesting_paths)[:200],
    }


def integrity_gate(index_report: dict[str, object], metadata_report: dict[str, object]) -> bool:
    return bool(
        index_report["official_counts_match"]
        and index_report["incomplete_packet_count"] == 0
        and index_report["duplicate_member_count"] == 0
        and index_report["unsafe_member_count"] == 0
        and index_report["symlink_member_count"] == 0
        and index_report["encrypted_member_count"] == 0
        and index_report["unexpected_packet_member_count"] == 0
        and metadata_report["complete_metadata_scan"]
        and metadata_report["malformed_count"] == 0
        and metadata_report["scene_disjoint"]
        and metadata_report["duplicate_global_id_count"] == 0
    )
