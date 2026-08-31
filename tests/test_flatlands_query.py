from __future__ import annotations

import struct
import unittest
import zlib

import numpy as np

from pathrel.flatlands_query import (
    ManifestObservation,
    construct_natural_queries,
    decode_binary_grayscale_png,
    mask_relation_counts,
    score_natural_queries,
    select_scene_observations,
)
from pathrel.labels import clearance_radius_map, maximum_clearance_map


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def grayscale_png(array: np.ndarray, filter_type: int = 0) -> bytes:
    image = np.asarray(array, dtype=np.uint8)
    height, width = image.shape
    previous = np.zeros(width, dtype=np.uint8)
    rows = bytearray()
    for row in image:
        encoded = np.empty(width, dtype=np.uint8)
        for column in range(width):
            left = int(row[column - 1]) if column else 0
            above = int(previous[column])
            upper_left = int(previous[column - 1]) if column else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                predictor = (left, above, upper_left)[int(np.argmin(distances))]
            else:
                raise ValueError(filter_type)
            encoded[column] = (int(row[column]) - predictor) & 0xFF
        rows.append(filter_type)
        rows.extend(encoded.tobytes())
        previous = row
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(bytes(rows))),
            png_chunk(b"IEND", b""),
        )
    )


def manifest_row(split: str, source: str, scene: str, observation: int) -> ManifestObservation:
    global_id = f"obs_{observation:06d}"
    return ManifestObservation(
        global_id=global_id,
        provenance_split=split,
        archive_split=split,
        source_dataset=source,
        scene_id=scene,
        packet_directory=f"{split}/{global_id}",
        metadata_member=f"{split}/{global_id}/metadata.json",
        original_observation_id="obs_000",
        quality_category="LEARNABLE",
        resolution=0.1,
        camera_px=(10, 10),
    )


class FlatLandsNaturalQueryAuditTest(unittest.TestCase):
    def test_binary_png_decoder_supports_all_standard_grayscale_filters(self) -> None:
        expected = np.asarray(
            [[0, 255, 0, 255], [255, 255, 0, 0], [0, 0, 255, 255]], dtype=np.uint8
        )
        for filter_type in range(5):
            with self.subTest(filter_type=filter_type):
                decoded = decode_binary_grayscale_png(grayscale_png(expected, filter_type))
                np.testing.assert_array_equal(decoded, expected == 255)

    def test_png_decoder_rejects_nonbinary_values_and_crc_damage(self) -> None:
        with self.assertRaisesRegex(ValueError, "not binary"):
            decode_binary_grayscale_png(grayscale_png(np.asarray([[0, 127]], dtype=np.uint8)))
        damaged = bytearray(grayscale_png(np.asarray([[0, 255]], dtype=np.uint8)))
        damaged[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            decode_binary_grayscale_png(bytes(damaged))

    def test_scene_selection_is_stratified_stable_and_order_independent(self) -> None:
        observations = []
        index = 1
        for split in ("train", "validation", "test"):
            for source in ("ScanNet", "ZInD"):
                for scene_index in range(3):
                    for _ in range(2):
                        observations.append(
                            manifest_row(split, source, f"scene-{scene_index}", index)
                        )
                        index += 1
        first = select_scene_observations(observations, scenes_per_stratum=2, seed=17)
        second = select_scene_observations(
            list(reversed(observations)), scenes_per_stratum=2, seed=17
        )
        self.assertEqual([row.global_id for row in first], [row.global_id for row in second])
        self.assertEqual(len(first), 12)
        self.assertEqual(
            len({(row.provenance_split, row.source_dataset, row.scene_id) for row in first}),
            12,
        )

    def test_query_selection_is_target_blind_and_target_scoring_separates_topology(self) -> None:
        shape = (21, 21)
        observed = np.zeros(shape, dtype=bool)
        observed[10, 10] = True
        hidden = np.ones(shape, dtype=bool)
        hidden[10, 10] = False
        valid = np.ones(shape, dtype=bool)
        queries = construct_natural_queries(
            observed,
            hidden,
            valid,
            camera_px=(10, 10),
            resolution_m=0.1,
            distances_m=(0.5,),
            angles_deg=(0, 180),
        )
        self.assertEqual([query.selection_status for query in queries], ["selected", "selected"])
        self.assertEqual([(query.goal_row, query.goal_col) for query in queries], [(10, 15), (10, 5)])

        open_floor = np.ones(shape, dtype=bool)
        divided_floor = open_floor.copy()
        divided_floor[:, 12] = False
        open_scores = score_natural_queries(open_floor, valid, queries, radii_cells=(0, 1))
        divided_scores = score_natural_queries(divided_floor, valid, queries, radii_cells=(0, 1))
        self.assertEqual(
            [score["selection_status"] for score in open_scores],
            [score["selection_status"] for score in divided_scores],
        )
        self.assertEqual(open_scores[0]["target_status"], "high_clearance_positive")
        self.assertEqual(divided_scores[0]["target_status"], "disconnected_radius_zero")
        self.assertEqual(divided_scores[1]["target_status"], "high_clearance_positive")

    def test_epistemic_mask_is_applied_as_invalid_support(self) -> None:
        observed = np.zeros((11, 11), dtype=bool)
        observed[5, 5] = True
        hidden = np.ones_like(observed)
        hidden[5, 5] = False
        valid = np.ones_like(observed)
        valid[:, 7] = False
        queries = construct_natural_queries(
            observed,
            hidden,
            valid,
            camera_px=(5, 5),
            resolution_m=0.1,
            distances_m=(0.3,),
            angles_deg=(0,),
        )
        self.assertEqual(queries[0].selection_status, "selected")
        floor = np.ones_like(observed)
        scores = score_natural_queries(floor, valid, queries, radii_cells=(0, 1))
        self.assertEqual(scores[0]["target_status"], "disconnected_radius_zero")
        counts = mask_relation_counts(observed, floor, hidden, valid)
        self.assertEqual(counts["floor_outside_epistemic"], 11)

    def test_linear_edt_and_single_source_capacity_match_bruteforce(self) -> None:
        generator = np.random.default_rng(5)
        for _ in range(8):
            free = generator.random((9, 10)) > 0.25
            start = (4, 4)
            free[start] = True
            actual = clearance_radius_map(free)
            expected = np.full(free.shape, -1, dtype=np.int64)
            obstacles = np.argwhere(~np.pad(free, 1, constant_values=False))
            for row, col in np.argwhere(free):
                delta = obstacles - np.asarray([row + 1, col + 1])
                squared = int(np.min(np.sum(delta * delta, axis=1)))
                expected[row, col] = int(np.ceil(np.sqrt(squared))) - 1
            np.testing.assert_array_equal(actual, expected)

            capacity = maximum_clearance_map(free, start, clearance=actual)
            for goal in ((0, 0), (4, 4), (8, 9)):
                if not free[goal]:
                    self.assertEqual(capacity[goal], -1)


if __name__ == "__main__":
    unittest.main()
