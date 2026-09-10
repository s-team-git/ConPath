"""CPU-only loss/RNG/state and interrupted multi-optimizer recovery checks."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pathrel.formal_checkpoint import (FormalCheckpointStore, StatefulBatchSampler,
    capture_training_state, recipe_hash, restore_training_state)


def make_objects():
    random.seed(37)
    np.random.seed(37)
    torch.manual_seed(37)
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(.25), torch.nn.Linear(6, 1))
    second = torch.nn.Linear(1, 1)
    optimizers = {name: torch.optim.AdamW(module.parameters(), lr=.003, weight_decay=.02)
                  for name, module in (("generator", model), ("discriminator", second))}
    schedulers = {name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
                  for name, optimizer in optimizers.items()}
    return {"models": {"generator": model, "discriminator": second}, "optimizers": optimizers,
            "schedulers": schedulers, "sampler": StatefulBatchSampler(7, 4, seed=17),
            "generators": {"noise": torch.Generator().manual_seed(71),
                           "sample": torch.Generator().manual_seed(19)}}


def complete_update(objects, step, interrupt=False):
    ids = objects["sampler"].next_batch()
    noise = torch.randn((4, 3), generator=objects["generators"]["noise"])
    x = noise + torch.tensor(ids)[:, None] / 7
    target = torch.rand((4, 1), generator=objects["generators"]["sample"])
    target += random.random() + float(np.random.random())
    a, b = objects["models"].values()
    oa, ob = objects["optimizers"].values()
    oa.zero_grad(set_to_none=True)
    b.requires_grad_(False)
    prediction = a(x)
    loss_a = (b(prediction) - target).square().mean()
    loss_a.backward()
    oa.step()
    if interrupt:
        # Deliberately emulate a real GAN interrupt between G and D: model,
        # optimizer, RNG, sampler and discriminator requires_grad are now dirty.
        raise KeyboardInterrupt("between two optimizers")
    b.requires_grad_(True)
    ob.zero_grad(set_to_none=True)
    loss_b = (b(prediction.detach()) - target).square().mean()
    loss_b.backward()
    ob.step()
    for scheduler in objects["schedulers"].values():
        scheduler.step()
    return {"step": step, "ids": ids, "loss_a": float(loss_a.detach()), "loss_b": float(loss_b.detach())}


class FormalCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_interrupted_second_optimizer_replays_exact_losses_and_full_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = {"method": "two_optimizer_cpu_recovery", "seed": 37}
            baseline = make_objects()
            expected = []
            for step in range(1, 9):
                expected.append(complete_update(baseline, step))
            expected_state = capture_training_state(completed_steps=8, **baseline)
            split = make_objects()
            with FormalCheckpointStore(root / "split", recipe) as store:
                store.initialize(capture_training_state(completed_steps=0, **split))
                for step in range(1, 4):
                    row = complete_update(split, step)
                    store.commit_completed_update(capture_training_state(completed_steps=step, **split), row)
                with self.assertRaises(KeyboardInterrupt):
                    complete_update(split, 4, interrupt=True)
                receipt = store.write_interrupted("mid GAN update")
                self.assertEqual(receipt["completed_steps"], 3)
                self.assertEqual(receipt["latest_sha256"], receipt["interrupted_sha256"])
                # Restoring into these same dirty objects also repairs mode and
                # requires_grad flags; a fresh-process construction is optional.
                restore_training_state(store.load_latest()["state"], **split)
                self.assertTrue(all(p.requires_grad for p in split["models"]["discriminator"].parameters()))
            resumed = make_objects()
            random.seed(999)
            torch.rand(100)
            np.random.rand(100)
            with FormalCheckpointStore(root / "split", recipe, resume=True) as store:
                restored = restore_training_state(store.load_latest()["state"], **resumed)
                self.assertEqual(restored["completed_steps"], 3)
                for step in range(4, 9):
                    row = complete_update(resumed, step)
                    store.commit_completed_update(capture_training_state(completed_steps=step, **resumed), row)
                actual_state = store.load_latest()["state"]
            actual = [json.loads(line) for line in (root / "split/progress.jsonl").read_text().splitlines()]
            self.assertEqual(expected, actual)
            self.assert_nested_equal(expected_state, actual_state)

    def test_sampler_epoch_crossing_saved_cursor_and_private_rng(self):
        sampler = StatefulBatchSampler(5, 12, seed=3)
        first = sampler.next_batch()
        self.assertEqual(sorted(first[:5]), list(range(5)))
        self.assertEqual(sorted(first[5:10]), list(range(5)))
        self.assertEqual((sampler.epoch, sampler.cursor), (2, 2))
        saved = sampler.state_dict()
        expected = [sampler.next_batch() for _ in range(4)]
        restored = StatefulBatchSampler(5, 12, seed=999)
        restored.load_state_dict(saved)
        self.assertEqual(expected, [restored.next_batch() for _ in range(4)])
        with self.assertRaisesRegex(ValueError, "batch size"):
            StatefulBatchSampler(5, 11, seed=3).load_state_dict(saved)
        saved["permutation"][0] = saved["permutation"][1]
        with self.assertRaisesRegex(ValueError, "permutation"):
            restored.load_state_dict(saved)

    def test_checkpoint_capture_is_immutable_after_mutation(self):
        objects = make_objects()
        saved = capture_training_state(completed_steps=0, **objects)
        original = saved["models"]["generator"]["0.weight"].clone()
        complete_update(objects, 1)
        self.assertTrue(torch.equal(original, saved["models"]["generator"]["0.weight"]))
        self.assertEqual(saved["optimizers"]["generator"]["state"], {})
        self.assertEqual(saved["sampler"]["cursor"], 0)

    def test_nonempty_directory_and_recipe_mismatch_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = {"seed": 4, "lr": .1}
            with FormalCheckpointStore(root / "run", recipe) as store:
                store.initialize(capture_training_state(completed_steps=0, **make_objects()))
                with self.assertRaises(BlockingIOError):
                    FormalCheckpointStore(root / "run", recipe, resume=True)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "run").iterdir()}
            with self.assertRaises(FileExistsError):
                FormalCheckpointStore(root / "run", recipe)
            with self.assertRaisesRegex(ValueError, "recipe hash"):
                FormalCheckpointStore(root / "run", {"seed": 5, "lr": .1}, resume=True)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "run").iterdir()}
            self.assertEqual(before, after)
            with self.assertRaises(FileNotFoundError):
                FormalCheckpointStore(root / "missing", recipe, resume=True)

    def test_failed_checkpoint_archives_ahead_journal_and_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = {"method": "journal_fault"}
            objects = make_objects()
            with FormalCheckpointStore(root, recipe) as store:
                store.initialize(capture_training_state(completed_steps=0, **objects))
                row = complete_update(objects, 1)
                state = capture_training_state(completed_steps=1, **objects)
                with mock.patch("pathrel.formal_checkpoint.torch.save", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        store.commit_completed_update(state, row)
                self.assertEqual(store.load_latest()["progress_index"], 0)
                self.assertIn(b'"step": 1', (root / "progress.jsonl").read_bytes())
                store.write_interrupted("interrupted checkpoint serialization")
            with FormalCheckpointStore(root, recipe, resume=True) as store:
                self.assertEqual((root / "progress.jsonl").read_bytes(), b"")
                archives = list((root / "recovery_journal").glob("uncommitted_tail_*.jsonl"))
                self.assertEqual(len(archives), 1)
                self.assertEqual(json.loads(archives[0].read_text()), row)
                restore_training_state(store.load_latest()["state"], **objects)
                replay = complete_update(objects, 1)
                self.assertEqual(replay, row)
                store.commit_completed_update(capture_training_state(completed_steps=1, **objects), replay)

    def test_committed_journal_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects = make_objects()
            with FormalCheckpointStore(root, {"seed": 37}) as store:
                store.initialize(capture_training_state(completed_steps=0, **objects))
                row = complete_update(objects, 1)
                store.commit_completed_update(capture_training_state(completed_steps=1, **objects), row)
            (root / "progress.jsonl").write_text('{"step": 1, "loss": 0}\n')
            with self.assertRaisesRegex(ValueError, "missing or modified"):
                FormalCheckpointStore(root, {"seed": 37}, resume=True)

    def test_truncated_uncommitted_line_archived_without_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with FormalCheckpointStore(root, {}) as store:
                store.initialize(capture_training_state(completed_steps=0, **make_objects()))
            tail = b'{"step": 1, "unfinished'
            with (root / "progress.jsonl").open("ab") as stream:
                stream.write(tail)
            with FormalCheckpointStore(root, {}, resume=True):
                self.assertEqual((root / "progress.jsonl").read_bytes(), b"")
                archive = next((root / "recovery_journal").glob("uncommitted_tail_*.jsonl"))
                self.assertEqual(archive.read_bytes(), tail)

    def test_recipe_hash_stability_and_nonfinite_rejection(self):
        self.assertEqual(recipe_hash({"b": 2, "a": "中文"}), recipe_hash({"a": "中文", "b": 2}))
        with self.assertRaises(ValueError):
            recipe_hash({"loss": float("nan")})

    def test_registered_object_mismatch_is_explicit(self):
        objects = make_objects()
        saved = capture_training_state(completed_steps=0, **objects)
        objects["schedulers"] = {}
        with self.assertRaisesRegex(ValueError, "schedulers"):
            restore_training_state(saved, **objects)

    def test_periodic_checkpoint_rolls_back_and_archives_multiple_completed_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects = make_objects()
            expected = []
            with FormalCheckpointStore(root, {}, checkpoint_interval=3) as store:
                store.initialize(capture_training_state(completed_steps=0, **objects))
                for step in range(1, 6):
                    row = complete_update(objects, step)
                    expected.append(row)
                    store.append_progress(row)
                    if step == 3:
                        store.commit_completed_update(capture_training_state(completed_steps=3, **objects))
                receipt = store.write_interrupted("unsafe interruption before update six completes")
                self.assertEqual(receipt["completed_steps"], 3)
                self.assertEqual(receipt["journaled_updates_to_replay"], 2)
            resumed = make_objects()
            with FormalCheckpointStore(root, {}, checkpoint_interval=3, resume=True) as store:
                self.assertEqual(len((root / "progress.jsonl").read_text().splitlines()), 3)
                restored = restore_training_state(store.load_latest()["state"], **resumed)
                self.assertEqual(restored["completed_steps"], 3)
                for step in (4, 5):
                    row = complete_update(resumed, step)
                    self.assertEqual(row, expected[step - 1])
                    store.append_progress(row)
                # A safe pause can commit before the scheduled checkpoint.
                store.commit_completed_update(capture_training_state(completed_steps=5, **resumed))
                self.assertEqual(store.write_interrupted("safe pause")["journaled_updates_to_replay"], 0)
            with self.assertRaisesRegex(ValueError, "checkpoint interval"):
                FormalCheckpointStore(root, {}, checkpoint_interval=2, resume=True)

    def test_checkpoint_interval_cap_and_failed_commit_require_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects = make_objects()
            with FormalCheckpointStore(root, {}, checkpoint_interval=2) as store:
                store.initialize(capture_training_state(completed_steps=0, **objects))
                for step in (1, 2):
                    store.append_progress(complete_update(objects, step))
                with self.assertRaisesRegex(RuntimeError, "checkpoint interval"):
                    store.append_progress({"step": 3})
                with mock.patch("pathrel.formal_checkpoint.torch.save", side_effect=OSError("disk fault")):
                    with self.assertRaises(OSError):
                        store.commit_completed_update(capture_training_state(completed_steps=2, **objects))
                with self.assertRaisesRegex(RuntimeError, "explicit resume"):
                    store.commit_completed_update(capture_training_state(completed_steps=2, **objects))
                self.assertEqual(store.write_interrupted("I/O failure")["completed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
