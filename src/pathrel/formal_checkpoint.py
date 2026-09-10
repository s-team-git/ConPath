"""Transactional, exact-resume checkpoints for the formal external experiments.

The driver calls ``initialize`` before sampling the first batch, ``append_progress``
after every complete update, and ``commit_completed_update`` at the frozen
checkpoint interval or a safe pause boundary. If an update/write is interrupted,
``write_interrupted`` copies the last durable checkpoint, never the live (possibly
half-updated GAN) objects. On resume the uncommitted journal tail is preserved in
an archive, then removed from the active journal before deterministic replay.

The checkpoint interval is frozen in run.json: interruption during an update can
roll back at most that many journaled updates (plus the unfinished update). Those
updates are replayed with exact RNG/sampler state, never counted twice. Full-state
I/O time must be included in training runtime. Feed
``StatefulBatchSampler.next_batch()`` directly, with no workers or prefetch queue.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import torch


SCHEMA_VERSION = 1


def recipe_hash(recipe: Mapping[str, Any]) -> str:
    """Hash a JSON-only recipe, rejecting NaNs and implicit object conversions."""
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_file(path: Path, writer) -> None:
    """Replace one file durably; an interrupted write leaves the previous file."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path | str, value: Any) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _atomic_file(Path(path), lambda stream: stream.write(payload))


class StatefulBatchSampler:
    """Full batches from consecutive independently shuffled dataset passes.

    A batch may straddle epochs, including when batch_size > dataset size. The
    permutation and cursor describe exactly the next sample after a checkpoint.
    The private CPU generator is independent of augmentation/model RNG streams.
    """

    def __init__(self, size: int, batch_size: int, seed: int):
        if not isinstance(size, int) or not isinstance(batch_size, int) or size < 1 or batch_size < 1:
            raise ValueError("Sampler size and batch_size must be positive integers")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self.permutation = torch.randperm(size, generator=self.generator)

    def next_batch(self) -> list[int]:
        pieces = []
        remaining = self.batch_size
        while remaining:
            if self.cursor == self.size:
                self.epoch += 1
                self.cursor = 0
                self.permutation = torch.randperm(self.size, generator=self.generator)
            count = min(remaining, self.size - self.cursor)
            pieces.extend(self.permutation[self.cursor:self.cursor + count].tolist())
            self.cursor += count
            remaining -= count
        return pieces

    def state_dict(self) -> dict[str, Any]:
        return {"size": self.size, "batch_size": self.batch_size, "epoch": self.epoch,
                "cursor": self.cursor, "permutation": self.permutation.clone(),
                "generator_state": self.generator.get_state().clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("Sampler dataset size or batch size differs from checkpoint")
        permutation = state["permutation"].cpu()
        if (permutation.dtype != torch.int64 or permutation.shape != (self.size,)
                or not torch.equal(permutation.sort().values, torch.arange(self.size))):
            raise ValueError("Invalid saved sampler permutation")
        if not isinstance(state["cursor"], int) or not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid saved sampler cursor")
        if not isinstance(state["epoch"], int) or state["epoch"] < 0:
            raise ValueError("Invalid saved sampler epoch")
        self.generator.set_state(state["generator_state"].cpu())
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.permutation = permutation.clone()


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def capture_training_state(*, completed_steps: int, models: Mapping[str, torch.nn.Module],
                           optimizers: Mapping[str, torch.optim.Optimizer],
                           schedulers: Mapping[str, Any], sampler: StatefulBatchSampler,
                           generators: Mapping[str, torch.Generator],
                           extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Capture independent CPU copies at a completed optimizer-update boundary.

    ``models`` includes every trained module (LaMa generator AND discriminator).
    ``generators`` includes all named training/evaluation Torch RNG streams.
    ``extra`` can carry best-selection statistics, elapsed time or scaler state.
    No CUDA context is created for a CPU-only run. In a CUDA run, capture all
    already-initialized device RNG states and synchronize before copying weights.
    """
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise ValueError("completed_steps must be a nonnegative integer")
    if not models or not optimizers:
        raise ValueError("Models and optimizers must be explicitly registered")
    cuda_initialized = torch.cuda.is_initialized()
    if cuda_initialized:
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
    state = {
        "schema_version": SCHEMA_VERSION, "completed_steps": completed_steps,
        "models": {name: model.state_dict() for name, model in models.items()},
        "model_training_modes": {name: {key: child.training for key, child in model.named_modules()}
                                 for name, model in models.items()},
        "parameter_requires_grad": {name: {key: parameter.requires_grad for key, parameter in model.named_parameters()}
                                    for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
        "sampler": sampler.state_dict(),
        "generators": {name: {"device": str(generator.device), "state": generator.get_state()}
                       for name, generator in generators.items()},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if cuda_initialized else [],
                "cuda_initialized": cuda_initialized},
        "extra": dict(extra or {}),
    }
    return _cpu_copy(state)


def restore_training_state(state: Mapping[str, Any], *, models: Mapping[str, torch.nn.Module],
                           optimizers: Mapping[str, torch.optim.Optimizer],
                           schedulers: Mapping[str, Any], sampler: StatefulBatchSampler,
                           generators: Mapping[str, torch.Generator]) -> dict[str, Any]:
    """Restore into freshly constructed objects, then restore RNG LAST.

    CUDA devices and generator devices must match. Construct the scheduler before
    this call; optimizer.load_state_dict maps momentum buffers onto parameters.
    Gradients are discarded: every complete next update must zero its gradients.
    """
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported training-state schema")
    for name, objects in (("models", models), ("optimizers", optimizers),
                          ("schedulers", schedulers), ("generators", generators)):
        if set(state[name]) != set(objects):
            raise ValueError(f"Registered {name} differ from checkpoint")
    for name, generator in generators.items():
        if str(generator.device) != state["generators"][name]["device"]:
            raise ValueError(f"Generator device differs for {name}")
    cuda_states = state["rng"]["torch_cuda"]
    if cuda_states and (not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda_states)):
        raise ValueError("CUDA device count differs from checkpoint")
    for name, model in models.items():
        model.load_state_dict(state["models"][name], strict=True)
        if set(dict(model.named_modules())) != set(state["model_training_modes"][name]):
            raise ValueError(f"Model module names differ for {name}")
        if set(dict(model.named_parameters())) != set(state["parameter_requires_grad"][name]):
            raise ValueError(f"Model parameter names differ for {name}")
        for key, child in model.named_modules():
            child.training = state["model_training_modes"][name][key]
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(state["parameter_requires_grad"][name][key])
            parameter.grad = None
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(state["schedulers"][name])
    sampler.load_state_dict(state["sampler"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name]["state"].cpu())
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch_cpu"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    return {"completed_steps": state["completed_steps"], "extra": copy.deepcopy(state["extra"])}


class FormalCheckpointStore:
    """Exclusive output directory and atomic completed-update checkpoint store.

    Use as a context manager (or call ``close``). A new run may use an empty
    existing directory; any nonempty directory requires resume=True, the exact
    same recipe, and a valid latest.pt. There is no overwrite or reset option.
    """

    def __init__(self, output: Path | str, recipe: Mapping[str, Any], *, resume: bool = False,
                 checkpoint_interval: int = 1):
        if not isinstance(checkpoint_interval, int) or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        self.output = Path(output)
        self.recipe = copy.deepcopy(dict(recipe))
        self.recipe_sha256 = recipe_hash(recipe)
        self._lock = None
        self._step = None
        self._journal_step = None
        self._needs_resume = False
        self.checkpoint_interval = checkpoint_interval
        existed = self.output.exists()
        if existed and not self.output.is_dir():
            raise ValueError("Output path is not a directory")
        if existed and any(self.output.iterdir()) and not resume:
            raise FileExistsError(f"Nonempty run directory; refusing overwrite: {self.output}")
        if resume and not existed:
            raise FileNotFoundError("Cannot resume a missing run directory")
        self.output.mkdir(parents=True, exist_ok=True)
        # Exclusive creation claims fresh directories without a check/mkdir race.
        try:
            self._lock = (self.output / ".run.lock").open("a+b" if resume else "x+b")
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                receipt = json.loads((self.output / "run.json").read_text())
                if receipt["recipe_sha256"] != self.recipe_sha256 or recipe_hash(receipt["recipe"]) != self.recipe_sha256:
                    raise ValueError("Resume recipe hash differs from frozen run recipe")
                if receipt["checkpoint_interval"] != checkpoint_interval:
                    raise ValueError("Resume checkpoint interval differs from frozen run")
                checkpoint = self.load_latest()
                self._step = checkpoint["state"]["completed_steps"]
                self._reconcile_journal(checkpoint)
                self._journal_step = self._step
            else:
                atomic_json(self.output / "run.json", {"schema_version": SCHEMA_VERSION,
                    "recipe": self.recipe, "recipe_sha256": self.recipe_sha256,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_policy": "periodic complete update; interrupted copies last durable state",
                    "checkpoint_interval": checkpoint_interval,
                    "maximum_journaled_updates_replayed_after_unsafe_interruption": checkpoint_interval,
                    "data_workers": 0, "prefetch": False})
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _require_open(self) -> None:
        if self._lock is None:
            raise RuntimeError("Checkpoint store is closed")

    def _envelope(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported training-state schema")
        return {"schema_version": SCHEMA_VERSION, "recipe_sha256": self.recipe_sha256,
                "progress_index": state["completed_steps"], "state": state}

    def initialize(self, state: Mapping[str, Any]) -> None:
        self._require_open()
        if self._step is not None or (self.output / "latest.pt").exists():
            raise FileExistsError("Initial checkpoint already exists")
        if state["completed_steps"] != 0:
            raise ValueError("A fresh run must start at zero completed updates")
        envelope = self._envelope(state)
        envelope["journal_sha256"] = hashlib.sha256(b"").hexdigest()
        _atomic_file(self.output / "progress.jsonl", lambda stream: stream.write(b""))
        _atomic_file(self.output / "latest.pt", lambda stream: torch.save(envelope, stream))
        self._step = 0
        self._journal_step = 0

    def append_progress(self, progress: Mapping[str, Any]) -> None:
        """Durably journal ONE fully completed update; never call mid GAN step.

        No live tensors are saved here. A later checkpoint commits the entire
        journal prefix. An unsafe interruption rolls back and archives its tail.
        """
        self._require_open()
        if self._needs_resume:
            raise RuntimeError("Failed checkpoint or journal write requires explicit resume")
        if self._step is None:
            raise RuntimeError("Initialize the store before the first update")
        step = progress.get("step")
        if step != self._journal_step + 1:
            raise ValueError("Progress must describe the next complete step")
        if step - self._step > self.checkpoint_interval:
            raise RuntimeError("Frozen checkpoint interval reached; commit before the next update")
        row = (json.dumps(dict(progress), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        journal = self.output / "progress.jsonl"
        self._needs_resume = True
        with journal.open("ab") as stream:
            stream.write(row)
            stream.flush()
            os.fsync(stream.fileno())
        self._journal_step = step
        self._needs_resume = False

    def commit_completed_update(self, state: Mapping[str, Any],
                                progress: Mapping[str, Any] | None = None) -> None:
        """Save complete live state at the last journaled step, atomically.

        Optional ``progress`` appends that one last row first (convenient for an
        interval of one). At larger intervals use append_progress after each
        update, then call this without progress. A failed call requires reopening
        with resume=True before further updates; do not continue in dirty memory.
        """
        self._require_open()
        if self._needs_resume:
            raise RuntimeError("Failed checkpoint or journal write requires explicit resume")
        if self._step is None:
            raise RuntimeError("Initialize the store before the first update")
        step = state["completed_steps"]
        expected = self._journal_step + (1 if progress is not None else 0)
        if step != expected or step <= self._step or (progress is not None and progress.get("step") != step):
            raise ValueError("Checkpoint must describe the last fully journaled complete step")
        envelope = self._envelope(state)
        if progress is not None:
            self.append_progress(progress)
        journal = self.output / "progress.jsonl"
        envelope["journal_sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
        # On failure, the journal may extend past latest.pt. A resume archives
        # that exact tail before replay, whether it contains one or many updates.
        self._needs_resume = True
        _atomic_file(self.output / "latest.pt", lambda stream: torch.save(envelope, stream))
        self._step = step
        self._needs_resume = False

    def load_latest(self) -> dict[str, Any]:
        self._require_open()
        # Only trusted project-owned checkpoints are accepted. Python/NumPy RNG
        # states require weights_only=False; this is not an upload reader.
        checkpoint = torch.load(self.output / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("recipe_sha256") != self.recipe_sha256:
            raise ValueError("Checkpoint schema or recipe hash mismatch")
        if checkpoint["progress_index"] != checkpoint["state"]["completed_steps"]:
            raise ValueError("Checkpoint progress index mismatch")
        return checkpoint

    def write_interrupted(self, reason: str) -> dict[str, Any]:
        """Copy the last completed step, even if live optimizers are half updated.

        The driver should exit after calling this. Loading latest.pt on a later
        explicit resume is authoritative; interrupted.pt is an identical durable
        recovery copy. SIGINT/SIGTERM handlers should defer further signals while
        this method runs; a hard kill still leaves latest.pt recoverable.
        """
        self._require_open()
        checkpoint = self.load_latest()
        latest = self.output / "latest.pt"
        def copy_file(stream):
            with latest.open("rb") as source:
                shutil.copyfileobj(source, stream)
        _atomic_file(self.output / "interrupted.pt", copy_file)
        receipt = {"reason": str(reason), "completed_steps": checkpoint["progress_index"],
                   "journaled_updates_to_replay": max(0, (self._journal_step or 0) - checkpoint["progress_index"]),
                   "maximum_journaled_updates_to_replay": self.checkpoint_interval,
                   "saved_utc": datetime.now(timezone.utc).isoformat(),
                   "saved_state": "last durable complete update; unfinished work will be replayed",
                   "latest_sha256": _file_hash(latest),
                   "interrupted_sha256": _file_hash(self.output / "interrupted.pt")}
        atomic_json(self.output / "interruption.json", receipt)
        return receipt

    def _reconcile_journal(self, checkpoint: Mapping[str, Any]) -> None:
        journal = self.output / "progress.jsonl"
        content = journal.read_bytes()
        lines = content.splitlines(keepends=True)
        step = checkpoint["progress_index"]
        kept = b"".join(lines[:step])
        if len(lines) < step or hashlib.sha256(kept).hexdigest() != checkpoint["journal_sha256"]:
            raise ValueError("Committed progress journal is missing or modified")
        for expected, line in enumerate(lines[:step], 1):
            if not line.endswith(b"\n") or json.loads(line)["step"] != expected:
                raise ValueError("Committed progress journal has inconsistent step order")
        tail = content[len(kept):]
        if tail:
            folder = self.output / "recovery_journal"
            folder.mkdir(exist_ok=True)
            token = uuid.uuid4().hex
            archive = folder / f"uncommitted_tail_{token}.jsonl"
            _atomic_file(archive, lambda stream: stream.write(tail))
            atomic_json(folder / f"recovery_{token}.json", {
                "restored_completed_steps": step, "uncommitted_tail": archive.name,
                "tail_bytes": len(tail), "tail_sha256": hashlib.sha256(tail).hexdigest(),
                "reason": "journal extends past last durable optimizer checkpoint",
                "created_utc": datetime.now(timezone.utc).isoformat()})
            _atomic_file(journal, lambda stream: stream.write(kept))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
