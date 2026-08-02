#!/usr/bin/env python3
"""Run a resumable two-worker Shadowhand18Tilted gravity curriculum.

The workers share one FIFO target queue. As soon as either worker succeeds or
uses up its per-target budget, it claims the next target. A new target resumes
from the geodesically closest checkpoint that has *already* succeeded; a policy
that is still training is deliberately not considered a parent.
"""

from __future__ import print_function

import argparse
import copy
import datetime
import fcntl
import json
import math
import os
import queue
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE_DIR / "curricula" / "shadowhand18_tilt_42.yaml"
NEGATIVE_REWARD_SENTINEL = -1000000000.0
STATE_VERSION = 1


class ManifestValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Target:
    target_id: str
    phi_deg: float
    theta_deg: float
    axis: tuple
    object_offset: object
    manifest_index: int

    @property
    def direction(self):
        return spherical_direction(self.theta_deg, self.phi_deg)


@dataclass(frozen=True)
class ParentCandidate:
    source_id: str
    checkpoint: Path
    direction: tuple
    completion_seq: int
    manifest_index: int


@dataclass
class ActiveJob:
    worker_id: int
    target_id: str
    process: subprocess.Popen
    command: list
    run_name: str
    output_checkpoint: Path
    staged_checkpoint: Path
    log_path: Path
    started_monotonic: float
    deadline_monotonic: float
    timed_out: bool = False
    timeout_message: str = ""


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def spherical_direction(theta_deg, phi_deg):
    theta = math.radians(float(theta_deg))
    phi = math.radians(float(phi_deg))
    return (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )


def angular_distance(a, b):
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return math.acos(max(-1.0, min(1.0, dot)))


def expected_axis(phi_deg):
    phi = math.radians(float(phi_deg))
    return (-math.sin(phi), math.cos(phi), 0.0)


def _numeric_vector(value, field, errors, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        errors.append("{} must be a three-number list".format(field))
        return None
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            errors.append("{} must contain only numbers".format(field))
            return None
        if not math.isfinite(float(component)):
            errors.append("{} must contain only finite numbers".format(field))
            return None
        result.append(float(component))
    return tuple(result)


def _expected_target_order():
    result = []
    result.extend((0, theta) for theta in (30, 60, 90, 120, 150, 180))
    for ring_index, phi in enumerate((45, 90, 135, 180, 225, 270, 315), start=1):
        thetas = (150, 120, 90, 60, 30) if ring_index % 2 == 1 else (30, 60, 90, 120, 150)
        result.extend((phi, theta) for theta in thetas)
    result.append((0, 0))
    return result


def load_manifest(path, require_offsets=True, require_seed=True):
    path = Path(path).resolve()
    errors = []
    try:
        with path.open("r") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestValidationError(["cannot read manifest {}: {}".format(path, exc)])

    if not isinstance(raw, dict):
        raise ManifestValidationError(["manifest root must be a mapping"])
    if raw.get("version") != 1:
        errors.append("manifest version must be 1")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("manifest name must be a non-empty string")

    seed_raw = raw.get("seed")
    if not isinstance(seed_raw, dict):
        errors.append("seed must be a mapping")
        seed_raw = {}
    # "checkpoint" remains accepted for manifests created by the first
    # launcher version; new manifests use the more explicit editable name.
    seed_checkpoint_value = seed_raw.get("start_checkpoint", seed_raw.get("checkpoint"))
    if not isinstance(seed_checkpoint_value, str) or not seed_checkpoint_value:
        errors.append("seed.start_checkpoint must be a path")
        seed_checkpoint = path.parent / "missing-seed-checkpoint"
    else:
        seed_checkpoint = Path(seed_checkpoint_value).expanduser()
        if not seed_checkpoint.is_absolute():
            seed_checkpoint = (path.parent / seed_checkpoint).resolve()
        if require_seed and not seed_checkpoint.is_file():
            errors.append("seed checkpoint does not exist: {}".format(seed_checkpoint))

    try:
        seed_theta = float(seed_raw.get("theta_deg"))
        seed_phi = float(seed_raw.get("phi_deg"))
    except (TypeError, ValueError):
        errors.append("seed theta_deg and phi_deg must be numbers")
        seed_theta, seed_phi = 0.0, 0.0
    seed_axis = _numeric_vector(seed_raw.get("base_tilt_axis"), "seed.base_tilt_axis", errors)
    if seed_axis is not None:
        _validate_axis(seed_axis, seed_phi, "seed.base_tilt_axis", errors)

    training = raw.get("training")
    if not isinstance(training, dict):
        errors.append("training must be a mapping")
        training = {}
    required_positive_ints = (
        "num_envs",
        "minibatch_size",
        "max_iterations",
        "episode_length",
        "save_best_after",
        "workers",
        "timeout_seconds",
    )
    for key in required_positive_ints:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("training.{} must be a positive integer".format(key))
    if training.get("workers") != 2:
        errors.append("training.workers must be 2 for this cooperative launcher")
    threshold = training.get("score_to_win")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        errors.append("training.score_to_win must be numeric")
    object_type = training.get("object_type")
    if not isinstance(object_type, str) or not object_type:
        errors.append("training.object_type must be a non-empty string")
    gpu = training.get("gpu")
    if not isinstance(gpu, (str, int)):
        errors.append("training.gpu must be a string or integer")

    existing_start_rows = raw.get("existing_start_checkpoints", [])
    if not isinstance(existing_start_rows, list):
        errors.append("existing_start_checkpoints must be a list")
        existing_start_rows = []
    existing_start_checkpoints = []
    seen_start_ids = set()
    seen_start_paths = {str(seed_checkpoint)}
    for index, row in enumerate(existing_start_rows):
        prefix = "existing_start_checkpoints[{}]".format(index)
        if not isinstance(row, dict):
            errors.append("{} must be a mapping".format(prefix))
            continue
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id:
            errors.append("{}.id must be a non-empty string".format(prefix))
            source_id = "invalid-start-{}".format(index)
        elif source_id in seen_start_ids:
            errors.append("duplicate existing start checkpoint id: {}".format(source_id))
        seen_start_ids.add(source_id)

        checkpoint_value = row.get("checkpoint")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            errors.append("{}.checkpoint must be a path".format(prefix))
            checkpoint = path.parent / (source_id + "-missing.pth")
        else:
            checkpoint = Path(checkpoint_value).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (path.parent / checkpoint).resolve()
            if require_seed and not checkpoint.is_file():
                errors.append("existing start checkpoint does not exist: {}".format(checkpoint))
            if str(checkpoint) in seen_start_paths:
                errors.append("duplicate existing start checkpoint path: {}".format(checkpoint))
            seen_start_paths.add(str(checkpoint))

        try:
            start_theta = float(row.get("theta_deg"))
            start_phi = float(row.get("phi_deg"))
        except (TypeError, ValueError):
            errors.append("{} theta_deg and phi_deg must be numbers".format(prefix))
            start_theta, start_phi = 0.0, 0.0
        if not 0.0 <= start_theta <= 180.0:
            errors.append("{}.theta_deg must be between 0 and 180".format(prefix))
        start_axis = _numeric_vector(row.get("base_tilt_axis"), prefix + ".base_tilt_axis", errors)
        if start_axis is not None:
            _validate_axis(start_axis, start_phi, prefix + ".base_tilt_axis", errors)
        recorded_reward = row.get("recorded_reward")
        if isinstance(recorded_reward, bool) or not isinstance(recorded_reward, (int, float)):
            errors.append("{}.recorded_reward must be numeric".format(prefix))
        elif isinstance(threshold, (int, float)) and float(recorded_reward) <= float(threshold):
            errors.append(
                "{}.recorded_reward must be strictly above training.score_to_win".format(prefix)
            )
        existing_start_checkpoints.append(
            ParentCandidate(
                source_id,
                checkpoint,
                spherical_direction(start_theta, start_phi),
                0,
                -9999 + index,
            )
        )

    target_rows = raw.get("targets")
    if not isinstance(target_rows, list):
        errors.append("targets must be a list")
        target_rows = []
    if len(target_rows) != 42:
        errors.append("targets must contain exactly 42 rows, found {}".format(len(target_rows)))

    targets = []
    seen_ids = set()
    seen_directions = {}
    for index, row in enumerate(target_rows):
        prefix = "targets[{}]".format(index)
        if not isinstance(row, dict):
            errors.append("{} must be a mapping".format(prefix))
            continue
        target_id = row.get("id")
        if not isinstance(target_id, str) or not target_id:
            errors.append("{}.id must be a non-empty string".format(prefix))
            target_id = "invalid-{}".format(index)
        elif target_id in seen_ids:
            errors.append("duplicate target id: {}".format(target_id))
        seen_ids.add(target_id)
        try:
            phi = float(row.get("phi_deg"))
            theta = float(row.get("theta_deg"))
        except (TypeError, ValueError):
            errors.append("{} phi_deg and theta_deg must be numbers".format(prefix))
            phi, theta = 0.0, -1.0
        if not 0.0 <= theta <= 180.0:
            errors.append("{}.theta_deg must be between 0 and 180".format(prefix))
        axis = _numeric_vector(row.get("base_tilt_axis"), prefix + ".base_tilt_axis", errors)
        if axis is not None:
            _validate_axis(axis, phi, prefix + ".base_tilt_axis", errors)
        offset = _numeric_vector(
            row.get("object_palm_offset"),
            prefix + ".object_palm_offset",
            errors,
            allow_none=not require_offsets,
        )
        if require_offsets and row.get("object_palm_offset") is None:
            # _numeric_vector already reports the invalid shape; replace it with
            # a more useful manual-calibration message.
            if errors and errors[-1] == prefix + ".object_palm_offset must be a three-number list":
                errors[-1] = prefix + ".object_palm_offset is unset; enter the calibrated palm-local offset"
        direction = spherical_direction(theta, phi)
        direction_key = tuple(round(component, 7) for component in direction)
        if direction_key in seen_directions:
            errors.append(
                "{} duplicates gravity direction from {}".format(prefix, seen_directions[direction_key])
            )
        else:
            seen_directions[direction_key] = prefix
        targets.append(Target(target_id, phi, theta, axis or (0.0, 1.0, 0.0), offset, index))

    expected_order = _expected_target_order()
    actual_order = [(int(round(t.phi_deg)) % 360, int(round(t.theta_deg))) for t in targets]
    if actual_order != expected_order:
        errors.append("target FIFO order does not match the documented 42-direction serpentine curriculum")

    ring_counts = {}
    pole_count = 0
    for target in targets:
        theta_key = int(round(target.theta_deg))
        if theta_key in (0, 180):
            pole_count += 1
        else:
            ring_counts[theta_key] = ring_counts.get(theta_key, 0) + 1
    if pole_count != 2 or any(ring_counts.get(theta) != 8 for theta in (30, 60, 90, 120, 150)):
        errors.append("coverage must be two poles plus five 8-azimuth rings")

    if errors:
        raise ManifestValidationError(errors)

    return {
        "path": path,
        "name": name,
        "seed_checkpoint": seed_checkpoint,
        "seed_direction": spherical_direction(seed_theta, seed_phi),
        "seed_theta_deg": seed_theta,
        "seed_phi_deg": seed_phi,
        "seed_axis": seed_axis,
        "existing_start_checkpoints": existing_start_checkpoints,
        "training": copy.deepcopy(training),
        "targets": targets,
    }


def _validate_axis(axis, phi_deg, field, errors):
    norm = math.sqrt(sum(component * component for component in axis))
    if abs(norm - 1.0) > 1e-5:
        errors.append("{} must be normalized (norm is {:.8f})".format(field, norm))
    expected = expected_axis(phi_deg)
    if max(abs(actual - wanted) for actual, wanted in zip(axis, expected)) > 1e-5:
        errors.append("{} does not match [-sin(phi), cos(phi), 0]".format(field))


def choose_nearest_parent(target, candidates):
    if not candidates:
        raise RuntimeError("no successful checkpoint is available for {}".format(target.target_id))
    return min(
        candidates,
        key=lambda candidate: (
            round(angular_distance(target.direction, candidate.direction), 12),
            -candidate.completion_seq,
            candidate.manifest_index,
        ),
    )


def available_parents(manifest, state):
    candidates = [
        ParentCandidate(
            "seed",
            manifest["seed_checkpoint"],
            manifest["seed_direction"],
            0,
            -10000,
        )
    ]
    candidates.extend(manifest.get("existing_start_checkpoints", []))
    targets_by_id = {target.target_id: target for target in manifest["targets"]}
    for target_id, record in state["targets"].items():
        if record["status"] != "succeeded":
            continue
        checkpoint = Path(record["output_checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError(
                "successful target {} is missing checkpoint {}".format(target_id, checkpoint)
            )
        target = targets_by_id[target_id]
        candidates.append(
            ParentCandidate(
                target_id,
                checkpoint,
                target.direction,
                int(record.get("completion_seq", 0)),
                target.manifest_index,
            )
        )
    return candidates


def stage_checkpoint(source, destination):
    """Copy a full RL Games checkpoint and reset only run bookkeeping."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to stage checkpoints; run this launcher from the Isaac Gym environment"
        ) from exc

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        # Loading from the temporary copy makes it impossible for this function
        # to modify the successful parent checkpoint in place.
        with source.open("rb") as source_stream, temporary_path.open("wb") as destination_stream:
            while True:
                chunk = source_stream.read(16 * 1024 * 1024)
                if not chunk:
                    break
                destination_stream.write(chunk)
        state = torch.load(str(temporary_path), map_location="cpu")
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError("checkpoint is not an RL Games full-state checkpoint")
        state["epoch"] = 0
        state["frame"] = 0
        state["last_mean_rewards"] = NEGATIVE_REWARD_SENTINEL
        torch.save(state, str(temporary_path))
        os.replace(str(temporary_path), str(destination))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def read_checkpoint_reward(checkpoint):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect training checkpoints") from exc
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        return None
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict) or "last_mean_rewards" not in state:
        raise ValueError("checkpoint {} has no last_mean_rewards".format(checkpoint))
    reward = state["last_mean_rewards"]
    if hasattr(reward, "item"):
        reward = reward.item()
    return float(reward)


def _format_number(value):
    value = float(value)
    if abs(value) < 5e-12:
        value = 0.0
    return "{:.8g}".format(value)


def _format_vector(vector):
    return "[{}]".format(",".join(_format_number(value) for value in vector))


def build_command(python_executable, target, staged_checkpoint, run_name, training):
    return [
        str(python_executable),
        "train.py",
        "num_envs={}".format(training["num_envs"]),
        "train.params.config.minibatch_size={}".format(training["minibatch_size"]),
        "max_iterations={}".format(training["max_iterations"]),
        "task=Shadowhand18Tilted",
        "headless=True",
        "task.env.objectType={}".format(training["object_type"]),
        "checkpoint={}".format(Path(staged_checkpoint).resolve()),
        "task.env.episodeLength={}".format(training["episode_length"]),
        "task.env.baseTiltAngleDeg={}".format(_format_number(target.theta_deg)),
        "task.env.objectPalmOffset={}".format(_format_vector(target.object_offset)),
        "task.env.baseTiltAxis={}".format(_format_vector(target.axis)),
        "train.params.config.score_to_win={}".format(_format_number(training["score_to_win"])),
        "train.params.config.save_best_after={}".format(training["save_best_after"]),
        "experiment={}".format(run_name),
        "+full_experiment_name={}".format(run_name),
    ]


def shell_join(command):
    return " ".join(shlex.quote(str(token)) for token in command)


def make_run_name(target, attempt):
    return "sh18tilt_{:02d}_{}_a{:02d}".format(
        target.manifest_index + 1, target.target_id, attempt
    )


def new_state(manifest):
    now = utc_now()
    return {
        "version": STATE_VERSION,
        "manifest": str(manifest["path"]),
        "curriculum": manifest["name"],
        "created_at": now,
        "updated_at": now,
        "completion_counter": 0,
        "targets": {
            target.target_id: {
                "manifest_index": target.manifest_index,
                "status": "pending",
                "attempts": 0,
                "worker_id": None,
                "parent_id": None,
                "parent_checkpoint": None,
                "staged_checkpoint": None,
                "run_name": None,
                "output_checkpoint": None,
                "log_path": None,
                "best_reward": None,
                "started_at": None,
                "completed_at": None,
                "completion_seq": None,
                "message": None,
            }
            for target in manifest["targets"]
        },
    }


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at"] = utc_now()
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _recorded_process_is_alive(record):
    pid = record.get("pid")
    run_name = record.get("run_name")
    if not isinstance(pid, int) or pid <= 0 or not run_name:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    try:
        command_line = Path("/proc") / str(pid) / "cmdline"
        return run_name in command_line.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        # If /proc cannot be inspected, avoiding a duplicate 10k-environment
        # process is safer than assuming the recorded PID is stale.
        return True


def load_or_create_state(manifest, state_path, retries):
    state_path = Path(state_path)
    changed = False
    if state_path.is_file():
        with state_path.open("r") as stream:
            state = json.load(stream)
        if state.get("version") != STATE_VERSION:
            raise RuntimeError("unsupported state version in {}".format(state_path))
        expected_ids = [target.target_id for target in manifest["targets"]]
        actual_ids = [
            target_id
            for target_id, _ in sorted(
                state.get("targets", {}).items(), key=lambda item: item[1].get("manifest_index", -1)
            )
        ]
        if actual_ids != expected_ids:
            raise RuntimeError("state targets do not match the manifest; use a new --state-dir")
    else:
        state = new_state(manifest)
        changed = True

    retry_set = set(retries)
    if "all" in retry_set:
        retry_set.update(("timed_out", "failed"))
    for record in state["targets"].values():
        if record["status"] == "running":
            if _recorded_process_is_alive(record):
                raise RuntimeError(
                    "recorded training process PID {} for {} is still alive; wait for it or terminate it "
                    "before restarting the launcher".format(record.get("pid"), record.get("run_name"))
                )
            reward = None
            output_checkpoint = record.get("output_checkpoint")
            if output_checkpoint:
                try:
                    reward = read_checkpoint_reward(output_checkpoint)
                except (OSError, RuntimeError, ValueError):
                    reward = None
            if reward is not None and reward > float(manifest["training"]["score_to_win"]):
                state["completion_counter"] = int(state.get("completion_counter", 0)) + 1
                record["status"] = "succeeded"
                record["best_reward"] = reward
                record["completed_at"] = utc_now()
                record["completion_seq"] = state["completion_counter"]
                record["message"] = "recovered completed success after the previous launcher stopped"
            else:
                record["status"] = "pending"
                record["message"] = "requeued because the previous launcher stopped while this job was running"
            record["worker_id"] = None
            record.pop("pid", None)
            staged_checkpoint = record.get("staged_checkpoint")
            if staged_checkpoint and Path(staged_checkpoint).is_file():
                Path(staged_checkpoint).unlink()
            record["staged_checkpoint"] = None
            changed = True
        elif record["status"] in retry_set:
            previous_status = record["status"]
            record["status"] = "pending"
            record["message"] = "requeued by --retry {}".format(previous_status)
            record["worker_id"] = None
            changed = True
    if changed:
        atomic_write_json(state_path, state)
    return state


def acquire_launcher_lock(state_dir):
    lock_path = Path(state_dir) / "launcher.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_stream.close()
        raise RuntimeError("another launcher already holds {}".format(lock_path))
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write("{}\n".format(os.getpid()))
    lock_stream.flush()
    return lock_stream


def _next_pending_target(manifest, state):
    for target in manifest["targets"]:
        if state["targets"][target.target_id]["status"] == "pending":
            return target
    return None


def _cleanup_staged_checkpoint(record):
    value = record.get("staged_checkpoint")
    if not value:
        return
    path = Path(value)
    if path.is_file():
        path.unlink()
    record["staged_checkpoint"] = None


def _watch_process(active_job, completion_queue):
    return_code = active_job.process.wait()
    completion_queue.put((time.monotonic(), active_job.worker_id, active_job.process.pid, return_code))


def _terminate_process(active_job, grace_seconds):
    process = active_job.process
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _print_status_counts(state):
    counts = {}
    for record in state["targets"].values():
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    print("Curriculum status: {}".format(", ".join("{}={}".format(k, counts[k]) for k in sorted(counts))))
    return counts


def run_curriculum(manifest, state_dir, python_executable, timeout_seconds=None, grace_seconds=30.0):
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = acquire_launcher_lock(state_dir)
    state_path = state_dir / "state.json"
    state = load_or_create_state(manifest, state_path, run_curriculum.retries)
    training = manifest["training"]
    timeout_seconds = float(training["timeout_seconds"] if timeout_seconds is None else timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if grace_seconds < 0:
        raise ValueError("terminate_grace_seconds cannot be negative")
    threshold = float(training["score_to_win"])
    completion_queue = queue.Queue()
    active = {}
    stop_scheduling = False

    checkpoints_dir = state_dir / "staged_checkpoints"
    logs_dir = state_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    def start_next(worker_id):
        nonlocal stop_scheduling
        target = _next_pending_target(manifest, state)
        if target is None or stop_scheduling:
            return False
        record = state["targets"][target.target_id]
        try:
            parent = choose_nearest_parent(target, available_parents(manifest, state))
            attempt = int(record["attempts"]) + 1
            run_name = make_run_name(target, attempt)
            staged_checkpoint = checkpoints_dir / (run_name + "_parent.pth")
            output_checkpoint = PACKAGE_DIR / "runs" / run_name / "nn" / (run_name + ".pth")
            log_path = logs_dir / (run_name + ".log")
            stage_checkpoint(parent.checkpoint, staged_checkpoint)
            command = build_command(
                python_executable, target, staged_checkpoint, run_name, training
            )

            record.update(
                {
                    "status": "running",
                    "attempts": attempt,
                    "worker_id": worker_id,
                    "parent_id": parent.source_id,
                    "parent_checkpoint": str(parent.checkpoint),
                    "staged_checkpoint": str(staged_checkpoint),
                    "run_name": run_name,
                    "output_checkpoint": str(output_checkpoint),
                    "log_path": str(log_path),
                    "best_reward": None,
                    "started_at": utc_now(),
                    "completed_at": None,
                    "completion_seq": None,
                    "message": None,
                }
            )
            atomic_write_json(state_path, state)

            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(training["gpu"])
            with log_path.open("a", buffering=1) as log_stream:
                log_stream.write("COMMAND: {}\n".format(shell_join(command)))
                log_stream.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(PACKAGE_DIR),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            now_monotonic = time.monotonic()
            job = ActiveJob(
                worker_id,
                target.target_id,
                process,
                command,
                run_name,
                output_checkpoint,
                staged_checkpoint,
                log_path,
                now_monotonic,
                now_monotonic + timeout_seconds,
            )
            active[worker_id] = job
            record["pid"] = process.pid
            atomic_write_json(state_path, state)
            watcher = threading.Thread(
                target=_watch_process, args=(job, completion_queue), name="watch-worker-{}".format(worker_id)
            )
            watcher.daemon = True
            watcher.start()
            distance_deg = math.degrees(angular_distance(target.direction, parent.direction))
            print(
                "Worker {} started {} (theta={}, phi={}) from {} at {:.2f} deg; log {}".format(
                    worker_id,
                    target.target_id,
                    _format_number(target.theta_deg),
                    _format_number(target.phi_deg),
                    parent.source_id,
                    distance_deg,
                    log_path,
                ),
                flush=True,
            )
            return True
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "worker_id": worker_id,
                    "completed_at": utc_now(),
                    "message": "launch preparation failed: {}".format(exc),
                }
            )
            _cleanup_staged_checkpoint(record)
            atomic_write_json(state_path, state)
            print("Worker {} could not start {}: {}".format(worker_id, target.target_id, exc), file=sys.stderr)
            stop_scheduling = True
            return False

    def finish_job(job, return_code):
        nonlocal stop_scheduling
        record = state["targets"][job.target_id]
        reward = None
        verification_error = None
        try:
            reward = read_checkpoint_reward(job.output_checkpoint)
        except Exception as exc:
            verification_error = str(exc)
        record["best_reward"] = reward
        record["completed_at"] = utc_now()
        record["return_code"] = return_code
        record["worker_id"] = None
        record.pop("pid", None)

        if reward is not None and reward > threshold:
            state["completion_counter"] = int(state.get("completion_counter", 0)) + 1
            record["status"] = "succeeded"
            record["completion_seq"] = state["completion_counter"]
            record["message"] = "strict reward threshold passed: {} > {}".format(reward, threshold)
            print("Worker {} succeeded on {} with reward {:.6g}".format(job.worker_id, job.target_id, reward), flush=True)
        elif verification_error is not None:
            record["status"] = "failed"
            record["message"] = "output checkpoint verification failed: {}".format(verification_error)
            stop_scheduling = True
            print("Worker {} failed on {}: {}".format(job.worker_id, job.target_id, record["message"]), file=sys.stderr)
        elif job.timed_out:
            record["status"] = "timed_out"
            record["message"] = job.timeout_message
            print("Worker {} timed out on {}; continuing the queue".format(job.worker_id, job.target_id), flush=True)
        elif return_code == 0:
            record["status"] = "timed_out"
            record["message"] = (
                "training ended cleanly without passing the strict reward threshold "
                "(the max-iteration budget was likely exhausted)"
            )
            print("Worker {} exhausted the training budget on {}; continuing the queue".format(job.worker_id, job.target_id), flush=True)
        else:
            record["status"] = "failed"
            record["message"] = "training process exited unexpectedly with code {}".format(return_code)
            stop_scheduling = True
            print("Worker {} crashed on {} with code {}; no new jobs will start".format(job.worker_id, job.target_id, return_code), file=sys.stderr)

        _cleanup_staged_checkpoint(record)
        atomic_write_json(state_path, state)

    try:
        for worker_id in range(int(training["workers"])):
            start_next(worker_id)

        while active:
            try:
                _, worker_id, pid, return_code = completion_queue.get(timeout=1.0)
            except queue.Empty:
                now_monotonic = time.monotonic()
                for job in list(active.values()):
                    if now_monotonic >= job.deadline_monotonic and not job.timed_out:
                        job.timed_out = True
                        job.timeout_message = "per-target wall-clock timeout of {:.0f} seconds reached".format(timeout_seconds)
                        print("Terminating timed-out worker {} on {}".format(job.worker_id, job.target_id), flush=True)
                        _terminate_process(job, grace_seconds)
                continue

            job = active.get(worker_id)
            if job is None or job.process.pid != pid:
                continue
            del active[worker_id]
            finish_job(job, return_code)
            if not stop_scheduling:
                start_next(worker_id)

    except KeyboardInterrupt:
        print("Launcher interrupted; terminating active training process groups", file=sys.stderr, flush=True)
        for job in list(active.values()):
            record = state["targets"][job.target_id]
            record["status"] = "pending"
            record["worker_id"] = None
            record["message"] = "requeued after launcher interruption"
            _terminate_process(job, grace_seconds)
            _cleanup_staged_checkpoint(record)
        atomic_write_json(state_path, state)
        return 130
    finally:
        lock_stream.close()

    counts = _print_status_counts(state)
    if stop_scheduling:
        return 3
    if counts.get("pending", 0) or counts.get("timed_out", 0) or counts.get("failed", 0):
        return 2
    return 0


# Set by main before run_curriculum. Keeping retries out of the public call
# signature makes the scheduler easy to call from small local tests.
run_curriculum.retries = []


def dry_run(manifest, python_executable):
    training = manifest["training"]
    print("Validated {} targets; two workers share CUDA_VISIBLE_DEVICES={}.".format(
        len(manifest["targets"]), training["gpu"]
    ))
    print("The first two targets start together; later parent choices depend on completion order.")
    for target in manifest["targets"]:
        run_name = make_run_name(target, 1)
        placeholder = PACKAGE_DIR / "curriculum_runs" / manifest["name"] / "staged_checkpoints" / (run_name + "_parent.pth")
        command = build_command(python_executable, target, placeholder, run_name, training)
        print("{:02d} {}".format(target.manifest_index + 1, shell_join(command)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Persistent state/log/staging directory (default: curriculum_runs/<manifest name>)",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for train.py")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--terminate-grace-seconds", type=float, default=30.0)
    parser.add_argument(
        "--retry",
        action="append",
        choices=("timed_out", "failed", "all"),
        default=[],
        help="Requeue a prior terminal status; for example: --retry timed_out",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true", help="Validate the manifest and exit")
    mode.add_argument("--dry-run", action="store_true", help="Print deterministic commands without launching")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest, require_offsets=True, require_seed=True)
    except ManifestValidationError as exc:
        print("Manifest validation failed:", file=sys.stderr)
        for error in exc.errors:
            print("  - {}".format(error), file=sys.stderr)
        return 2

    if args.validate:
        print("Manifest is valid: {} unique targets, seed {}".format(
            len(manifest["targets"]), manifest["seed_checkpoint"]
        ))
        return 0
    if args.dry_run:
        dry_run(manifest, args.python)
        return 0

    state_dir = args.state_dir
    if state_dir is None:
        state_dir = PACKAGE_DIR / "curriculum_runs" / manifest["name"]
    run_curriculum.retries = list(args.retry)
    try:
        return run_curriculum(
            manifest,
            state_dir,
            args.python,
            timeout_seconds=args.timeout_seconds,
            grace_seconds=args.terminate_grace_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print("Launcher failed: {}".format(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
