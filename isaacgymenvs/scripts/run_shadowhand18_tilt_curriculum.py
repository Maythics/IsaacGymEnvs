#!/usr/bin/env python3
"""Run a resumable multi-GPU fixed-tilt hand curriculum.

Workers share one FIFO target queue. As soon as a worker succeeds or
uses up its per-target budget, it claims the next target. A new target resumes
from the geodesically closest checkpoint that has *already* succeeded; a policy
that is still training is deliberately not considered a parent.
"""

from __future__ import print_function

import argparse
import copy
import concurrent.futures
import datetime
import fcntl
import itertools
import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE_DIR / "curricula" / "shadowhand18_tilt_42.yaml"
NEGATIVE_REWARD_SENTINEL = -1000000000.0
STATE_VERSION = 5
PROMOTION_MODES = ("reward_only", "reward_and_physical", "physical_only")
OFFSET_PROBE_DEFAULTS = {
    "enabled": False,
    "num_envs": 65536,
    "episodes": 65536,
    "step_m": 0.03,
    "timeout_seconds": 1800,
    "seed": 314159,
}


class ManifestValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def promotion_mode(training):
    """Return the explicitly configured checkpoint promotion rule."""
    return str(training.get("promotion_mode", "physical_only"))


def reward_passes_score_gate(training, reward):
    return reward is not None and float(reward) > float(training["score_to_win"])


def certification_passes(training, certification):
    if certification is None:
        return False
    rate = certification.get(
        "minimum_object_success_rate",
        certification.get("retained_success_rate", -1.0),
    )
    return float(rate) >= float(training["certification_success_rate"])


def checkpoint_passes_promotion(training, reward, certification=None):
    mode = promotion_mode(training)
    if mode == "reward_only":
        return reward_passes_score_gate(training, reward)
    if mode == "reward_and_physical":
        return (
            reward_passes_score_gate(training, reward)
            and certification_passes(training, certification)
        )
    if mode == "physical_only":
        return certification_passes(training, certification)
    raise ValueError("unsupported promotion_mode: {}".format(mode))


@dataclass(frozen=True)
class Target:
    target_id: str
    phi_deg: float
    theta_deg: float
    axis: tuple
    object_offset: object
    manifest_index: int
    offset_source: str = "manifest"
    stage: str = "block"
    object_type_pool: tuple = ()
    # The physical yaw is chosen per parent transition. Zero preserves the
    # historical manifest pose for direct/manual task invocation.
    base_yaw_deg: float = 0.0

    @property
    def direction(self):
        return spherical_direction(self.theta_deg, self.phi_deg)

    @property
    def base_rotation(self):
        return base_rotation_quat(self.theta_deg, self.axis, self.base_yaw_deg)


@dataclass(frozen=True)
class ParentCandidate:
    source_id: str
    checkpoint: Path
    direction: tuple
    completion_seq: int
    manifest_index: int
    base_rotation: tuple = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ParentTransition:
    parent: ParentCandidate
    child_base_yaw_deg: float
    hand_rotation_distance: float
    gravity_direction_distance: float


@dataclass(frozen=True)
class PendingTargetSelection:
    """One pending target plus its best currently available parent transition."""
    target: Target
    transition: ParentTransition
    within_transition_limit: bool


@dataclass
class ActiveJob:
    worker_id: int
    gpu_id: str
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


def quat_multiply(a, b):
    """Quaternion product for scalar-last tuples, independent of torch."""
    ax, ay, az, aw = (float(value) for value in a)
    bx, by, bz, bw = (float(value) for value in b)
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def axis_angle_quat(angle_deg, axis):
    axis = tuple(float(value) for value in axis)
    norm = math.sqrt(sum(value * value for value in axis))
    if norm < 1.0e-12:
        raise ValueError("base tilt axis must be non-zero")
    half_angle = math.radians(float(angle_deg)) * 0.5
    scale = math.sin(half_angle) / norm
    return (
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
        math.cos(half_angle),
    )


def base_rotation_quat(theta_deg, axis, base_yaw_deg=0.0):
    """World rotation applied to the default hand pose for one target."""
    tilt = axis_angle_quat(theta_deg, axis)
    yaw = axis_angle_quat(base_yaw_deg, (0.0, 0.0, 1.0))
    return quat_multiply(yaw, tilt)


def rotation_distance(a, b):
    """Shortest SO(3) distance between scalar-last unit quaternions."""
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def optimal_world_yaw(parent_rotation, target_zero_yaw_rotation):
    """Pick the gravity-preserving world-Z yaw closest to the parent pose.

    Left-multiplying a target by a world-Z yaw leaves world gravity unchanged,
    hence gravity expressed in the palm stays unchanged. It resolves the
    arbitrary roll/gauge discontinuity of the historical horizontal-axis
    representation, especially near the south pole.
    """
    parent = tuple(float(value) for value in parent_rotation)
    target = tuple(float(value) for value in target_zero_yaw_rotation)
    z_times_target = quat_multiply((0.0, 0.0, 1.0, 0.0), target)
    cosine_coefficient = sum(a * b for a, b in zip(parent, target))
    sine_coefficient = sum(a * b for a, b in zip(parent, z_times_target))
    half_yaw = math.atan2(sine_coefficient, cosine_coefficient)
    yaw_deg = math.degrees(2.0 * half_yaw)
    # A normalized representative keeps state and command output stable.
    yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
    child = base_rotation_quat(
        0.0, (0.0, 0.0, 1.0), yaw_deg
    )
    child = quat_multiply(child, target)
    return yaw_deg, rotation_distance(parent, child)


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
            relocated = _relocate_checkpoint(seed_checkpoint, raw.get("checkpoint_search_roots", []), path)
            if relocated is None:
                errors.append("seed checkpoint does not exist: {}".format(seed_checkpoint))
            else:
                seed_checkpoint = relocated

    try:
        seed_theta = float(seed_raw.get("theta_deg"))
        seed_phi = float(seed_raw.get("phi_deg"))
    except (TypeError, ValueError):
        errors.append("seed theta_deg and phi_deg must be numbers")
        seed_theta, seed_phi = 0.0, 0.0
    seed_axis = _numeric_vector(seed_raw.get("base_tilt_axis"), "seed.base_tilt_axis", errors)
    if seed_axis is not None:
        _validate_axis(seed_axis, seed_phi, "seed.base_tilt_axis", errors)
    try:
        seed_base_yaw = float(seed_raw.get("base_yaw_deg", 0.0))
        if not math.isfinite(seed_base_yaw):
            raise ValueError
    except (TypeError, ValueError):
        errors.append("seed.base_yaw_deg must be a finite number")
        seed_base_yaw = 0.0

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
        "timeout_seconds",
    )
    for key in required_positive_ints:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("training.{} must be a positive integer".format(key))
    gpu_ids = training.get("gpu_ids")
    if gpu_ids is None:
        gpu_ids = [training.get("gpu", "0")]
    if not isinstance(gpu_ids, list) or not gpu_ids:
        errors.append("training.gpu_ids must be a non-empty list")
        gpu_ids = [0]
    elif any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in gpu_ids):
        errors.append("training.gpu_ids entries must be integer or string CUDA ids")
    training["gpu_ids"] = [str(value) for value in gpu_ids]
    training["workers"] = len(training["gpu_ids"])
    # Retain the legacy scalar for older callers, but gpu_ids is authoritative.
    training["gpu"] = training["gpu_ids"][0]
    training.setdefault("horizon_length", 8)
    for key, default in (
        ("object_gravity_compensation_seconds", 0.2),
        ("object_gravity_ramp_seconds", 0.1),
        ("certification_episodes", 128),
        ("certification_num_envs", 128),
    ):
        training.setdefault(key, default)
    if float(training["object_gravity_compensation_seconds"]) < 0:
        errors.append("training.object_gravity_compensation_seconds must be non-negative")
    if float(training["object_gravity_ramp_seconds"]) < 0:
        errors.append("training.object_gravity_ramp_seconds must be non-negative")
    offset_probe = training.get("offset_probe", {})
    if offset_probe is None:
        offset_probe = {}
    if not isinstance(offset_probe, dict):
        errors.append("training.offset_probe must be a mapping")
        offset_probe = {}
    offset_probe = dict(offset_probe)
    for key, default in OFFSET_PROBE_DEFAULTS.items():
        offset_probe.setdefault(key, default)
    if not isinstance(offset_probe["enabled"], bool):
        errors.append("training.offset_probe.enabled must be true or false")
    for key in ("num_envs", "episodes", "timeout_seconds", "seed"):
        value = offset_probe[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append("training.offset_probe.{} must be a positive integer".format(key))
    step_m = offset_probe["step_m"]
    if (
        isinstance(step_m, bool)
        or not isinstance(step_m, (int, float))
        or not math.isfinite(float(step_m))
        or float(step_m) <= 0.0
    ):
        errors.append("training.offset_probe.step_m must be a positive finite number")
    training["offset_probe"] = offset_probe
    success_rate = training.setdefault("certification_success_rate", 0.60)
    if not isinstance(success_rate, (int, float)) or not 0.0 <= float(success_rate) <= 1.0:
        errors.append("training.certification_success_rate must be between 0 and 1")
    mode = training.setdefault("promotion_mode", "physical_only")
    if mode not in PROMOTION_MODES:
        errors.append(
            "training.promotion_mode must be one of {}".format(
                ", ".join(PROMOTION_MODES)
            )
        )
    max_transition = training.setdefault("max_parent_transition_deg", 30.0)
    if (
        isinstance(max_transition, bool)
        or not isinstance(max_transition, (int, float))
        or not math.isfinite(float(max_transition))
        or not 0.0 < float(max_transition) <= 180.0
    ):
        errors.append(
            "training.max_parent_transition_deg must be a finite number in (0, 180]"
        )
    threshold = training.get("score_to_win")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        errors.append("training.score_to_win must be numeric")
    discovered_min_reward = training.setdefault(
        "discovered_parent_min_reward", threshold if isinstance(threshold, (int, float)) else 0.0
    )
    if (
        isinstance(discovered_min_reward, bool)
        or not isinstance(discovered_min_reward, (int, float))
        or not math.isfinite(float(discovered_min_reward))
    ):
        errors.append("training.discovered_parent_min_reward must be a finite number")
    allow_unscored_discovered = training.setdefault(
        "allow_unscored_discovered_parents", False
    )
    if not isinstance(allow_unscored_discovered, bool):
        errors.append("training.allow_unscored_discovered_parents must be true or false")
    object_type = training.get("object_type")
    if not isinstance(object_type, str) or not object_type:
        errors.append("training.object_type must be a non-empty string")
    horizon_length = training.get("horizon_length")
    if isinstance(horizon_length, bool) or not isinstance(horizon_length, int) or horizon_length <= 0:
        errors.append("training.horizon_length must be a positive integer")
    elif (
        isinstance(training.get("num_envs"), int)
        and isinstance(training.get("minibatch_size"), int)
        and training["num_envs"] * horizon_length % training["minibatch_size"] != 0
    ):
        errors.append("training.minibatch_size must divide num_envs * horizon_length")

    resource_profiles = training.get("resource_profiles", [])
    if not isinstance(resource_profiles, list):
        errors.append("training.resource_profiles must be a list")
        resource_profiles = []
    normalized_profiles = []
    for index, profile in enumerate(resource_profiles):
        prefix = "training.resource_profiles[{}]".format(index)
        if not isinstance(profile, dict):
            errors.append("{} must be a mapping".format(prefix))
            continue
        normalized = {}
        for key in ("min_free_memory_mb", "num_envs", "minibatch_size"):
            value = profile.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append("{}.{} must be a positive integer".format(prefix, key))
            else:
                normalized[key] = value
        if len(normalized) == 3:
            if isinstance(horizon_length, int) and (
                normalized["num_envs"] * horizon_length
                % normalized["minibatch_size"] != 0
            ):
                errors.append(
                    "{}.minibatch_size must divide num_envs * horizon_length".format(prefix)
                )
            normalized_profiles.append(normalized)
    if len({profile["min_free_memory_mb"] for profile in normalized_profiles}) != len(normalized_profiles):
        errors.append("training.resource_profiles min_free_memory_mb values must be unique")
    training["resource_profiles"] = sorted(
        normalized_profiles, key=lambda profile: profile["min_free_memory_mb"], reverse=True
    )

    existing_start_rows = raw.get("existing_start_checkpoints", [])
    if not isinstance(existing_start_rows, list):
        errors.append("existing_start_checkpoints must be a list")
        existing_start_rows = []
    existing_start_checkpoints = []
    unavailable_existing_start_checkpoints = []
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
        optional = row.get("optional", False)
        if not isinstance(optional, bool):
            errors.append("{}.optional must be true or false".format(prefix))
            optional = False

        checkpoint_value = row.get("checkpoint")
        checkpoint_unavailable = False
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            errors.append("{}.checkpoint must be a path".format(prefix))
            checkpoint = path.parent / (source_id + "-missing.pth")
        else:
            checkpoint = Path(checkpoint_value).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = (path.parent / checkpoint).resolve()
            if not checkpoint.is_file():
                relocated = _relocate_checkpoint(
                    checkpoint, raw.get("checkpoint_search_roots", []), path
                )
                if relocated is None:
                    checkpoint_unavailable = True
                    if optional:
                        unavailable_existing_start_checkpoints.append({
                            "id": source_id,
                            "checkpoint": str(checkpoint),
                        })
                    elif require_seed:
                        errors.append("existing start checkpoint does not exist: {}".format(checkpoint))
                else:
                    checkpoint = relocated
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
        try:
            start_base_yaw = float(row.get("base_yaw_deg", 0.0))
            if not math.isfinite(start_base_yaw):
                raise ValueError
        except (TypeError, ValueError):
            errors.append("{}.base_yaw_deg must be a finite number".format(prefix))
            start_base_yaw = 0.0
        recorded_reward = row.get("recorded_reward")
        if isinstance(recorded_reward, bool) or not isinstance(recorded_reward, (int, float)):
            errors.append("{}.recorded_reward must be numeric".format(prefix))
        elif isinstance(threshold, (int, float)) and float(recorded_reward) <= float(threshold):
            errors.append(
                "{}.recorded_reward must be strictly above training.score_to_win".format(prefix)
            )
        # Optional historical runs may be kept in the manifest even while a
        # different machine has not mounted their storage.  They remain
        # eligible automatically after relocation succeeds, but never block a
        # complete curriculum that has other valid warm starts.
        if not checkpoint_unavailable:
            existing_start_checkpoints.append(
                ParentCandidate(
                    source_id,
                    checkpoint,
                    spherical_direction(start_theta, start_phi),
                    0,
                    -9999 + index,
                    base_rotation_quat(start_theta, start_axis or (0.0, 1.0, 0.0), start_base_yaw),
                )
            )

    target_rows = raw.get("targets")
    if not isinstance(target_rows, list):
        errors.append("targets must be a list")
        target_rows = []
    if len(target_rows) < 42:
        errors.append("targets must contain at least the 42 verified rows, found {}".format(len(target_rows)))

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
        try:
            base_yaw = float(row.get("base_yaw_deg", 0.0))
            if not math.isfinite(base_yaw):
                raise ValueError
        except (TypeError, ValueError):
            errors.append("{}.base_yaw_deg must be a finite number".format(prefix))
            base_yaw = 0.0
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
        targets.append(Target(
            target_id, phi, theta, axis or (0.0, 1.0, 0.0), offset, index,
            str(row.get("offset_source", target_id)), base_yaw_deg=base_yaw,
        ))

    expected_order = _expected_target_order()
    actual_order = [(int(round(t.phi_deg)) % 360, int(round(t.theta_deg))) for t in targets]
    if actual_order[:42] != expected_order:
        errors.append("the first 42 targets must retain the verified serpentine curriculum")

    ring_counts = {}
    pole_count = 0
    for target in targets:
        theta_key = int(round(target.theta_deg))
        if theta_key in (0, 180):
            pole_count += 1
        else:
            ring_counts[theta_key] = ring_counts.get(theta_key, 0) + 1
    if len(targets) == 42 and (pole_count != 2 or any(ring_counts.get(theta) != 8 for theta in (30, 60, 90, 120, 150))):
        errors.append("coverage must be two poles plus five 8-azimuth rings")

    if errors:
        raise ManifestValidationError(errors)

    density = raw.get("density", {}) or {}
    coverage = float(density.get("coverage_radius_deg", 0.0))
    if coverage > 0.0:
        targets = densify_targets(targets, coverage)

    multi_object_stage = raw.get("multi_object_stage", {}) or {}
    if not isinstance(multi_object_stage, dict):
        errors.append("multi_object_stage must be a mapping")
    elif bool(multi_object_stage.get("enabled", False)):
        object_types = multi_object_stage.get("object_types", ["block", "egg", "pen"])
        if (
            not isinstance(object_types, list)
            or len(object_types) < 2
            or any(not isinstance(value, str) or not value for value in object_types)
        ):
            errors.append("multi_object_stage.object_types must contain at least two names")
        else:
            for source in list(targets):
                targets.append(Target(
                    "multi_" + source.target_id,
                    source.phi_deg,
                    source.theta_deg,
                    source.axis,
                    source.object_offset,
                    len(targets),
                    source.offset_source,
                    "multi_object",
                    tuple(object_types),
                    source.base_yaw_deg,
                ))

    if errors:
        raise ManifestValidationError(errors)

    manifest = {
        "path": path,
        "name": name,
        "seed_checkpoint": seed_checkpoint,
        "seed_direction": spherical_direction(seed_theta, seed_phi),
        "seed_theta_deg": seed_theta,
        "seed_phi_deg": seed_phi,
        "seed_axis": seed_axis,
        "seed_base_yaw_deg": seed_base_yaw,
        "existing_start_checkpoints": existing_start_checkpoints,
        "unavailable_existing_start_checkpoints": unavailable_existing_start_checkpoints,
        "training": copy.deepcopy(training),
        "targets": targets,
        "checkpoint_search_roots": _resolve_search_roots(
            raw.get("checkpoint_search_roots", []), path
        ),
    }
    manifest["discovered_checkpoints"] = discover_checkpoint_parents(manifest)
    return manifest


def _resolve_search_roots(values, manifest_path):
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        root = Path(str(value)).expanduser()
        if not root.is_absolute():
            root = (Path(manifest_path).parent / root).resolve()
        result.append(root)
    return result


def _relocate_checkpoint(original, search_roots, manifest_path):
    """Find a moved checkpoint by run directory and file name."""
    original = Path(original)
    run_name = original.parent.parent.name if original.parent.name == "nn" else None
    matches = []
    for root in _resolve_search_roots(search_roots, manifest_path):
        if not root.is_dir():
            continue
        if run_name:
            direct = root / run_name / "nn" / original.name
            if direct.is_file():
                # The run directory is part of the manifest identity.  Prefer
                # it over the generic filename fallback: names such as
                # ``Shadowhand18Tilted.pth`` occur in many runs, so treating
                # the direct match as one item in that global set makes a
                # perfectly unambiguous moved checkpoint look ambiguous.
                return direct.resolve()
        matches.extend(path.resolve() for path in root.glob("**/{}".format(original.name)) if path.is_file())
    unique = sorted(set(matches), key=str)
    return unique[0] if len(unique) == 1 else None


def _deep_get(value, keys, default=None):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def checkpoint_reward_from_filename(checkpoint):
    """Read RL-Games' optional ``_rew_<value>`` suffix without loading it."""
    match = re.search(
        r"(?:^|_)rew_(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$",
        Path(checkpoint).stem,
    )
    return float(match.group(1)) if match else None


def discover_checkpoint_parents(manifest):
    """Catalog compatible legacy runs from their resolved task configs.

    All checkpoint files remain load-compatible because the fixed-tilt tasks
    retain their original observation/action interfaces.  Automatically found
    parents are limited to files whose RL-Games reward suffix clears the
    configured score gate; explicit manifest parents remain supported for
    legacy generic filenames.
    """
    expected_wuji = manifest["name"].startswith("wujihand")
    expected_names = (
        {"WujiHand", "WujiHandFixedTilt"}
        if expected_wuji else {"Shadowhand18", "Shadowhand18Tilted"}
    )
    explicit_paths = {str(manifest["seed_checkpoint"].resolve())}
    explicit_paths.update(
        str(candidate.checkpoint.resolve())
        for candidate in manifest.get("existing_start_checkpoints", [])
    )
    candidates = []
    minimum_reward = float(manifest["training"]["discovered_parent_min_reward"])
    allow_unscored = bool(manifest["training"]["allow_unscored_discovered_parents"])
    seen = set(explicit_paths)
    for root in manifest.get("checkpoint_search_roots", []):
        if not root.is_dir():
            continue
        for config_path in sorted(root.glob("**/config.yaml")):
            try:
                with config_path.open("r") as stream:
                    config = yaml.safe_load(stream)
            except (OSError, yaml.YAMLError):
                continue
            task_name = _deep_get(config, ("task", "name"))
            if task_name is None:
                task_name = config.get("task_name") if isinstance(config, dict) else None
            if task_name not in expected_names:
                continue
            env = _deep_get(config, ("task", "env"), {})
            try:
                theta = float(env.get("baseTiltAngleDeg", 0.0))
                axis = env.get("baseTiltAxis", [0.0, 1.0, 0.0])
                axis_x, axis_y = float(axis[0]), float(axis[1])
                phi = math.degrees(math.atan2(-axis_x, axis_y)) % 360.0
                base_yaw = float(env.get("baseYawDeg", 0.0))
            except (TypeError, ValueError, IndexError):
                continue
            nn_dir = config_path.parent / "nn"
            if not nn_dir.is_dir():
                continue
            for checkpoint in sorted(nn_dir.glob("*.pth")):
                resolved = str(checkpoint.resolve())
                if resolved in seen:
                    continue
                filename_reward = checkpoint_reward_from_filename(checkpoint)
                if filename_reward is None and not allow_unscored:
                    continue
                if filename_reward is not None and filename_reward <= minimum_reward:
                    continue
                seen.add(resolved)
                # Full-state checkpoint structure is verified lazily when the
                # candidate is staged, avoiding eager multi-gigabyte reads.
                try:
                    completion_seq = int(checkpoint.stat().st_mtime)
                except OSError:
                    completion_seq = 0
                candidates.append(ParentCandidate(
                    "discovered:{}:{}".format(config_path.parent.name, checkpoint.stem),
                    checkpoint.resolve(), spherical_direction(theta, phi),
                    completion_seq, -20000 - len(candidates),
                    base_rotation_quat(theta, axis, base_yaw),
                ))
    return candidates


def _direction_to_angles(direction):
    x, y, z = (float(v) for v in direction)
    theta = math.degrees(math.acos(max(-1.0, min(1.0, z))))
    phi = math.degrees(math.atan2(y, x)) % 360.0
    return theta, phi


def densify_targets(verified_targets, coverage_radius_deg, probe_count=20000):
    """Preserve verified targets and add evenly spread maximin directions."""
    if coverage_radius_deg <= 0.0:
        return list(verified_targets)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    indices = np.arange(probe_count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / float(probe_count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = indices * golden_angle
    probes = np.stack((radius * np.cos(phi), radius * np.sin(phi), z), axis=1)
    verified_dirs = np.asarray([target.direction for target in verified_targets], dtype=np.float64)
    nearest_dot = np.max(probes.dot(verified_dirs.T), axis=1)
    coverage_cos = math.cos(math.radians(coverage_radius_deg))
    additions = []
    while float(np.min(nearest_dot)) < coverage_cos:
        candidate = probes[int(np.argmin(nearest_dot))]
        additions.append(tuple(float(v) for v in candidate))
        nearest_dot = np.maximum(nearest_dot, probes.dot(candidate))

    resolved = list(verified_targets)
    verified = list(verified_targets)
    for direction_number, direction in enumerate(additions, start=1):
        offset_parent = min(
            verified,
            key=lambda target: angular_distance(direction, target.direction),
        )
        theta, point_phi = _direction_to_angles(direction)
        resolved.append(Target(
            "dense_{:03d}".format(direction_number),
            point_phi,
            theta,
            expected_axis(point_phi),
            offset_parent.object_offset,
            len(resolved),
            offset_parent.target_id,
        ))
    return resolved


def _validate_axis(axis, phi_deg, field, errors):
    norm = math.sqrt(sum(component * component for component in axis))
    if abs(norm - 1.0) > 1e-5:
        errors.append("{} must be normalized (norm is {:.8f})".format(field, norm))
    expected = expected_axis(phi_deg)
    if max(abs(actual - wanted) for actual, wanted in zip(axis, expected)) > 1e-5:
        errors.append("{} does not match [-sin(phi), cos(phi), 0]".format(field))


def choose_parent_transition(target, candidates):
    if not candidates:
        raise RuntimeError("no successful checkpoint is available for {}".format(target.target_id))

    def candidate_tier(candidate):
        # Newly physically certified curriculum results outrank explicit known
        # starts, which in turn outrank merely discovered (compatible but not
        # yet certified) files when angular distance is identical.
        if candidate.manifest_index >= 0:
            return 2
        if candidate.manifest_index >= -10000:
            return 1
        return 0

    transitions = []
    target_base_rotation = target.base_rotation
    for candidate in candidates:
        additional_yaw, hand_distance = optimal_world_yaw(
            candidate.base_rotation, target_base_rotation
        )
        child_yaw = (target.base_yaw_deg + additional_yaw + 180.0) % 360.0 - 180.0
        transitions.append(ParentTransition(
            candidate,
            child_yaw,
            hand_distance,
            angular_distance(target.direction, candidate.direction),
        ))
    return min(
        transitions,
        key=lambda transition: (
            round(transition.hand_rotation_distance, 12),
            -candidate_tier(transition.parent),
            -transition.parent.completion_seq,
            transition.parent.manifest_index,
        ),
    )


def choose_nearest_parent(target, candidates):
    """Backward-compatible parent-only selector used by small tests/callers."""
    return choose_parent_transition(target, candidates).parent


def continuous_viewer_targets(manifest, targets):
    """Assign deterministic yaw values for a manually inspected target path.

    This is deliberately separate from live scheduling: the latter uses the
    checkpoint that actually won parent selection.  The viewer path instead
    uses its preceding displayed pose as the parent, so an ordered list of
    commands does not reintroduce a world-frame gauge jump merely because its
    next target is written with a different horizontal tilt axis.
    """
    previous = ParentCandidate(
        "viewer_seed",
        manifest["seed_checkpoint"],
        manifest["seed_direction"],
        0,
        -10000,
        base_rotation_quat(
            manifest["seed_theta_deg"],
            manifest["seed_axis"] or (0.0, 1.0, 0.0),
            manifest.get("seed_base_yaw_deg", 0.0),
        ),
    )
    resolved = []
    for index, target in enumerate(targets, start=1):
        transition = choose_parent_transition(target, [previous])
        displayed = replace(target, base_yaw_deg=transition.child_base_yaw_deg)
        resolved.append(displayed)
        previous = ParentCandidate(
            displayed.target_id,
            Path("viewer-continuity-placeholder.pth"),
            displayed.direction,
            index,
            displayed.manifest_index,
            displayed.base_rotation,
        )
    return resolved


def available_parents(manifest, state):
    candidates = [
        ParentCandidate(
            "seed",
            manifest["seed_checkpoint"],
            manifest["seed_direction"],
            0,
            -10000,
            base_rotation_quat(
                manifest["seed_theta_deg"],
                manifest["seed_axis"] or (0.0, 1.0, 0.0),
                manifest.get("seed_base_yaw_deg", 0.0),
            ),
        )
    ]
    candidates.extend(manifest.get("existing_start_checkpoints", []))
    candidates.extend(manifest.get("discovered_checkpoints", []))
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
                base_rotation_quat(
                    target.theta_deg,
                    target.axis,
                    float(record.get("base_yaw_deg", target.base_yaw_deg)),
                ),
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
        state = _load_full_checkpoint(torch, temporary_path)
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
    state = _load_full_checkpoint(torch, checkpoint)
    if not isinstance(state, dict) or "last_mean_rewards" not in state:
        raise ValueError("checkpoint {} has no last_mean_rewards".format(checkpoint))
    reward = state["last_mean_rewards"]
    if hasattr(reward, "item"):
        reward = reward.item()
    return float(reward)


def _load_full_checkpoint(torch_module, checkpoint):
    """Load a trusted RL-Games full-state checkpoint across PyTorch releases."""
    try:
        # Explicitly retain full-state loading: curriculum staging needs model,
        # optimizer, RMS, and other state, not merely tensor weights. Passing
        # the flag silences PyTorch 2.4+'s future-default warning.
        return torch_module.load(
            str(checkpoint), map_location="cpu", weights_only=False
        )
    except TypeError:
        # Isaac Gym installations commonly carry an older PyTorch that predates
        # the weights_only keyword.
        return torch_module.load(str(checkpoint), map_location="cpu")


def _format_number(value):
    value = float(value)
    if abs(value) < 5e-12:
        value = 0.0
    return "{:.8g}".format(value)


def _format_vector(vector):
    return "[{}]".format(",".join(_format_number(value) for value in vector))


def offset_probe_candidates(center, step_m):
    """Return the centered 3 x 3 x 3 palm-local offset stencil.

    The manifest value remains the center/initial guess.  Keeping all three
    coordinates in the stencil lets a probe recover diagonal corrections
    without inventing an angle-dependent offset rule.
    """
    center = tuple(float(value) for value in center)
    if len(center) != 3:
        raise ValueError("offset probe center must contain three values")
    step_m = float(step_m)
    return tuple(
        tuple(center[axis] + delta[axis] * step_m for axis in range(3))
        for delta in itertools.product((-1.0, 0.0, 1.0), repeat=3)
    )


def offset_probe_seed(target_id, attempt, base_seed):
    """Stable per-target probe seed; unlike hash(), this survives restarts."""
    value = int(base_seed) + 1009 * int(attempt)
    for character in str(target_id).encode("utf-8"):
        value = (value * 33 + character) % 2147483647
    return value


def offset_probe_rank(summary):
    """Prefer successful, long-lived, then high-reward parent rollouts."""
    return (
        float(summary.get("retained_success_rate", 0.0)),
        float(summary.get("mean_episode_steps", 0.0)),
        float(summary.get("mean_episode_reward", float("-inf"))),
    )


def select_offset_probe_winner(result):
    summaries = result.get("per_offset", [])
    if not summaries:
        raise ValueError("offset probe result contains no per_offset summaries")
    winner = max(
        summaries,
        key=lambda summary: (
            offset_probe_rank(summary),
            -int(summary.get("candidate_index", 0)),
        ),
    )
    offset = winner.get("offset")
    if not isinstance(offset, list) or len(offset) != 3:
        raise ValueError("offset probe winner has no three-number offset")
    return tuple(float(value) for value in offset), winner


def reusable_offset_probe(record, parent_checkpoint, candidates, probe_config):
    """A probe is valid only for the exact warm start and stencil settings."""
    probe = record.get("offset_probe")
    if not isinstance(probe, dict) or probe.get("status") != "succeeded":
        return None
    if probe.get("parent_checkpoint") != str(Path(parent_checkpoint).resolve()):
        return None
    if probe.get("candidates") != [list(candidate) for candidate in candidates]:
        return None
    expected = {
        key: probe_config[key]
        for key in ("num_envs", "episodes", "step_m", "seed")
    }
    if probe.get("settings") != expected:
        return None
    selected = probe.get("selected_offset")
    if not isinstance(selected, list) or len(selected) != 3:
        return None
    return tuple(float(value) for value in selected)


def build_command(python_executable, target, staged_checkpoint, run_name, training):
    command = [
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
        "task.env.baseYawDeg={}".format(_format_number(target.base_yaw_deg)),
        "task.env.objectGravityCompensationSeconds={}".format(
            _format_number(training["object_gravity_compensation_seconds"])
        ),
        "task.env.objectGravityRampSeconds={}".format(
            _format_number(training["object_gravity_ramp_seconds"])
        ),
        "train.params.config.score_to_win={}".format(_format_number(training["score_to_win"])),
        "train.params.config.save_best_after={}".format(training["save_best_after"]),
        "experiment={}".format(run_name),
        "+full_experiment_name={}".format(run_name),
    ]
    if target.object_type_pool:
        command.append(
            "task.env.objectTypePool=[{}]".format(
                ",".join(target.object_type_pool)
            )
        )
    else:
        command.append("task.env.objectTypePool=[]")
    return command


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
                "stage": target.stage,
                "object_type_pool": list(target.object_type_pool),
                "base_yaw_deg": target.base_yaw_deg,
                "hand_rotation_distance_deg": None,
                "gravity_direction_distance_deg": None,
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
            certification = record.get("certification")
            if checkpoint_passes_promotion(manifest["training"], reward, certification):
                state["completion_counter"] = int(state.get("completion_counter", 0)) + 1
                record["status"] = "succeeded"
                record["best_reward"] = reward
                record["completed_at"] = utc_now()
                record["completion_seq"] = state["completion_counter"]
                record["message"] = "recovered checkpoint that passed the {} promotion gate".format(
                    promotion_mode(manifest["training"])
                )
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

    # A prior evaluator bug must never waste an otherwise winning checkpoint.
    # In reward-only mode an output above score_to_win is sufficient evidence
    # for promotion, irrespective of the status left by the old evaluator.
    if promotion_mode(manifest["training"]) == "reward_only":
        for record in state["targets"].values():
            if record["status"] not in ("pending", "timed_out", "failed"):
                continue
            output_checkpoint = record.get("output_checkpoint")
            if not output_checkpoint:
                continue
            try:
                reward = read_checkpoint_reward(output_checkpoint)
            except (OSError, RuntimeError, ValueError):
                continue
            if not reward_passes_score_gate(manifest["training"], reward):
                continue
            state["completion_counter"] = int(state.get("completion_counter", 0)) + 1
            record["status"] = "succeeded"
            record["best_reward"] = reward
            record["completed_at"] = utc_now()
            record["completion_seq"] = state["completion_counter"]
            record["worker_id"] = None
            record.pop("pid", None)
            record["message"] = (
                "recovered existing output by score gate: {:.6g} > {:.6g}"
            ).format(reward, manifest["training"]["score_to_win"])
            _cleanup_staged_checkpoint(record)
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
    """Return the legacy FIFO candidate for inspection/tests only.

    Live training uses ``choose_next_pending_transition`` below so every new
    job expands from the closest successful parent rather than from YAML row
    order.  Keeping this tiny helper preserves callers that only need to
    inspect the current stage barrier.
    """
    active_stages = {
        target.stage
        for target in manifest["targets"]
        if state["targets"][target.target_id]["status"] in ("pending", "running")
    }
    if not active_stages:
        return None
    stage_order = {"block": 0, "multi_object": 1}
    current_stage = min(active_stages, key=lambda value: stage_order.get(value, 999))
    for target in manifest["targets"]:
        if (
            target.stage == current_stage
            and state["targets"][target.target_id]["status"] == "pending"
        ):
            return target
    return None


def choose_next_pending_transition(manifest, state):
    """Choose the nearest trainable pending target in the current stage.

    All pending targets in the active stage are scored against the *actual*
    available parent checkpoints.  The score is the yaw-optimized hand SO(3)
    distance, so the world-frame MLP input changes as little as possible.
    Prefer transitions inside max_parent_transition_deg (30 degrees by
    default). If no such parent is available, immediately use the globally
    nearest one as an explicit, logged fallback. The threshold is a soft
    continuity preference rather than a worker-blocking requirement: a single
    timeout must never strand all later angles or leave GPUs idle.
    """
    active_stages = {
        target.stage
        for target in manifest["targets"]
        if state["targets"][target.target_id]["status"] in ("pending", "running")
    }
    if not active_stages:
        return None
    stage_order = {"block": 0, "multi_object": 1}
    current_stage = min(active_stages, key=lambda value: stage_order.get(value, 999))
    parents = available_parents(manifest, state)
    pending = [
        target
        for target in manifest["targets"]
        if target.stage == current_stage
        and state["targets"][target.target_id]["status"] == "pending"
    ]
    if not pending:
        return None

    transition_limit = math.radians(
        float(manifest["training"]["max_parent_transition_deg"])
    )
    candidates = [
        (target, choose_parent_transition(target, parents)) for target in pending
    ]
    within_limit = [
        item for item in candidates
        if item[1].hand_rotation_distance <= transition_limit + 1e-12
    ]
    pool = within_limit or candidates
    target, transition = min(
        pool,
        key=lambda item: (
            round(item[1].hand_rotation_distance, 12),
            round(item[1].gravity_direction_distance, 12),
            item[0].manifest_index,
        ),
    )
    return PendingTargetSelection(target, transition, bool(within_limit))


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


def gpu_free_memory_mb(training):
    """Return current free memory for configured physical GPU ids.

    If nvidia-smi is unavailable (for example inside a restricted container),
    values are ``None`` and the manifest fallback profile is used.
    """
    configured = [str(value) for value in training["gpu_ids"]]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=10.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {gpu_id: None for gpu_id in configured}
    free_by_id = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            try:
                free_by_id[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return {gpu_id: free_by_id.get(gpu_id, 0) for gpu_id in configured}


def available_gpu_ids(training):
    """Return configured devices with enough currently free memory."""
    free_by_id = gpu_free_memory_mb(training)
    minimum_free = int(training.get("min_free_memory_mb", 4096))
    return [
        gpu_id for gpu_id in training["gpu_ids"]
        if free_by_id.get(gpu_id) is None or free_by_id.get(gpu_id, 0) >= minimum_free
    ]


def training_profile_for_gpu(training, free_memory_mb):
    """Resolve env/minibatch sizes without changing optimizer/checkpoint state.

    Profiles are selected independently for each worker because usable GPUs can
    differ in capacity or current load. If memory cannot be queried, the
    manifest's top-level num_envs/minibatch_size values remain the safe fallback.
    """
    resolved = copy.deepcopy(training)
    selected = None
    if free_memory_mb is not None:
        for profile in training.get("resource_profiles", []):
            if free_memory_mb >= profile["min_free_memory_mb"]:
                selected = profile
                break
    if selected is not None:
        resolved["num_envs"] = selected["num_envs"]
        resolved["minibatch_size"] = selected["minibatch_size"]
        resolved["selected_resource_profile"] = copy.deepcopy(selected)
    else:
        resolved["selected_resource_profile"] = {
            "min_free_memory_mb": None,
            "num_envs": resolved["num_envs"],
            "minibatch_size": resolved["minibatch_size"],
        }
    return resolved


def evaluate_checkpoint(python_executable, target, checkpoint, result_path,
                        training, gpu_id, task_name, episodes=None, num_envs=None,
                        offset_candidates=None, seed=None, timeout_seconds=None):
    """Run the regular evaluator, optionally assigning one offset per env.

    Normal certification continues to pass a scalar target offset.  Offset
    probing supplies the full stencil in one large vectorized evaluator run.
    """
    command = [
        str(python_executable),
        str(PACKAGE_DIR / "scripts" / "evaluate_tilted_policy.py"),
        "--task", task_name,
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--result", str(Path(result_path).resolve()),
        "--episodes", str(
            training["certification_episodes"] if episodes is None else episodes
        ),
        "--num-envs", str(
            training["certification_num_envs"] if num_envs is None else num_envs
        ),
        "--episode-length", str(training["episode_length"]),
        "--angle-deg", _format_number(target.theta_deg),
        # CSV vectors may begin with a minus sign.  Use --name=value so
        # argparse cannot mistake them for a following option.
        "--axis={}".format(",".join(_format_number(value) for value in target.axis)),
        "--base-yaw-deg", _format_number(target.base_yaw_deg),
        "--offset={}".format(
            ",".join(_format_number(value) for value in target.object_offset)
        ),
        "--gravity-hold-seconds", _format_number(
            training["object_gravity_compensation_seconds"]
        ),
        "--gravity-ramp-seconds", _format_number(
            training["object_gravity_ramp_seconds"]
        ),
        "--object-type", str(training["object_type"]),
        "--headless",
    ]
    if offset_candidates:
        command.append(
            "--offset-candidates={}".format(
                ";".join(
                ",".join(_format_number(value) for value in candidate)
                for candidate in offset_candidates
                )
            )
        )
    if seed is not None:
        command.extend(("--seed", str(int(seed))))
    if target.object_type_pool:
        command.extend(("--object-type-pool", ",".join(target.object_type_pool)))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completed = subprocess.run(
        command,
        cwd=str(PACKAGE_DIR),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=float(
            training.get("certification_timeout_seconds", 1800)
            if timeout_seconds is None else timeout_seconds
        ),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "physical evaluator exited {}:\n{}".format(
                completed.returncode, completed.stdout[-4000:]
            )
        )
    with Path(result_path).open("r") as stream:
        return json.load(stream)


def run_offset_probe(python_executable, target, parent_checkpoint, record,
                     state_dir, training, gpu_id, task_name, attempt):
    """Probe the parent on a local 3 cm offset stencil before fine-tuning."""
    config = training["offset_probe"]
    candidates = offset_probe_candidates(target.object_offset, config["step_m"])
    parent_checkpoint = Path(parent_checkpoint).resolve()
    reused = reusable_offset_probe(record, parent_checkpoint, candidates, config)
    if reused is not None:
        return reused, True

    seed = offset_probe_seed(target.target_id, attempt, config["seed"])
    result_path = (
        Path(state_dir) / "offset_probes"
        / "{}_{:02d}.json".format(target.target_id, int(attempt))
    )
    result = evaluate_checkpoint(
        python_executable,
        target,
        parent_checkpoint,
        result_path,
        training,
        gpu_id,
        task_name,
        episodes=config["episodes"],
        num_envs=config["num_envs"],
        offset_candidates=candidates,
        seed=seed,
        timeout_seconds=config["timeout_seconds"],
    )
    selected_offset, winner = select_offset_probe_winner(result)
    record["offset_probe"] = {
        "status": "succeeded",
        "parent_checkpoint": str(parent_checkpoint),
        "candidates": [list(candidate) for candidate in candidates],
        "settings": {
            key: config[key]
            for key in ("num_envs", "episodes", "step_m", "seed")
        },
        "seed": seed,
        "result_path": str(result_path),
        "selected_offset": list(selected_offset),
        "winner": winner,
        "completed_at": utc_now(),
    }
    return selected_offset, False


def _is_recertifiable_failure(record):
    """Whether an existing output only failed while running the evaluator.

    A trainer may have already produced a useful checkpoint when a launcher
    bug, an interrupted evaluator, or a transient evaluator failure marks the
    target failed.  Such a checkpoint must be evaluated again, not trained
    again from scratch.
    """
    if record.get("status") != "failed":
        return False
    checkpoint = record.get("output_checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        return False
    message = str(record.get("message") or "")
    return (
        "physical certification failed:" in message
        or "physical recertification failed:" in message
    )


def recertify_failed_outputs(manifest, state_dir, python_executable):
    """Re-run only failed physical certifications, preserving checkpoints.

    This is intentionally a separate operation from ``--retry failed``:
    retrying trains again, while recertification consumes the already-produced
    checkpoint and updates its persistent state in place.
    """
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = acquire_launcher_lock(state_dir)
    try:
        state_path = state_dir / "state.json"
        state = load_or_create_state(manifest, state_path, [])
        training = manifest["training"]
        free_memory_by_gpu = gpu_free_memory_mb(training)
        minimum_free_memory = int(training.get("min_free_memory_mb", 4096))
        gpu_ids = [
            gpu_id for gpu_id in training["gpu_ids"]
            if free_memory_by_gpu.get(gpu_id) is None
            or free_memory_by_gpu.get(gpu_id, 0) >= minimum_free_memory
        ]
        if not gpu_ids:
            raise RuntimeError(
                "none of the configured GPU ids {} currently has enough free memory".format(
                    training["gpu_ids"]
                )
            )

        targets_by_id = {target.target_id: target for target in manifest["targets"]}
        candidates = [
            (target_id, record)
            for target_id, record in state["targets"].items()
            if _is_recertifiable_failure(record)
        ]
        if not candidates:
            print("No failed physical certifications with an existing output checkpoint to re-evaluate.")
            return 0

        task_name = (
            "WujiHandFixedTilt"
            if manifest["name"].startswith("wujihand")
            else "Shadowhand18Tilted"
        )
        evaluations_dir = state_dir / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Re-certifying {} existing checkpoint(s) on GPU(s): {}".format(
                len(candidates), ", ".join(gpu_ids)
            ),
            flush=True,
        )

        def evaluate_one(target_id, record, gpu_id):
            target = targets_by_id[target_id]
            target = replace(
                target,
                base_yaw_deg=float(record.get("base_yaw_deg", target.base_yaw_deg)),
            )
            run_name = record.get("run_name") or target_id
            result_path = evaluations_dir / (run_name + ".json")
            certification = evaluate_checkpoint(
                python_executable,
                target,
                Path(record["output_checkpoint"]),
                result_path,
                training,
                gpu_id,
                task_name,
            )
            return target_id, certification, result_path

        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {
                executor.submit(evaluate_one, target_id, record, gpu_ids[index % len(gpu_ids)]): target_id
                for index, (target_id, record) in enumerate(candidates)
            }
            for future in concurrent.futures.as_completed(futures):
                target_id = futures[future]
                record = state["targets"][target_id]
                try:
                    _, certification, result_path = future.result()
                    certified_rate = float(
                        certification.get(
                            "minimum_object_success_rate",
                            certification.get("retained_success_rate", -1.0),
                        )
                    )
                    record["certification"] = certification
                    record["certification_path"] = str(result_path)
                    record["completed_at"] = utc_now()
                    if certified_rate >= float(training["certification_success_rate"]):
                        state["completion_counter"] = int(
                            state.get("completion_counter", 0)
                        ) + 1
                        record["status"] = "succeeded"
                        record["completion_seq"] = state["completion_counter"]
                        record["message"] = (
                            "physical retained-success gate passed after recertification: "
                            "{:.3f} >= {:.3f}"
                        ).format(
                            certified_rate, training["certification_success_rate"]
                        )
                        print(
                            "Re-certification succeeded on {}: retained success {:.1%}".format(
                                target_id, certified_rate
                            ),
                            flush=True,
                        )
                    else:
                        failures += 1
                        record["message"] = (
                            "physical retained-success gate failed after recertification: "
                            "{:.3f} < {:.3f}"
                        ).format(
                            certified_rate, training["certification_success_rate"]
                        )
                        print(
                            "Re-certification did not pass on {}: retained success {:.1%}".format(
                                target_id, certified_rate
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                except Exception as exc:
                    failures += 1
                    record["completed_at"] = utc_now()
                    record["message"] = "physical recertification failed: {}".format(exc)
                    print(
                        "Re-certification failed on {}: {}".format(target_id, exc),
                        file=sys.stderr,
                        flush=True,
                    )
                atomic_write_json(state_path, state)
        _print_status_counts(state)
        return 0 if failures == 0 else 2
    finally:
        lock_stream.close()


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

    free_memory_by_gpu = gpu_free_memory_mb(training)
    minimum_free_memory = int(training.get("min_free_memory_mb", 4096))
    gpu_ids = [
        gpu_id for gpu_id in training["gpu_ids"]
        if free_memory_by_gpu.get(gpu_id) is None
        or free_memory_by_gpu.get(gpu_id, 0) >= minimum_free_memory
    ]
    if not gpu_ids:
        raise RuntimeError(
            "none of the configured GPU ids {} currently has enough free memory".format(
                training["gpu_ids"]
            )
        )
    worker_training = {
        gpu_id: training_profile_for_gpu(training, free_memory_by_gpu.get(gpu_id))
        for gpu_id in gpu_ids
    }
    print("Using one curriculum worker on each available GPU: {}".format(", ".join(gpu_ids)))
    for gpu_id in gpu_ids:
        profile = worker_training[gpu_id]
        free_label = (
            "unknown" if free_memory_by_gpu.get(gpu_id) is None
            else str(free_memory_by_gpu[gpu_id])
        )
        print(
            "  GPU {}: free={} MiB, num_envs={}, minibatch_size={}".format(
                gpu_id, free_label, profile["num_envs"], profile["minibatch_size"]
            )
        )
    task_name = "WujiHandFixedTilt" if manifest["name"].startswith("wujihand") else "Shadowhand18Tilted"

    def start_next(worker_id, gpu_id):
        nonlocal stop_scheduling
        if stop_scheduling:
            return False
        record = None
        try:
            selection = choose_next_pending_transition(manifest, state)
            if selection is None:
                return False
            target = selection.target
            record = state["targets"][target.target_id]
            transition = selection.transition
            parent = transition.parent
            run_target = replace(target, base_yaw_deg=transition.child_base_yaw_deg)
            attempt = int(record["attempts"]) + 1
            run_name = make_run_name(target, attempt)
            probe_reused = False
            probe_error = None
            if training["offset_probe"]["enabled"]:
                try:
                    selected_offset, probe_reused = run_offset_probe(
                        python_executable,
                        run_target,
                        parent.checkpoint,
                        record,
                        state_dir,
                        training,
                        str(gpu_id),
                        task_name,
                        attempt,
                    )
                    run_target = replace(run_target, object_offset=selected_offset)
                except Exception as exc:
                    # A probe is a training aid, not a new way to strand the
                    # queue.  Preserve the manifest guess and save enough
                    # context to diagnose the evaluator failure later.
                    probe_error = str(exc)
                    record["offset_probe"] = {
                        "status": "failed",
                        "parent_checkpoint": str(Path(parent.checkpoint).resolve()),
                        "error": probe_error,
                        "completed_at": utc_now(),
                    }
            staged_checkpoint = checkpoints_dir / (run_name + "_parent.pth")
            output_checkpoint = PACKAGE_DIR / "runs" / run_name / "nn" / (run_name + ".pth")
            log_path = logs_dir / (run_name + ".log")
            stage_checkpoint(parent.checkpoint, staged_checkpoint)
            job_training = worker_training[str(gpu_id)]
            command = build_command(
                python_executable, run_target, staged_checkpoint, run_name, job_training
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
                    "gpu_id": str(gpu_id),
                    "num_envs": job_training["num_envs"],
                    "minibatch_size": job_training["minibatch_size"],
                    "base_yaw_deg": run_target.base_yaw_deg,
                    "hand_rotation_distance_deg": math.degrees(
                        transition.hand_rotation_distance
                    ),
                    "gravity_direction_distance_deg": math.degrees(
                        transition.gravity_direction_distance
                    ),
                    "transition_within_limit": selection.within_transition_limit,
                    "transition_limit_deg": training["max_parent_transition_deg"],
                    "selected_object_palm_offset": list(run_target.object_offset),
                }
            )
            atomic_write_json(state_path, state)

            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
                str(gpu_id),
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
            print(
                "Worker {} on GPU {} started {} (theta={}, phi={}, yaw={}) from {} "
                "at hand/SO3 {:.2f} deg (gravity {:.2f} deg, {} {:.2f} deg limit); log {}".format(
                    worker_id,
                    gpu_id,
                    target.target_id,
                    _format_number(target.theta_deg),
                    _format_number(target.phi_deg),
                    _format_number(run_target.base_yaw_deg),
                    parent.source_id,
                    math.degrees(transition.hand_rotation_distance),
                    math.degrees(transition.gravity_direction_distance),
                    "within" if selection.within_transition_limit else "fallback beyond",
                    float(training["max_parent_transition_deg"]),
                    log_path,
                ),
                flush=True,
            )
            if training["offset_probe"]["enabled"]:
                if probe_error is not None:
                    print(
                        "  Offset probe failed; training {} from manifest offset {}: {}".format(
                            target.target_id,
                            _format_vector(run_target.object_offset),
                            probe_error,
                        ),
                        flush=True,
                    )
                else:
                    print(
                        "  Offset probe {} selected {}".format(
                            "reused" if probe_reused else "selected",
                            _format_vector(run_target.object_offset),
                        ),
                        flush=True,
                    )
            return True
        except Exception as exc:
            if record is None:
                raise
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
        checkpoint_error = None
        try:
            reward = read_checkpoint_reward(job.output_checkpoint)
        except Exception as exc:
            checkpoint_error = str(exc)
        record["best_reward"] = reward
        record["completed_at"] = utc_now()
        record["return_code"] = return_code
        record["worker_id"] = None
        record.pop("pid", None)

        mode = promotion_mode(training)
        certification = None
        evaluation_error = None
        if reward is not None and mode != "reward_only":
            try:
                target = next(target for target in manifest["targets"] if target.target_id == job.target_id)
                target = replace(
                    target,
                    base_yaw_deg=float(record.get("base_yaw_deg", target.base_yaw_deg)),
                )
                certification_path = state_dir / "evaluations" / (job.run_name + ".json")
                certification = evaluate_checkpoint(
                    python_executable, target, job.output_checkpoint,
                    certification_path, training, job.gpu_id, task_name,
                )
                record["certification"] = certification
                record["certification_path"] = str(certification_path)
            except Exception as exc:
                evaluation_error = "physical certification failed: {}".format(exc)
                record["certification_error"] = evaluation_error

        certified_rate = float(
            certification.get(
                "minimum_object_success_rate",
                certification.get("retained_success_rate", -1.0),
            )
            if certification is not None else -1.0
        )
        passed = checkpoint_passes_promotion(training, reward, certification)
        if passed:
            state["completion_counter"] = int(state.get("completion_counter", 0)) + 1
            record["status"] = "succeeded"
            record["completion_seq"] = state["completion_counter"]
            if mode == "reward_only":
                record["message"] = "score gate passed: {:.6g} > {:.6g}".format(
                    reward, training["score_to_win"]
                )
                print(
                    "Worker {} succeeded on {}: reward {:.6g} > {:.6g}".format(
                        job.worker_id, job.target_id, reward, training["score_to_win"]
                    ),
                    flush=True,
                )
            else:
                record["message"] = "{} promotion gate passed: retained success {:.3f}, reward {:.6g}".format(
                    mode, certified_rate, reward
                )
                print(
                    "Worker {} succeeded on {}: retained success {:.1%}, reward {:.6g}".format(
                        job.worker_id, job.target_id, certified_rate, reward
                    ),
                    flush=True,
                )
        elif checkpoint_error is not None:
            record["status"] = "failed"
            record["message"] = "output checkpoint verification failed: {}".format(checkpoint_error)
            stop_scheduling = True
            print("Worker {} failed on {}: {}".format(job.worker_id, job.target_id, record["message"]), file=sys.stderr)
        elif evaluation_error is not None:
            # Certification is diagnostic/protective.  A broken evaluator must
            # not strand every later curriculum target or throw away the
            # already saved output checkpoint.
            record["status"] = "failed"
            record["message"] = evaluation_error
            print(
                "Worker {} could not certify {}; continuing the queue: {}".format(
                    job.worker_id, job.target_id, evaluation_error
                ),
                file=sys.stderr,
                flush=True,
            )
        elif job.timed_out:
            record["status"] = "timed_out"
            record["message"] = job.timeout_message
            print("Worker {} timed out on {}; continuing the queue".format(job.worker_id, job.target_id), flush=True)
        elif return_code == 0:
            record["status"] = "timed_out"
            record["message"] = (
                "training ended cleanly without passing the {} promotion gate "
                "(the max-iteration budget was likely exhausted)"
            ).format(mode)
            print("Worker {} exhausted the training budget on {}; continuing the queue".format(job.worker_id, job.target_id), flush=True)
        else:
            record["status"] = "failed"
            record["message"] = "training process exited unexpectedly with code {}".format(return_code)
            stop_scheduling = True
            print("Worker {} crashed on {} with code {}; no new jobs will start".format(job.worker_id, job.target_id, return_code), file=sys.stderr)

        _cleanup_staged_checkpoint(record)
        atomic_write_json(state_path, state)

    try:
        for worker_id, gpu_id in enumerate(gpu_ids):
            start_next(worker_id, gpu_id)

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
                # Refill every idle slot. This is required at the block ->
                # multi-object barrier, where early finishers deliberately sit
                # idle until the final block worker completes.
                for idle_worker_id, idle_gpu_id in enumerate(gpu_ids):
                    if idle_worker_id not in active:
                        start_next(idle_worker_id, idle_gpu_id)

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
    print("Validated {} targets; one worker per configured GPU: {}.".format(
        len(manifest["targets"]), ", ".join(training["gpu_ids"])
    ))
    print(
        "These commands use deterministic preceding-pose world-Z yaw. Live scheduling "
        "recomputes yaw from the actual parent checkpoint."
    )
    for target in continuous_viewer_targets(manifest, manifest["targets"]):
        run_name = make_run_name(target, 1)
        placeholder = PACKAGE_DIR / "curriculum_runs" / manifest["name"] / "staged_checkpoints" / (run_name + "_parent.pth")
        command = build_command(python_executable, target, placeholder, run_name, training)
        print("{:02d} {}".format(target.manifest_index + 1, shell_join(command)))


def inspect_manifest(manifest):
    targets = manifest["targets"]
    verified_count = min(42, len(targets))
    block_targets = [target for target in targets if target.stage == "block"]
    multi_targets = [target for target in targets if target.stage == "multi_object"]
    inherited_count = sum(
        target.offset_source != target.target_id for target in block_targets[42:]
    )
    print("Curriculum: {}".format(manifest["name"]))
    print(
        "Directions: {} total ({} verified, {} generated); generated offsets inherited: {}".format(
            len(block_targets), verified_count, len(block_targets) - verified_count, inherited_count
        )
    )
    if multi_targets:
        print(
            "Multi-object stage: {} targets after block barrier; objects={}".format(
                len(multi_targets), ",".join(multi_targets[0].object_type_pool)
            )
        )
    print(
        "GPUs: {}; fallback num_envs={}, minibatch_size={}".format(
            ", ".join(manifest["training"]["gpu_ids"]),
            manifest["training"]["num_envs"],
            manifest["training"]["minibatch_size"],
        )
    )
    print(
        "Scheduler: closest successful-parent frontier <= {} deg; "
        "discovered parents require reward > {}{}".format(
            _format_number(manifest["training"]["max_parent_transition_deg"]),
            _format_number(manifest["training"]["discovered_parent_min_reward"]),
            " (unscored allowed)"
            if manifest["training"]["allow_unscored_discovered_parents"]
            else "",
        )
    )
    for profile in manifest["training"].get("resource_profiles", []):
        print(
            "  profile >= {} MiB: num_envs={}, minibatch_size={}".format(
                profile["min_free_memory_mb"], profile["num_envs"], profile["minibatch_size"]
            )
        )
    print("Checkpoint search roots:")
    for root in manifest["checkpoint_search_roots"]:
        print("  {}{}".format(root, "" if root.is_dir() else " (missing)"))
    unavailable = manifest.get("unavailable_existing_start_checkpoints", [])
    if unavailable:
        print("Temporarily unavailable optional historical checkpoints:")
        for item in unavailable:
            print("  {} -> {}".format(item["id"], item["checkpoint"]))
    discovered = manifest.get("discovered_checkpoints", [])
    print("Compatible discovered checkpoints: {}".format(len(discovered)))
    for candidate in discovered:
        print("  {} -> {}".format(candidate.source_id, candidate.checkpoint))


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
    mode.add_argument(
        "--inspect",
        action="store_true",
        help="Report dense targets, GPU profiles, and compatible checkpoint discovery; missing seeds are allowed",
    )
    mode.add_argument(
        "--recertify-failed",
        action="store_true",
        help="Re-evaluate existing checkpoints that failed only during physical certification; never retrains them",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        manifest = load_manifest(
            args.manifest, require_offsets=True, require_seed=not args.inspect
        )
    except ManifestValidationError as exc:
        print("Manifest validation failed:", file=sys.stderr)
        for error in exc.errors:
            print("  - {}".format(error), file=sys.stderr)
        return 2

    if args.validate:
        print("Manifest is valid: {} unique targets, seed {}".format(
            len(manifest["targets"]), manifest["seed_checkpoint"]
        ))
        for item in manifest.get("unavailable_existing_start_checkpoints", []):
            print(
                "Warning: optional historical checkpoint is unavailable and will be skipped "
                "until restored: {} -> {}".format(item["id"], item["checkpoint"])
            )
        return 0
    if args.inspect:
        inspect_manifest(manifest)
        return 0
    if args.dry_run:
        dry_run(manifest, args.python)
        return 0

    state_dir = args.state_dir
    if state_dir is None:
        state_dir = PACKAGE_DIR / "curriculum_runs" / (manifest["name"] + "_dense_v5")
    if args.recertify_failed:
        try:
            return recertify_failed_outputs(manifest, state_dir, args.python)
        except (OSError, RuntimeError, ValueError) as exc:
            print("Re-certification failed: {}".format(exc), file=sys.stderr)
            return 3
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
