#!/usr/bin/env python3
"""Resumable multi-GPU fixed-root gravity-in-palm curriculum."""

import argparse
import copy
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GravityTarget:
    target_id: str
    gravity_in_palm: tuple


def canonical_targets_42():
    """Old 42 sphere locations, explicitly stored as physical gravity vectors."""
    rows = [("south_pole", 0, 0), ("north_pole", 0, 180)]
    rows += [("p{:03d}_t{:03d}".format(phi, theta), phi, theta)
             for theta in (30, 60, 90, 120, 150) for phi in range(0, 360, 45)]
    targets = []
    for target_id, phi_deg, theta_deg in rows:
        theta, phi = math.radians(theta_deg), math.radians(phi_deg)
        # The historical tilt manifest represented palm-frame up; use -up for
        # physical acceleration / gravity.
        up = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
        targets.append(GravityTarget(target_id, tuple(0.0 if abs(v) < 1e-12 else -v for v in up)))
    return targets


def angular_distance(a, b):
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def minimum_target_spacing(targets):
    return min(angular_distance(a.gravity_in_palm, b.gravity_in_palm)
               for i, a in enumerate(targets) for b in targets[i + 1:])


def ordered_targets(native_gravity_in_palm):
    return sorted(canonical_targets_42(), key=lambda t: (
        angular_distance(native_gravity_in_palm, t.gravity_in_palm), t.target_id))


def format_vector(vector):
    return "[{}]".format(",".join("{:.8g}".format(value) for value in vector))


def build_command(python, task, checkpoint, target, training, run_name, gpu_id=None):
    command = [
        str(python), "train.py", "task={}".format(task), "headless=True",
        "num_envs={}".format(training["num_envs"]),
        "train.params.config.minibatch_size={}".format(training["minibatch_size"]),
        "max_iterations={}".format(training["max_iterations"]),
        "checkpoint={}".format(Path(checkpoint).resolve()),
        "task.env.episodeLength={}".format(training["episode_length"]),
        "task.env.objectType={}".format(training["object_type"]),
        "task.env.gravityInPalm={}".format(format_vector(target.gravity_in_palm)),
        "task.env.objectGravityHoldSeconds=0.2", "task.env.objectGravityRampSeconds=0.2",
        "train.params.config.score_to_win={}".format(training["score_to_win"]),
        "train.params.config.save_best_after={}".format(training["save_best_after"]),
        "experiment={}".format(run_name), "+full_experiment_name={}".format(run_name),
    ]
    if gpu_id is not None:
        command += ["sim_device=cuda:{}".format(gpu_id), "rl_device=cuda:{}".format(gpu_id),
                    "graphics_device_id={}".format(gpu_id)]
    return command


def nearest_parent(target, parents, max_distance_deg):
    candidates = [(angular_distance(target.gravity_in_palm, parent["gravity_in_palm"]), parent)
                  for parent in parents]
    allowed = [(distance, parent) for distance, parent in candidates if distance <= max_distance_deg + 1e-8]
    if not allowed:
        return None
    return min(allowed, key=lambda pair: (pair[0], pair[1]["completion_seq"]))


def checkpoint_reward(path):
    """Read a checkpoint's reward for the score gate; not for selecting it."""
    import torch
    try:
        state = torch.load(str(path), map_location="cpu")
        reward = state.get("last_mean_rewards") if isinstance(state, dict) else None
        if hasattr(reward, "item"):
            reward = reward.item()
        return None if reward is None else float(reward)
    except (OSError, RuntimeError, ValueError, EOFError, TypeError):
        return None


def preferred_checkpoint(nn_dir, run_name):
    """Resolve rl_games' best checkpoint for one experiment.

    The normal save is exactly ``<experiment>.pth``.  ``*_last.pth`` is a
    continuation snapshot and is deliberately never selected when the normal
    save exists.
    """
    direct = nn_dir / "{}.pth".format(run_name)
    if direct.is_file():
        return direct
    # Compatibility fallback for old runs with a different checkpoint prefix.
    candidates = [path for path in nn_dir.glob("*.pth") if "last" not in path.stem.lower()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def checkpoint_from_run(package_dir, run_name):
    nn_dir = package_dir / "runs" / run_name / "nn"
    checkpoint = preferred_checkpoint(nn_dir, run_name) if nn_dir.is_dir() else None
    return (str(checkpoint.resolve()), checkpoint_reward(checkpoint)) if checkpoint else (None, None)


def checkpoint_from_target_runs(package_dir, target_id):
    """Find the best checkpoint from all previous gravity runs for one target.

    Experiment directories, rather than the old parent recorded in state, are
    authoritative here.  Prefer explicitly named ``gravity_*.pth`` files when
    present, then fall back to normal rl_games task-named checkpoint files.
    """
    candidates = []
    for nn_dir in (package_dir / "runs").glob("gravity_*_{}_a*/nn".format(target_id)):
        checkpoint = preferred_checkpoint(nn_dir, nn_dir.parent.name)
        if checkpoint:
            candidates.append(checkpoint)
    if not candidates:
        return None, None
    # Each candidate is already its run's best checkpoint.  An interrupted
    # retry should continue the most recently written such best model, rather
    # than choose a *_last continuation snapshot or compare stale metrics.
    checkpoint = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(checkpoint.resolve()), checkpoint_reward(checkpoint)


def free_gpu_memory_mb(gpu_id):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            text=True, stderr=subprocess.DEVNULL,
        )
        return int(output.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def training_profile(training, gpu_id):
    resolved = copy.deepcopy(training)
    free_memory = free_gpu_memory_mb(gpu_id)
    for profile in sorted(training.get("resource_profiles", []), key=lambda row: row["min_free_memory_mb"], reverse=True):
        if free_memory is not None and free_memory >= int(profile["min_free_memory_mb"]):
            resolved.update(profile)
            break
    return resolved, free_memory


def initial_state(targets):
    return {"version": 1, "completion_seq": 0, "targets": {
        target.target_id: {"gravity_in_palm": list(target.gravity_in_palm), "status": "pending", "attempts": 0}
        for target in targets
    }}


def load_state(path, targets):
    if not path.exists():
        return initial_state(targets)
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = {target.target_id for target in targets}
    if set(state.get("targets", {})) != expected:
        raise ValueError("state targets do not match the canonical 42-target manifest")
    # A terminated launcher has no live child processes; retry its interrupted jobs.
    for record in state["targets"].values():
        if record["status"] == "running":
            record["status"] = "pending"
            record["interrupted"] = True
    return state


def apply_retry(state, retry):
    """Requeue unsuccessful work while preserving its run/checkpoint history."""
    if retry is None:
        return
    if retry != "all":
        raise ValueError("unsupported retry mode: {}".format(retry))
    for record in state["targets"].values():
        # A passed score gate is a curriculum parent and must not be retrained
        # by retry-all.  Everything else is eligible to continue.
        if record["status"] != "succeeded":
            record["status"] = "pending"


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args():
    package_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=package_dir / "curricula" / "gravity_42.yaml")
    parser.add_argument("--task", choices=("Shadowhand18Gravity", "WujiHandGravity"), default="Shadowhand18Gravity")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--retry", nargs="?", const="all", choices=("all",), default=None,
                        help="requeue all non-successful targets; their best existing gravity checkpoint is used")
    # Accept the natural spelling "--retry --all" as well as "--retry all".
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    package_dir = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    targets = canonical_targets_42()
    training = manifest["training"]
    native = tuple(manifest["native_gravity_in_palm"][args.task])
    seed = (manifest_path.parent / manifest["seed_checkpoint"][args.task]).resolve()
    if not seed.is_file():
        raise FileNotFoundError("seed checkpoint does not exist: {}".format(seed))
    gpu_ids = [int(value) for value in training["gpu_ids"]]
    max_distance = float(training.get("max_parent_distance_deg", 2.0 * minimum_target_spacing(targets)))
    if args.state_dir is None:
        args.state_dir = package_dir / "curriculum_runs" / (manifest["name"] + "_" + args.task)
    state_path = args.state_dir / "state.json"

    if args.dry_run:
        try:
            print("minimum spacing={:.2f} deg; reusable-parent radius={:.2f} deg".format(
                minimum_target_spacing(targets), max_distance))
            for index, target in enumerate(ordered_targets(native), start=1):
                command = build_command(args.python, args.task, seed, target, training,
                                        "gravity_{:02d}_{}".format(index, target.target_id), gpu_ids[(index - 1) % len(gpu_ids)])
                print(" ".join(shlex.quote(str(token)) for token in command))
        except BrokenPipeError:
            return 0
        return 0

    state = load_state(state_path, targets)
    if args.all:
        if args.retry not in (None, "all"):
            raise ValueError("--all is only valid with --retry")
        args.retry = "all"
    apply_retry(state, args.retry)
    save_state(state_path, state)
    seed_parent = {"source_id": "seed", "checkpoint": str(seed), "gravity_in_palm": list(native), "completion_seq": -1}
    running = {}
    launched = 0
    try:
        while True:
            for gpu_id, job in list(running.items()):
                code = job["process"].poll()
                if code is None:
                    continue
                record = state["targets"][job["target"].target_id]
                checkpoint, reward = checkpoint_from_run(package_dir, job["run_name"])
                threshold = float(training["score_to_win"])
                if code == 0 and checkpoint and reward is not None and reward > threshold:
                    state["completion_seq"] += 1
                    record.update(status="succeeded", checkpoint=checkpoint,
                                  parent=job["parent"]["source_id"], completion_seq=state["completion_seq"], reward=reward)
                else:
                    reason = "trainer exited without a scored checkpoint" if code == 0 and checkpoint is None else (
                        "score gate failed: {} <= {}".format(reward, threshold) if code == 0 else "trainer exited with {}".format(code))
                    record.update(status="failed", exit_code=code, reward=reward, failure=reason)
                del running[gpu_id]
                save_state(state_path, state)

            if args.max_targets is not None and launched >= args.max_targets and not running:
                return 0
            parents = [seed_parent]
            parents += [{"source_id": target_id, "checkpoint": record["checkpoint"],
                         "gravity_in_palm": record["gravity_in_palm"], "completion_seq": record["completion_seq"]}
                        for target_id, record in state["targets"].items() if record["status"] == "succeeded"]
            pending = [target for target in targets if state["targets"][target.target_id]["status"] == "pending"]
            for gpu_id in gpu_ids:
                if gpu_id in running or not pending or (args.max_targets is not None and launched >= args.max_targets):
                    continue
                choices = []
                for target in pending:
                    record = state["targets"][target.target_id]
                    # An attempted target resumes itself first.  This avoids
                    # throwing away a near-successful policy just because it
                    # did not yet pass score_to_win.
                    checkpoint, reward = checkpoint_from_target_runs(package_dir, target.target_id)
                    if record["attempts"] and checkpoint:
                        parent = {"source_id": "resume:{}".format(target.target_id), "checkpoint": checkpoint,
                                  "gravity_in_palm": list(target.gravity_in_palm), "completion_seq": -1,
                                  "reward": reward}
                        choices.append(((0.0, parent), target))
                    else:
                        candidate = nearest_parent(target, parents, max_distance)
                        if candidate is not None:
                            choices.append((candidate, target))
                if not choices:
                    break
                (distance, parent), target = min(choices, key=lambda item: (item[0][0], item[1].target_id))
                job_training, free_memory = training_profile(training, gpu_id)
                if free_memory is not None and free_memory < int(training["min_free_memory_mb"]):
                    continue
                record = state["targets"][target.target_id]
                record["status"], record["attempts"] = "running", record["attempts"] + 1
                run_name = "gravity_{:02d}_{}_a{:02d}".format(launched + 1, target.target_id, record["attempts"])
                record.update(run_name=run_name, checkpoint_start=parent["checkpoint"], parent=parent["source_id"],
                              started_at=time.time(), interrupted=False)
                command = build_command(args.python, args.task, parent["checkpoint"], target, job_training, run_name, gpu_id)
                print("GPU {}: {} <- {} ({:.2f} deg; {} envs, {} MiB free)".format(gpu_id, target.target_id, parent["source_id"], distance, job_training["num_envs"], free_memory), flush=True)
                running[gpu_id] = {"process": subprocess.Popen(command, cwd=str(package_dir)), "target": target,
                                   "parent": parent, "run_name": run_name, "started": time.monotonic()}
                pending.remove(target)
                launched += 1
                save_state(state_path, state)
            if not running:
                if not pending:
                    return 0
                raise RuntimeError("no pending target has a reusable parent within {:.2f} deg".format(max_distance))
            timeout = float(training["timeout_seconds"])
            for gpu_id, job in list(running.items()):
                if time.monotonic() - job["started"] > timeout:
                    job["process"].terminate()
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("Interrupted: preserving state and stopping active trainers...", flush=True)
        for job in running.values():
            job["process"].terminate()
            record = state["targets"][job["target"].target_id]
            record.update(status="pending", interrupted=True, run_name=job["run_name"])
        save_state(state_path, state)
        return 130


if __name__ == "__main__":
    sys.exit(main())
