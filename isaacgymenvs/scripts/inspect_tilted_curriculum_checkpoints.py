#!/usr/bin/env python3
"""List curriculum checkpoints and generate exact viewer commands from state."""

from __future__ import print_function

import argparse
import json
import shlex
import sys
from pathlib import Path

try:
    from . import run_shadowhand18_tilt_curriculum as curriculum
except ImportError:
    import run_shadowhand18_tilt_curriculum as curriculum


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=curriculum.DEFAULT_MANIFEST)
    parser.add_argument("--python", default="python", help="Python executable written to viewer commands")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--format", choices=("commands", "table"), default="commands",
        help="commands prints one non-headless train.py command per checkpoint",
    )
    parser.add_argument(
        "--include", choices=("succeeded-and-timeout", "succeeded", "all-output"),
        default="succeeded-and-timeout",
        help=(
            "succeeded-and-timeout (default) prints promoted policies first, then "
            "timed-out runs; all-output also includes failed outputs"
        ),
    )
    parser.add_argument("--target", action="append", default=[], help="Target id to include; repeatable")
    args = parser.parse_args(argv)
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    return args


def _read_state(state_dir):
    state_path = Path(state_dir).resolve() / "state.json"
    if not state_path.is_file():
        raise RuntimeError("state.json does not exist: {}".format(state_path))
    with state_path.open("r") as stream:
        state = json.load(stream)
    if not isinstance(state, dict) or not isinstance(state.get("targets"), dict):
        raise RuntimeError("invalid curriculum state: {}".format(state_path))
    return state_path, state


def _command_from_log(record):
    log_path = record.get("log_path")
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        with path.open("r") as stream:
            for line in stream:
                if line.startswith("COMMAND: "):
                    return shlex.split(line[len("COMMAND: "):].strip())
    except OSError:
        return None
    return None


def _replace_or_append(command, key, value):
    prefix = key + "="
    for index, token in enumerate(command):
        if token.startswith(prefix):
            command[index] = prefix + str(value)
            return
    command.append(prefix + str(value))


def viewer_command(record, target, manifest, python_executable, num_envs, checkpoint=None):
    """Create a test=True, headless=False command for one saved output."""
    checkpoint = Path(checkpoint or record["output_checkpoint"]).resolve()
    command = _command_from_log(record)
    if command is None:
        task_name = (
            "WujiHandFixedTilt"
            if manifest["name"].startswith("wujihand")
            else "Shadowhand18Tilted"
        )
        command = [
            str(python_executable), "train.py", "task=" + task_name,
            "task.env.objectType={}".format(manifest["training"]["object_type"]),
            "task.env.baseTiltAngleDeg={}".format(curriculum._format_number(target.theta_deg)),
            "task.env.baseTiltAxis={}".format(curriculum._format_vector(target.axis)),
            "task.env.objectPalmOffset={}".format(curriculum._format_vector(target.object_offset)),
            "task.env.baseYawDeg={}".format(
                curriculum._format_number(record.get("base_yaw_deg", target.base_yaw_deg))
            ),
            "task.env.objectGravityCompensationSeconds={}".format(
                curriculum._format_number(manifest["training"]["object_gravity_compensation_seconds"])
            ),
            "task.env.objectGravityRampSeconds={}".format(
                curriculum._format_number(manifest["training"]["object_gravity_ramp_seconds"])
            ),
        ]
        if target.object_type_pool:
            command.append("task.env.objectTypePool=[{}]".format(",".join(target.object_type_pool)))

    command[0] = str(python_executable)
    _replace_or_append(command, "num_envs", num_envs)
    _replace_or_append(command, "train.params.config.minibatch_size", num_envs)
    _replace_or_append(command, "checkpoint", checkpoint)
    _replace_or_append(command, "headless", "False")
    _replace_or_append(command, "test", "True")
    # Viewer commands should not create or overwrite an experiment directory.
    command = [
        token for token in command
        if not token.startswith("experiment=")
        and not token.startswith("+full_experiment_name=")
    ]
    return command


def timeout_checkpoint_candidates(record):
    """Return saved checkpoints from all attempts of one timed-out target."""
    output = Path(record["output_checkpoint"])
    candidates = [output]
    run_name = record.get("run_name")
    # Output is .../runs/<run_name>/nn/<run_name>.pth.  Earlier attempts have
    # the same prefix and may be better than the last attempt kept in state.
    if run_name and "_a" in run_name and len(output.parents) >= 3:
        run_prefix = run_name.rsplit("_a", 1)[0] + "_a"
        runs_dir = output.parents[2]
        if runs_dir.is_dir():
            for run_dir in sorted(runs_dir.glob(run_prefix + "*")):
                nn_dir = run_dir / "nn"
                if nn_dir.is_dir():
                    candidates.extend(sorted(nn_dir.glob("*.pth")))
    seen = set()
    return [path for path in candidates if not (str(path) in seen or seen.add(str(path)))]


def checkpoint_for_record(record):
    """Choose the highest-reward saved checkpoint for a timeout record.

    A normal/succeeded record points directly at the policy that the launcher
    used.  A target retried after timeouts has several run directories, so for
    a final timeout inspect every saved attempt and use the best reward.
    """
    configured = Path(record["output_checkpoint"])
    if record.get("status") != "timed_out":
        return configured, record.get("best_reward")

    best_path = None
    best_reward = None
    for path in timeout_checkpoint_candidates(record):
        if not path.is_file():
            continue
        try:
            reward = curriculum.read_checkpoint_reward(path)
        except Exception:
            continue
        if best_path is None or reward > best_reward:
            best_path, best_reward = path, reward
    if best_path is not None:
        return best_path, best_reward
    # The user should still see a diagnostic command and the missing marker if
    # RL-Games had not produced a checkpoint before termination.
    return configured, record.get("best_reward")


def selected_records(manifest, state, include, target_ids):
    targets_by_id = {target.target_id: target for target in manifest["targets"]}
    result = []
    wanted = set(target_ids)
    for target_id, record in state["targets"].items():
        target = targets_by_id.get(target_id)
        checkpoint = record.get("output_checkpoint")
        if target is None or not checkpoint:
            continue
        if wanted and target_id not in wanted:
            continue
        status = record.get("status")
        if include == "succeeded" and status != "succeeded":
            continue
        if include == "succeeded-and-timeout" and status not in ("succeeded", "timed_out"):
            continue
        selected_checkpoint, selected_reward = checkpoint_for_record(record)
        result.append((target, record, selected_checkpoint, selected_reward))
    # Explicitly keep timeout diagnostics below all passing policies, rather
    # than interleaving them by the manifest's spherical ordering.
    status_order = {"succeeded": 0, "timed_out": 1, "failed": 2}
    return sorted(
        result,
        key=lambda item: (status_order.get(item[1].get("status"), 3), item[0].manifest_index),
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        manifest = curriculum.load_manifest(
            args.manifest, require_offsets=True, require_seed=False
        )
        state_path, state = _read_state(args.state_dir)
    except (OSError, RuntimeError, curriculum.ManifestValidationError) as exc:
        print("Checkpoint inspection failed: {}".format(exc), file=sys.stderr)
        return 2

    records = selected_records(manifest, state, args.include, args.target)
    if not records:
        print("No matching checkpoint records in {}".format(state_path), file=sys.stderr)
        return 1

    if args.format == "table":
        print("# state: {}".format(state_path))
        print("# idx target status attempt reward hand_deg parent checkpoint")
        for target, record, checkpoint, selected_reward in records:
            exists = checkpoint.is_file()
            print(
                "{idx:03d} {target_id} {status} a{attempt:02d} {reward} {hand} {parent} {checkpoint}{missing}".format(
                    idx=target.manifest_index + 1,
                    target_id=target.target_id,
                    status=record.get("status", "unknown"),
                    attempt=int(record.get("attempts") or 0),
                    reward=curriculum._format_number(
                        selected_reward if selected_reward is not None else record.get("best_reward") or 0.0
                    ),
                    hand=curriculum._format_number(record.get("hand_rotation_distance_deg") or 0.0),
                    parent=record.get("parent_id") or "-",
                    checkpoint=checkpoint,
                    missing=" [MISSING]" if not exists else "",
                )
            )
        return 0

    print("# state: {}".format(state_path))
    print("# Close one viewer before executing the next command.")
    previous_status = None
    for target, record, checkpoint, selected_reward in records:
        status = record.get("status", "unknown")
        if status != previous_status:
            print("# === {} checkpoint(s) ===".format(
                "PROMOTED / SUCCEEDED" if status == "succeeded" else
                "TIMED OUT (diagnostic only; best saved attempt)" if status == "timed_out" else
                "FAILED (diagnostic only)"
            ))
            previous_status = status
        exists = checkpoint.is_file()
        print(
            "# {:03d} {} | {} | {} | attempt a{:02d} | reward {} | hand/SO3 {} deg{}".format(
                target.manifest_index + 1,
                target.target_id,
                status,
                record.get("run_name") or "-",
                int(record.get("attempts") or 0),
                curriculum._format_number(
                    selected_reward if selected_reward is not None else record.get("best_reward") or 0.0
                ),
                curriculum._format_number(record.get("hand_rotation_distance_deg") or 0.0),
                " | CHECKPOINT MISSING" if not exists else "",
            )
        )
        print(curriculum.shell_join(viewer_command(
            record, target, manifest, args.python, args.num_envs, checkpoint
        )))
    return 0


if __name__ == "__main__":
    sys.exit(main())
