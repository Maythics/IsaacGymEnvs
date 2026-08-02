import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from isaacgymenvs.scripts import run_shadowhand18_tilt_curriculum as launcher


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "isaacgymenvs" / "curricula" / "shadowhand18_tilt_42.yaml"


def calibrated_manifest_file(directory):
    with MANIFEST_PATH.open("r") as stream:
        raw = yaml.safe_load(stream)
    for row in raw["targets"]:
        row["object_palm_offset"] = [-0.04, 0.0, 0.0]
    output = Path(directory) / "calibrated.yaml"
    with output.open("w") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False)
    return output


class ManifestTests(unittest.TestCase):
    def test_checked_in_manifest_has_full_unique_serpentine_coverage(self):
        manifest = launcher.load_manifest(MANIFEST_PATH, require_offsets=False, require_seed=True)
        targets = manifest["targets"]
        self.assertEqual(42, len(targets))
        self.assertEqual(["p000_t030", "p000_t060"], [targets[0].target_id, targets[1].target_id])
        self.assertEqual("north_pole", targets[-1].target_id)
        unique_directions = {
            tuple(round(component, 7) for component in target.direction) for target in targets
        }
        self.assertEqual(42, len(unique_directions))

    def test_all_tuned_offsets_pass_strict_validation(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=True, require_seed=True
        )
        self.assertEqual(42, len(manifest["targets"]))

    def test_calibrated_copy_passes_validation_and_builds_requested_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=True
            )
            target = manifest["targets"][0]
            command = launcher.build_command(
                "python",
                target,
                Path(temporary_directory) / "parent.pth",
                "test_run",
                manifest["training"],
            )
        self.assertIn("num_envs=10240", command)
        self.assertIn("train.params.config.minibatch_size=10240", command)
        self.assertIn("max_iterations=200000", command)
        self.assertIn("task.env.baseTiltAngleDeg=30", command)
        self.assertIn("task.env.baseTiltAxis=[0,1,0]", command)
        self.assertIn("task.env.objectPalmOffset=[-0.04,0,0]", command)
        self.assertIn("train.params.config.score_to_win=2500", command)
        self.assertIn("experiment=test_run", command)
        self.assertIn("+full_experiment_name=test_run", command)

    def test_all_configured_existing_start_checkpoints_are_strict_winners(self):
        manifest = launcher.load_manifest(MANIFEST_PATH, require_offsets=False, require_seed=True)
        threshold = float(manifest["training"]["score_to_win"])
        checkpoints = [manifest["seed_checkpoint"]]
        checkpoints.extend(candidate.checkpoint for candidate in manifest["existing_start_checkpoints"])
        self.assertEqual(12, len(checkpoints))
        for checkpoint in checkpoints:
            self.assertGreater(launcher.read_checkpoint_reward(checkpoint), threshold, str(checkpoint))


class ParentSelectionTests(unittest.TestCase):
    def test_equal_distance_prefers_most_recent_success(self):
        target = launcher.Target("target", 0.0, 90.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 10)
        older = launcher.ParentCandidate(
            "older", Path("older.pth"), launcher.spherical_direction(60, 0), 3, 1
        )
        newer = launcher.ParentCandidate(
            "newer", Path("newer.pth"), launcher.spherical_direction(120, 0), 8, 2
        )
        self.assertEqual("newer", launcher.choose_nearest_parent(target, [older, newer]).source_id)

    def test_running_targets_are_not_available_parents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=True
            )
            state = launcher.new_state(manifest)
            succeeded = manifest["targets"][0]
            running = manifest["targets"][1]
            successful_checkpoint = Path(temporary_directory) / "successful.pth"
            successful_checkpoint.touch()
            state["targets"][succeeded.target_id].update(
                {
                    "status": "succeeded",
                    "output_checkpoint": str(successful_checkpoint),
                    "completion_seq": 1,
                }
            )
            state["targets"][running.target_id].update(
                {
                    "status": "running",
                    "output_checkpoint": str(Path(temporary_directory) / "running.pth"),
                }
            )
            source_ids = {candidate.source_id for candidate in launcher.available_parents(manifest, state)}
        self.assertIn("seed", source_ids)
        self.assertIn(succeeded.target_id, source_ids)
        self.assertNotIn(running.target_id, source_ids)

    def test_existing_exact_checkpoint_is_selected_before_curriculum_training(self):
        manifest = launcher.load_manifest(MANIFEST_PATH, require_offsets=False, require_seed=True)
        target = next(target for target in manifest["targets"] if target.target_id == "p000_t060")
        state = launcher.new_state(manifest)
        parent = launcher.choose_nearest_parent(target, launcher.available_parents(manifest, state))
        self.assertEqual("existing_p000_t060", parent.source_id)


class CheckpointTests(unittest.TestCase):
    def test_staging_preserves_training_state_and_resets_only_bookkeeping(self):
        original = {
            "model": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer": {"state": {7: {"step": 123}}, "param_groups": [{"lr": 5e-4}]},
            "running_mean_std": {"mean": torch.tensor([3.0])},
            "env_state": {"difficulty": 9},
            "epoch": 18543,
            "frame": 622428160,
            "last_mean_rewards": 4706.7954,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.pth"
            staged = Path(temporary_directory) / "staged.pth"
            torch.save(original, str(source))
            launcher.stage_checkpoint(source, staged)
            source_state = torch.load(str(source), map_location="cpu")
            staged_state = torch.load(str(staged), map_location="cpu")

        self.assertEqual(18543, source_state["epoch"])
        self.assertEqual(4706.7954, source_state["last_mean_rewards"])
        self.assertEqual(0, staged_state["epoch"])
        self.assertEqual(0, staged_state["frame"])
        self.assertEqual(launcher.NEGATIVE_REWARD_SENTINEL, staged_state["last_mean_rewards"])
        self.assertTrue(torch.equal(original["model"]["weight"], staged_state["model"]["weight"]))
        self.assertEqual(original["optimizer"], staged_state["optimizer"])
        self.assertEqual(original["env_state"], staged_state["env_state"])

    def test_checkpoint_reward_is_read_for_strict_threshold_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pth"
            torch.save({"last_mean_rewards": 2500.0}, str(checkpoint))
            self.assertEqual(2500.0, launcher.read_checkpoint_reward(checkpoint))
            torch.save({"last_mean_rewards": 2500.01}, str(checkpoint))
            self.assertGreater(launcher.read_checkpoint_reward(checkpoint), 2500.0)


class StateTests(unittest.TestCase):
    def test_running_jobs_requeue_and_timed_out_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=True
            )
            state_path = Path(temporary_directory) / "state.json"
            state = launcher.new_state(manifest)
            running_id = manifest["targets"][0].target_id
            timed_out_id = manifest["targets"][1].target_id
            state["targets"][running_id]["status"] = "running"
            state["targets"][timed_out_id]["status"] = "timed_out"
            launcher.atomic_write_json(state_path, state)

            resumed = launcher.load_or_create_state(manifest, state_path, [])
            self.assertEqual("pending", resumed["targets"][running_id]["status"])
            self.assertEqual("timed_out", resumed["targets"][timed_out_id]["status"])
            retried = launcher.load_or_create_state(manifest, state_path, ["timed_out"])
            self.assertEqual("pending", retried["targets"][timed_out_id]["status"])
            self.assertIn("timed_out", retried["targets"][timed_out_id]["message"])

    def test_completed_success_is_recovered_after_launcher_interruption(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=True
            )
            state_path = Path(temporary_directory) / "state.json"
            state = launcher.new_state(manifest)
            target_id = manifest["targets"][0].target_id
            checkpoint = Path(temporary_directory) / "completed.pth"
            staged = Path(temporary_directory) / "staged.pth"
            torch.save({"last_mean_rewards": 2500.01}, str(checkpoint))
            staged.touch()
            state["targets"][target_id].update(
                {
                    "status": "running",
                    "run_name": "interrupted_run",
                    "output_checkpoint": str(checkpoint),
                    "staged_checkpoint": str(staged),
                }
            )
            launcher.atomic_write_json(state_path, state)
            resumed = launcher.load_or_create_state(manifest, state_path, [])

        self.assertEqual("succeeded", resumed["targets"][target_id]["status"])
        self.assertEqual(2500.01, resumed["targets"][target_id]["best_reward"])
        self.assertIsNone(resumed["targets"][target_id]["staged_checkpoint"])


if __name__ == "__main__":
    unittest.main()
