import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from isaacgymenvs.scripts import run_shadowhand18_tilt_curriculum as launcher


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "isaacgymenvs" / "curricula" / "shadowhand18_tilt_42.yaml"
REMOTE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "isaacgymenvs" / "curricula" / "remote_shadowhand18_tilted.yaml"
)


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
    def test_remote_block_manifest_uses_dense_yaw_compatible_curriculum_settings(self):
        manifest = launcher.load_manifest(
            REMOTE_MANIFEST_PATH, require_offsets=True, require_seed=False
        )
        self.assertEqual("remote_shadowhand18_tilted_block", manifest["name"])
        block_targets = [target for target in manifest["targets"] if target.stage == "block"]
        self.assertGreater(len(block_targets), 42)
        self.assertFalse(any(target.stage == "multi_object" for target in manifest["targets"]))
        self.assertEqual("reward_only", manifest["training"]["promotion_mode"])
        self.assertEqual(
            (10240, 10240),
            tuple(
                launcher.training_profile_for_gpu(manifest["training"], 24135)[key]
                for key in ("num_envs", "minibatch_size")
            ),
        )
        self.assertEqual(
            {
                "enabled": True,
                "num_envs": 65536,
                "episodes": 65536,
                "step_m": 0.03,
                "timeout_seconds": 1800,
                "seed": 314159,
            },
            manifest["training"]["offset_probe"],
        )
        displayed = launcher.continuous_viewer_targets(manifest, block_targets[:7])
        self.assertAlmostEqual(-90.0, displayed[6].base_yaw_deg, places=6)

    def test_checked_in_curriculum_advances_on_strict_reward_gate(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=False, require_seed=False
        )
        training = manifest["training"]
        self.assertEqual("reward_only", training["promotion_mode"])
        self.assertTrue(launcher.checkpoint_passes_promotion(training, 2500.01))
        self.assertFalse(launcher.checkpoint_passes_promotion(training, 2500.0))

    def test_checked_in_manifest_has_full_unique_serpentine_coverage(self):
        manifest = launcher.load_manifest(MANIFEST_PATH, require_offsets=False, require_seed=False)
        targets = [
            target for target in manifest["targets"] if target.stage == "block"
        ]
        self.assertGreater(len(targets), 42)
        self.assertEqual(["p000_t030", "p000_t060"], [targets[0].target_id, targets[1].target_id])
        self.assertEqual("north_pole", targets[41].target_id)
        unique_directions = {
            tuple(round(component, 7) for component in target.direction) for target in targets
        }
        self.assertEqual(len(targets), len(unique_directions))
        self.assertFalse(any(target.target_id.startswith("bridge_") for target in targets))

    def test_all_tuned_offsets_pass_strict_validation(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=True, require_seed=False
        )
        self.assertGreater(len(manifest["targets"]), 42)
        self.assertTrue(all(target.object_offset is not None for target in manifest["targets"]))
        self.assertTrue(any(target.stage == "multi_object" for target in manifest["targets"]))

    def test_generated_targets_inherit_one_exact_verified_offset(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=True, require_seed=False
        )
        block_targets = [
            target for target in manifest["targets"] if target.stage == "block"
        ]
        verified = block_targets[:42]
        verified_by_id = {target.target_id: target.object_offset for target in verified}
        for target in block_targets[42:]:
            self.assertIn(target.offset_source, verified_by_id)
            self.assertEqual(verified_by_id[target.offset_source], target.object_offset)

    def test_calibrated_copy_passes_validation_and_builds_requested_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=False
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
        manifest = launcher.load_manifest(MANIFEST_PATH, require_offsets=False, require_seed=False)
        threshold = float(manifest["training"]["score_to_win"])
        with MANIFEST_PATH.open("r") as stream:
            raw = yaml.safe_load(stream)
        rewards = [row["recorded_reward"] for row in raw["existing_start_checkpoints"]]
        self.assertEqual(11, len(rewards))
        self.assertTrue(all(float(reward) > threshold for reward in rewards))


class ResourceProfileTests(unittest.TestCase):
    def test_gpu_profile_scales_env_and_minibatch_together(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=False, require_seed=False
        )
        training = manifest["training"]
        large = launcher.training_profile_for_gpu(training, 24000)
        small = launcher.training_profile_for_gpu(training, 7000)
        unknown = launcher.training_profile_for_gpu(training, None)
        self.assertEqual((10240, 10240), (large["num_envs"], large["minibatch_size"]))
        self.assertEqual((2048, 2048), (small["num_envs"], small["minibatch_size"]))
        self.assertEqual(
            (training["num_envs"], training["minibatch_size"]),
            (unknown["num_envs"], unknown["minibatch_size"]),
        )
        for resolved in (large, small, unknown):
            self.assertEqual(
                0,
                resolved["num_envs"] * training["horizon_length"]
                % resolved["minibatch_size"],
            )


class OffsetProbeTests(unittest.TestCase):
    def test_three_centimeter_stencil_keeps_manifest_offset_as_center(self):
        center = (0.08, 0.06, 0.0)
        candidates = launcher.offset_probe_candidates(center, 0.03)

        self.assertEqual(27, len(candidates))
        self.assertIn(center, candidates)
        self.assertIn((0.05, 0.03, -0.03), candidates)
        self.assertIn((0.11, 0.09, 0.03), candidates)

    def test_probe_winner_prefers_retention_then_lingering_then_reward(self):
        result = {
            "per_offset": [
                {
                    "candidate_index": 0,
                    "offset": [0.08, 0.06, 0.0],
                    "retained_success_rate": 0.10,
                    "mean_episode_steps": 300.0,
                    "mean_episode_reward": 100.0,
                },
                {
                    "candidate_index": 1,
                    "offset": [0.11, 0.06, 0.0],
                    "retained_success_rate": 0.10,
                    "mean_episode_steps": 300.0,
                    "mean_episode_reward": 200.0,
                },
                {
                    "candidate_index": 2,
                    "offset": [0.05, 0.06, 0.0],
                    "retained_success_rate": 0.11,
                    "mean_episode_steps": 1.0,
                    "mean_episode_reward": -1000.0,
                },
            ]
        }

        selected, winner = launcher.select_offset_probe_winner(result)

        self.assertEqual((0.05, 0.06, 0.0), selected)
        self.assertEqual(2, winner["candidate_index"])

    def test_same_parent_and_settings_reuses_saved_probe(self):
        candidates = launcher.offset_probe_candidates((0.08, 0.06, 0.0), 0.03)
        checkpoint = Path("/tmp/parent.pth")
        config = {
            "num_envs": 65536,
            "episodes": 65536,
            "step_m": 0.03,
            "seed": 314159,
        }
        record = {
            "offset_probe": {
                "status": "succeeded",
                "parent_checkpoint": str(checkpoint.resolve()),
                "candidates": [list(candidate) for candidate in candidates],
                "settings": config,
                "selected_offset": [0.11, 0.06, 0.0],
            }
        }

        self.assertEqual(
            (0.11, 0.06, 0.0),
            launcher.reusable_offset_probe(record, checkpoint, candidates, config),
        )
        self.assertIsNone(
            launcher.reusable_offset_probe(record, Path("/tmp/other.pth"), candidates, config)
        )

    def test_vectorized_probe_passes_all_candidates_to_evaluator(self):
        target = launcher.Target(
            "target", 45.0, 30.0, (-0.70710678, 0.70710678, 0.0),
            (0.08, 0.06, 0.0), 0,
        )
        training = {
            "certification_episodes": 128,
            "certification_num_envs": 128,
            "episode_length": 300,
            "object_gravity_compensation_seconds": 0.2,
            "object_gravity_ramp_seconds": 0.1,
            "object_type": "block",
        }
        candidates = launcher.offset_probe_candidates(target.object_offset, 0.03)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "parent.pth"
            checkpoint.touch()
            result_path = root / "probe.json"
            result_path.write_text("{}")
            with mock.patch.object(
                launcher.subprocess, "run",
                return_value=launcher.subprocess.CompletedProcess([], 0, ""),
            ) as run:
                launcher.evaluate_checkpoint(
                    "python",
                    target,
                    checkpoint,
                    result_path,
                    training,
                    "0",
                    "Shadowhand18Tilted",
                    episodes=65536,
                    num_envs=65536,
                    offset_candidates=candidates,
                    seed=123,
                    timeout_seconds=1800,
                )

        command = run.call_args.args[0]
        self.assertTrue(any(token.startswith("--offset-candidates=") for token in command))
        self.assertIn("--axis=-0.70710678,0.70710678,0", command)
        self.assertIn("--episodes", command)
        self.assertIn("65536", command)
        candidate_argument = next(
            token for token in command if token.startswith("--offset-candidates=")
        )
        self.assertIn("0.05,0.03,-0.03", candidate_argument)


class ParentSelectionTests(unittest.TestCase):
    def test_scheduler_prefers_continuous_frontier_but_never_waits_on_a_failure_chain(self):
        far = launcher.Target("far", 0.0, 90.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0)
        near = launcher.Target("near", 0.0, 10.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 1)
        manifest = {
            "path": Path("/tmp/frontier.yaml"),
            "name": "frontier",
            "targets": [far, near],
            "training": {"max_parent_transition_deg": 15.0},
            "seed_checkpoint": Path("seed.pth"),
            "seed_direction": launcher.spherical_direction(0.0, 0.0),
            "seed_theta_deg": 0.0,
            "seed_phi_deg": 0.0,
            "seed_axis": (0.0, 1.0, 0.0),
            "seed_base_yaw_deg": 0.0,
            "existing_start_checkpoints": [],
            "discovered_checkpoints": [],
        }
        state = launcher.new_state(manifest)

        selection = launcher.choose_next_pending_transition(manifest, state)

        self.assertEqual("near", selection.target.target_id)
        self.assertTrue(selection.within_transition_limit)
        self.assertAlmostEqual(10.0, math.degrees(selection.transition.hand_rotation_distance))

        state["targets"]["near"]["status"] = "running"
        fallback = launcher.choose_next_pending_transition(manifest, state)
        self.assertEqual("far", fallback.target.target_id)
        self.assertFalse(fallback.within_transition_limit)

    def test_continuous_viewer_path_uses_the_south_pole_yaw_fix(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=True, require_seed=False
        )
        targets = [target for target in manifest["targets"] if target.stage == "block"][:7]
        displayed = launcher.continuous_viewer_targets(manifest, targets)

        self.assertEqual("south_pole", displayed[5].target_id)
        self.assertEqual("p045_t150", displayed[6].target_id)
        self.assertAlmostEqual(0.0, displayed[5].base_yaw_deg, places=6)
        self.assertAlmostEqual(-90.0, displayed[6].base_yaw_deg, places=6)

    def test_world_yaw_removes_south_pole_roll_discontinuity(self):
        south = launcher.ParentCandidate(
            "south", Path("south.pth"), launcher.spherical_direction(180, 0), 1, 0,
            launcher.base_rotation_quat(180, (0.0, 1.0, 0.0)),
        )
        next_target = launcher.Target(
            "p045_t150", 45.0, 150.0, launcher.expected_axis(45),
            (0.08, 0.1, 0.01), 1,
        )
        transition = launcher.choose_parent_transition(next_target, [south])
        uncorrected = launcher.rotation_distance(
            south.base_rotation, next_target.base_rotation
        )
        self.assertAlmostEqual(93.8409657, math.degrees(uncorrected), places=4)
        self.assertAlmostEqual(30.0, math.degrees(transition.hand_rotation_distance), places=5)
        self.assertAlmostEqual(-90.0, transition.child_base_yaw_deg, places=5)
        self.assertAlmostEqual(
            30.0, math.degrees(transition.gravity_direction_distance), places=5
        )

    def test_equal_distance_prefers_most_recent_success(self):
        target = launcher.Target("target", 0.0, 90.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 10)
        older = launcher.ParentCandidate(
            "older", Path("older.pth"), launcher.spherical_direction(60, 0), 3, 1
        )
        newer = launcher.ParentCandidate(
            "newer", Path("newer.pth"), launcher.spherical_direction(120, 0), 8, 2
        )
        self.assertEqual("newer", launcher.choose_nearest_parent(target, [older, newer]).source_id)

    def test_explicit_seed_outranks_uncertified_discovery_at_equal_angle(self):
        target = launcher.Target("target", 0.0, 30.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 10)
        seed = launcher.ParentCandidate(
            "seed", Path("seed.pth"), launcher.spherical_direction(0, 0), 0, -10000
        )
        discovered = launcher.ParentCandidate(
            "discovered", Path("discovered.pth"), launcher.spherical_direction(0, 0), 999999, -20000
        )
        self.assertEqual(
            "seed",
            launcher.choose_nearest_parent(target, [discovered, seed]).source_id,
        )

    def test_running_targets_are_not_available_parents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=False
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
        manifest = launcher.load_manifest(MANIFEST_PATH, require_offsets=False, require_seed=False)
        target = next(target for target in manifest["targets"] if target.target_id == "p000_t060")
        state = launcher.new_state(manifest)
        parent = launcher.choose_nearest_parent(target, launcher.available_parents(manifest, state))
        self.assertEqual("existing_p000_t060", parent.source_id)


class CheckpointTests(unittest.TestCase):
    def test_discovered_reward_suffix_is_available_without_loading_checkpoint(self):
        self.assertEqual(
            89.64116,
            launcher.checkpoint_reward_from_filename(
                Path("last_Shadowhand18Tilted_ep_18600_rew_89.64116.pth")
            ),
        )
        self.assertIsNone(
            launcher.checkpoint_reward_from_filename(Path("Shadowhand18Tilted.pth"))
        )

    def test_only_evaluator_failures_with_an_output_are_recertifiable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "output.pth"
            checkpoint.touch()
            record = {
                "status": "failed",
                "output_checkpoint": str(checkpoint),
                "message": "output checkpoint verification failed: physical certification failed: old evaluator bug",
            }
            self.assertTrue(launcher._is_recertifiable_failure(record))

            record["message"] = "training process exited unexpectedly with code 1"
            self.assertFalse(launcher._is_recertifiable_failure(record))
            record["message"] = "physical certification failed: missing output"
            record["output_checkpoint"] = str(checkpoint.with_name("missing.pth"))
            self.assertFalse(launcher._is_recertifiable_failure(record))

    def test_missing_optional_existing_start_does_not_block_the_curriculum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            with MANIFEST_PATH.open("r") as stream:
                raw = yaml.safe_load(stream)
            seed = temporary_path / "seed.pth"
            seed.touch()
            raw["seed"]["start_checkpoint"] = str(seed)
            raw["existing_start_checkpoints"] = [{
                "id": "optional_missing",
                "optional": True,
                "checkpoint": str(temporary_path / "not-mounted.pth"),
                "theta_deg": 30,
                "phi_deg": 0,
                "base_tilt_axis": [0.0, 1.0, 0.0],
                "recorded_reward": 3000.0,
            }]
            manifest_path = temporary_path / "manifest.yaml"
            with manifest_path.open("w") as stream:
                yaml.safe_dump(raw, stream, sort_keys=False)

            manifest = launcher.load_manifest(manifest_path, require_offsets=True)

        self.assertEqual([], manifest["existing_start_checkpoints"])
        self.assertEqual(
            ["optional_missing"],
            [item["id"] for item in manifest["unavailable_existing_start_checkpoints"]],
        )

    def test_checkpoint_relocation_prefers_the_manifest_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "runs"
            direct = root / "known_run" / "nn" / "Shadowhand18Tilted.pth"
            unrelated = root / "other_run" / "nn" / "Shadowhand18Tilted.pth"
            direct.parent.mkdir(parents=True)
            unrelated.parent.mkdir(parents=True)
            direct.touch()
            unrelated.touch()

            relocated = launcher._relocate_checkpoint(
                Path("/old/location/known_run/nn/Shadowhand18Tilted.pth"),
                [root],
                Path(temporary_directory) / "manifest.yaml",
            )

        self.assertEqual(direct.resolve(), relocated)

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
    def test_failed_old_evaluator_output_above_score_is_recovered_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=False
            )
            state_path = Path(temporary_directory) / "state.json"
            state = launcher.new_state(manifest)
            target_id = manifest["targets"][0].target_id
            checkpoint = Path(temporary_directory) / "winner.pth"
            torch.save({"last_mean_rewards": 2500.01}, str(checkpoint))
            state["targets"][target_id].update(
                {
                    "status": "failed",
                    "output_checkpoint": str(checkpoint),
                    "message": "physical certification failed: evaluator API mismatch",
                }
            )
            launcher.atomic_write_json(state_path, state)

            resumed = launcher.load_or_create_state(manifest, state_path, [])

        self.assertEqual("succeeded", resumed["targets"][target_id]["status"])
        self.assertEqual(2500.01, resumed["targets"][target_id]["best_reward"])

    def test_multi_object_stage_waits_for_running_block_targets(self):
        manifest = launcher.load_manifest(
            MANIFEST_PATH, require_offsets=True, require_seed=False
        )
        state = launcher.new_state(manifest)
        block_targets = [target for target in manifest["targets"] if target.stage == "block"]
        multi_targets = [target for target in manifest["targets"] if target.stage == "multi_object"]
        for target in block_targets:
            state["targets"][target.target_id]["status"] = "succeeded"
        state["targets"][block_targets[-1].target_id]["status"] = "running"
        self.assertIsNone(launcher._next_pending_target(manifest, state))
        state["targets"][block_targets[-1].target_id]["status"] = "succeeded"
        self.assertEqual(
            multi_targets[0].target_id,
            launcher._next_pending_target(manifest, state).target_id,
        )

    def test_running_jobs_requeue_and_timed_out_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = launcher.load_manifest(
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=False
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
                calibrated_manifest_file(temporary_directory), require_offsets=True, require_seed=False
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
                    "certification": {"retained_success_rate": 0.60},
                }
            )
            launcher.atomic_write_json(state_path, state)
            resumed = launcher.load_or_create_state(manifest, state_path, [])

        self.assertEqual("succeeded", resumed["targets"][target_id]["status"])
        self.assertEqual(2500.01, resumed["targets"][target_id]["best_reward"])
        self.assertIsNone(resumed["targets"][target_id]["staged_checkpoint"])


if __name__ == "__main__":
    unittest.main()
