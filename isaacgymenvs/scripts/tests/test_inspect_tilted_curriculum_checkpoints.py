import tempfile
import unittest
from pathlib import Path
from unittest import mock

from isaacgymenvs.scripts import inspect_tilted_curriculum_checkpoints as inspector
from isaacgymenvs.scripts import run_shadowhand18_tilt_curriculum as launcher


class CheckpointInspectorTests(unittest.TestCase):
    def test_viewer_command_reuses_logged_training_parameters(self):
        target = launcher.Target(
            "p180_t030", 180.0, 30.0, (0.0, -1.0, 0.0),
            (0.04, 0.09, 0.01), 21,
        )
        manifest = {
            "name": "remote_shadowhand18_tilted_block",
            "training": {
                "object_type": "block",
                "object_gravity_compensation_seconds": 0.2,
                "object_gravity_ramp_seconds": 0.1,
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "model.pth"
            output.touch()
            log = root / "run.log"
            log.write_text(
                "COMMAND: python train.py task=Shadowhand18Tilted num_envs=10240 "
                "train.params.config.minibatch_size=10240 checkpoint=/old/parent.pth "
                "task.env.baseTiltAngleDeg=30 task.env.baseTiltAxis='[0,-1,0]' "
                "task.env.objectPalmOffset='[0.04,0.09,0.01]' "
                "task.env.baseYawDeg=174.4678 headless=True experiment=old_run\n"
            )
            command = inspector.viewer_command(
                {"output_checkpoint": str(output), "log_path": str(log)},
                target, manifest, "custom-python", 64,
            )

        self.assertEqual("custom-python", command[0])
        self.assertIn("num_envs=64", command)
        self.assertIn("train.params.config.minibatch_size=64", command)
        self.assertIn("checkpoint={}".format(output.resolve()), command)
        self.assertIn("task.env.baseYawDeg=174.4678", command)
        self.assertIn("headless=False", command)
        self.assertIn("test=True", command)
        self.assertFalse(any(token.startswith("experiment=") for token in command))

    def test_selection_uses_state_status_and_manifest_order(self):
        late = launcher.Target("late", 0.0, 60.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 4)
        early = launcher.Target("early", 0.0, 30.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 1)
        manifest = {"targets": [late, early]}
        state = {
            "targets": {
                "late": {"status": "succeeded", "output_checkpoint": "/tmp/late.pth"},
                "early": {"status": "timed_out", "output_checkpoint": "/tmp/early.pth"},
            }
        }

        succeeded = inspector.selected_records(manifest, state, "succeeded", [])
        default_records = inspector.selected_records(manifest, state, "succeeded-and-timeout", [])
        all_outputs = inspector.selected_records(manifest, state, "all-output", [])

        self.assertEqual(["late"], [item[0].target_id for item in succeeded])
        self.assertEqual(["late", "early"], [item[0].target_id for item in default_records])
        self.assertEqual(["late", "early"], [item[0].target_id for item in all_outputs])

    def test_timeout_uses_highest_reward_checkpoint_from_all_attempts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = Path(temporary_directory) / "runs"
            first = runs / "sh18tilt_22_p180_t030_a01" / "nn" / "first.pth"
            latest = runs / "sh18tilt_22_p180_t030_a02" / "nn" / "latest.pth"
            first.parent.mkdir(parents=True)
            latest.parent.mkdir(parents=True)
            first.touch()
            latest.touch()
            record = {
                "status": "timed_out",
                "run_name": "sh18tilt_22_p180_t030_a02",
                "output_checkpoint": str(latest),
            }
            rewards = {str(first): 1800.0, str(latest): 1200.0}
            with mock.patch.object(
                inspector.curriculum,
                "read_checkpoint_reward",
                side_effect=lambda path: rewards[str(path)],
            ):
                checkpoint, reward = inspector.checkpoint_for_record(record)

        self.assertEqual(first, checkpoint)
        self.assertEqual(1800.0, reward)


if __name__ == "__main__":
    unittest.main()
