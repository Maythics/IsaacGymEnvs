import math
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

_SCRIPT = Path(__file__).resolve().parents[1] / "run_gravity_curriculum.py"
_SPEC = importlib.util.spec_from_file_location("run_gravity_curriculum", _SCRIPT)
launcher = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = launcher
_SPEC.loader.exec_module(launcher)


class GravityCurriculumTest(unittest.TestCase):
    def test_canonical_targets_cover_sphere_without_duplicates(self):
        targets = launcher.canonical_targets_42()
        self.assertEqual(42, len(targets))
        self.assertEqual(42, len({target.target_id for target in targets}))
        self.assertEqual(42, len({tuple(round(v, 7) for v in t.gravity_in_palm) for t in targets}))
        for target in targets:
            self.assertAlmostEqual(1.0, math.sqrt(sum(v * v for v in target.gravity_in_palm)))

    def test_native_order_starts_at_matching_target(self):
        shadow = launcher.ordered_targets((0.0, 1.0, 0.0))
        wuji = launcher.ordered_targets((-1.0, 0.0, 0.0))
        self.assertEqual("p270_t090", shadow[0].target_id)
        self.assertEqual("p000_t090", wuji[0].target_id)

    def test_command_contains_only_gravity_task_parameters(self):
        target = launcher.GravityTarget("check", (0.0, 0.0, -1.0))
        command = launcher.build_command(
            "python", "Shadowhand18Gravity", "/tmp/checkpoint.pth", target,
            {"num_envs": 64, "minibatch_size": 64, "max_iterations": 1,
             "episode_length": 300, "object_type": "block", "score_to_win": 2500,
             "save_best_after": 1}, "run",
        )
        rendered = " ".join(command)
        self.assertIn("task.env.gravityInPalm=[0,0,-1]", rendered)
        self.assertIn("task.env.objectGravityHoldSeconds=0.2", rendered)
        self.assertIn("task.env.objectGravityRampSeconds=0.2", rendered)
        self.assertIn("train.params.config.score_to_win=2500", rendered)
        self.assertNotIn("baseTilt", rendered)
        self.assertNotIn("baseYaw", rendered)

    def test_parent_reuse_radius_is_two_minimum_intervals(self):
        targets = launcher.canonical_targets_42()
        spacing = launcher.minimum_target_spacing(targets)
        self.assertAlmostEqual(22.0621911575, spacing, places=5)
        seed = {"source_id": "seed", "gravity_in_palm": [0.0, 1.0, 0.0], "completion_seq": -1}
        near = next(target for target in targets if target.target_id == "p270_t120")
        far = next(target for target in targets if target.target_id == "p090_t090")
        self.assertIsNotNone(launcher.nearest_parent(near, [seed], 2.0 * spacing))
        self.assertIsNone(launcher.nearest_parent(far, [seed], 2.0 * spacing))

    def test_retry_all_preserves_successes_and_requeues_unsuccessful_targets(self):
        state = {"targets": {
            "passed": {"status": "succeeded", "attempts": 1, "checkpoint": "/tmp/passed.pth"},
            "near": {"status": "failed", "attempts": 1, "run_name": "gravity_01_near_a01"},
            "stopped": {"status": "pending", "attempts": 2, "run_name": "gravity_02_stopped_a02"},
        }}
        launcher.apply_retry(state, "all")
        self.assertEqual("succeeded", state["targets"]["passed"]["status"])
        self.assertEqual("pending", state["targets"]["near"]["status"])
        self.assertEqual("gravity_01_near_a01", state["targets"]["near"]["run_name"])
        self.assertEqual("pending", state["targets"]["stopped"]["status"])

    def test_target_resume_uses_run_named_best_checkpoint_not_last(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            run_name = "gravity_01_p000_t090_a01"
            nn_dir = package / "runs" / run_name / "nn"
            nn_dir.mkdir(parents=True)
            torch.save({"last_mean_rewards": 200.0}, nn_dir / (run_name + ".pth"))
            torch.save({"last_mean_rewards": 999.0}, nn_dir / (run_name + "_last.pth"))
            checkpoint, reward = launcher.checkpoint_from_target_runs(package, "p000_t090")
            self.assertTrue(checkpoint.endswith(run_name + ".pth"))
            self.assertEqual(200.0, reward)


if __name__ == "__main__":
    unittest.main()
