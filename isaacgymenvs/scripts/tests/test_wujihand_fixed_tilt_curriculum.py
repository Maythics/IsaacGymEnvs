import unittest
from pathlib import Path

from isaacgymenvs.scripts import run_shadowhand18_tilt_curriculum as core
from isaacgymenvs.scripts import run_wujihand_fixed_tilt_curriculum as wuji_launcher


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WUJI_MANIFEST = (
    REPOSITORY_ROOT
    / "isaacgymenvs"
    / "curricula"
    / "wujihand_fixed_tilt_42.yaml"
)
SHADOW_MANIFEST = (
    REPOSITORY_ROOT
    / "isaacgymenvs"
    / "curricula"
    / "shadowhand18_tilt_42.yaml"
)


class WujiManifestTests(unittest.TestCase):
    def test_zero_degree_seed_is_real_and_above_threshold(self):
        manifest = core.load_manifest(
            WUJI_MANIFEST, require_offsets=False, require_seed=True
        )
        self.assertEqual(0.0, manifest["seed_theta_deg"])
        self.assertEqual(0.0, manifest["seed_phi_deg"])
        self.assertEqual("WujiHand.pth", manifest["seed_checkpoint"].name)
        reward = core.read_checkpoint_reward(manifest["seed_checkpoint"])
        self.assertGreater(reward, float(manifest["training"]["score_to_win"]))

    def test_all_42_offsets_are_copied_exactly_from_shadow_manifest(self):
        wuji = core.load_manifest(
            WUJI_MANIFEST, require_offsets=False, require_seed=True
        )
        shadow = core.load_manifest(
            SHADOW_MANIFEST, require_offsets=False, require_seed=True
        )
        wuji_offsets = {
            target.target_id: target.object_offset for target in wuji["targets"]
        }
        shadow_offsets = {
            target.target_id: target.object_offset for target in shadow["targets"]
        }
        self.assertEqual(shadow_offsets, wuji_offsets)
        self.assertEqual(42, len(wuji_offsets))
        self.assertEqual(0, sum(offset is None for offset in wuji_offsets.values()))

    def test_wuji_command_targets_new_fixed_tilt_task(self):
        manifest = core.load_manifest(
            WUJI_MANIFEST, require_offsets=False, require_seed=True
        )
        target = manifest["targets"][0]
        command = wuji_launcher.build_wuji_command(
            "python",
            target,
            Path("/tmp/parent.pth"),
            "wujitilt_test",
            manifest["training"],
        )
        self.assertIn("task=WujiHandFixedTilt", command)
        self.assertNotIn("task=Shadowhand18Tilted", command)
        self.assertIn("task.env.baseTiltAngleDeg=30", command)
        self.assertIn("task.env.objectPalmOffset=[0.08,0,-0.01]", command)
        self.assertEqual(
            "wujitilt_01_p000_t030_a01",
            wuji_launcher.make_wuji_run_name(target, 1),
        )

    def test_copied_offsets_pass_strict_validation(self):
        manifest = core.load_manifest(
            WUJI_MANIFEST, require_offsets=True, require_seed=True
        )
        self.assertEqual(42, len(manifest["targets"]))


if __name__ == "__main__":
    unittest.main()
