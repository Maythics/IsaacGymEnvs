import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from isaacgymenvs.scripts import run_shadowhand18_tilt_curriculum as core
from isaacgymenvs.scripts import run_wujihand_fixed_tilt_curriculum as wuji_launcher


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WUJI_MANIFEST = (
    REPOSITORY_ROOT
    / "isaacgymenvs"
    / "curricula"
    / "wujihand_fixed_tilt_42.yaml"
)
WUJI_TASK_CONFIG = (
    REPOSITORY_ROOT
    / "isaacgymenvs"
    / "cfg"
    / "task"
    / "WujiHandFixedTilt.yaml"
)
FIXED_WRIST_ASSET = (
    REPOSITORY_ROOT
    / "assets"
    / "urdf"
    / "wuji"
    / "wuji_right_fixed_wrist_compat.urdf"
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

    def test_all_42_wuji_offsets_are_calibrated(self):
        wuji = core.load_manifest(
            WUJI_MANIFEST, require_offsets=False, require_seed=True
        )
        wuji_offsets = {
            target.target_id: target.object_offset for target in wuji["targets"][:42]
        }
        self.assertEqual(42, len(wuji_offsets))
        self.assertEqual(0, sum(offset is None for offset in wuji_offsets.values()))

    def test_fixed_tilt_uses_physically_fixed_compatibility_asset(self):
        with WUJI_TASK_CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            "wuji_right_fixed_wrist_compat.urdf",
            config["env"]["assetFileNameWuji"],
        )
        self.assertTrue(config["env"]["compatibilityDummyWristDofs"])

        root = ET.parse(str(FIXED_WRIST_ASSET)).getroot()
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        self.assertEqual("revolute", joints["right_hand_WRJ2"].attrib["type"])
        self.assertEqual("revolute", joints["right_hand_WRJ1"].attrib["type"])
        self.assertEqual("fixed", joints["right_hand_WRJ2_fixed"].attrib["type"])
        self.assertEqual("fixed", joints["right_hand_WRJ1_fixed"].attrib["type"])

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
        self.assertIn("task.env.objectPalmOffset=[0,0.05,0]", command)
        self.assertIn("task.env.objectGravityCompensationSeconds=0.2", command)
        self.assertIn("task.env.objectGravityRampSeconds=0.1", command)
        self.assertEqual(
            "wujitilt_01_p000_t030_a01",
            wuji_launcher.make_wuji_run_name(target, 1),
        )

    def test_calibrated_offsets_pass_strict_validation(self):
        manifest = core.load_manifest(
            WUJI_MANIFEST, require_offsets=True, require_seed=True
        )
        self.assertGreater(len(manifest["targets"]), 42)
        self.assertEqual(42, len(manifest["targets"][:42]))


if __name__ == "__main__":
    unittest.main()
