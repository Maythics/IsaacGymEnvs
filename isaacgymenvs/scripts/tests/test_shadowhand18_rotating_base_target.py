"""Unit tests plus an opt-in Isaac Gym smoke test for the additive task."""
from __future__ import annotations

from isaacgym import gymapi  # noqa: F401  (must precede torch)

import math
import os
import unittest
from pathlib import Path

import torch
import yaml

from isaacgymenvs.tasks.shadow_hand_18_rotating_base_target import (
    ShadowHand18RotatingBaseTarget,
    compose_pose_xyzw,
    relative_pose_xyzw,
    world_y_root_quat,
)


class PoseHelperTests(unittest.TestCase):
    def test_relative_compose_round_trip(self):
        parent = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
        local = torch.tensor([[0.04, 0.02, -0.01, 0.0, 0.0,
                               math.sin(0.2), math.cos(0.2)]])
        world = compose_pose_xyzw(parent, local)
        recovered = relative_pose_xyzw(world, parent)
        self.assertTrue(torch.allclose(recovered, local, atol=1.0e-6))

    def test_world_y_rotation_is_absolute(self):
        initial = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        angle = torch.tensor([math.pi / 2.0])
        got = world_y_root_quat(initial, angle)
        expected = torch.tensor([[0.0, math.sin(math.pi / 4.0), 0.0,
                                  math.cos(math.pi / 4.0)]])
        self.assertTrue(torch.allclose(got, expected, atol=1.0e-6))

    def test_config_preserves_direct_target_contract(self):
        path = Path(__file__).resolve().parents[2] / "cfg/task/Shadowhand18RotatingBaseTarget.yaml"
        with path.open() as stream:
            cfg = yaml.safe_load(stream)["env"]
        self.assertEqual(1, cfg["controlFrequencyInv"])
        self.assertEqual(1.0, cfg["actionSpeedScale"] * cfg["actionsMovingAverage"])
        self.assertFalse(cfg["useRelativeControl"])


@unittest.skipUnless(
    os.environ.get("RUN_ROTATING_BASE_SMOKE") == "1",
    "set RUN_ROTATING_BASE_SMOKE=1 to launch the GPU Isaac Gym smoke test",
)
class IsaacGymSmokeTest(unittest.TestCase):
    def test_task_steps_with_runtime_registration(self):
        import isaacgymenvs
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from isaacgymenvs.tasks import isaacgym_task_map

        isaacgym_task_map["Shadowhand18RotatingBaseTarget"] = (
            ShadowHand18RotatingBaseTarget
        )
        cfg_dir = Path(isaacgymenvs.__file__).resolve().parent / "cfg"
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "task=Shadowhand18RotatingBaseTarget",
                    "train=Shadowhand18PPO",
                    "task.env.numEnvs=1",
                    "task.env.objectType=block",
                    "task.env.baseAngularVelocityRadS=0.2",
                    "headless=true",
                    "test=true",
                    "sim_device=cuda:0",
                    "rl_device=cuda:0",
                ],
            )
        env = isaacgymenvs.make(
            seed=0, task=cfg.task_name, num_envs=1,
            sim_device=cfg.sim_device, rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id, headless=True,
            multi_gpu=False, virtual_screen_capture=False,
            force_render=False, cfg=cfg,
        )
        task = env
        while not hasattr(task, "target_object_pose_palm"):
            task = task.task if hasattr(task, "task") else task.env
        self.assertEqual(18, task.num_actions)
        target = task.target_object_pose_palm.clone()
        task.set_object_in_palm_target(target)
        actions = torch.zeros((1, 18), device=task.device)
        for _ in range(4):
            env.step(actions)
        self.assertTrue(torch.isfinite(task.base_angle_rad).all())
        self.assertTrue(torch.allclose(task.root_state_tensor[task.hand_indices, :3],
                                       task.base_root_pos, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
