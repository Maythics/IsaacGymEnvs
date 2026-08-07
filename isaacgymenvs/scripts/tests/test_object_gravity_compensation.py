import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = Path(__file__).resolve().parents[2] / "tasks" / "object_gravity_compensation.py"
SPEC = importlib.util.spec_from_file_location("object_gravity_compensation_for_test", str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ObjectGravityCompensationMixin = MODULE.ObjectGravityCompensationMixin


class GravityCompensationTimingTests(unittest.TestCase):
    def _task(self, hold, ramp, progress):
        task = ObjectGravityCompensationMixin()
        task.object_gravity_compensation_seconds = hold
        task.object_gravity_ramp_seconds = ramp
        task.progress_buf = torch.tensor(progress, dtype=torch.long)
        task.control_freq_inv = 6
        task.dt = 1.0 / 60.0
        return task

    def test_zero_hold_is_exactly_disabled(self):
        task = self._task(0.0, 0.5, [0, 10, 100])
        value = task._gravity_compensation_fraction_for_substep(0)
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_tuned_point_two_second_hold_and_point_one_second_ramp(self):
        # One policy step is 0.1 seconds at dt=1/60 and decimation 6.
        task = self._task(0.2, 0.1, [0, 1, 2, 3])
        value = task._gravity_compensation_fraction_for_substep(0)
        expected = torch.tensor([1.0, 1.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(value, expected, atol=1.0e-6))
        midpoint = task._gravity_compensation_fraction_for_substep(3)
        self.assertAlmostEqual(0.5, midpoint[2].item(), places=5)

    def test_substep_time_is_included(self):
        task = self._task(0.1, 0.1, [1])
        first = task._gravity_compensation_fraction_for_substep(0)
        last = task._gravity_compensation_fraction_for_substep(5)
        self.assertAlmostEqual(1.0, first.item(), places=6)
        self.assertLess(last.item(), first.item())

    def test_enabled_compensation_uses_each_environment_object_mass(self):
        class FakeGym:
            def find_actor_handle(self, env_ptr, name):
                self.assert_name = name
                return env_ptr

            def get_actor_rigid_body_properties(self, env_ptr, object_handle):
                return [SimpleNamespace(mass=float(env_ptr))]

        task = ObjectGravityCompensationMixin()
        task.device = "cpu"
        task.num_envs = 2
        task.rb_forces = torch.zeros((2, 5, 3))
        task.object_rb_handles = torch.tensor([3], dtype=torch.long)
        task.envs = [1.25, 2.5]
        task.gym = FakeGym()
        task._configure_object_gravity_compensation({
            "env": {
                "objectGravityCompensationSeconds": 0.2,
                "objectGravityRampSeconds": 0.1,
            },
            "sim": {"gravity": [0.0, 0.0, -9.81]},
        })
        self.assertTrue(torch.allclose(
            task._gravity_object_masses[:, 0], torch.tensor([1.25, 2.5])
        ))
        self.assertTrue(task.has_sim_step_forces)


if __name__ == "__main__":
    unittest.main()
