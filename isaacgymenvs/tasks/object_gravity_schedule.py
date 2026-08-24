"""Per-object gravity schedules expressed in an initial palm frame.

The simulator gravity is deliberately left unchanged.  This lets a task vary
the gravity experienced by the manipulated object without rotating the hand
root or moving the object spawn distribution in world coordinates.
"""

import math

import torch

from isaacgymenvs.utils.torch_jit_utils import quat_rotate, quat_rotate_inverse


class ObjectGravityScheduleMixin:
    """Apply an object-only gravity vector with a reset-local transition."""

    def _configure_object_gravity_schedule(self, cfg):
        env_cfg = cfg["env"]
        self.object_gravity_hold_seconds = float(
            env_cfg.get("objectGravityHoldSeconds", 0.2)
        )
        self.object_gravity_ramp_seconds = float(
            env_cfg.get("objectGravityRampSeconds", 0.2)
        )
        if self.object_gravity_hold_seconds < 0.0:
            raise ValueError("objectGravityHoldSeconds must be non-negative")
        if self.object_gravity_ramp_seconds < 0.0:
            raise ValueError("objectGravityRampSeconds must be non-negative")

        gravity = cfg.get("sim", {}).get("gravity", [0.0, 0.0, -9.81])
        if len(gravity) != 3:
            raise ValueError("sim.gravity must contain exactly three values")
        self.sim_gravity_world = torch.tensor(
            gravity, dtype=torch.float, device=self.device
        )
        gravity_norm = torch.linalg.vector_norm(self.sim_gravity_world)
        if gravity_norm.item() <= 1.0e-8:
            raise ValueError("sim.gravity must have non-zero magnitude")
        self._gravity_magnitude = gravity_norm

        # This is intentionally measured from Isaac Gym's resolved rigid-body
        # state.  ShadowHand and Wuji use different palm axes, so neither task
        # may assume that native gravity is palm -Z.
        self.initial_palm_quat = self.rigid_body_states[
            :, self.palm_body_idx, 3:7
        ].clone()
        native = quat_rotate_inverse(
            self.initial_palm_quat,
            self.sim_gravity_world.unsqueeze(0).expand(self.num_envs, -1),
        ) / self._gravity_magnitude
        self.native_gravity_in_palm = native

        configured = env_cfg.get("gravityInPalm", None)
        if configured is None:
            target = native[0]
        else:
            try:
                values = [float(value) for value in configured]
            except (TypeError, ValueError) as exc:
                raise ValueError("gravityInPalm must contain exactly three numbers") from exc
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError("gravityInPalm must contain exactly three finite numbers")
            target = torch.tensor(values, dtype=torch.float, device=self.device)
            target_norm = torch.linalg.vector_norm(target)
            if abs(target_norm.item() - 1.0) > 1.0e-4:
                raise ValueError("gravityInPalm must be a unit vector")
            target = target / target_norm

        self.target_gravity_in_palm = target.unsqueeze(0).expand(
            self.num_envs, -1
        ).clone()
        self.target_gravity_world = quat_rotate(
            self.initial_palm_quat,
            self.target_gravity_in_palm * self._gravity_magnitude,
        )
        self._gravity_schedule_forces = torch.zeros_like(self.rb_forces)
        self.gravity_schedule_fraction = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        handles = self.object_rb_handles.reshape(-1)
        masses = torch.zeros(
            (self.num_envs, handles.numel()), dtype=torch.float, device=self.device
        )
        for env_index, env_ptr in enumerate(self.envs):
            object_handle = self.gym.find_actor_handle(env_ptr, "object")
            properties = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            if len(properties) > handles.numel():
                raise RuntimeError("object has more rigid bodies than force slots")
            if properties:
                masses[env_index, :len(properties)] = torch.tensor(
                    [prop.mass for prop in properties], dtype=torch.float,
                    device=self.device,
                )
        self._gravity_schedule_object_masses = masses
        self.has_sim_step_forces = True
        print(
            "Object gravity schedule: native_g_palm={} target_g_palm={} "
            "hold={:.6g}s ramp={:.6g}s".format(
                native[0].detach().cpu().tolist(), target.detach().cpu().tolist(),
                self.object_gravity_hold_seconds, self.object_gravity_ramp_seconds,
            )
        )

    def _gravity_schedule_fraction_for_substep(self, substep):
        elapsed = (
            self.progress_buf.to(dtype=torch.float) * float(self.control_freq_inv)
            + float(substep)
        ) * float(self.dt)
        hold = self.object_gravity_hold_seconds
        ramp = self.object_gravity_ramp_seconds
        if ramp <= 0.0:
            return (elapsed >= hold).to(dtype=torch.float)
        return torch.clamp((elapsed - hold) / ramp, min=0.0, max=1.0)

    def effective_object_gravity_world(self, substep=0):
        """Return the per-environment object acceleration for this substep."""
        fraction = self._gravity_schedule_fraction_for_substep(substep)
        gravity = self.sim_gravity_world.unsqueeze(0) + fraction.unsqueeze(-1) * (
            self.target_gravity_world - self.sim_gravity_world.unsqueeze(0)
        )
        return gravity, fraction

    def effective_object_gravity_in_palm(self, substep=0):
        """Return the current effective acceleration in the *initial* palm frame."""
        gravity, _ = self.effective_object_gravity_world(substep)
        return quat_rotate_inverse(self.initial_palm_quat, gravity) / self._gravity_magnitude

    def apply_sim_step_forces(self, substep):
        gravity, fraction = self.effective_object_gravity_world(substep)
        self.gravity_schedule_fraction[:] = fraction
        forces = self._gravity_schedule_forces
        forces.zero_()
        handles = self.object_rb_handles.reshape(-1)
        delta = gravity - self.sim_gravity_world.unsqueeze(0)
        forces[:, handles, :] = (
            self._gravity_schedule_object_masses.unsqueeze(-1) * delta.unsqueeze(1)
        )
        from isaacgym import gymapi, gymtorch
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE
        )

    def _publish_gravity_schedule_metrics(self):
        self.extras["object_gravity_schedule_fraction"] = (
            self.gravity_schedule_fraction.mean()
        )
