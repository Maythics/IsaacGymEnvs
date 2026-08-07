"""Per-object gravity grace period shared by fixed-tilt hand tasks."""

import torch


class ObjectGravityCompensationMixin:
    """Cancel object gravity briefly after each asynchronous environment reset."""

    def _configure_object_gravity_compensation(self, cfg):
        env_cfg = cfg["env"]
        self.object_gravity_compensation_seconds = float(
            env_cfg.get("objectGravityCompensationSeconds", 0.0)
        )
        self.object_gravity_ramp_seconds = float(
            env_cfg.get("objectGravityRampSeconds", 0.1)
        )
        if self.object_gravity_compensation_seconds < 0.0:
            raise ValueError("objectGravityCompensationSeconds must be non-negative")
        if self.object_gravity_ramp_seconds < 0.0:
            raise ValueError("objectGravityRampSeconds must be non-negative")

        gravity = cfg.get("sim", {}).get("gravity", [0.0, 0.0, -9.81])
        if len(gravity) != 3:
            raise ValueError("sim.gravity must contain exactly three values")
        self._gravity_vector = torch.tensor(
            gravity, dtype=torch.float, device=self.device
        )
        self._gravity_compensation_forces = torch.zeros_like(self.rb_forces)
        self.has_sim_step_forces = self.object_gravity_compensation_seconds > 0.0
        handles = self.object_rb_handles.reshape(-1)
        self._gravity_object_masses = None
        if self.object_gravity_compensation_seconds <= 0.0:
            # Preserve legacy startup cost as well as legacy physics when the
            # feature is disabled; actor mass queries are unnecessary.
            self.gravity_compensation_fraction = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
            print(
                "Object gravity compensation: hold=0s ramp={:.6g}s "
                "(hold=0 disables it)".format(self.object_gravity_ramp_seconds)
            )
            return
        per_env_masses = torch.zeros(
            (self.num_envs, handles.numel()), dtype=torch.float, device=self.device
        )
        # The parent tasks retain a single mass vector for random disturbances.
        # Gravity cancellation needs the actual actor masses in every env,
        # because multi-object pools and per-env scale randomization differ.
        for env_index, env_ptr in enumerate(self.envs):
            object_handle = self.gym.find_actor_handle(env_ptr, "object")
            properties = self.gym.get_actor_rigid_body_properties(
                env_ptr, object_handle
            )
            if len(properties) > handles.numel():
                raise RuntimeError(
                    "object has {} rigid bodies but only {} force slots".format(
                        len(properties), handles.numel()
                    )
                )
            if len(properties) > 0:
                per_env_masses[env_index, :len(properties)] = torch.tensor(
                    [prop.mass for prop in properties],
                    dtype=torch.float,
                    device=self.device,
                )
        self._gravity_object_masses = per_env_masses
        self.gravity_compensation_fraction = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        print(
            "Object gravity compensation: hold={:.6g}s ramp={:.6g}s "
            "(hold=0 disables it)".format(
                self.object_gravity_compensation_seconds,
                self.object_gravity_ramp_seconds,
            )
        )

    def _gravity_compensation_fraction_for_substep(self, substep):
        elapsed = (
            self.progress_buf.to(dtype=torch.float) * float(self.control_freq_inv)
            + float(substep)
        ) * float(self.dt)
        hold = self.object_gravity_compensation_seconds
        ramp = self.object_gravity_ramp_seconds
        if hold <= 0.0:
            return torch.zeros_like(elapsed)
        if ramp <= 0.0:
            return (elapsed < hold).to(dtype=torch.float)
        return torch.where(
            elapsed < hold,
            torch.ones_like(elapsed),
            torch.clamp(1.0 - (elapsed - hold) / ramp, min=0.0, max=1.0),
        )

    def apply_sim_step_forces(self, substep):
        if self.object_gravity_compensation_seconds <= 0.0:
            return
        fraction = self._gravity_compensation_fraction_for_substep(substep)
        self.gravity_compensation_fraction[:] = fraction
        if not bool(torch.any(fraction > 0.0)):
            return

        from isaacgym import gymapi, gymtorch

        forces = self._gravity_compensation_forces
        forces.zero_()
        masses = self._gravity_object_masses
        handles = self.object_rb_handles.reshape(-1)
        if masses.shape != (self.num_envs, handles.numel()):
            raise RuntimeError(
                "object rigid-body handles/masses differ: {} handles, mass shape {}".format(
                    handles.numel(), tuple(masses.shape)
                )
            )
        compensation = (
            -self._gravity_vector.view(1, 1, 3)
            * masses.unsqueeze(-1)
            * fraction.view(-1, 1, 1)
        )
        forces[:, handles, :] = compensation
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            None,
            gymapi.ENV_SPACE,
        )

    def _publish_gravity_compensation_metrics(self):
        self.extras["object_gravity_compensation_fraction"] = (
            self.gravity_compensation_fraction.mean()
        )
        fall_distance = torch.linalg.vector_norm(
            self.object_pos - self._fall_ref_pos(), dim=-1
        )
        dropped = fall_distance >= float(self.fall_dist)
        timed_out = self.progress_buf >= self.max_episode_length - 1
        has_goal = self.successes > 0
        done = self.reset_buf != 0
        # Per-environment tensors are intentionally exposed for the standalone
        # evaluator. RL Games' observer ignores non-scalar direct metrics.
        self.extras["tilt_goal_hit"] = has_goal
        self.extras["tilt_dropped"] = dropped
        self.extras["tilt_timed_out"] = timed_out
        self.extras["tilt_retained_success"] = done & has_goal & ~dropped
        self.extras["tilt_timeout_without_success"] = done & timed_out & ~has_goal
        self.extras["tilt_object_palm_distance"] = fall_distance
        self.extras["tilt_object_type_index"] = self.env_object_type_idx
