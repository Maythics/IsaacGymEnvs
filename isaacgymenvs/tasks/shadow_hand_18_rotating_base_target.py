"""Evaluation-only Shadowhand18 task with a rotating base and palm-frame goal.

The task keeps the 18-action frozen-wrist interface, rotates the hand root at a
constant signed angular velocity about the *world* Y axis, and accepts an
externally supplied object pose in the live palm frame.  It is intentionally
not registered in :mod:`isaacgymenvs.tasks`; the downstream evaluator imports
and registers it in memory so existing IsaacGymEnvs source files stay intact.

Quaternion convention in this module is Isaac Gym's scalar-last
``[qx, qy, qz, qw]`` convention.
"""
from __future__ import annotations

import math

import torch
from isaacgym import gymtorch

from isaacgymenvs.tasks.shadow_hand import ShadowHand
from isaacgymenvs.tasks.shadow_hand_tilted import ShadowHandTilted
from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)


def compose_pose_xyzw(parent: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Compose scalar-last poses: ``world = parent * local``."""
    pos = parent[..., :3] + quat_rotate(parent[..., 3:7], local[..., :3])
    quat = quat_mul(parent[..., 3:7], local[..., 3:7])
    return torch.cat([pos, quat], dim=-1)


def relative_pose_xyzw(world: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Express a scalar-last world pose in ``frame`` coordinates."""
    pos = quat_rotate_inverse(frame[..., 3:7], world[..., :3] - frame[..., :3])
    quat = quat_mul(quat_conjugate(frame[..., 3:7]), world[..., 3:7])
    return torch.cat([pos, quat], dim=-1)


def world_y_root_quat(initial_quat: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Apply an absolute world-Y rotation to each initial root quaternion."""
    axis = torch.zeros((*angle_rad.shape, 3), dtype=initial_quat.dtype,
                       device=initial_quat.device)
    axis[..., 1] = 1.0
    return quat_mul(quat_from_angle_axis(angle_rad, axis), initial_quat)


class ShadowHand18RotatingBaseTarget(ShadowHandTilted):
    """Shadowhand18 direct-target evaluation under deterministic base rotation."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        env_cfg = cfg["env"]
        self.base_angular_velocity_cfg = float(
            env_cfg.get("baseAngularVelocityRadS", 0.2)
        )
        self.base_initial_angle_rad = math.radians(
            float(env_cfg.get("baseInitialAngleDeg", 0.0))
        )
        self.base_motion_delay_s = float(env_cfg.get("baseMotionDelaySec", 0.5))
        if self.base_motion_delay_s < 0.0:
            raise ValueError("baseMotionDelaySec must be non-negative")

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                         virtual_screen_capture, force_render)

        if not self.freeze_wrist or not self.reduce_wrist_actions:
            raise RuntimeError(
                "ShadowHand18RotatingBaseTarget requires frozen wrist and 18 actions"
            )
        if self.control_freq_inv != 1:
            raise RuntimeError(
                "ShadowHand18RotatingBaseTarget requires controlFrequencyInv=1 "
                "so the base advances once per 60 Hz physics step"
            )
        effective_beta = self.action_speed_scale * self.act_moving_average
        if self.use_relative_control or abs(effective_beta - 1.0) > 1.0e-8:
            raise RuntimeError(
                "Incoming actions are effective absolute PD targets; configure "
                "useRelativeControl=False and actionSpeedScale*"
                "actionsMovingAverage=1"
            )

        self.base_root_pos = self.root_state_tensor[self.hand_indices, 0:3].clone()
        self.base_elapsed_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.base_angle_rad = torch.full(
            (self.num_envs,), self.base_initial_angle_rad,
            dtype=torch.float, device=self.device,
        )

        # Default external target is the object's actual reset pose in the palm
        # frame.  The evaluator replaces it before the benchmark starts.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        palm = self.rigid_body_states[:, self.palm_body_idx, 0:7]
        obj = self.root_state_tensor[self.object_indices, 0:7]
        self.target_object_pose_palm = relative_pose_xyzw(obj, palm).clone()

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._write_absolute_base_pose(all_env_ids)
        self._sync_world_goal_from_palm_target(all_env_ids, push_actor=True)

    def _randomize_base_rotation(self, env_ids):
        """Keep the parent buffers but make axis/speed deterministic."""
        self.hand_rot_axis[env_ids] = 0.0
        self.hand_rot_axis[env_ids, 1] = 1.0
        self.hand_rot_speed[env_ids] = self.base_angular_velocity_cfg

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)
        if hasattr(self, "base_elapsed_time"):
            self.reset_base_motion(env_ids)

    def reset_base_motion(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset scripted base time/orientation without resetting hand or object."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            if env_ids.ndim == 0:
                env_ids = env_ids.unsqueeze(0)
        self.base_elapsed_time[env_ids] = 0.0
        self.base_angle_rad[env_ids] = self.base_initial_angle_rad
        self._write_absolute_base_pose(env_ids)

    def reset_target_pose(self, env_ids, apply_reset=False, from_goal_reach=False,
                          base_pos=None):
        # During the parent constructor the external target buffer does not yet
        # exist, so retain ordinary initialization.  Afterwards all goals are
        # controlled by set_object_in_palm_target().
        if not hasattr(self, "target_object_pose_palm"):
            return super().reset_target_pose(
                env_ids, apply_reset=apply_reset,
                from_goal_reach=from_goal_reach, base_pos=base_pos,
            )
        self.reset_goal_buf[env_ids] = 0

    def set_object_in_palm_target(
        self,
        target_pose_palm: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set absolute object targets in the live palm frame.

        ``target_pose_palm`` accepts ``(7,)``, ``(1,7)``, or ``(N,7)`` and uses
        Isaac Gym scalar-last quaternions. A single row broadcasts to all
        selected environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            if env_ids.ndim == 0:
                env_ids = env_ids.unsqueeze(0)

        pose = torch.as_tensor(
            target_pose_palm, dtype=torch.float, device=self.device
        )
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        if pose.ndim != 2 or pose.shape[1] != 7:
            raise ValueError(
                f"target_pose_palm must have shape (7,), (1,7), or (N,7); "
                f"got {tuple(pose.shape)}"
            )
        if pose.shape[0] == 1 and env_ids.numel() > 1:
            pose = pose.expand(env_ids.numel(), -1)
        if pose.shape[0] != env_ids.numel():
            raise ValueError(
                f"target rows ({pose.shape[0]}) must equal selected envs "
                f"({env_ids.numel()})"
            )
        if not bool(torch.isfinite(pose).all().item()):
            raise ValueError("target_pose_palm contains non-finite values")

        quat_norm = pose[:, 3:7].norm(dim=-1, keepdim=True)
        if bool((quat_norm < 1.0e-8).any().item()):
            raise ValueError("target_pose_palm quaternion norm must be non-zero")
        pose = pose.clone()
        pose[:, 3:7] /= quat_norm
        self.target_object_pose_palm[env_ids] = pose

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._sync_world_goal_from_palm_target(env_ids, push_actor=True)

    def _write_absolute_base_pose(self, env_ids: torch.Tensor) -> None:
        root_quat = world_y_root_quat(
            self.hand_initial_quat[env_ids], self.base_angle_rad[env_ids]
        )
        self.hand_current_quat[env_ids] = root_quat
        self.root_state_tensor[self.hand_indices[env_ids], 0:3] = self.base_root_pos[env_ids]
        self.root_state_tensor[self.hand_indices[env_ids], 3:7] = root_quat
        self.root_state_tensor[self.hand_indices[env_ids], 7:13] = 0.0

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(hand_indices),
            len(hand_indices),
        )

    def _advance_base_motion(self) -> None:
        self.base_elapsed_time += float(self.dt)
        active_time = torch.clamp(
            self.base_elapsed_time - self.base_motion_delay_s, min=0.0
        )
        self.base_angle_rad = (
            self.base_initial_angle_rad
            + self.hand_rot_speed * active_time
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._write_absolute_base_pose(env_ids)

    def _sync_world_goal_from_palm_target(
        self, env_ids: torch.Tensor, *, push_actor: bool
    ) -> None:
        palm = self.rigid_body_states[env_ids, self.palm_body_idx, 0:7]
        goal_world = compose_pose_xyzw(
            palm, self.target_object_pose_palm[env_ids]
        )
        self.goal_states[env_ids, 0:7] = goal_world
        self.goal_states[env_ids, 7:13] = 0.0

        goal_actor_ids = self.goal_object_indices[env_ids]
        self.root_state_tensor[goal_actor_ids, 0:3] = (
            goal_world[:, 0:3] + self.goal_displacement_tensor
        )
        self.root_state_tensor[goal_actor_ids, 3:7] = goal_world[:, 3:7]
        self.root_state_tensor[goal_actor_ids, 7:13] = 0.0
        if push_actor:
            actor_ids_i32 = goal_actor_ids.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(actor_ids_i32),
                len(actor_ids_i32),
            )

    def pre_physics_step(self, actions):
        # A reached external target must never trigger the parent's random goal
        # sampler. ShadowHand.pre_physics_step supplies reset handling, direct
        # target scaling, wrist freezing, and optional disturbance forces.
        self.reset_goal_buf.zero_()
        ShadowHand.pre_physics_step(self, actions)
        self._advance_base_motion()

    def compute_observations(self):
        # The parent must see the world goal composed from the palm pose after
        # the just-completed base/physics step.
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sync_world_goal_from_palm_target(env_ids, push_actor=True)
        super().compute_observations()

        palm = self.rigid_body_states[:, self.palm_body_idx, 0:7]
        object_palm = relative_pose_xyzw(self.object_pose, palm)
        pos_err = (object_palm[:, :3] - self.target_object_pose_palm[:, :3]).norm(dim=-1)
        q_err = quat_mul(
            object_palm[:, 3:7],
            quat_conjugate(self.target_object_pose_palm[:, 3:7]),
        )
        rot_err = 2.0 * torch.asin(
            torch.clamp(q_err[:, :3].norm(dim=-1), max=1.0)
        )
        palm_dist = (self.object_pos - palm[:, :3]).norm(dim=-1)

        self.extras["base_angle_rad"] = self.base_angle_rad.clone()
        self.extras["base_angular_velocity_rad_s"] = self.hand_rot_speed.clone()
        self.extras["object_palm_pos_error"] = pos_err
        self.extras["object_palm_rot_error"] = rot_err
        self.extras["target_reached"] = (
            (pos_err <= self.pos_success_tolerance)
            & (rot_err <= self.success_tolerance)
        )
        self.extras["object_palm_distance"] = palm_dist
        self.extras["dropped"] = palm_dist >= self.fall_dist
