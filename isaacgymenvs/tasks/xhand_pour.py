"""XHandPour: XHandHand variant with the same pour task as ShadowPour.

Differs only in the underlying robot (XHand has 14 DOFs and 175-dim
observations); the sub-goal trajectory, phase tracking, and reward share
the same helpers as ShadowPour.
"""

import math

import torch

from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.xhand_hand import XHandHand
from isaacgymenvs.tasks._pour_common import build_subgoal_quats, compute_pour_reward
from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    to_torch,
    torch_rand_float,
)


class XHandPour(XHandHand):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        cfg["env"]["objectType"] = "bottle"
        cfg["env"]["objectTypePool"] = ["bottle"]

        self.roll_angle_min = float(cfg["env"].get("rollAngleMin", math.radians(80.0)))
        self.roll_angle_max = float(cfg["env"].get("rollAngleMax", math.radians(100.0)))
        self._roll_axis_cfg = cfg["env"].get("rollAxis", [1.0, 0.0, 0.0])

        self.num_subgoals = int(cfg["env"].get("numSubGoals", 5))
        self.pour_final_pitch = math.radians(float(cfg["env"].get("pourFinalPitchDeg", 90.0)))
        self._pour_pitch_axis_cfg = cfg["env"].get("pourPitchAxis", [0.0, 1.0, 0.0])

        self.bottle_spawn_pos_noise = float(cfg["env"].get("bottleSpawnPosNoise", 0.03))
        # World-frame biases on top of the palm anchor (match ShadowPour defaults)
        self.bottle_spawn_x_bias = float(cfg["env"].get("bottleSpawnXBias", 0.03))
        self.bottle_spawn_y_bias = float(cfg["env"].get("bottleSpawnYBias", 0.08))
        self.bottle_spawn_z_offset = float(cfg["env"].get("bottleSpawnZOffset", 0.10))
        self.bottle_spawn_rot_noise = float(cfg["env"].get("bottleSpawnRotNoise", 0.2))

        # Two ways to align XHand's palm with ShadowPour's palm:
        #   - palmWorldTarget: target palm world position; the hand base is
        #     auto-translated so the palm lands here. Read ShadowPour's
        #     printed palm world position and paste it into the YAML.
        #   - handBasePosOverride: direct base-position override (fallback).
        # palmWorldTarget takes precedence when both are set.
        self._palm_world_target_cfg = cfg["env"].get("palmWorldTarget", None)
        self._hand_base_pos_override = cfg["env"].get("handBasePosOverride", None)

        self.grasp_proximity_scale = float(cfg["env"].get("graspProximityScale", -5.0))
        self.grasp_contact_scale = float(cfg["env"].get("graspContactScale", -2.0))
        self.grasp_bonus = float(cfg["env"].get("graspBonus", 50.0))
        self.grasp_proximity_threshold = float(cfg["env"].get("graspProximityThreshold", 0.10))
        self.grasp_linvel_threshold = float(cfg["env"].get("graspLinvelThreshold", 0.20))
        self.grasp_angvel_threshold = float(cfg["env"].get("graspAngvelThreshold", 1.50))

        self.subgoal_rot_scale = float(cfg["env"].get("subgoalRotScale", 2.0))
        self.subgoal_pos_scale = float(cfg["env"].get("subgoalPosScale", -1.0))
        self.subgoal_advance_bonus = float(cfg["env"].get("subgoalAdvanceBonus", 100.0))
        self.subgoal_success_tolerance = float(cfg["env"].get("subgoalSuccessTolerance", 0.20))
        self.subgoal_hold_frames = int(cfg["env"].get("subgoalHoldFrames", 10))
        self.final_success_bonus = float(cfg["env"].get("finalSuccessBonus", 400.0))

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        device = self.device
        N = self.num_envs

        self.roll_axis_tensor = to_torch(self._roll_axis_cfg, device=device, dtype=torch.float)
        self.roll_axis_tensor = self.roll_axis_tensor / (self.roll_axis_tensor.norm() + 1e-8)
        self.pour_pitch_axis_tensor = to_torch(self._pour_pitch_axis_cfg, device=device, dtype=torch.float)
        self.pour_pitch_axis_tensor = self.pour_pitch_axis_tensor / (self.pour_pitch_axis_tensor.norm() + 1e-8)

        self.hand_initial_quat = self.root_state_tensor[self.hand_indices, 3:7].clone()
        self.hand_initial_pos = self.root_state_tensor[self.hand_indices, 0:3].clone()

        palm_world_init = self.rigid_body_states[:, self.palm_body_idx, 0:3]
        palm_world_rot_init = self.rigid_body_states[:, self.palm_body_idx, 3:7]
        palm_offset_world = (palm_world_init[0] - self.hand_initial_pos[0]).clone()
        q_init_inv = quat_conjugate(self.hand_initial_quat[0:1])
        self.palm_local_offset = quat_apply(q_init_inv, palm_offset_world.unsqueeze(0)).squeeze(0)
        self.palm_local_rot = quat_mul(q_init_inv, palm_world_rot_init[0:1]).squeeze(0)

        # Resolve hand base override
        override = None
        if self._palm_world_target_cfg is not None and len(self._palm_world_target_cfg) == 3:
            target = to_torch(self._palm_world_target_cfg, device=device, dtype=torch.float)
            # palm_world = hand_pos + R_init * palm_local_offset → solve for hand_pos
            palm_offset_world_init = quat_apply(
                self.hand_initial_quat[0:1],
                self.palm_local_offset.unsqueeze(0),
            ).squeeze(0)
            override = target - palm_offset_world_init
        elif self._hand_base_pos_override is not None and len(self._hand_base_pos_override) == 3:
            override = to_torch(self._hand_base_pos_override, device=device, dtype=torch.float)

        if override is not None:
            self.hand_initial_pos[:] = override.unsqueeze(0).expand(N, 3)
            self.root_state_tensor[self.hand_indices, 0:3] = self.hand_initial_pos
            self.root_state_tensor[self.hand_indices, 7:13] = 0.0
            self._push_root_state(self.hand_indices.to(torch.int32))
            palm_world_default = (
                self.hand_initial_pos[0]
                + quat_apply(self.hand_initial_quat[0:1],
                             self.palm_local_offset.unsqueeze(0)).squeeze(0)
            )
            print(f"[XHandPour] palm world position (at default rotation) = "
                  f"{palm_world_default.tolist()}")

        self.hand_current_quat = self.hand_initial_quat.clone()

        self.phase = torch.zeros(N, dtype=torch.long, device=device)
        self.is_grasped = torch.zeros(N, dtype=torch.bool, device=device)
        self.subgoal_hold_buf = torch.zeros(N, dtype=torch.long, device=device)

        canonical = build_subgoal_quats(
            self.num_subgoals, self.pour_final_pitch,
            self.pour_pitch_axis_tensor, device,
        )
        self.subgoal_quats = canonical.unsqueeze(0).expand(N, self.num_subgoals, 4).contiguous()

        all_env_ids = torch.arange(N, device=device)
        self._apply_hand_roll(all_env_ids)
        self._respawn_bottle(all_env_ids)
        self._push_root_state(torch.cat([
            self.hand_indices[all_env_ids],
            self.object_indices[all_env_ids],
        ]).to(torch.int32))

    # ------------------------------------------------------------------
    # Helpers (mirror ShadowPour with xhand_dof_vel substitution downstream)
    # ------------------------------------------------------------------

    def _apply_hand_roll(self, env_ids):
        n = len(env_ids)
        roll = torch_rand_float(self.roll_angle_min, self.roll_angle_max,
                                (n, 1), device=self.device).squeeze(-1)
        axis = self.roll_axis_tensor.unsqueeze(0).expand(n, 3)
        roll_q = quat_from_angle_axis(roll, axis)
        new_q = quat_mul(roll_q, self.hand_initial_quat[env_ids])
        self.hand_current_quat[env_ids] = new_q
        self.root_state_tensor[self.hand_indices[env_ids], 0:3] = self.hand_initial_pos[env_ids]
        self.root_state_tensor[self.hand_indices[env_ids], 3:7] = new_q
        self.root_state_tensor[self.hand_indices[env_ids], 7:13] = 0.0

    def _respawn_bottle(self, env_ids):
        n = len(env_ids)
        palm_offset_world = quat_apply(
            self.hand_current_quat[env_ids],
            self.palm_local_offset.unsqueeze(0).expand(n, 3),
        )
        palm_world = self.hand_initial_pos[env_ids] + palm_offset_world

        # World-frame biases and noise (mirrors ShadowPour). Same seed + same
        # YAML values + same palm world position → identical bottle world pose.
        rand = torch_rand_float(-1.0, 1.0, (n, 3), device=self.device)
        bottle_pos = palm_world.clone()
        bottle_pos[:, 0] += self.bottle_spawn_x_bias + self.bottle_spawn_pos_noise * rand[:, 0]
        bottle_pos[:, 1] += self.bottle_spawn_y_bias + self.bottle_spawn_pos_noise * rand[:, 1]
        bottle_pos[:, 2] += self.bottle_spawn_z_offset + self.bottle_spawn_pos_noise * rand[:, 2]

        rand = torch_rand_float(-1.0, 1.0, (n, 3), device=self.device)
        rx = quat_from_angle_axis(rand[:, 0] * self.bottle_spawn_rot_noise, self.x_unit_tensor[env_ids])
        ry = quat_from_angle_axis(rand[:, 1] * self.bottle_spawn_rot_noise, self.y_unit_tensor[env_ids])
        rz = quat_from_angle_axis(rand[:, 2] * self.bottle_spawn_rot_noise, self.z_unit_tensor[env_ids])
        bottle_rot = quat_mul(rz, quat_mul(ry, rx))

        self.root_state_tensor[self.object_indices[env_ids], 0:3] = bottle_pos
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = bottle_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0.0

    def _push_root_state(self, indices_int32):
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(indices_int32),
            len(indices_int32),
        )

    def _fall_ref_pos(self):
        return self.rigid_body_states[:, self.palm_body_idx, 0:3]

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)
        self._apply_hand_roll(env_ids)
        self._respawn_bottle(env_ids)
        self.phase[env_ids] = 0
        self.is_grasped[env_ids] = False
        self.subgoal_hold_buf[env_ids] = 0
        self.successes[env_ids] = 0
        indices = torch.cat([
            self.hand_indices[env_ids],
            self.object_indices[env_ids],
        ]).to(torch.int32)
        self._push_root_state(indices)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.root_state_tensor[self.hand_indices, 3:7] = self.hand_current_quat
        self.root_state_tensor[self.hand_indices, 7:13] = 0.0
        all_hand_indices = self.hand_indices.to(torch.int32)
        self._push_root_state(all_hand_indices)

    def compute_observations(self):
        env_arange = torch.arange(self.num_envs, device=self.device)
        cur_subgoal_quat = self.subgoal_quats[env_arange, self.phase]
        self.goal_states[:, 0:3] = self.rigid_body_states[:, self.palm_body_idx, 0:3]
        self.goal_states[:, 3:7] = cur_subgoal_quat
        super().compute_observations()

    def compute_reward(self, actions):
        palm_pos = self.rigid_body_states[:, self.palm_body_idx, 0:3]
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        (rew, resets, new_phase, new_is_grasped, new_hold,
         new_successes, cons_successes, rot_dist, palm_to_obj) = compute_pour_reward(
            self.reset_buf, self.progress_buf, self.phase, self.is_grasped, self.subgoal_hold_buf,
            self.successes, self.consecutive_successes,
            float(self.max_episode_length),
            self.object_pos, self.object_rot, self.object_linvel, self.object_angvel,
            self.subgoal_quats,
            palm_pos,
            fingertip_pos,
            self.actions, self.xhand_dof_vel,
            self.grasp_proximity_scale,
            self.grasp_contact_scale,
            self.grasp_bonus,
            self.grasp_proximity_threshold,
            self.grasp_linvel_threshold,
            self.grasp_angvel_threshold,
            self.subgoal_rot_scale,
            self.subgoal_pos_scale,
            self.subgoal_advance_bonus,
            self.subgoal_success_tolerance,
            self.subgoal_hold_frames,
            self.final_success_bonus,
            self.rot_eps,
            self.action_penalty_scale,
            self.dof_vel_penalty_scale,
            self.obj_linvel_penalty_scale,
            self.obj_angvel_penalty_scale,
            self.obj_linvel_limit,
            self.obj_angvel_limit,
            self.dof_vel_limit,
            self.fall_dist,
            self.fall_penalty,
            float(self.av_factor),
        )

        self.rew_buf[:] = rew
        self.reset_buf[:] = resets
        self.phase[:] = new_phase
        self.is_grasped[:] = new_is_grasped
        self.subgoal_hold_buf[:] = new_hold
        self.successes[:] = new_successes
        self.consecutive_successes[:] = cons_successes

        self.extras['consecutive_successes'] = cons_successes.mean()
        self.extras['phase_mean'] = new_phase.float().mean()
        self.extras['grasp_rate'] = new_is_grasped.float().mean()
