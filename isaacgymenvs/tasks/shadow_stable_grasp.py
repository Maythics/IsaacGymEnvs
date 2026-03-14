# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.shadow_hand import ShadowHand
from isaacgymenvs.utils.torch_jit_utils import (
    quat_mul, quat_conjugate, quat_from_angle_axis, quat_rotate, quat_rotate_inverse,
    torch_rand_float, to_torch
)


class ShadowStableGrasp(ShadowHand):
    """Shadow Hand task where the agent must stably grasp an object that starts
    at rest in the palm and falls due to gravity. The hand pose is randomly
    tilted at each episode start to vary the gravity direction in palm space.

    Observation space: 219 dims (full_state + palm state + relative vel).
    Action space: 20 dims (same as ShadowHand).
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Read stable-grasp-specific config before calling super
        self._max_init_angle_rad = cfg["env"].get("maxInitOrientationDeg", 30) * (3.14159265 / 180.0)

        self._stable_grasp_scale = cfg["env"].get("stableGraspScale", 5.0)
        self._contact_reward_scale = cfg["env"].get("contactRewardScale", 0.001)
        self._fall_dist_grasp = cfg["env"].get("fallDistance", 0.3)
        self._dof_force_penalty_scale = cfg["env"].get("dofForcePenaltyScale", 0.0)

        # Force full_state obs type — the 219-dim space includes palm state
        cfg["env"]["observationType"] = "full_state"

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # Patch obs size to 219 (full_state 211 replaces 11 goal dims with 16 palm dims + 3 rel)
        # Layout: DOF(72) + obj(13) + palm(13) + rel(6) + fingertip(65) + FT(30) + actions(20) = 219
        self.num_observations = 219
        self.obs_space = spaces.Box(np.ones(219) * -np.Inf, np.ones(219) * np.Inf)
        self.obs_buf = torch.zeros((self.num_envs, 219), device=self.device, dtype=torch.float)

        # Palm rigid body index (found after envs are created by super)
        self.palm_handle = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.shadow_hands[0], "robot0:palm", gymapi.DOMAIN_ENV
        )

        # Snapshot initial hand orientation from physics state
        self.hand_initial_quat = self.root_state_tensor[self.hand_indices, 3:7].clone()

        # ----------------------------------------------------------------
        # Derive palm kinematics from actual physics state (env 0).
        # This avoids hardcoding any offsets — the MJCF chain
        # hand_mount → forearm → wrist → palm is several links away from root.
        # ----------------------------------------------------------------
        root_pos_0  = self.root_state_tensor[self.hand_indices[0], 0:3].clone()   # (3,)
        root_quat_0 = self.root_state_tensor[self.hand_indices[0], 3:7].clone()   # (4,)
        palm_world_pos_0  = self.rigid_body_states[0, self.palm_handle, 0:3].clone()  # (3,)
        palm_world_quat_0 = self.rigid_body_states[0, self.palm_handle, 3:7].clone()  # (4,)

        # Palm center expressed in the root body's local frame
        self._palm_in_root = quat_rotate_inverse(
            root_quat_0.unsqueeze(0), (palm_world_pos_0 - root_pos_0).unsqueeze(0)
        ).squeeze(0)  # (3,)

        # Palm orientation expressed in the root body's local frame (fixed relative transform)
        self._palm_quat_in_root = quat_mul(
            quat_conjugate(root_quat_0).unsqueeze(0), palm_world_quat_0.unsqueeze(0)
        ).squeeze(0)  # (4,)

        # Object spawn offset in palm local frame.
        # The parent's default object init state is the tuned "object resting in palm" position,
        # so we use it to derive the palm-relative spawn point — no axis guessing required.
        obj_default_world_pos = self.object_init_state[0, 0:3]  # world pos from parent init
        self._obj_in_palm_local = quat_rotate_inverse(
            palm_world_quat_0.unsqueeze(0),
            (obj_default_world_pos - palm_world_pos_0).unsqueeze(0)
        ).squeeze(0)  # (3,)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # Goal pose needed by parent reward functions (kept for compatibility)
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # Palm state from rigid body tensor: (num_envs, 13)
        palm_state = self.rigid_body_states[:, self.palm_handle, 0:13]
        palm_pos = palm_state[:, 0:3]
        palm_quat = palm_state[:, 3:7]
        palm_linvel = palm_state[:, 7:10]
        palm_angvel = palm_state[:, 10:13]

        # Store palm pos for reward computation
        self._palm_pos = palm_pos

        obs = self.obs_buf
        # [0:24]   DOF positions (unscaled)
        obs[:, 0:24] = self.shadow_hand_dof_pos
        # [24:48]  DOF velocities
        obs[:, 24:48] = self.vel_obs_scale * self.shadow_hand_dof_vel
        # [48:72]  DOF forces
        obs[:, 48:72] = self.force_torque_obs_scale * self.dof_force_tensor

        # [72:79]  Object pose (pos + quat)
        obs[:, 72:79] = self.object_pose
        # [79:82]  Object linvel
        obs[:, 79:82] = self.object_linvel
        # [82:85]  Object angvel
        obs[:, 82:85] = self.vel_obs_scale * self.object_angvel

        # [85:88]  Palm position
        obs[:, 85:88] = palm_pos
        # [88:92]  Palm orientation
        obs[:, 88:92] = palm_quat
        # [92:95]  Palm linvel
        obs[:, 92:95] = self.vel_obs_scale * palm_linvel
        # [95:98]  Palm angvel
        obs[:, 95:98] = self.vel_obs_scale * palm_angvel

        # [98:101] Relative position: object - palm
        obs[:, 98:101] = self.object_pos - palm_pos
        # [101:104] Relative linvel: object - palm
        obs[:, 101:104] = self.object_linvel - palm_linvel

        # [104:169] Fingertip state (13 * 5 = 65)
        obs[:, 104:169] = self.fingertip_state.reshape(self.num_envs, 65)
        # [169:199] Fingertip force-torque sensors (6 * 5 = 30)
        obs[:, 169:199] = self.force_torque_obs_scale * self.vec_sensor_tensor
        # [199:219] Last actions (20)
        obs[:, 199:219] = self.actions

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, actions):
        palm_state = self.rigid_body_states[:, self.palm_handle, 0:13]
        palm_linvel = palm_state[:, 7:10]

        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:] = compute_stable_grasp_reward(
            self.rew_buf, self.reset_buf, self.progress_buf,
            self.object_pos, self.object_linvel, self.object_angvel,
            self._palm_pos, palm_linvel,
            self.shadow_hand_dof_vel, self.dof_force_tensor,
            self.vec_sensor_tensor, actions,
            self._stable_grasp_scale,
            self._contact_reward_scale,
            self.action_penalty_scale,
            self.palm_dist_penalty_scale,
            self.obj_linvel_penalty_scale,
            self.obj_angvel_penalty_scale,
            self.dof_vel_penalty_scale,
            self._dof_force_penalty_scale,
            self._fall_dist_grasp,
            self.max_episode_length,
        )

        # Keep reset_goal_buf clear (no goal in this task)
        self.reset_goal_buf[:] = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids, goal_env_ids):
        # Parent handles DOF reset, object init state, goal object placement
        super().reset_idx(env_ids, goal_env_ids)

        n = len(env_ids)

        # 1. Sample random hand orientation (axis-angle within max_init_angle_rad)
        axes = torch.randn((n, 3), device=self.device)
        axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
        angles = torch_rand_float(0.0, self._max_init_angle_rad, (n, 1), device=self.device).squeeze(-1)
        delta_quat = quat_from_angle_axis(angles, axes)
        new_quat = quat_mul(delta_quat, self.hand_initial_quat[env_ids])
        new_quat = new_quat / (new_quat.norm(dim=-1, keepdim=True) + 1e-8)

        # 2. Apply new hand orientation (kinematic body)
        self.root_state_tensor[self.hand_indices[env_ids], 3:7] = new_quat
        self.root_state_tensor[self.hand_indices[env_ids], 7:13] = 0.0
        hand_indices_i32 = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(hand_indices_i32),
            len(hand_indices_i32),
        )

        # 3. Compute palm world position and orientation analytically.
        #    palm_in_root and palm_quat_in_root are fixed relative transforms derived from
        #    actual physics state at init — no hardcoded MJCF offsets.
        hand_base_pos = self.root_state_tensor[self.hand_indices[env_ids], 0:3]  # (n, 3) fixed positions
        palm_in_root_batch = self._palm_in_root.unsqueeze(0).expand(n, -1)           # (n, 3)
        palm_quat_in_root_batch = self._palm_quat_in_root.unsqueeze(0).expand(n, -1) # (n, 4)

        palm_world_pos  = hand_base_pos + quat_rotate(new_quat, palm_in_root_batch)
        palm_world_quat = quat_mul(new_quat, palm_quat_in_root_batch)

        # 4. Place object on the inward side of the palm.
        #    _obj_in_palm_local was computed from the default object init pos — the tuned
        #    "object resting in palm" location in palm's local frame.
        obj_in_palm_batch = self._obj_in_palm_local.unsqueeze(0).expand(n, -1)  # (n, 3)
        object_pos = palm_world_pos + quat_rotate(palm_world_quat, obj_in_palm_batch)

        self.root_state_tensor[self.object_indices[env_ids], 0:3] = object_pos
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = self.object_init_state[env_ids, 3:7]
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0.0
        obj_indices_i32 = self.object_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj_indices_i32),
            len(obj_indices_i32),
        )


# ---------------------------------------------------------------------------
# JIT reward function
# ---------------------------------------------------------------------------

@torch.jit.script
def compute_stable_grasp_reward(
    rew_buf, reset_buf, progress_buf,
    object_pos, object_linvel, object_angvel,
    palm_pos, palm_linvel,
    dof_vel, dof_force,
    vec_sensor_tensor, actions,
    stable_grasp_scale: float,
    contact_reward_scale: float,
    action_penalty_scale: float,
    palm_dist_penalty_scale: float,
    obj_linvel_penalty_scale: float,
    obj_angvel_penalty_scale: float,
    dof_vel_penalty_scale: float,
    dof_force_penalty_scale: float,
    fall_dist: float,
    max_episode_length: float,
):
    # Primary stability reward: exp decay with object-palm relative velocity magnitude
    rel_vel = object_linvel - palm_linvel
    rel_vel_norm = torch.norm(rel_vel, dim=-1)
    stable_reward = torch.exp(-rel_vel_norm * stable_grasp_scale)

    # Keep object near palm
    palm_dist = torch.norm(object_pos - palm_pos, dim=-1)
    palm_dist_penalty = palm_dist * palm_dist_penalty_scale

    # Quasi-static: penalize object motion (L2 of linvel and angvel vectors)
    obj_linvel_penalty = torch.sum(object_linvel ** 2, dim=-1) * obj_linvel_penalty_scale
    obj_angvel_penalty = torch.sum(object_angvel ** 2, dim=-1) * obj_angvel_penalty_scale

    # Suppress hand shaking: penalize joint velocities and joint torques
    dof_vel_penalty   = torch.sum(dof_vel   ** 2, dim=-1) * dof_vel_penalty_scale
    dof_force_penalty = torch.sum(dof_force ** 2, dim=-1) * dof_force_penalty_scale

    # Smooth actions penalty
    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale

    # Small bonus for active fingertip contacts
    contact_reward = (vec_sensor_tensor.abs() > 0.01).float().sum(dim=-1) * contact_reward_scale

    reward = (stable_reward
              + palm_dist_penalty
              + obj_linvel_penalty
              + obj_angvel_penalty
              + dof_vel_penalty
              + dof_force_penalty
              + action_penalty
              + contact_reward)

    # Reset when object falls too far or episode times out
    resets = torch.where(palm_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    return reward, resets, progress_buf
