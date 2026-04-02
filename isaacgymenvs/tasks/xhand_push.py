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

"""XPush: in-hand repositioning task for the XHand robot.

Mirrors ShadowPush but inherits from XHandHand. Each episode a random XY push
direction is drawn; the goal advances along that direction on every goal reset.

Observation space: 175 dims (full_state, same as XHandHand).
Action space: 14 dims.
"""

import math
import torch

from isaacgym import gymtorch
from isaacgymenvs.tasks.xhand_hand import XHandHand
from isaacgymenvs.utils.torch_jit_utils import quat_from_angle_axis, torch_rand_float


class XHandPush(XHandHand):
    """In-hand repositioning (push) task for the XHand.

    Identical logic to ShadowPush — only the parent class differs.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        # Read push-specific config before calling super
        self.yaw_reward_scale = cfg["env"].get("yawRewardScale", 1.0)
        self.yaw_eps = cfg["env"].get("yawEps", 0.1)
        self.pos_success_tolerance = cfg["env"].get("posSuccessTolerance", 0.02)
        self.yaw_success_tolerance = cfg["env"].get("yawSuccessTolerance", 0.1)
        self.target_pos_range_xy = cfg["env"].get("targetPosRangeXY", 0.06)
        self.target_pos_range_z = cfg["env"].get("targetPosRangeZ", 0.04)
        self.upright_penalty_scale = cfg["env"].get("uprightPenaltyScale", -0.5)
        self.flip_angle = cfg["env"].get("flipAngle", math.pi / 2.0)
        self.push_step_size = cfg["env"].get("pushStepSize", 0.015)

        # Force full_state obs type
        cfg["env"]["observationType"] = "full_state"

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                         virtual_screen_capture, force_render)

        # Per-environment push direction: unit XY vector, fixed within each episode
        self.push_direction = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device)

    # ------------------------------------------------------------------
    # Goal reset: sample nearby position + yaw target
    # ------------------------------------------------------------------

    def reset_target_pose(self, env_ids, apply_reset=False):
        n = len(env_ids)

        if not apply_reset:
            # Full episode reset: sample a fresh random goal (first waypoint)
            rand = torch_rand_float(-1.0, 1.0, (n, 4), device=self.device)
            goal_pos = self.goal_init_state[env_ids, 0:3].clone()
            goal_pos[:, 0] = goal_pos[:, 0] + rand[:, 0] * self.target_pos_range_xy
            goal_pos[:, 1] = goal_pos[:, 1] + rand[:, 1] * self.target_pos_range_xy
            goal_pos[:, 2] = goal_pos[:, 2] + rand[:, 2] * self.target_pos_range_z
            goal_yaw = rand[:, 3] * math.pi
            goal_yaw_quat = quat_from_angle_axis(goal_yaw, self.z_unit_tensor[env_ids])
        else:
            # Goal-only reset: advance previous goal along the stored push direction
            goal_pos = self.goal_states[env_ids, 0:3].clone()
            goal_pos = goal_pos + self.push_step_size * self.push_direction[env_ids]
            goal_yaw_quat = self.goal_states[env_ids, 3:7].clone()

        self.goal_states[env_ids, 0:3] = goal_pos
        self.goal_states[env_ids, 3:7] = goal_yaw_quat

        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = \
            goal_pos + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = goal_yaw_quat
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = 0.0

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(env_ids),
            )
        self.reset_goal_buf[env_ids] = 0

    # ------------------------------------------------------------------
    # Episode reset: yaw-only object orientation at start
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)

        n = len(env_ids)
        rand_yaw = torch_rand_float(-math.pi, math.pi, (n, 1), device=self.device)
        new_object_rot = quat_from_angle_axis(rand_yaw[:, 0], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0.0

        obj_indices = self.object_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj_indices),
            len(obj_indices),
        )

        angle = torch_rand_float(0.0, 2.0 * math.pi, (n, 1), device=self.device)[:, 0]
        self.push_direction[env_ids, 0] = torch.cos(angle)
        self.push_direction[env_ids, 1] = torch.sin(angle)
        self.push_direction[env_ids, 2] = 0.0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, actions):
        (self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:],
         self.progress_buf[:], self.successes[:], self.consecutive_successes[:]) = \
            compute_push_reward(
                self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf,
                self.successes, self.consecutive_successes,
                self.max_episode_length,
                self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
                self.dist_reward_scale, self.yaw_reward_scale, self.yaw_eps,
                self.actions, self.action_penalty_scale,
                self.pos_success_tolerance, self.yaw_success_tolerance,
                self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
                self.upright_penalty_scale, self.flip_angle,
                self.max_consecutive_successes, self.av_factor,
                self.object_linvel, self.object_angvel, self.xhand_dof_vel,
                self.rigid_body_states[:, self.palm_body_idx, 0:3],
                self.obj_linvel_penalty_scale, self.obj_angvel_penalty_scale,
                self.dof_vel_penalty_scale, self.palm_dist_penalty_scale,
            )

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))


# ---------------------------------------------------------------------------
# JIT reward function (identical to ShadowPush)
# ---------------------------------------------------------------------------

@torch.jit.script
def compute_push_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float,
    object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, yaw_reward_scale: float, yaw_eps: float,
    actions, action_penalty_scale: float,
    pos_success_tolerance: float, yaw_success_tolerance: float,
    reach_goal_bonus: float, fall_dist: float, fall_penalty: float,
    upright_penalty_scale: float, flip_angle: float,
    max_consecutive_successes: int, av_factor: float,
    object_linvel, object_angvel, dof_vel, hand_pos,
    obj_linvel_penalty_scale: float, obj_angvel_penalty_scale: float,
    dof_vel_penalty_scale: float, palm_dist_penalty_scale: float,
):
    # --- Position distance reward ---
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    dist_rew = goal_dist * dist_reward_scale

    # --- Yaw reward: extract yaw from quaternions (XYZW format) ---
    ox, oy, oz, ow = object_rot[:, 0], object_rot[:, 1], object_rot[:, 2], object_rot[:, 3]
    obj_yaw = torch.atan2(2.0 * (ow * oz + ox * oy), 1.0 - 2.0 * (oy * oy + oz * oz))

    tx, ty, tz, tw = target_rot[:, 0], target_rot[:, 1], target_rot[:, 2], target_rot[:, 3]
    goal_yaw = torch.atan2(2.0 * (tw * tz + tx * ty), 1.0 - 2.0 * (ty * ty + tz * tz))

    yaw_diff = torch.atan2(torch.sin(obj_yaw - goal_yaw), torch.cos(obj_yaw - goal_yaw))
    yaw_rew = 1.0 / (torch.abs(yaw_diff) + yaw_eps) * yaw_reward_scale

    # --- Upright penalty ---
    roll  = torch.atan2(2.0 * (ow * ox + oy * oz), 1.0 - 2.0 * (ox * ox + oy * oy))
    pitch = torch.asin(torch.clamp(2.0 * (ow * oy - oz * ox), -1.0, 1.0))
    upright_penalty = (roll ** 2 + pitch ** 2) * upright_penalty_scale

    # --- Auxiliary penalties ---
    action_penalty     = torch.sum(actions       ** 2, dim=-1) * action_penalty_scale
    obj_linvel_penalty = torch.sum(object_linvel  ** 2, dim=-1) * obj_linvel_penalty_scale
    obj_angvel_penalty = torch.sum(object_angvel  ** 2, dim=-1) * obj_angvel_penalty_scale
    dof_vel_penalty    = torch.sum(dof_vel         ** 2, dim=-1) * dof_vel_penalty_scale
    palm_dist_penalty  = torch.norm(object_pos - hand_pos, p=2, dim=-1) * palm_dist_penalty_scale

    reward = (dist_rew + yaw_rew + upright_penalty
              + action_penalty + obj_linvel_penalty + obj_angvel_penalty
              + dof_vel_penalty + palm_dist_penalty)

    # --- Success condition ---
    pos_ok = goal_dist <= pos_success_tolerance
    yaw_ok = torch.abs(yaw_diff) <= yaw_success_tolerance
    goal_resets = torch.where(pos_ok & yaw_ok, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # --- Fall check ---
    fall_check_dist = torch.norm(object_pos - hand_pos, p=2, dim=-1)
    reward = torch.where(fall_check_dist >= fall_dist, reward + fall_penalty, reward)

    # --- Flip termination ---
    flipped = (torch.abs(roll) > flip_angle) | (torch.abs(pitch) > flip_angle)

    # --- Resets ---
    resets = torch.where(fall_check_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(flipped, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        progress_buf = torch.where(pos_ok & yaw_ok, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(
            successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    resets = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        reward = torch.where(
            progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
