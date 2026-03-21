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

import math
import torch

from isaacgym import gymtorch
from isaacgymenvs.tasks.shadow_hand import ShadowHand
from isaacgymenvs.utils.torch_jit_utils import (
    quat_from_angle_axis, quat_mul, quat_conjugate, torch_rand_float,
)


class ShadowFlip(ShadowHand):
    """In-hand flip task: flip a block to one of 4 target orientations, then hold
    it there for settle_frames consecutive frames without dropping it.

    Episode flow:
      Pre-flip  → orientation reward drives the block toward the target 90° rotation.
      Settling  → once flip detected (rot_dist < flip_tolerance), settle_buf counts up.
                  A stable_bonus_per_frame is awarded each frame the block stays near
                  target and in-hand.
      Success   → settle_buf >= settle_frames while not fallen → reach_goal_bonus + reset.
      Failure   → object falls out of hand (palm dist >= fall_dist) or timeout.

    Observation space: 211 dims (identical layout to ShadowHand full_state) for
    checkpoint / architecture compatibility.
    Action space: 20 dims (same as ShadowHand).
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        # Read flip-specific config before calling super
        self.settle_frames = cfg["env"].get("settleFrames", 30)
        self.flip_tolerance = cfg["env"].get("flipTolerance", 0.4)       # ~23 deg
        self.stable_bonus_per_frame = cfg["env"].get("stableBonusPerFrame", 0.5)
        self.orient_reward_scale = cfg["env"].get("orientRewardScale", 1.0)
        self.orient_eps = cfg["env"].get("orientEps", 0.1)

        # Force full_state obs — guarantees 211-dim observation vector
        cfg["env"]["observationType"] = "full_state"

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                         virtual_screen_capture, force_render)

        # Phase-tracking buffers (long/int for TorchScript compatibility)
        self.settle_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.flip_detected = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Goal reset: sample one of 4 axis-aligned 90° flip targets
    # ------------------------------------------------------------------

    def reset_target_pose(self, env_ids, apply_reset=False):
        n = len(env_ids)

        # Pick one of 4 flip directions uniformly:
        #   0: +π/2 around Y  (flip forward)
        #   1: -π/2 around Y  (flip backward)
        #   2: +π/2 around X  (flip left)
        #   3: -π/2 around X  (flip right)
        flip_choice = torch.randint(0, 4, (n,), device=self.device)

        angles = torch.where(
            flip_choice % 2 == 0,
            torch.full((n,), math.pi / 2.0, device=self.device),
            torch.full((n,), -math.pi / 2.0, device=self.device),
        )
        axes = torch.zeros(n, 3, device=self.device)
        axes[:, 1] = (flip_choice < 2).float()   # Y axis for choices 0, 1
        axes[:, 0] = (flip_choice >= 2).float()  # X axis for choices 2, 3

        goal_quat = quat_from_angle_axis(angles, axes)  # (n, 4) XYZW

        # Goal position: same as initial object position (flip in place)
        goal_pos = self.goal_init_state[env_ids, 0:3].clone()

        self.goal_states[env_ids, 0:3] = goal_pos
        self.goal_states[env_ids, 3:7] = goal_quat

        # Update visual goal object
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = \
            goal_pos + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = goal_quat
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
    # Episode reset: upright start with small random yaw
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids, goal_env_ids):
        # Parent handles DOF reset, goal pose (via our reset_target_pose), and
        # object init state.
        super().reset_idx(env_ids, goal_env_ids)

        # Reset phase-tracking buffers
        self.settle_buf[env_ids] = 0
        self.flip_detected[env_ids] = 0

        # Override the object rotation set by parent: upright with small random yaw
        # so the agent must learn to flip from varied initial orientations.
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

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, actions):
        (self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:],
         self.progress_buf[:], self.settle_buf[:], self.flip_detected[:],
         self.successes[:], self.consecutive_successes[:]) = \
            compute_flip_reward(
                self.rew_buf, self.reset_buf, self.reset_goal_buf,
                self.progress_buf, self.settle_buf, self.flip_detected,
                self.successes, self.consecutive_successes,
                self.max_episode_length,
                self.object_pos, self.object_rot, self.goal_rot,
                self.orient_reward_scale, self.orient_eps,
                self.actions, self.action_penalty_scale,
                self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
                self.settle_frames, self.stable_bonus_per_frame,
                self.flip_tolerance,
                self.max_consecutive_successes, self.av_factor,
                self.rigid_body_states[:, self.palm_body_idx, 0:3],
                self.object_linvel, self.object_angvel, self.shadow_hand_dof_vel,
                self.obj_linvel_penalty_scale, self.obj_angvel_penalty_scale,
                self.dof_vel_penalty_scale,
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
# JIT reward function
# ---------------------------------------------------------------------------

@torch.jit.script
def compute_flip_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf,
    settle_buf, flip_detected,
    successes, consecutive_successes,
    max_episode_length: float,
    object_pos, object_rot, target_rot,
    orient_reward_scale: float, orient_eps: float,
    actions, action_penalty_scale: float,
    reach_goal_bonus: float, fall_dist: float, fall_penalty: float,
    settle_frames: int, stable_bonus_per_frame: float,
    flip_tolerance: float,
    max_consecutive_successes: int, av_factor: float,
    hand_pos,
    obj_linvel, obj_angvel, dof_vel,
    obj_linvel_penalty_scale: float, obj_angvel_penalty_scale: float,
    dof_vel_penalty_scale: float,
):
    # --- Orientation distance to target (quaternion angle) ---
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    # --- Orientation reward: higher when closer to target ---
    orient_rew = 1.0 / (rot_dist + orient_eps) * orient_reward_scale

    # --- Auxiliary penalties ---
    action_penalty     = torch.sum(actions    ** 2, dim=-1) * action_penalty_scale
    obj_linvel_penalty = torch.sum(obj_linvel ** 2, dim=-1) * obj_linvel_penalty_scale
    obj_angvel_penalty = torch.sum(obj_angvel ** 2, dim=-1) * obj_angvel_penalty_scale
    dof_vel_penalty    = torch.sum(dof_vel    ** 2, dim=-1) * dof_vel_penalty_scale

    reward = orient_rew + action_penalty + obj_linvel_penalty + obj_angvel_penalty + dof_vel_penalty

    # --- Flip detection: once detected, stays detected until reset ---
    flipped_now = rot_dist < flip_tolerance
    flip_detected = torch.where(flipped_now, torch.ones_like(flip_detected), flip_detected)

    # --- Settling: increment counter each frame while flip is active ---
    settle_buf = torch.where(flip_detected == 1, settle_buf + 1, settle_buf)

    # --- Stable bonus: reward each settling frame the block stays near target ---
    stable_bonus = torch.where(
        (flip_detected == 1) & flipped_now,
        torch.full_like(reward, stable_bonus_per_frame),
        torch.zeros_like(reward),
    )
    reward = reward + stable_bonus

    # --- Fall check: object too far from palm ---
    fall_check_dist = torch.norm(object_pos - hand_pos, p=2, dim=-1)
    fallen = fall_check_dist >= fall_dist
    reward = torch.where(fallen, reward + fall_penalty, reward)

    # --- Success: settled for settle_frames without falling ---
    success = (settle_buf >= settle_frames) & (fall_check_dist < fall_dist)
    reward = torch.where(success, reward + reach_goal_bonus, reward)
    successes = successes + success.float()

    # --- Resets ---
    resets = torch.where(fallen, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(success, torch.ones_like(resets), resets)
    resets = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets
    )

    # --- Consecutive successes tracking ---
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (reward, resets, reset_goal_buf, progress_buf,
            settle_buf, flip_detected, successes, cons_successes)
