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

import torch
from isaacgym import gymtorch
from isaacgymenvs.tasks.wuji_hand import WujiHand
from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_from_angle_axis, torch_rand_float


class WujiHandTilted(WujiHand):
    """WujiHand variant where each env's robot base continuously rotates about a
    random axis at a random angular speed in [baseAngVelMin, baseAngVelMax] rad/s.

    Observation and action dimensions are identical to WujiHand (207 obs, 22 actions),
    so pretrained WujiHand checkpoints can be loaded directly for fine-tuning.
    The SE3 variant additionally appends the hand root pose to obs (+7 dims = 214).
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self._base_ang_vel_min = cfg["env"].get("baseAngVelMin", 0.05)
        self._base_ang_vel_max = cfg["env"].get("baseAngVelMax", 0.30)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.survive_reward_scale = cfg["env"].get("surviveRewardScale", 0.0)
        self.survive_start_frame = cfg["env"].get("surviveStartFrame", 400)

        self.wrist_dof_indices = cfg["env"].get("wristDofIndices", [0, 1])
        self.wrist_accel_penalty_scale = cfg["env"].get("wristAccelPenaltyScale", 0.0)
        self.prev_wrist_actions = torch.zeros(
            (self.num_envs, len(self.wrist_dof_indices)), device=self.device
        )

        self.hand_rot_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot_speed = torch.zeros(self.num_envs, device=self.device)

        self.hand_initial_quat = self.root_state_tensor[self.hand_indices, 3:7].clone()
        self.hand_current_quat = self.hand_initial_quat.clone()

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_base_rotation(all_env_ids)

    def _randomize_base_rotation(self, env_ids):
        n = len(env_ids)
        axes = torch.tensor([[0, 1.0, 0]], device=self.device)
        axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
        self.hand_rot_axis[env_ids] = axes
        self.hand_rot_speed[env_ids] = torch_rand_float(
            self._base_ang_vel_min, self._base_ang_vel_max,
            (n, 1), device=self.device
        ).squeeze(-1)

    def _fall_ref_pos(self):
        return self.rigid_body_states[:, self.palm_body_idx, 0:3]

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)
        self.hand_current_quat[env_ids] = self.hand_initial_quat[env_ids]
        self._randomize_base_rotation(env_ids)
        self.prev_wrist_actions[env_ids] = 0.0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        angle_delta = self.hand_rot_speed * self.dt
        q_delta = quat_from_angle_axis(angle_delta, self.hand_rot_axis)

        self.hand_current_quat = quat_mul(q_delta, self.hand_current_quat)
        self.hand_current_quat = self.hand_current_quat / (
            self.hand_current_quat.norm(dim=-1, keepdim=True) + 1e-8
        )

        self.root_state_tensor[self.hand_indices, 3:7] = self.hand_current_quat
        self.root_state_tensor[self.hand_indices, 7:13] = 0.0

        all_hand_indices = self.hand_indices.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

    def compute_reward(self, actions):
        palm_linvel = self.rigid_body_states[:, self.palm_body_idx, 7:10]
        palm_angvel = self.rigid_body_states[:, self.palm_body_idx, 10:13]
        orig_linvel = self.object_linvel
        orig_angvel = self.object_angvel
        self.object_linvel = self.object_linvel - palm_linvel
        self.object_angvel = self.object_angvel - palm_angvel

        super().compute_reward(actions)

        self.object_linvel = orig_linvel
        self.object_angvel = orig_angvel

        survive_reward = self.survive_reward_scale * torch.clamp(
            self.progress_buf.float() - self.survive_start_frame, min=0.0
        )
        self.rew_buf[:] += survive_reward

        wrist_actions = actions[:, self.wrist_dof_indices]
        wrist_accel_penalty = self.wrist_accel_penalty_scale * torch.sum(
            (wrist_actions - self.prev_wrist_actions) ** 2, dim=-1
        )
        self.rew_buf[:] += wrist_accel_penalty
        self.prev_wrist_actions[:] = wrist_actions
