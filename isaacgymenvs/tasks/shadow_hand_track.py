"""Goal-conditioned ShadowHand with a soft SE(3) path-following reward.

The policy interface is deliberately identical to :class:`ShadowHand`: the full
state observation remains 211-dimensional and the action remains 20-dimensional.
The reference path is internal reward-shaping state, so baseline ShadowHand PPO
checkpoints and observation-normalization statistics can be reused directly.
"""

import torch

from isaacgymenvs.tasks.shadow_hand import ShadowHand
from isaacgymenvs.utils.se3_path import closest_se3_path_distance


class ShadowHandTrack(ShadowHand):
    """ShadowHand reorientation with geometric SE(3) path reward shaping.

    Each time the goal advances, the previous goal and new goal define a static
    constant-twist SE(3) segment. At episode reset, the reset object pose and first
    goal define the initial segment. The object receives a bounded soft reward for
    lying close to any point on that segment; success and termination are unchanged.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        env_cfg = cfg["env"]
        self.path_reward_scale = float(env_cfg.get("pathRewardScale", 5.0))
        self.path_position_sigma = float(env_cfg.get("pathPositionSigma", 0.025))
        self.path_rotation_sigma = float(env_cfg.get("pathRotationSigma", 0.35))
        self.path_sample_count = int(env_cfg.get("pathSampleCount", 9))
        if self.path_sample_count < 2:
            raise ValueError("pathSampleCount must be at least 2")
        if self.path_position_sigma <= 0.0 or self.path_rotation_sigma <= 0.0:
            raise ValueError("pathPositionSigma and pathRotationSigma must be positive")

        # Exact checkpoint compatibility requires the original ShadowHand layout.
        env_cfg["observationType"] = "full_state"
        env_cfg["appendHandBasePose"] = False

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                         virtual_screen_capture, force_render)

        self.path_start_pose = torch.zeros((self.num_envs, 7), dtype=torch.float,
                                           device=self.device)
        self.path_end_pose = torch.zeros_like(self.path_start_pose)
        self.path_fractions = torch.linspace(0.0, 1.0, self.path_sample_count,
                                             dtype=torch.float, device=self.device)

        self.path_start_pose[:] = self.root_state_tensor[self.object_indices, 0:7]
        self.path_end_pose[:] = self.goal_states[:, 0:7]

    def reset_target_pose(self, env_ids, apply_reset=False, from_goal_reach=False,
                          base_pos=None):
        """Advance goals normally and update the reward-only path endpoints."""
        if from_goal_reach:
            previous_goal = self.goal_states[env_ids, 0:7].clone()

        super().reset_target_pose(
            env_ids,
            apply_reset=apply_reset,
            from_goal_reach=from_goal_reach,
            base_pos=base_pos,
        )

        if from_goal_reach:
            self.path_start_pose[env_ids] = previous_goal
            self.path_end_pose[env_ids] = self.goal_states[env_ids, 0:7]

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)

        # The initial reference segment begins at the actual randomized object
        # reset pose, including its randomized orientation.
        self.path_start_pose[env_ids] = \
            self.root_state_tensor[self.object_indices[env_ids], 0:7]
        self.path_end_pose[env_ids] = self.goal_states[env_ids, 0:7]

    def compute_reward(self, actions):
        # Preserve every baseline term, reset rule, success counter, and bonus.
        super().compute_reward(actions)

        object_pose = torch.cat((self.object_pos, self.object_rot), dim=-1)
        path_cost, path_pos_error, path_rot_error = closest_se3_path_distance(
            object_pose,
            self.path_start_pose,
            self.path_end_pose,
            self.path_fractions,
            self.path_position_sigma,
            self.path_rotation_sigma,
        )
        path_reward = self.path_reward_scale * torch.exp(-path_cost)
        self.rew_buf[:] = self.rew_buf + path_reward

        self.extras["path_reward"] = path_reward.mean()
        self.extras["path_cost"] = path_cost.mean()
        self.extras["path_pos_error"] = path_pos_error.mean()
        self.extras["path_rot_error"] = path_rot_error.mean()
