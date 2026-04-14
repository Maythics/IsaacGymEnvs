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

"""XHandHand: in-hand reorientation task for the XHand robot.

Mirrors ShadowHand but uses the XHand URDF (12 finger DOFs + 2 wrist DOFs = 14
total) and full_state observations (175 dims).

Observation layout (full_state, 175 dims):
  [0:14]    DOF positions (unscaled to [-1, 1])
  [14:28]   DOF velocities × 0.2
  [28:42]   DOF forces × 10.0
  [42:49]   Object pose (pos + quat)
  [49:52]   Object linear velocity
  [52:55]   Object angular velocity × 0.2
  [55:62]   Goal pose (pos + quat)
  [62:66]   quat_mul(obj_rot, conj(goal_rot))
  [66:131]  5 fingertips × 13 = 65
  [131:161] 5 force-torque sensors × 6 = 30
  [161:175] Last 14 actions
"""

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import (
    scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, quat_apply,
    to_torch, get_axis_params, torch_rand_float, tensor_clamp,
)
from isaacgymenvs.tasks.base.vec_task import VecTask


# Number of DOFs in the XHand with wrist: 2 wrist + 12 fingers
_XHAND_NUM_DOFS = 14
# Number of actions = all DOFs (no tendon coupling)
_XHAND_NUM_ACTIONS = 14

# Fingertip rigid-body names after collapse_fixed_joints=True.
# The fixed "tip" links are merged into the last revolute-joint child bodies.
_XHAND_FINGERTIPS = [
    "right_hand_index_rota_link2",
    "right_hand_mid_link2",
    "right_hand_ring_link2",
    "right_hand_pinky_link2",
    "right_hand_thumb_rota_link2",
]

# Palm link name (used for fall detection / palm-distance penalty)
_XHAND_PALM = "right_hand_link"


class XHandHand(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2
        self.force_torque_obs_scale = 10.0

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.obj_linvel_penalty_scale = self.cfg["env"].get("objLinvelPenaltyScale", 0.0)
        self.obj_angvel_penalty_scale = self.cfg["env"].get("objAngvelPenaltyScale", 0.0)
        self.dof_vel_penalty_scale = self.cfg["env"].get("dofVelPenaltyScale", 0.0)
        self.palm_dist_penalty_scale = self.cfg["env"].get("palmDistPenaltyScale", 0.0)
        self.obj_linvel_limit = self.cfg["env"].get("objLinvelLimit", 0.3)
        self.obj_angvel_limit = self.cfg["env"].get("objAngvelLimit", 1.5)
        self.dof_vel_limit = self.cfg["env"].get("dofVelLimit", 2.0)

        self.xhand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.action_speed_scale = self.cfg["env"].get("actionSpeedScale", 1.0)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        object_type_pool = self.cfg["env"].get("objectTypePool", [])
        if not object_type_pool:
            object_type_pool = [self.object_type]
        self.object_type_pool = object_type_pool

        for t in self.object_type_pool:
            if t not in {"block", "egg", "pen"} and not t.startswith("ycb_"):
                raise ValueError(f"Unknown objectType '{t}'. Must be block/egg/pen or ycb_*.")

        self.object_scale_ranges = self.cfg["env"].get("objectScaleRanges", {})

        # Object asset files (same sources as ShadowHand)
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg":   "mjcf/open_ai_assets/hand/egg.xml",
            "pen":   "mjcf/open_ai_assets/hand/pen.xml",
        }
        for t in self.object_type_pool:
            if t.startswith("ycb_") and t not in self.asset_files_dict:
                self.asset_files_dict[t] = f"urdf/ycb_balls/{t}.urdf"

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
                "assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get(
                "assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get(
                "assetFileNamePen", self.asset_files_dict["pen"])

        self.ignore_z = (self.object_type_pool == ["pen"])

        self.obs_type = self.cfg["env"]["observationType"]
        if self.obs_type != "full_state":
            raise Exception("XHandHand only supports observationType='full_state'")

        print("Obs type:", self.obs_type)

        self.num_xhand_dofs = _XHAND_NUM_DOFS

        # full_state obs: 3*num_dofs + 13 (obj) + 11 (goal+rel_quat) + 5*13 (fingertips) + 5*6 (ft sensors) + num_actions
        # = 42 + 13 + 11 + 65 + 30 + 14 = 175
        num_obs = 3 * self.num_xhand_dofs + 13 + 11 + 5 * 13 + 5 * 6 + _XHAND_NUM_ACTIONS

        self.up_axis = 'z'

        self.fingertips = _XHAND_FINGERTIPS
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = num_obs

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = _XHAND_NUM_ACTIONS

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device,
            graphics_device_id=graphics_device_id, headless=headless,
            virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time:", self.reset_time)
            print("New episode length:", self.max_episode_length)

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_xhand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.xhand_default_dof_pos = torch.zeros(
            self.num_xhand_dofs, dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.xhand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_xhand_dofs]
        self.xhand_dof_pos = self.xhand_dof_state[..., 0]
        self.xhand_dof_vel = self.xhand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"],
                          int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # XHand URDF is self-contained in assets/urdf/xhand/
        base_asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        xhand_asset_root = os.path.join(base_asset_root, "urdf", "xhand")
        xhand_asset_file = "xhand_right_with_wrist.urdf"

        # Load XHand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        xhand_asset = self.gym.load_asset(
            self.sim, xhand_asset_root, xhand_asset_file, asset_options)

        self.num_xhand_bodies = self.gym.get_asset_rigid_body_count(xhand_asset)
        self.num_xhand_shapes = self.gym.get_asset_rigid_shape_count(xhand_asset)
        self.num_xhand_dofs = self.gym.get_asset_dof_count(xhand_asset)

        # DEBUG: print actual DOF names in IsaacGym's internal order
        _dof_names = self.gym.get_asset_dof_names(xhand_asset)
        print(f"[XHandHand] DOF names ({len(_dof_names)}): {_dof_names}")

        # Set PD drive gains for all DOFs
        xhand_dof_props = self.gym.get_asset_dof_properties(xhand_asset)
        for i in range(self.num_xhand_dofs):
            xhand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            xhand_dof_props['stiffness'][i] = 2.0
            xhand_dof_props['damping'][i] = 0.1

        # All DOFs are actuated (wrist + fingers)
        self.actuated_dof_indices = to_torch(
            list(range(self.num_xhand_dofs)), dtype=torch.long, device=self.device)

        self.xhand_dof_lower_limits = []
        self.xhand_dof_upper_limits = []
        self.xhand_dof_default_pos = []
        self.xhand_dof_default_vel = []
        for i in range(self.num_xhand_dofs):
            self.xhand_dof_lower_limits.append(xhand_dof_props['lower'][i])
            self.xhand_dof_upper_limits.append(xhand_dof_props['upper'][i])
            self.xhand_dof_default_pos.append(0.0)
            self.xhand_dof_default_vel.append(0.0)

        self.xhand_dof_lower_limits = to_torch(self.xhand_dof_lower_limits, device=self.device)
        self.xhand_dof_upper_limits = to_torch(self.xhand_dof_upper_limits, device=self.device)
        self.xhand_dof_default_pos = to_torch(self.xhand_dof_default_pos, device=self.device)
        self.xhand_dof_default_vel = to_torch(self.xhand_dof_default_vel, device=self.device)

        # Fingertip rigid-body indices
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(xhand_asset, name)
            for name in self.fingertips
        ]
        self.palm_body_idx = self.gym.find_asset_rigid_body_index(xhand_asset, _XHAND_PALM)

        # Create force sensors at each fingertip
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(xhand_asset, ft_handle, sensor_pose)

        # Load object assets (same paths as ShadowHand)
        import random as _random
        unique_types = list(dict.fromkeys(self.object_type_pool))
        object_assets = {}
        max_obj_shapes = 0
        for obj_type in unique_types:
            obj_file = self.asset_files_dict[obj_type]
            opt = gymapi.AssetOptions()
            phys = self.gym.load_asset(self.sim, base_asset_root, obj_file, opt)
            opt2 = gymapi.AssetOptions()
            opt2.disable_gravity = True
            goal = self.gym.load_asset(self.sim, base_asset_root, obj_file, opt2)
            object_assets[obj_type] = (phys, goal)
            max_obj_shapes = max(max_obj_shapes, self.gym.get_asset_rigid_shape_count(phys))

        # Poses
        xhand_start_pose = gymapi.Transform()
        xhand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        # Align with ShadowHand: palm faces world -Z (down), fingers point world -Y.
        # URDF local frame: palm normal = +X, fingers extend in +Z.
        # Required: +X → -Z (world), +Z → -Y (world).
        # Quaternion (scalar-first): [qw=0.5, qx=0.5, qy=0.5, qz=-0.5]
        # IsaacGym scalar-last format: Quat(x, y, z, w) = Quat(0.5, 0.5, -0.5, 0.5)
        # Palm faces world +Z (up), fingers point world -Y.
        # URDF local frame: palm normal = +X, fingers extend in +Z.
        # Required: +X → +Z (world), +Z → -Y (world).
        # Quaternion (scalar-first): [qw=0.5, qx=0.5, qy=-0.5, qz=0.5]
        # IsaacGym scalar-last format: Quat(x, y, z, w) = Quat(0.5, -0.5, 0.5, 0.5)
        xhand_start_pose.r = gymapi.Quat(0.5, -0.5, 0.5, 0.5)
        # Translate hand base so object spawns at same world position as ShadowHand:
        # ShadowHand: hand.y=0, dy=-0.39 → object world-y=-0.39
        # XHand:      hand.y=-0.29, dy=-0.10 → object world-y=-0.39 ✓
        xhand_start_pose.p.y = -0.29

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        # Fingers point in world -Y; shift object into the cup (dy=-0.10) and
        # above the palm (dz=+0.10).
        pose_dy, pose_dz = -0.10, 0.10
        object_start_pose.p.x = xhand_start_pose.p.x
        object_start_pose.p.y = xhand_start_pose.p.y + pose_dy
        object_start_pose.p.z = xhand_start_pose.p.z + pose_dz

        if self.object_type_pool == ["pen"]:
            object_start_pose.p.z = xhand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z],
            device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.p.z -= 0.04

        max_agg_bodies = self.num_xhand_bodies + 2
        max_agg_shapes = self.num_xhand_shapes + max_obj_shapes * 2

        self.xhands = []
        self.envs = []
        self.object_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        xhand_rb_count = self.gym.get_asset_rigid_body_count(xhand_asset)
        max_obj_rb_count = max(
            self.gym.get_asset_rigid_body_count(object_assets[t][0]) for t in unique_types
        )
        self.object_rb_handles = list(range(xhand_rb_count, xhand_rb_count + max_obj_rb_count))

        env_type_indices = [_random.randrange(len(self.object_type_pool))
                            for _ in range(self.num_envs)]
        self.env_object_type_idx = torch.tensor(
            env_type_indices, dtype=torch.long, device=self.device)
        pen_pool_idx = (self.object_type_pool.index("pen")
                        if "pen" in self.object_type_pool else -1)
        self.is_pen = (
            (self.env_object_type_idx == pen_pool_idx) if pen_pool_idx >= 0
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            xhand_actor = self.gym.create_actor(
                env_ptr, xhand_asset, xhand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([
                xhand_start_pose.p.x, xhand_start_pose.p.y, xhand_start_pose.p.z,
                xhand_start_pose.r.x, xhand_start_pose.r.y,
                xhand_start_pose.r.z, xhand_start_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            self.gym.set_actor_dof_properties(env_ptr, xhand_actor, xhand_dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, xhand_actor)
            hand_idx = self.gym.get_actor_index(env_ptr, xhand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            obj_type = self.object_type_pool[env_type_indices[i]]
            phys_asset, goal_asset_inst = object_assets[obj_type]

            object_handle = self.gym.create_actor(
                env_ptr, phys_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([
                object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                object_start_pose.r.x, object_start_pose.r.y,
                object_start_pose.r.z, object_start_pose.r.w,
                0, 0, 0, 0, 0, 0,
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            goal_handle = self.gym.create_actor(
                env_ptr, goal_asset_inst, goal_start_pose, "goal_object",
                i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            scale_range = self.object_scale_ranges.get(
                obj_type, self.object_scale_ranges.get("default", [1.0, 1.0]))
            actor_scale = scale_range[0] + (scale_range[1] - scale_range[0]) * _random.random()
            self.gym.set_actor_scale(env_ptr, object_handle, actor_scale)
            self.gym.set_actor_scale(env_ptr, goal_handle, actor_scale)

            if obj_type in ("egg", "pen"):
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.xhands.append(xhand_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float).view(
                self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(
            self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(
            self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(
            self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device)

    def _fall_ref_pos(self):
        return self.goal_pos

    def compute_reward(self, actions):
        (self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:],
         self.progress_buf[:], self.successes[:], self.consecutive_successes[:]) = \
            compute_hand_reward(
                self.rew_buf, self.reset_buf, self.reset_goal_buf,
                self.progress_buf, self.successes, self.consecutive_successes,
                self.max_episode_length,
                self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
                self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
                self.actions, self.action_penalty_scale,
                self.success_tolerance, self.reach_goal_bonus,
                self.fall_dist, self.fall_penalty,
                self.max_consecutive_successes, self.av_factor,
                (self.object_type_pool == ["pen"]),
                self.object_linvel, self.object_angvel, self.xhand_dof_vel,
                self.root_state_tensor[self.hand_indices, 0:3],
                self.obj_linvel_penalty_scale, self.obj_angvel_penalty_scale,
                self.dof_vel_penalty_scale, self.palm_dist_penalty_scale,
                self.obj_linvel_limit, self.obj_angvel_limit, self.dof_vel_limit,
                self._fall_ref_pos(),
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

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        self.compute_full_state()
        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        buf = self.states_buf if asymm_obs else self.obs_buf
        n = self.num_xhand_dofs

        buf[:, 0:n] = unscale(
            self.xhand_dof_pos, self.xhand_dof_lower_limits, self.xhand_dof_upper_limits)
        buf[:, n:2*n] = self.vel_obs_scale * self.xhand_dof_vel
        buf[:, 2*n:3*n] = self.force_torque_obs_scale * self.dof_force_tensor

        obj_obs_start = 3 * n
        buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 13
        buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(
            self.object_rot, quat_conjugate(self.goal_rot))

        num_ft_states = 13 * self.num_fingertips       # 65
        num_ft_force_torques = 6 * self.num_fingertips  # 30

        fingertip_obs_start = goal_obs_start + 11
        buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = \
            self.fingertip_state.reshape(self.num_envs, num_ft_states)
        buf[:, fingertip_obs_start + num_ft_states:
               fingertip_obs_start + num_ft_states + num_ft_force_torques] = \
            self.force_torque_obs_scale * self.vec_sensor_tensor

        obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
        buf[:, obs_end:obs_end + self.num_actions] = self.actions

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1],
            self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        pen_mask = self.is_pen[env_ids]
        if pen_mask.any():
            pen_rot = randomize_rotation_pen(
                rand_floats[:, 0], rand_floats[:, 1], torch.tensor(0.3),
                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
                self.z_unit_tensor[env_ids])
            new_rot = torch.where(pen_mask.unsqueeze(-1), pen_rot, new_rot)

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = \
            self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = \
            self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = \
            torch.zeros_like(
                self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_xhand_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        self.rb_forces[env_ids, :, :] = 0.0

        # Reset object
        self.root_state_tensor[self.object_indices[env_ids]] = \
            self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = \
            self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = \
            self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(
            rand_floats[:, 3], rand_floats[:, 4],
            self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        pen_mask = self.is_pen[env_ids]
        if pen_mask.any():
            pen_rot = randomize_rotation_pen(
                rand_floats[:, 3], rand_floats[:, 4], torch.tensor(0.3),
                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
                self.z_unit_tensor[env_ids])
            new_object_rot = torch.where(pen_mask.unsqueeze(-1), pen_rot, new_object_rot)

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = \
            torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([
            self.object_indices[env_ids],
            self.goal_object_indices[env_ids],
            self.goal_object_indices[goal_env_ids],
        ]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices), len(object_indices))

        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1]))

        # Reset XHand DOFs
        delta_max = self.xhand_dof_upper_limits - self.xhand_dof_default_pos
        delta_min = self.xhand_dof_lower_limits - self.xhand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            rand_floats[:, 5:5 + self.num_xhand_dofs] + 1)

        pos = self.xhand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.xhand_dof_pos[env_ids, :] = pos
        self.xhand_dof_vel[env_ids, :] = (
            self.xhand_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[:, 5 + self.num_xhand_dofs:5 + self.num_xhand_dofs * 2])
        self.prev_targets[env_ids, :self.num_xhand_dofs] = pos
        self.cur_targets[env_ids, :self.num_xhand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = (self.prev_targets[:, self.actuated_dof_indices]
                       + self.xhand_dof_speed_scale * self.dt * self.action_speed_scale
                       * self.actions)
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.xhand_dof_lower_limits[self.actuated_dof_indices],
                self.xhand_dof_upper_limits[self.actuated_dof_indices])
        else:
            new_targets = scale(
                self.actions,
                self.xhand_dof_lower_limits[self.actuated_dof_indices],
                self.xhand_dof_upper_limits[self.actuated_dof_indices])
            eff_moving_average = self.act_moving_average * self.action_speed_scale
            self.cur_targets[:, self.actuated_dof_indices] = (
                eff_moving_average * new_targets
                + (1.0 - eff_moving_average)
                * self.prev_targets[:, self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.xhand_dof_lower_limits[self.actuated_dof_indices],
                self.xhand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = \
            self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            force_indices = (
                torch.rand(self.num_envs, device=self.device) < self.random_force_prob
            ).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * self.object_rb_masses * self.force_scale
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(
                    self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)
                ).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(
                    self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)
                ).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(
                    self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)
                ).cpu().numpy()
                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2],
                                    targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2],
                                    targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2],
                                    targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float,
    ignore_z_rot: bool,
    object_linvel, object_angvel, dof_vel, hand_pos,
    obj_linvel_penalty_scale: float, obj_angvel_penalty_scale: float,
    dof_vel_penalty_scale: float, palm_dist_penalty_scale: float,
    obj_linvel_limit: float, obj_angvel_limit: float, dof_vel_limit: float,
    fall_ref_pos,
):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    fall_check_dist = torch.norm(object_pos - fall_ref_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    dof_vel_penalty = torch.sum(
        torch.clamp(torch.exp(torch.abs(dof_vel) - dof_vel_limit) - 1.0,
                    min=0.0, max=100.0), dim=-1)
    obj_linspeed = torch.norm(object_linvel, p=2, dim=-1)
    obj_linvel_penalty = torch.clamp(
        torch.exp(obj_linspeed - obj_linvel_limit) - 1.0, min=0.0, max=500.0)
    obj_angspeed = torch.norm(object_angvel, p=2, dim=-1)
    obj_angvel_penalty = torch.clamp(
        torch.exp(obj_angspeed - obj_angvel_limit) - 1.0, min=0.0, max=1000.0)
    palm_dist_penalty = torch.norm(object_pos - hand_pos, p=2, dim=-1)

    reward = (dist_rew + rot_rew
              + action_penalty * action_penalty_scale
              + obj_linvel_penalty * obj_linvel_penalty_scale
              + obj_angvel_penalty * obj_angvel_penalty_scale
              + dof_vel_penalty * dof_vel_penalty_scale
              + palm_dist_penalty * palm_dist_penalty_scale)

    goal_resets = torch.where(
        torch.abs(rot_dist) <= success_tolerance,
        torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    reward = torch.where(fall_check_dist >= fall_dist, reward + fall_penalty, reward)

    resets = torch.where(
        fall_check_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        progress_buf = torch.where(
            torch.abs(rot_dist) <= success_tolerance,
            torch.zeros_like(progress_buf), progress_buf)
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
        consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(
        quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
        quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
