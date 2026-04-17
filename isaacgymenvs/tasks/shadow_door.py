"""ShadowDoor: single Shadow Hand with floating base opening a door handle.

Ported from DexterousHands/bidexhands/tasks/shadow_hand_door_open_inward.py,
simplified to one right hand grasping the right door handle.

Action space (26 dims):
  [0:3]   base translation forces
  [3:6]   base rotation torques
  [6:26]  20 actuated finger DOF position targets

Observation space (full_state, 219 dims):
  [0:24]    hand DOF positions (unscaled)
  [24:48]   hand DOF velocities x 0.2
  [48:72]   hand DOF forces x 10.0
  [72:137]  5 fingertips x 13 (pose + vel)
  [137:167] 5 fingertip force-torque x 6
  [167:170] hand base position
  [170:173] hand base orientation (euler xyz)
  [173:199] last 26 actions
  [199:206] door root pose (pos + quat)
  [206:209] door linear velocity
  [209:212] door angular velocity x 0.2
  [212:215] right handle position
  [215:219] right handle rotation (quat)
"""

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import (
    scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, quat_apply,
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, get_euler_xyz,
)
from isaacgymenvs.tasks.base.vec_task import VecTask


# Shadow Hand constants
_NUM_HAND_DOFS = 24       # total DOFs (with tendon coupling)
_NUM_ACTUATED_DOFS = 20   # actuated finger joints
_NUM_ACTIONS = 26         # 6 base + 20 finger

_FINGERTIPS = [
    "robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal",
    "robot0:lfdistal", "robot0:thdistal",
]

# Observation dimensions
# hand: 3*24 + 65 + 30 + 3 + 3 + 26 = 199
# door: 7 + 3 + 3 + 3 + 4 = 20
_NUM_OBS = 219


class ShadowDoor(VecTask):

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

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.transition_scale = self.cfg["env"]["transitionScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.obs_type = self.cfg["env"]["observationType"]
        if self.obs_type != "full_state":
            raise Exception("ShadowDoor only supports observationType='full_state'")

        self.up_axis = 'z'

        self.fingertips = list(_FINGERTIPS)
        self.num_fingertips = len(self.fingertips)

        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = _NUM_OBS

        self.cfg["env"]["numObservations"] = _NUM_OBS
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = _NUM_ACTIONS

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device,
            graphics_device_id=graphics_device_id, headless=headless,
            virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Acquire GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, -1)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Wrapper tensors
        self.shadow_hand_default_dof_pos = torch.zeros(
            self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        # Door DOF state (after hand DOFs)
        self.door_dof_state = self.dof_state.view(
            self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs + self.num_door_dofs]
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_vel = self.door_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(
            self.num_envs * self.num_actors_per_env, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        # Force/torque buffers for floating base
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3),
                                        device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3),
                                        device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1

        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'],
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

        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))

        # --- Load hand asset (floating base) ---
        hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand.xml")
        if "asset" in self.cfg["env"]:
            hand_asset_file = os.path.normpath(
                self.cfg["env"]["asset"].get("assetFileName", hand_asset_file))

        hand_opts = gymapi.AssetOptions()
        hand_opts.flip_visual_attachments = False
        hand_opts.fix_base_link = False  # FLOATING BASE
        hand_opts.collapse_fixed_joints = True
        hand_opts.disable_gravity = True
        hand_opts.thickness = 0.001
        hand_opts.angular_damping = 100
        hand_opts.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_opts.use_physx_armature = True
        hand_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_opts)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(hand_asset)

        print("Shadow hand bodies:", self.num_shadow_hand_bodies)
        print("Shadow hand DOFs:", self.num_shadow_hand_dofs)
        print("Shadow hand actuators:", self.num_shadow_hand_actuators)

        # Tendon setup
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(hand_asset)
        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(hand_asset, tendon_props)

        actuated_dof_names = [
            self.gym.get_asset_actuator_joint_name(hand_asset, i)
            for i in range(self.num_shadow_hand_actuators)
        ]
        self.actuated_dof_indices = [
            self.gym.find_asset_dof_index(hand_asset, name) for name in actuated_dof_names
        ]

        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name) for name in self.fingertips
        ]

        # Create fingertip force sensors
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(hand_asset, ft_handle, sensor_pose)

        # --- Load door asset ---
        door_asset_file = os.path.normpath("mjcf/door/mobility.urdf")
        if "asset" in self.cfg["env"]:
            door_asset_file = os.path.normpath(
                self.cfg["env"]["asset"].get("assetFileNameDoor", door_asset_file))

        door_opts = gymapi.AssetOptions()
        door_opts.density = 500
        door_opts.fix_base_link = True
        door_opts.disable_gravity = True
        door_opts.use_mesh_materials = True
        door_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        door_opts.override_com = True
        door_opts.override_inertia = True
        door_opts.vhacd_enabled = True
        door_opts.vhacd_params = gymapi.VhacdParams()
        door_opts.vhacd_params.resolution = 100000
        door_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        door_asset = self.gym.load_asset(self.sim, asset_root, door_asset_file, door_opts)

        self.num_door_bodies = self.gym.get_asset_rigid_body_count(door_asset)
        self.num_door_shapes = self.gym.get_asset_rigid_shape_count(door_asset)
        self.num_door_dofs = self.gym.get_asset_dof_count(door_asset)

        print("Door bodies:", self.num_door_bodies)
        print("Door DOFs:", self.num_door_dofs)

        # Door DOF properties
        door_dof_props = self.gym.get_asset_dof_properties(door_asset)

        # Right handle body index within the door asset
        # Door structure: base(0), link_0(1), link_1=right_handle(2), link_2=left_handle(3)
        self.door_right_handle_body_idx = self.num_shadow_hand_bodies + 2

        # --- Start poses ---
        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(0.55, 0.2, 0.6)
        hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 1.57, 1.57)

        door_start_pose = gymapi.Transform()
        door_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.7)
        door_start_pose.r = gymapi.Quat().from_euler_zyx(0, 3.14159, 0.0)

        # Aggregate sizes
        max_agg_bodies = self.num_shadow_hand_bodies + self.num_door_bodies
        max_agg_shapes = self.num_shadow_hand_shapes + self.num_door_shapes

        self.num_actors_per_env = 2  # hand + door

        self.shadow_hands = []
        self.envs = []

        self.door_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.door_indices = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Add hand
            hand_actor = self.gym.create_actor(
                env_ptr, hand_asset, hand_start_pose, "hand", i, 0, 0)
            self.hand_start_states.append([
                hand_start_pose.p.x, hand_start_pose.p.y, hand_start_pose.p.z,
                hand_start_pose.r.x, hand_start_pose.r.y, hand_start_pose.r.z, hand_start_pose.r.w,
                0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

            # Add door
            door_actor = self.gym.create_actor(
                env_ptr, door_asset, door_start_pose, "object", i, 0, 0)
            self.door_init_state.append([
                door_start_pose.p.x, door_start_pose.p.y, door_start_pose.p.z,
                door_start_pose.r.x, door_start_pose.r.y, door_start_pose.r.z, door_start_pose.r.w,
                0, 0, 0, 0, 0, 0])
            door_idx = self.gym.get_actor_index(env_ptr, door_actor, gymapi.DOMAIN_SIM)
            self.door_indices.append(door_idx)

            # Set door DOF properties (stiffness/damping for hinges)
            actor_door_dof_props = self.gym.get_actor_dof_properties(env_ptr, door_actor)
            for prop in actor_door_dof_props:
                prop['stiffness'] = 100
                prop['damping'] = 100
                prop['effort'] = 5
                prop['friction'] = 1
            self.gym.set_actor_dof_properties(env_ptr, door_actor, actor_door_dof_props)

            # Set door friction
            door_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, door_actor)
            for sp in door_shape_props:
                sp.friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, door_actor, door_shape_props)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(hand_actor)

        self.door_init_state = to_torch(self.door_init_state, device=self.device,
                                         dtype=torch.float).view(self.num_envs, 13)
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.door_indices = to_torch(self.door_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        (self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:],
         self.progress_buf[:], self.successes[:], self.consecutive_successes[:]) = \
            compute_door_reward(
                self.rew_buf, self.reset_buf, self.reset_goal_buf,
                self.progress_buf, self.successes, self.consecutive_successes,
                self.max_episode_length,
                self.door_right_handle_pos,
                self.right_hand_ff_pos, self.right_hand_mf_pos,
                self.right_hand_rf_pos, self.right_hand_lf_pos,
                self.right_hand_th_pos,
                self.dist_reward_scale, self.actions,
                self.action_penalty_scale, self.success_tolerance,
                self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
                self.max_consecutive_successes, self.av_factor,
                self.door_dof_pos,
            )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

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

        # Door state from root tensor
        self.door_pose = self.root_state_tensor[self.door_indices, 0:7]
        self.door_pos = self.root_state_tensor[self.door_indices, 0:3]
        self.door_rot = self.root_state_tensor[self.door_indices, 3:7]
        self.door_linvel = self.root_state_tensor[self.door_indices, 7:10]
        self.door_angvel = self.root_state_tensor[self.door_indices, 10:13]

        # Right handle position (with offset in handle local frame)
        self.door_right_handle_pos = self.rigid_body_states[:, self.door_right_handle_body_idx, 0:3].clone()
        self.door_right_handle_rot = self.rigid_body_states[:, self.door_right_handle_body_idx, 3:7]
        self.door_right_handle_pos = self.door_right_handle_pos + quat_apply(
            self.door_right_handle_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.5)
        self.door_right_handle_pos = self.door_right_handle_pos + quat_apply(
            self.door_right_handle_rot,
            to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.39)
        self.door_right_handle_pos = self.door_right_handle_pos + quat_apply(
            self.door_right_handle_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        # Hand palm position (body index 3 = wrist/palm area in shadow hand)
        self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(
            self.right_hand_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(
            self.right_hand_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # Fingertip positions (with small offset along z)
        self.right_hand_ff_pos = self.rigid_body_states[:, 7, 0:3] + quat_apply(
            self.rigid_body_states[:, 7, 3:7],
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, 11, 0:3] + quat_apply(
            self.rigid_body_states[:, 11, 3:7],
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, 15, 0:3] + quat_apply(
            self.rigid_body_states[:, 15, 3:7],
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, 20, 0:3] + quat_apply(
            self.rigid_body_states[:, 20, 3:7],
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, 25, 0:3] + quat_apply(
            self.rigid_body_states[:, 25, 3:7],
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        # Fingertip state (for obs)
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]

        self.compute_full_state()

    def compute_full_state(self):
        num_ft_states = 13 * self.num_fingertips  # 65
        num_ft_force_torques = 6 * self.num_fingertips  # 30

        # Hand DOF pos/vel/force
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(
            self.shadow_hand_dof_pos,
            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, 24:48] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 48:72] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]

        # Fingertip states
        fingertip_obs_start = 72
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = \
            self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:
                     fingertip_obs_start + num_ft_states + num_ft_force_torques] = \
            self.force_torque_obs_scale * self.vec_sensor_tensor

        # Hand base pose
        hand_pose_start = fingertip_obs_start + num_ft_states + num_ft_force_torques  # 167
        hand_root_pos = self.root_state_tensor[self.hand_indices, 0:3]
        hand_root_rot = self.root_state_tensor[self.hand_indices, 3:7]
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = hand_root_pos
        euler = get_euler_xyz(hand_root_rot)
        self.obs_buf[:, hand_pose_start + 3] = euler[0]
        self.obs_buf[:, hand_pose_start + 4] = euler[1]
        self.obs_buf[:, hand_pose_start + 5] = euler[2]

        # Last actions
        action_obs_start = hand_pose_start + 6  # 173
        self.obs_buf[:, action_obs_start:action_obs_start + _NUM_ACTIONS] = self.actions

        # Door state
        door_obs_start = action_obs_start + _NUM_ACTIONS  # 199
        self.obs_buf[:, door_obs_start:door_obs_start + 7] = self.door_pose
        self.obs_buf[:, door_obs_start + 7:door_obs_start + 10] = self.door_linvel
        self.obs_buf[:, door_obs_start + 10:door_obs_start + 13] = self.vel_obs_scale * self.door_angvel
        self.obs_buf[:, door_obs_start + 13:door_obs_start + 16] = self.door_right_handle_pos
        self.obs_buf[:, door_obs_start + 16:door_obs_start + 20] = self.door_right_handle_rot

    def reset_idx(self, env_ids, goal_env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # Reset door root state
        self.root_state_tensor[self.door_indices[env_ids]] = self.door_init_state[env_ids].clone()
        self.root_state_tensor[self.door_indices[env_ids], 7:13] = 0

        # Reset hand root state
        self.root_state_tensor[self.hand_indices[env_ids]] = self.hand_start_states[env_ids].clone()

        all_indices = torch.unique(torch.cat([
            self.hand_indices[env_ids], self.door_indices[env_ids]
        ]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices), len(all_indices))

        # Reset hand DOFs
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            rand_floats[:, 5:5 + self.num_shadow_hand_dofs] + 1)
        pos = self.shadow_hand_dof_default_pos + self.reset_dof_pos_noise * rand_delta

        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        # Reset door DOFs to closed
        self.door_dof_pos[env_ids, :] = 0
        self.door_dof_vel[env_ids, :] = 0
        door_dof_start = self.num_shadow_hand_dofs
        self.prev_targets[env_ids, door_dof_start:door_dof_start + self.num_door_dofs] = 0
        self.cur_targets[env_ids, door_dof_start:door_dof_start + self.num_door_dofs] = 0

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        door_indices_i32 = self.door_indices[env_ids].to(torch.int32)
        all_dof_indices = torch.unique(torch.cat([hand_indices, door_indices_i32]))

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_dof_indices), len(all_dof_indices))

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_dof_indices), len(all_dof_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        # Zero out force buffers
        self.apply_forces[:] = 0
        self.apply_torque[:] = 0

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + \
                self.shadow_hand_dof_speed_scale * self.dt * self.actions[:, 6:26]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            new_targets = scale(
                self.actions[:, 6:26],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = \
                self.act_moving_average * new_targets + \
                (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        # Apply floating base forces/torques to forearm body (index 1)
        self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
        self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            for i in range(self.num_envs):
                p0 = self.door_right_handle_pos[i].cpu().numpy()
                p1 = self.right_hand_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]],
                                   [0.85, 0.1, 0.1])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_door_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float,
    door_right_handle_pos,
    right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos,
    right_hand_lf_pos, right_hand_th_pos,
    dist_reward_scale: float, actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float,
    fall_dist: float, fall_penalty: float,
    max_consecutive_successes: int, av_factor: float,
    door_dof_pos,
):
    # Sum of distances from each fingertip to the right handle
    finger_dist = (
        torch.norm(door_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_mf_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) +
        torch.norm(door_right_handle_pos - right_hand_th_pos, p=2, dim=-1))

    # Opening reward: how far the right handle joint has rotated
    # door_dof_pos[:, 0] is the right handle joint angle (range 0 to pi/2)
    handle_angle = door_dof_pos[:, 0]
    up_rew = torch.zeros_like(finger_dist)
    up_rew = torch.where(finger_dist < 0.5,
                         handle_angle * 2.0, up_rew)

    action_penalty = torch.sum(actions ** 2, dim=-1)

    reward = (2.0 - finger_dist) * dist_reward_scale + up_rew + action_penalty * action_penalty_scale

    # Success: handle opened beyond threshold
    successes = torch.where(successes == 0,
                            torch.where(handle_angle > success_tolerance,
                                        torch.ones_like(successes), successes),
                            successes)

    # Reset conditions
    resets = torch.where(finger_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
