# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""ShadowHandTrack — reference-trajectory tracking controller for the Shadow Hand.

A TopoRetarget-style (paper §4 / Appendix A.5) minimal RL tracking controller: a
PPO policy issues a residual action on a per-frame reference finger-joint trajectory
so the (physically simulated) object and fingers track the reference as closely as
possible under contact dynamics.

Design (see the project plan):
  * Fixed hand base; everything tracked in the PALM (= base) frame.  The object is
    placed and rewarded relative to the current palm body pose, matching how the
    reference clip was built (``pose_to_ego(object_pose, palm_pose)`` on the demo).
  * Action = 18-DOF finger residual on the reference joints; wrist held at 0.
  * Reference clips are produced offline by ``scripts/make_track_reference.py`` in the
    purifier repo (a demo H5 formatted to radians + palm-frame object pose, resampled
    to the controller rate).  Quaternions in the .npz are WXYZ (project convention) and
    are converted to IsaacGym XYZW on load.
  * Reference tracked-link (fingertip) positions are FK-baked from the reference finger
    configs using the simulator's own kinematics at init (no external FK / URDF-name
    mismatch).
"""

import os
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import (
    to_torch, scale, tensor_clamp, quat_mul, quat_conjugate,
    quat_rotate, quat_rotate_inverse, quat_from_angle_axis,
)
from isaacgymenvs.tasks.base.vec_task import VecTask


# ─────────────────────────────────────────────────────────────────────────────
# Reference library
# ─────────────────────────────────────────────────────────────────────────────
class HandReferenceLib:
    """Loads a tracking-reference .npz onto the GPU and serves clip frames.

    Stored tensors (padded to ``t_max`` across clips):
      finger_ref    (N, Tmax, 18)  finger joint angles (radians)
      obj_pose_palm (N, Tmax, 7)   object pose in palm frame [xyz, qx,qy,qz,qw]  (XYZW!)
      lengths       (N,)           valid frame count per clip
      link_ref      (N, Tmax, L, 3) FK-baked reference fingertip positions (palm frame),
                                    filled later by the env via ``set_link_ref``.
    """

    def __init__(self, npz_path, device):
        data = np.load(npz_path, allow_pickle=True)
        self.device = device
        finger = torch.from_numpy(data["finger_ref"]).float().to(device)      # (N,Tmax,18)
        obj = torch.from_numpy(data["obj_pose_palm"]).float().to(device)       # (N,Tmax,7) wxyz
        # wxyz -> xyzw for IsaacGym.
        pos, quat_wxyz = obj[..., :3], obj[..., 3:7]
        quat_xyzw = torch.cat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)
        self.finger_ref = finger
        self.obj_pose_palm = torch.cat([pos, quat_xyzw], dim=-1)               # (N,Tmax,7) xyzw
        self.lengths = torch.from_numpy(data["clip_lengths"]).long().to(device)
        self.finger_lower = torch.from_numpy(data["finger_lower"]).float().to(device)
        self.finger_upper = torch.from_numpy(data["finger_upper"]).float().to(device)
        self.target_freq = float(data["target_freq"])
        self.demo_freq = float(data["demo_freq"])
        self.obj_bbox_min = torch.from_numpy(data["obj_bbox_min"]).float().to(device)
        self.obj_bbox_max = torch.from_numpy(data["obj_bbox_max"]).float().to(device)
        self.n_clips, self.t_max = finger.shape[0], finger.shape[1]
        self.link_ref = None  # set after FK-bake: (N, Tmax, L, 3)

    def set_link_ref(self, link_ref):
        self.link_ref = link_ref

    def clamp_k(self, clip_id, k):
        """Clamp a (possibly look-ahead) frame index to the clip's last valid frame."""
        return torch.minimum(k, (self.lengths[clip_id] - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────
class ShadowHandTrack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        self.cfg = cfg

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"].get("aggregateMode", 1)
        self.max_episode_length = int(self.cfg["env"]["episodeLength"])

        # ── explicit frequency block (recorded, no implicit rates) ──
        self.control_freq = float(self.cfg["env"]["controlFreq"])
        self.sim_freq = float(self.cfg["env"]["simFreq"])
        decimation = int(round(self.sim_freq / self.control_freq))
        self.cfg["sim"]["dt"] = 1.0 / self.sim_freq
        self.cfg["env"]["controlFrequencyInv"] = decimation
        print(f"[track] control={self.control_freq}Hz  sim={self.sim_freq}Hz  "
              f"decimation={decimation}")

        # ── reward weights / kernel sigmas (Table 4) ──
        rw = self.cfg["env"]["reward"]
        self.w_obj = float(rw["wObj"]);      self.sig_obj = float(rw["sigmaObj"])
        self.w_link = float(rw["wLink"]);    self.sig_link = float(rw["sigmaLink"])
        self.w_joint = float(rw["wJoint"]);  self.sig_joint = float(rw["sigmaJoint"])
        self.w_smooth = float(rw["wSmooth"])
        self.axis_len = float(self.cfg["env"].get("axisPointLen", 0.04))

        # ── termination thresholds (Table 4) ──
        tm = self.cfg["env"]["termination"]
        self.term_obj_pos = float(tm["objPosErr"])
        self.term_obj_rot = float(np.deg2rad(tm["objRotErrDeg"]))
        self.term_axis = float(tm["axisPointErr"])
        self.term_linvel = float(tm["objLinvel"])
        self.term_angvel = float(tm["objAngvel"])

        # ── lookahead offsets ──
        self.lookahead = list(self.cfg["env"].get("lookahead", [1, 3, 5]))

        self.reset_obj_pos_noise = float(self.cfg["env"].get("resetObjPosNoise", 0.005))
        self.reset_obj_rot_noise = float(np.deg2rad(self.cfg["env"].get("resetObjRotNoiseDeg", 0.03 * 57.2958)))
        self.reset_dof_noise = float(self.cfg["env"].get("resetDofNoise", 0.02))
        self.action_scale = float(self.cfg["env"].get("actionScale", 0.3))
        self.stiffness_scale = float(self.cfg["env"].get("stiffnessScale", 1.0))
        self.obs_noise = float(self.cfg["env"].get("obsNoise", 0.0))

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.6, 1.8])  # seconds
        self.torque_scale = self.cfg["env"].get("torqueScale", 0.0)

        self.up_axis = 'z'
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal",
                           "robot0:lfdistal", "robot0:thdistal"]
        self.num_track_links = len(self.fingertips)   # L = 5

        # ── load reference (needs sizes to define obs) ──
        ref_path = self.cfg["env"]["referenceFile"]
        if not os.path.isabs(ref_path):
            ref_path = os.path.abspath(ref_path)
        self._ref_path = ref_path
        _tmp = np.load(ref_path, allow_pickle=True)
        ref_target_freq = float(_tmp["target_freq"])
        del _tmp
        assert abs(ref_target_freq - self.control_freq) < 1e-6, (
            f"reference target_freq={ref_target_freq} != controlFreq={self.control_freq}; "
            "regenerate the .npz with --target_freq matching the env controlFreq.")

        # ── observation / action dims ──
        n_prop = 18 + 18 + 18                       # dof_pos, dof_vel, prev_action
        n_obj = 6 * 3                               # object axis-points (palm frame)
        n_ref_cur = 18 + 6 * 3 + self.num_track_links * 3   # joints + obj-axis + links
        n_ref_look = len(self.lookahead) * (18 + 6 * 3)     # joints + obj-axis per lookahead
        num_obs = n_prop + n_obj + n_ref_cur + n_ref_look

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = 0
        self.cfg["env"]["numActions"] = 18

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(0.6, 0.0, 1.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # ── acquire GPU state tensors ──
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # finger DOF indices (drop the 2 wrist actuators) into the 24-DOF state.
        self.finger_dof_indices = self.actuated_dof_indices[2:]   # (18,)
        self.wrist_dof_indices = self.actuated_dof_indices[:2]    # (2,)

        # per-env reference bookkeeping
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.k_t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_action = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device)
        self.prev_prev_action = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.num_envs, 18), dtype=torch.float, device=self.device)

        # object random-force machinery (Table 5 external disturbances)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.rb_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # cache the (constant, fixed-base) palm body world pose
        palm_state = self.rigid_body_states[:, self.palm_body_idx]
        self.palm_pos = palm_state[:, 0:3].clone()   # (N,3)
        self.palm_rot = palm_state[:, 3:7].clone()    # (N,4) xyzw

        # local object axis-point offsets: ±axis_len along x,y,z  → 6 points (3,) each
        self.axis_offsets = to_torch([
            [ self.axis_len, 0, 0], [-self.axis_len, 0, 0],
            [0,  self.axis_len, 0], [0, -self.axis_len, 0],
            [0, 0,  self.axis_len], [0, 0, -self.axis_len],
        ], device=self.device)                        # (6,3)

        # FK-bake reference fingertip positions in the palm frame.
        self._bake_link_reference()

    # ── sim / envs ──────────────────────────────────────────────────────────
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand.xml")
        object_asset_file = self.cfg["env"].get("objectAsset", "urdf/objects/cube_multicolor.urdf")
        if "asset" in self.cfg["env"]:
            hand_asset_file = os.path.normpath(
                self.cfg["env"]["asset"].get("assetFileName", hand_asset_file))

        # hand asset (fixed base, MJCF position actuators)
        ao = gymapi.AssetOptions()
        ao.flip_visual_attachments = False
        ao.fix_base_link = True
        ao.collapse_fixed_joints = True
        ao.disable_gravity = True
        ao.thickness = 0.001
        ao.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            ao.use_physx_armature = True
        ao.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, ao)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(hand_asset)
        num_tendons = self.gym.get_asset_tendon_count(hand_asset)

        # tendon coupling (as in ShadowHand)
        relevant = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tp = self.gym.get_asset_tendon_properties(hand_asset)
        for i in range(num_tendons):
            if self.gym.get_asset_tendon_name(hand_asset, i) in relevant:
                tp[i].limit_stiffness = 30
                tp[i].damping = 0.1
        self.gym.set_asset_tendon_properties(hand_asset, tp)

        actuated_names = [self.gym.get_asset_actuator_joint_name(hand_asset, i)
                          for i in range(self.num_shadow_hand_actuators)]
        actuated_dof_indices = [self.gym.find_asset_dof_index(hand_asset, n) for n in actuated_names]

        dof_props = self.gym.get_asset_dof_properties(hand_asset)
        dof_props['stiffness'] *= self.stiffness_scale

        lower_l, upper_l = [], []
        for i in range(self.num_shadow_hand_dofs):
            lower_l.append(dof_props['lower'][i])
            upper_l.append(dof_props['upper'][i])
        self.actuated_dof_indices = to_torch(actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(lower_l, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(upper_l, device=self.device)

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(hand_asset, n) for n in self.fingertips]
        self.palm_body_idx = self.gym.find_asset_rigid_body_index(hand_asset, "robot0:palm")

        # object asset (gravity ON) — cube_multicolor matches the block demos
        oo = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, oo)
        obj_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        obj_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        hand_start = gymapi.Transform()
        hand_start.p = gymapi.Vec3(0.0, 0.0, 0.5)
        object_start = gymapi.Transform()
        object_start.p = gymapi.Vec3(0.0, -0.1, 0.5)   # provisional; RSI overrides at reset

        max_agg_bodies = self.num_shadow_hand_bodies + obj_rb_count
        max_agg_shapes = self.num_shadow_hand_shapes + obj_shapes

        self.envs = []
        self.hand_indices = []
        self.object_indices = []
        hand_rb_count = self.num_shadow_hand_bodies
        self.object_rb_handles = list(range(hand_rb_count, hand_rb_count + obj_rb_count))

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, dof_props)
            self.hand_indices.append(self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM))

            obj_actor = self.gym.create_actor(env_ptr, object_asset, object_start, "object", i, 0, 0)
            self.object_indices.append(self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        obj_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, obj_actor)
        self.object_rb_masses = to_torch([p.mass for p in obj_rb_props], dtype=torch.float, device=self.device)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        # load the reference now that the device is known
        self.ref = HandReferenceLib(self._ref_path, self.device)

    # ── FK-bake reference fingertip positions (palm frame) ────────────────────
    def _bake_link_reference(self):
        N, Tmax, L = self.ref.n_clips, self.ref.t_max, self.num_track_links
        flat = self.ref.finger_ref.reshape(-1, 18)              # (M,18) radians
        M = flat.shape[0]
        link_flat = torch.zeros(M, L, 3, device=self.device)

        # move all objects far away so the bake never contacts them
        self.root_state_tensor[self.object_indices, 2] = -20.0
        self.root_state_tensor[self.object_indices, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(self.object_indices.to(torch.int32)), len(self.object_indices))

        chunk = self.num_envs
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            m = e - s
            env_ids = torch.arange(m, device=self.device)
            dof_pos = torch.zeros(m, self.num_shadow_hand_dofs, device=self.device)
            dof_pos[:, self.finger_dof_indices] = flat[s:e]
            self.shadow_hand_dof_pos[env_ids] = dof_pos
            self.shadow_hand_dof_vel[env_ids] = 0.0
            self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos
            self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos
            hand_int = self.hand_indices[env_ids].to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(hand_int), m)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_int), m)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            ft_world = self.rigid_body_states[env_ids][:, self.fingertip_handles, 0:3]  # (m,L,3)
            palm_pos = self.palm_pos[env_ids].unsqueeze(1)                                # (m,1,3)
            palm_rot = self.palm_rot[env_ids].unsqueeze(1).expand(-1, L, -1).reshape(-1, 4)
            rel = (ft_world - palm_pos).reshape(-1, 3)
            link_flat[s:e] = quat_rotate_inverse(palm_rot, rel).reshape(m, L, 3)

        self.ref.set_link_ref(link_flat.reshape(N, Tmax, L, 3))
        print(f"[track] FK-baked reference fingertips for {M} frames "
              f"({N} clips × up to {Tmax}).")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _object_axis_points(self, obj_pos_palm, obj_rot_palm):
        """(N,3),(N,4 xyzw) → (N,6,3) axis points in the palm frame."""
        n = obj_pos_palm.shape[0]
        off = self.axis_offsets.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)   # (N*6,3)
        rot = obj_rot_palm.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4)
        pts = quat_rotate(rot, off).reshape(n, 6, 3) + obj_pos_palm.unsqueeze(1)
        return pts

    def _obj_pose_in_palm(self):
        """Measured object pose expressed in the (fixed) palm frame."""
        obj_pos = self.root_state_tensor[self.object_indices, 0:3]
        obj_rot = self.root_state_tensor[self.object_indices, 3:7]
        pos_palm = quat_rotate_inverse(self.palm_rot, obj_pos - self.palm_pos)
        rot_palm = quat_mul(quat_conjugate(self.palm_rot), obj_rot)
        return pos_palm, rot_palm

    def _fingertips_in_palm(self):
        ft_world = self.rigid_body_states[:, self.fingertip_handles, 0:3]   # (N,L,3)
        L = self.num_track_links
        palm_rot = self.palm_rot.unsqueeze(1).expand(-1, L, -1).reshape(-1, 4)
        rel = (ft_world - self.palm_pos.unsqueeze(1)).reshape(-1, 3)
        return quat_rotate_inverse(palm_rot, rel).reshape(-1, L, 3)

    def _ref_at(self, offset=0):
        """Reference (finger, obj_pose_palm, link) at k_t+offset, clamped to clip end."""
        k = self.ref.clamp_k(self.clip_id, self.k_t + offset)
        finger = self.ref.finger_ref[self.clip_id, k]               # (N,18)
        obj = self.ref.obj_pose_palm[self.clip_id, k]               # (N,7) xyzw
        link = self.ref.link_ref[self.clip_id, k]                  # (N,L,3)
        return finger, obj, link

    # ── stepping ────────────────────────────────────────────────────────────
    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.prev_prev_action = self.prev_action.clone()
        self.prev_action = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        finger_ref, _, _ = self._ref_at(0)                          # (N,18) radians
        target = finger_ref + self.action_scale * self.actions
        target = tensor_clamp(target,
                              self.shadow_hand_dof_lower_limits[self.finger_dof_indices],
                              self.shadow_hand_dof_upper_limits[self.finger_dof_indices])
        self.cur_targets[:, self.finger_dof_indices] = target
        self.cur_targets[:, self.wrist_dof_indices] = 0.0
        self.prev_targets[:] = self.cur_targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # intermittent external object disturbance (Table 5)
        if self.force_scale > 0.0 or self.torque_scale > 0.0:
            steps = max(int(self.control_freq * float(np.mean(self.force_prob_range))), 1)
            prob = 1.0 / steps
            hit = (torch.rand(self.num_envs, device=self.device) < prob).nonzero(as_tuple=False).squeeze(-1)
            self.rb_forces[:] = 0.0
            self.rb_torques[:] = 0.0
            if len(hit) > 0:
                obj_rb = self.object_rb_handles[0]
                if self.force_scale > 0.0:
                    self.rb_forces[hit, obj_rb, :] = torch.randn(len(hit), 3, device=self.device) * self.force_scale
                if self.torque_scale > 0.0:
                    self.rb_torques[hit, obj_rb, :] = torch.randn(len(hit), 3, device=self.device) * self.torque_scale
                self.gym.apply_rigid_body_force_tensors(
                    self.sim, gymtorch.unwrap_tensor(self.rb_forces),
                    gymtorch.unwrap_tensor(self.rb_torques), gymapi.ENV_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.k_t += 1
        self.compute_observations()
        self.compute_reward()

    # ── observation ───────────────────────────────────────────────────────────
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        finger_pos = self.shadow_hand_dof_pos[:, self.finger_dof_indices]   # (N,18)
        finger_vel = self.shadow_hand_dof_vel[:, self.finger_dof_indices]
        obj_pos_palm, obj_rot_palm = self._obj_pose_in_palm()
        obj_axis = self._object_axis_points(obj_pos_palm, obj_rot_palm)     # (N,6,3)

        cur_finger_ref, cur_obj_ref, cur_link_ref = self._ref_at(0)
        cur_obj_ref_axis = self._object_axis_points(cur_obj_ref[:, :3], cur_obj_ref[:, 3:7])

        obs = [finger_pos, finger_vel, self.actions,
               obj_axis.reshape(self.num_envs, -1),
               cur_finger_ref, cur_obj_ref_axis.reshape(self.num_envs, -1),
               cur_link_ref.reshape(self.num_envs, -1)]
        for off in self.lookahead:
            f, o, _ = self._ref_at(off)
            o_axis = self._object_axis_points(o[:, :3], o[:, 3:7])
            obs.append(f)
            obs.append(o_axis.reshape(self.num_envs, -1))
        obs_buf = torch.cat(obs, dim=-1)
        if self.obs_noise > 0.0:
            obs_buf = obs_buf + self.obs_noise * torch.randn_like(obs_buf)
        self.obs_buf[:] = obs_buf

        # stash for reward
        self._obj_pos_palm = obj_pos_palm
        self._obj_rot_palm = obj_rot_palm
        self._obj_axis = obj_axis
        self._finger_pos = finger_pos

    # ── reward + termination ──────────────────────────────────────────────────
    def compute_reward(self):
        cur_finger_ref, cur_obj_ref, cur_link_ref = self._ref_at(0)
        ref_axis = self._object_axis_points(cur_obj_ref[:, :3], cur_obj_ref[:, 3:7])

        # object axis-point tracking
        axis_err = (self._obj_axis - ref_axis).norm(dim=-1).mean(dim=-1)          # (N,)
        r_obj = torch.exp(-(axis_err / self.sig_obj) ** 2)

        # fingertip link tracking
        meas_link = self._fingertips_in_palm()                                    # (N,L,3)
        link_err = (meas_link - cur_link_ref).norm(dim=-1)                        # (N,L)
        r_link = torch.exp(-(link_err / self.sig_link) ** 2).mean(dim=-1)

        # normalized joint tracking
        span = (self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_lower_limits)[self.finger_dof_indices]
        joint_err = ((self._finger_pos - cur_finger_ref) / span)                  # (N,18)
        r_joint = torch.exp(-(joint_err / self.sig_joint) ** 2).mean(dim=-1)

        # action smoothness
        r_smooth = ((self.actions - self.prev_action) ** 2).sum(-1) + \
                   ((self.actions - 2 * self.prev_action + self.prev_prev_action) ** 2).sum(-1)

        reward = (self.w_obj * r_obj + self.w_link * r_link
                  + self.w_joint * r_joint + self.w_smooth * r_smooth)

        # ── terminations ──
        obj_pos_err = (self._obj_pos_palm - cur_obj_ref[:, :3]).norm(dim=-1)
        quat_diff = quat_mul(self._obj_rot_palm, quat_conjugate(cur_obj_ref[:, 3:7]))
        rot_err = 2.0 * torch.asin(torch.clamp(quat_diff[:, 0:3].norm(dim=-1), max=1.0))
        max_axis_err = (self._obj_axis - ref_axis).norm(dim=-1).max(dim=-1).values
        linvel = self.root_state_tensor[self.object_indices, 7:10].norm(dim=-1)
        angvel = self.root_state_tensor[self.object_indices, 10:13].norm(dim=-1)

        fail = (obj_pos_err > self.term_obj_pos) | (rot_err > self.term_obj_rot) | \
               (max_axis_err > self.term_axis) | (linvel > self.term_linvel) | \
               (angvel > self.term_angvel)

        clip_done = self.k_t >= (self.ref.lengths[self.clip_id] - 1)
        timeout = self.progress_buf >= self.max_episode_length - 1

        self.reset_buf[:] = (fail | clip_done | timeout).long()
        self.rew_buf[:] = reward
        self.extras["r_obj"] = r_obj.mean()
        self.extras["r_link"] = r_link.mean()
        self.extras["r_joint"] = r_joint.mean()
        self.extras["obj_pos_err"] = obj_pos_err.mean()

    # ── reset (RSI) ───────────────────────────────────────────────────────────
    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        n = len(env_ids)
        # RSI: uniform clip + uniform start frame
        clip = torch.randint(0, self.ref.n_clips, (n,), device=self.device)
        lengths = self.ref.lengths[clip]
        k0 = (torch.rand(n, device=self.device) * (lengths.float() - 1)).long()
        self.clip_id[env_ids] = clip
        self.k_t[env_ids] = k0

        finger0 = self.ref.finger_ref[clip, k0]                     # (n,18) radians
        obj0 = self.ref.obj_pose_palm[clip, k0]                     # (n,7) xyzw palm frame

        # reset hand dof to the reference finger config (+ small noise), wrist 0
        dof_pos = torch.zeros(n, self.num_shadow_hand_dofs, device=self.device)
        noisy = finger0 + self.reset_dof_noise * torch.randn(n, 18, device=self.device)
        noisy = tensor_clamp(noisy,
                             self.shadow_hand_dof_lower_limits[self.finger_dof_indices],
                             self.shadow_hand_dof_upper_limits[self.finger_dof_indices])
        dof_pos[:, self.finger_dof_indices] = noisy
        self.shadow_hand_dof_pos[env_ids] = dof_pos
        self.shadow_hand_dof_vel[env_ids] = 0.0
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos

        # place object: palm ∘ (ref palm-frame pose) + small reset noise
        pos_palm = obj0[:, :3] + self.reset_obj_pos_noise * torch.randn(n, 3, device=self.device)
        axis = torch.randn(n, 3, device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ang = self.reset_obj_rot_noise * torch.randn(n, device=self.device)
        dq = quat_from_angle_axis(ang, axis)                        # xyzw
        rot_palm = quat_mul(dq, obj0[:, 3:7])

        palm_pos = self.palm_pos[env_ids]
        palm_rot = self.palm_rot[env_ids]
        obj_world_pos = quat_rotate(palm_rot, pos_palm) + palm_pos
        obj_world_rot = quat_mul(palm_rot, rot_palm)
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = obj_world_pos
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = obj_world_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0.0

        hand_int = self.hand_indices[env_ids].to(torch.int32)
        obj_int = self.object_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_int), n)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_int), n)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj_int), n)

        self.prev_action[env_ids] = 0.0
        self.prev_prev_action[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
