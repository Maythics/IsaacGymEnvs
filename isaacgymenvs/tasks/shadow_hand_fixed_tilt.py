# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.

"""Fixed-base-tilt ShadowHand task variants.

This module deliberately keeps the parent ShadowHand observation and action
interfaces unchanged.  In particular, ``Shadowhand18Tilted`` is a physics-only
curriculum variant of Shadowhand18, so its 209-observation / 18-action
checkpoints can be resumed directly at a new fixed tilt angle.
"""

import math

import torch
from isaacgym import gymtorch

from isaacgymenvs.tasks.shadow_hand import (
    ShadowHand,
    randomize_rotation,
    randomize_rotation_pen,
)
from isaacgymenvs.tasks.object_gravity_compensation import (
    ObjectGravityCompensationMixin,
)
from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    torch_rand_float,
)


class ShadowHandFixedTilt(ObjectGravityCompensationMixin, ShadowHand):
    """ShadowHand with a constant root orientation and palm-anchored object reset.

    ``baseTiltAngleDeg`` and ``baseTiltAxis`` define a world-frame axis-angle
    rotation that is left-multiplied onto the asset's true initial root
    quaternion.  The object keeps the parent's randomized reset orientation,
    but its position is reconstructed in the tilted palm frame on every reset.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        self._base_tilt_angle_rad = math.radians(
            float(cfg["env"].get("baseTiltAngleDeg", 0.0))
        )
        self._base_tilt_axis_cfg = cfg["env"].get("baseTiltAxis", [0.0, 1.0, 0.0])
        self._base_yaw_angle_rad = math.radians(
            float(cfg["env"].get("baseYawDeg", 0.0))
        )
        self._object_palm_offset_cfg = cfg["env"].get("objectPalmOffset", [0.0, 0.0, 0.0])
        raw_object_palm_offset_candidates = cfg["env"].get(
            "objectPalmOffsetCandidates", []
        )

        if len(self._base_tilt_axis_cfg) != 3:
            raise ValueError("baseTiltAxis must contain exactly three values.")
        if len(self._object_palm_offset_cfg) != 3:
            raise ValueError("objectPalmOffset must contain exactly three values.")
        # Hydra keeps nested overrides as OmegaConf ListConfig instances.
        # Convert any iterable here so both ordinary YAML lists and the probe's
        # command-line candidate stencil follow the same task path.
        try:
            raw_candidates = list(raw_object_palm_offset_candidates)
        except TypeError as exc:
            raise ValueError(
                "objectPalmOffsetCandidates must be an iterable of three-number offsets."
            ) from exc
        self._object_palm_offset_candidates_cfg = []
        for offset in raw_candidates:
            try:
                values = [float(value) for value in offset]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "objectPalmOffsetCandidates entries must contain three numbers."
                ) from exc
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    "objectPalmOffsetCandidates entries must contain three finite numbers."
                )
            self._object_palm_offset_candidates_cfg.append(values)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                         virtual_screen_capture, force_render)
        self._configure_object_gravity_compensation(cfg)

        axis = torch.tensor(self._base_tilt_axis_cfg, dtype=torch.float,
                            device=self.device)
        axis_norm = torch.linalg.vector_norm(axis)
        if axis_norm.item() < 1.0e-8:
            raise ValueError("baseTiltAxis must have non-zero length.")
        self.base_tilt_axis = axis / axis_norm
        self.object_palm_offset = torch.tensor(
            self._object_palm_offset_cfg, dtype=torch.float, device=self.device
        )
        configured_candidates = self._object_palm_offset_candidates_cfg or [
            self._object_palm_offset_cfg
        ]
        self.object_palm_offset_candidates = torch.tensor(
            configured_candidates, dtype=torch.float, device=self.device
        )
        self.object_palm_offset_candidate_index = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            % self.object_palm_offset_candidates.shape[0]
        )

        # Snapshot actual physics transforms rather than actor creation poses:
        # the MJCF root/palm relationship is not safe to hard-code.
        self.hand_initial_pos = self.root_state_tensor[self.hand_indices, 0:3].clone()
        self.hand_initial_quat = self.root_state_tensor[self.hand_indices, 3:7].clone()
        palm_pos = self.rigid_body_states[:, self.palm_body_idx, 0:3].clone()
        palm_quat = self.rigid_body_states[:, self.palm_body_idx, 3:7].clone()

        self.palm_in_root = quat_rotate_inverse(
            self.hand_initial_quat,
            palm_pos - self.hand_initial_pos,
        )
        self.palm_quat_in_root = quat_mul(
            quat_conjugate(self.hand_initial_quat), palm_quat
        )
        self.initial_palm_quat = palm_quat
        self.default_object_in_palm = quat_rotate_inverse(
            palm_quat,
            self.object_init_state[:, 0:3] - palm_pos,
        )

        tilt_angles = torch.full(
            (self.num_envs,), self._base_tilt_angle_rad, dtype=torch.float,
            device=self.device,
        )
        tilt_axes = self.base_tilt_axis.unsqueeze(0).expand(self.num_envs, -1)
        self.tilt_quat = quat_from_angle_axis(tilt_angles, tilt_axes)
        yaw_angles = torch.full(
            (self.num_envs,), self._base_yaw_angle_rad, dtype=torch.float,
            device=self.device,
        )
        yaw_axes = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        yaw_axes[:, self.up_axis_idx] = 1.0
        self.base_yaw_quat = quat_from_angle_axis(yaw_angles, yaw_axes)
        # A world-Z yaw does not change gravity expressed in the palm frame.
        # It is selected by the curriculum to make world-frame MLP inputs
        # continuous with the parent checkpoint.
        self.tilted_hand_quat = quat_mul(
            self.base_yaw_quat, quat_mul(self.tilt_quat, self.hand_initial_quat)
        )

        # Preserve the parent task's zero-tilt meaning of these world-frame
        # vectors, but express them in the palm frame so they rotate with the
        # fixed base.  This makes the initial goal offset, subgoal z bound, and
        # visual goal displacement all consistent with the tilted hand.
        world_up = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        world_up[:, self.up_axis_idx] = 1.0
        self.goal_up_in_palm = quat_rotate_inverse(self.initial_palm_quat, world_up)
        self.initial_goal_offset_in_palm = -self.init_goal_down_offset * self.goal_up_in_palm
        self.goal_visual_displacement_in_palm = quat_rotate_inverse(
            self.initial_palm_quat,
            self.goal_displacement_tensor.unsqueeze(0).expand(self.num_envs, -1),
        )

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        _, _, initial_palm_pos, initial_palm_quat = self._tilted_palm_pose(all_env_ids)
        initial_object_pos = initial_palm_pos + quat_rotate(
            initial_palm_quat,
            self.default_object_in_palm + self._object_palm_offsets(all_env_ids),
        )
        self.goal_center = initial_object_pos + quat_rotate(
            initial_palm_quat, self.initial_goal_offset_in_palm,
        )

        print(
            "ShadowHandFixedTilt: angle={:.6g}deg axis={} yaw={:.6g}deg "
            "objectPalmOffset={}".format(
                math.degrees(self._base_tilt_angle_rad),
                list(self._base_tilt_axis_cfg),
                math.degrees(self._base_yaw_angle_rad),
                list(self._object_palm_offset_cfg),
            )
        )

    def post_physics_step(self):
        super().post_physics_step()
        self._publish_gravity_compensation_metrics()
        self.extras["tilt_object_palm_offset_candidate_index"] = (
            self.object_palm_offset_candidate_index
        )

    def _fall_ref_pos(self):
        """Measure falling relative to the actual palm rather than the goal."""
        return self.rigid_body_states[:, self.palm_body_idx, 0:3]

    def _tilted_palm_pose(self, env_ids):
        root_pos = self.hand_initial_pos[env_ids]
        root_quat = self.tilted_hand_quat[env_ids]
        palm_pos = root_pos + quat_rotate(root_quat, self.palm_in_root[env_ids])
        palm_quat = quat_mul(root_quat, self.palm_quat_in_root[env_ids])
        return root_pos, root_quat, palm_pos, palm_quat

    def _goal_visual_displacement(self, env_ids, palm_quat):
        return quat_rotate(
            palm_quat, self.goal_visual_displacement_in_palm[env_ids]
        )

    def _object_palm_offsets(self, env_ids):
        """Return the fixed probe offset assigned to each environment.

        With the default singleton candidate this is exactly the historical
        ``objectPalmOffset`` behavior.  The evaluator can supply a small
        candidate list to benchmark all offsets in one vectorized rollout.
        """
        return self.object_palm_offset_candidates[
            self.object_palm_offset_candidate_index[env_ids]
        ]

    def reset_target_pose(self, env_ids, apply_reset=False, from_goal_reach=False,
                          base_pos=None):
        """Sample logical goals entirely in the fixed tilted-palm frame."""
        n = len(env_ids)
        _, _, _, palm_quat = self._tilted_palm_pose(env_ids)

        if from_goal_reach:
            axis_local = torch.randn(n, 3, device=self.device)
            axis_local = axis_local / axis_local.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            angle = (torch.rand(n, device=self.device)
                     * (self.goal_rot_delta_max - self.goal_rot_delta_min)
                     + self.goal_rot_delta_min)
            delta_local = quat_from_angle_axis(angle, axis_local)
            current_local_rot = quat_mul(
                quat_conjugate(palm_quat), self.goal_states[env_ids, 3:7].clone()
            )
            new_rot = quat_mul(palm_quat, quat_mul(delta_local, current_local_rot))
        else:
            # Keep the legacy initial orientation distribution at zero tilt,
            # then carry that sampled orientation into the tilted palm frame.
            rand_floats = torch_rand_float(-1.0, 1.0, (n, 4), device=self.device)
            sampled_world_rot = randomize_rotation(
                rand_floats[:, 0], rand_floats[:, 1],
                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
            )
            pen_mask = self.is_pen[env_ids]
            if pen_mask.any():
                sampled_pen_rot = randomize_rotation_pen(
                    rand_floats[:, 0], rand_floats[:, 1], torch.tensor(0.3),
                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
                    self.z_unit_tensor[env_ids],
                )
                sampled_world_rot = torch.where(
                    pen_mask.unsqueeze(-1), sampled_pen_rot, sampled_world_rot
                )
            sampled_local_rot = quat_mul(
                quat_conjugate(self.initial_palm_quat[env_ids]), sampled_world_rot
            )
            new_rot = quat_mul(palm_quat, sampled_local_rot)

        if self.track_goal_pos:
            if from_goal_reach:
                pos_dir_local = torch.randn(n, 3, device=self.device)
                pos_dir_local = pos_dir_local / pos_dir_local.norm(
                    dim=-1, keepdim=True
                ).clamp(min=1e-8)
                pos_mag = (torch.rand(n, device=self.device)
                           * (self.goal_pos_delta_max - self.goal_pos_delta_min)
                           + self.goal_pos_delta_min)
                new_pos = self.goal_states[env_ids, 0:3].clone() + quat_rotate(
                    palm_quat, pos_dir_local * pos_mag.unsqueeze(-1)
                )

                # Clamp in palm-local coordinates around the relocated first
                # goal, rather than around the parent task's untilted object.
                offset_local = quat_rotate_inverse(
                    palm_quat, new_pos - self.goal_center[env_ids]
                )
                up_in_palm = self.goal_up_in_palm[env_ids]
                up_offset = torch.sum(offset_local * up_in_palm, dim=-1, keepdim=True)
                clamped_up_offset = torch.clamp(
                    up_offset, -self.goal_pos_z_max_radius, self.goal_pos_z_max_radius
                )
                offset_local = offset_local + (clamped_up_offset - up_offset) * up_in_palm
                offset_norm = offset_local.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                offset_local = offset_local * torch.clamp(
                    self.goal_pos_max_radius / offset_norm, max=1.0
                )
                new_pos = self.goal_center[env_ids] + quat_rotate(palm_quat, offset_local)
            elif base_pos is not None:
                # reset_idx replaces this provisional position after it has
                # relocated the object into the tilted palm frame.
                new_pos = base_pos.clone()
            else:
                new_pos = self.goal_center[env_ids].clone()
            self.goal_states[env_ids, 0:3] = new_pos
        else:
            self.goal_states[env_ids, 0:3] = self.goal_center[env_ids]

        self.goal_states[env_ids, 3:7] = new_rot
        goal_actor_indices = self.goal_object_indices[env_ids]
        self.root_state_tensor[goal_actor_indices, 0:3] = (
            self.goal_states[env_ids, 0:3]
            + self._goal_visual_displacement(env_ids, palm_quat)
        )
        self.root_state_tensor[goal_actor_indices, 3:7] = new_rot
        self.root_state_tensor[goal_actor_indices, 7:13] = 0.0

        if apply_reset:
            goal_actor_indices = goal_actor_indices.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_actor_indices), len(env_ids),
            )
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        # The parent resets joints, object orientation, and reward buffers.  It
        # also samples the reset-position noise; below we express that sampled
        # noise in the newly tilted palm frame instead of discarding it.
        super().reset_idx(env_ids, goal_env_ids)

        root_pos, root_quat, palm_pos, palm_quat = self._tilted_palm_pose(env_ids)
        object_actor_indices = self.object_indices[env_ids]

        parent_object_pos = self.root_state_tensor[object_actor_indices, 0:3].clone()
        parent_noise_world = parent_object_pos - self.object_init_state[env_ids, 0:3]
        parent_palm_quat = quat_mul(
            self.hand_initial_quat[env_ids], self.palm_quat_in_root[env_ids]
        )
        noise_in_palm = quat_rotate_inverse(parent_palm_quat, parent_noise_world)

        object_local_pos = (
            self.default_object_in_palm[env_ids]
            + self._object_palm_offsets(env_ids)
            + noise_in_palm
        )
        object_pos = palm_pos + quat_rotate(palm_quat, object_local_pos)

        # Parent reset leaves the randomized object orientation intact.  Only
        # relocate the hand and the position component of the object pose.
        hand_actor_indices = self.hand_indices[env_ids]
        self.root_state_tensor[hand_actor_indices, 0:3] = root_pos
        self.root_state_tensor[hand_actor_indices, 3:7] = root_quat
        self.root_state_tensor[hand_actor_indices, 7:13] = 0.0
        self.root_state_tensor[object_actor_indices, 0:3] = object_pos

        # Anchor every later subgoal around this first, relocated goal. The
        # initial goal offset has the legacy value at zero tilt and rotates with
        # the palm at nonzero tilt.
        self.goal_center[env_ids] = object_pos + quat_rotate(
            palm_quat, self.initial_goal_offset_in_palm[env_ids]
        )
        goal_pos = self.goal_center[env_ids]
        self.goal_states[env_ids, 0:3] = goal_pos

        goal_actor_indices = self.goal_object_indices[env_ids]
        self.root_state_tensor[goal_actor_indices, 0:3] = (
            goal_pos + self._goal_visual_displacement(env_ids, palm_quat)
        )
        self.root_state_tensor[goal_actor_indices, 7:13] = 0.0

        actor_indices = torch.unique(torch.cat([
            hand_actor_indices,
            object_actor_indices,
            goal_actor_indices,
        ]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )
