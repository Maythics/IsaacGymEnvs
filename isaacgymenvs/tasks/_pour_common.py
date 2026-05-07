"""Shared tensor helpers for ShadowPour / XHandPour.

These functions are pure tensor ops with no Isaac Gym dependencies, so they
can be JIT-compiled and reused across both task variants.
"""
import math

import torch

from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)


def build_subgoal_quats(
    num_subgoals: int,
    final_pitch_rad: float,
    pitch_axis,           # (3,) tensor
    device,
    dtype=torch.float,
):
    """Return (num_subgoals, 4) tensor of sub-goal orientations.

    Sub-goal 0 is identity (upright). Sub-goal N-1 is `final_pitch_rad`
    rotation around `pitch_axis`. Intermediate sub-goals are linearly spaced.
    """
    angles = torch.linspace(0.0, final_pitch_rad, num_subgoals,
                            device=device, dtype=dtype)
    axis = pitch_axis.to(device=device, dtype=dtype).unsqueeze(0).expand(num_subgoals, 3).contiguous()
    return quat_from_angle_axis(angles, axis)  # (num_subgoals, 4)


@torch.jit.script
def compute_pour_reward(
    reset_buf, progress_buf, phase, is_grasped, subgoal_hold_buf,
    successes, consecutive_successes,
    max_episode_length: float,
    object_pos, object_rot, object_linvel, object_angvel,
    subgoal_quats_per_env,    # (num_envs, num_subgoals, 4)
    palm_pos,
    fingertip_pos,            # (num_envs, num_fingertips, 3)
    actions, dof_vel,
    grasp_proximity_scale: float,
    grasp_contact_scale: float,
    grasp_bonus: float,
    grasp_proximity_threshold: float,
    grasp_linvel_threshold: float,
    grasp_angvel_threshold: float,
    subgoal_rot_scale: float,
    subgoal_pos_scale: float,
    subgoal_advance_bonus: float,
    subgoal_success_tolerance: float,
    subgoal_hold_frames: int,
    final_success_bonus: float,
    rot_eps: float,
    action_penalty_scale: float,
    dof_vel_penalty_scale: float,
    obj_linvel_penalty_scale: float,
    obj_angvel_penalty_scale: float,
    obj_linvel_limit: float,
    obj_angvel_limit: float,
    dof_vel_limit: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    num_envs = phase.shape[0]
    num_subgoals = subgoal_quats_per_env.shape[1]

    palm_to_obj = torch.norm(object_pos - palm_pos, p=2, dim=-1)
    fingertip_d = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1).mean(dim=-1)
    obj_linspeed = torch.norm(object_linvel, p=2, dim=-1)
    obj_angspeed = torch.norm(object_angvel, p=2, dim=-1)

    env_arange = torch.arange(num_envs, device=phase.device)
    cur_subgoal_quat = subgoal_quats_per_env[env_arange, phase]
    quat_diff = quat_mul(object_rot, quat_conjugate(cur_subgoal_quat))
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    near_palm = palm_to_obj < grasp_proximity_threshold
    obj_low_speed = (obj_linspeed < grasp_linvel_threshold) & (obj_angspeed < grasp_angvel_threshold)
    grasp_now = near_palm & obj_low_speed
    is_grasped_new = is_grasped | grasp_now
    just_grasped = is_grasped_new & (~is_grasped)

    subgoal_reached = torch.abs(rot_dist) < subgoal_success_tolerance
    hold_active = is_grasped_new & subgoal_reached
    new_hold = torch.where(
        hold_active,
        subgoal_hold_buf + 1,
        torch.zeros_like(subgoal_hold_buf),
    )
    not_final = phase < (num_subgoals - 1)
    advance = (new_hold >= subgoal_hold_frames) & not_final & is_grasped_new
    new_phase = torch.where(advance, phase + 1, phase)
    new_hold = torch.where(advance, torch.zeros_like(new_hold), new_hold)

    final_success = is_grasped_new & (~not_final) & subgoal_reached & (subgoal_hold_buf == 0) & hold_active

    grasp_focus_rew = palm_to_obj * grasp_proximity_scale + fingertip_d * grasp_contact_scale

    orient_rew = subgoal_rot_scale / (torch.abs(rot_dist) + rot_eps)
    pos_retention = palm_to_obj * subgoal_pos_scale
    post_grasp_rew = orient_rew + pos_retention
    post_grasp_rew = post_grasp_rew + torch.where(
        just_grasped, torch.full_like(post_grasp_rew, grasp_bonus),
        torch.zeros_like(post_grasp_rew))
    post_grasp_rew = post_grasp_rew + torch.where(
        advance, torch.full_like(post_grasp_rew, subgoal_advance_bonus),
        torch.zeros_like(post_grasp_rew))
    post_grasp_rew = post_grasp_rew + torch.where(
        final_success, torch.full_like(post_grasp_rew, final_success_bonus),
        torch.zeros_like(post_grasp_rew))

    reward = torch.where(is_grasped_new, post_grasp_rew, grasp_focus_rew)

    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale
    dof_vel_penalty = torch.sum(
        torch.clamp(torch.exp(torch.abs(dof_vel) - dof_vel_limit) - 1.0,
                    min=0.0, max=100.0), dim=-1) * dof_vel_penalty_scale
    obj_linvel_penalty = torch.clamp(
        torch.exp(obj_linspeed - obj_linvel_limit) - 1.0,
        min=0.0, max=500.0) * obj_linvel_penalty_scale
    obj_angvel_penalty = torch.clamp(
        torch.exp(obj_angspeed - obj_angvel_limit) - 1.0,
        min=0.0, max=1000.0) * obj_angvel_penalty_scale

    reward = reward + action_penalty + dof_vel_penalty + obj_linvel_penalty + obj_angvel_penalty

    fell = palm_to_obj > fall_dist
    reward = torch.where(fell, reward + fall_penalty, reward)

    resets = torch.where(fell, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(progress_buf >= max_episode_length - 1,
                         torch.ones_like(resets), resets)

    new_successes = successes + final_success.float()

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(new_successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (reward, resets, new_phase, is_grasped_new, new_hold,
            new_successes, cons_successes, rot_dist, palm_to_obj)
