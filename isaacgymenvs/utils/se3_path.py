"""Small batched SE(3) interpolation helpers used by reference-tracking tasks.

All poses use Isaac Gym's ``[x, y, z, qx, qy, qz, qw]`` convention.
"""

import torch


def _quat_conjugate(q):
    out = q.clone()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul(q, r):
    qx, qy, qz, qw = q.unbind(-1)
    rx, ry, rz, rw = r.unbind(-1)
    return torch.stack((
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
        qw * rw - qx * rx - qy * ry - qz * rz,
    ), dim=-1)


def _quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_rotate(q, v):
    qv = torch.cat((v, torch.zeros_like(v[..., :1])), dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., :3]


def _quat_to_matrix(q):
    q = _quat_normalize(q)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def _skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ), dim=-1).reshape(v.shape[:-1] + (3, 3))


def _se3_log(relative_pos, relative_rot):
    """Return the se(3) twist whose exponential is the relative pose."""
    relative_rot = _quat_normalize(relative_rot)
    # Choose the shortest SO(3) logarithm, avoiding q/-q discontinuities.
    relative_rot = torch.where(
        (relative_rot[..., 3:4] < 0), -relative_rot, relative_rot)
    vec = relative_rot[..., :3]
    scalar = relative_rot[..., 3].clamp(-1.0, 1.0)
    sin_half = vec.norm(dim=-1)
    angle = 2.0 * torch.atan2(sin_half, scalar)
    scale = torch.where(sin_half > 1e-6, angle / sin_half, 2.0 * torch.ones_like(angle))
    omega = vec * scale.unsqueeze(-1)

    omega_hat = _skew(omega)
    omega_hat2 = omega_hat @ omega_hat
    theta2 = (omega * omega).sum(dim=-1)
    theta = theta2.sqrt()
    sin_theta = theta.sin()
    cos_theta = theta.cos()
    # V^{-1} = I - 1/2 W + A W^2, with a stable small-angle limit.
    denom = (2.0 * theta * sin_theta).clamp_min(1e-8)
    a_regular = 1.0 / theta2.clamp_min(1e-8) - (1.0 + cos_theta) / denom
    a_small = 1.0 / 12.0 + theta2 / 720.0
    a = torch.where(theta < 1e-4, a_small, a_regular)
    eye = torch.eye(3, dtype=relative_pos.dtype, device=relative_pos.device)
    eye = eye.expand(relative_pos.shape[:-1] + (3, 3))
    v_inv = eye - 0.5 * omega_hat + a.unsqueeze(-1).unsqueeze(-1) * omega_hat2
    v = (v_inv @ relative_pos.unsqueeze(-1)).squeeze(-1)
    return omega, v


def _so3_exp(omega):
    """Quaternion exponential map for a batch of rotation vectors."""
    theta = omega.norm(dim=-1)
    half = 0.5 * theta
    scale = torch.where(theta > 1e-6, torch.sin(half) / theta, 0.5 * torch.ones_like(theta))
    return _quat_normalize(torch.cat((omega * scale.unsqueeze(-1), torch.cos(half).unsqueeze(-1)), dim=-1))


def interpolate_se3(start_pose, end_pose, fractions):
    """Interpolate batched poses along the constant-twist SE(3) geodesic.

    Args:
        start_pose: ``(N, 7)`` XYZW pose.
        end_pose: ``(N, 7)`` XYZW pose.
        fractions: ``(K,)`` values in ``[0, 1]``.
    Returns:
        ``(N, K, 3)`` positions and ``(N, K, 4)`` quaternions.
    """
    p0, q0 = start_pose[..., :3], _quat_normalize(start_pose[..., 3:7])
    p1, q1 = end_pose[..., :3], _quat_normalize(end_pose[..., 3:7])
    q0_inv = _quat_conjugate(q0)
    relative_pos = _quat_rotate(q0_inv, p1 - p0)
    relative_rot = _quat_mul(q0_inv, q1)
    omega, v = _se3_log(relative_pos, relative_rot)

    alpha = fractions.to(dtype=start_pose.dtype, device=start_pose.device)
    alpha = alpha.reshape((1, -1) + (1,) * (start_pose.ndim - 1))
    omega_a = omega.unsqueeze(1) * alpha
    v_a = v.unsqueeze(1) * alpha
    theta = omega_a.norm(dim=-1)
    theta2 = theta * theta
    omega_hat = _skew(omega_a)
    omega_hat2 = omega_hat @ omega_hat
    eye = torch.eye(3, dtype=start_pose.dtype, device=start_pose.device)
    eye = eye.reshape((1, 1, 3, 3)).expand(omega_a.shape[:-1] + (3, 3))
    a = torch.where(theta < 1e-4,
                    0.5 - theta2 / 24.0,
                    (1.0 - theta.cos()) / theta2.clamp_min(1e-8))
    b = torch.where(theta < 1e-4,
                    1.0 / 6.0 - theta2 / 120.0,
                    (theta - theta.sin()) / theta.clamp_min(1e-8).pow(3))
    v_matrix = eye + a.unsqueeze(-1).unsqueeze(-1) * omega_hat + b.unsqueeze(-1).unsqueeze(-1) * omega_hat2
    local_pos = (v_matrix @ v_a.unsqueeze(-1)).squeeze(-1)
    local_rot = _so3_exp(omega_a)
    pos = p0.unsqueeze(1) + _quat_rotate(
        q0.unsqueeze(1).expand_as(local_rot), local_pos)
    rot = _quat_mul(q0.unsqueeze(1).expand_as(local_rot), local_rot)
    return pos, _quat_normalize(rot)


def closest_se3_path_distance(object_pose, start_pose, end_pose, fractions,
                              position_sigma, rotation_sigma):
    """Minimum normalized pose error to sampled points on an SE(3) path."""
    path_pos, path_rot = interpolate_se3(start_pose, end_pose, fractions)
    pos_err = (object_pose[:, None, :3] - path_pos).norm(dim=-1)
    object_rot = _quat_normalize(object_pose[:, None, 3:7])
    path_rot = _quat_normalize(path_rot)
    dot = (object_rot * path_rot).sum(dim=-1).abs().clamp(0.0, 1.0)
    rot_err = 2.0 * torch.acos(dot)
    normalized = (pos_err / max(float(position_sigma), 1e-6)).square() + \
        (rot_err / max(float(rotation_sigma), 1e-6)).square()
    best = normalized.argmin(dim=-1)
    best_pos = pos_err.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    best_rot = rot_err.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    return normalized.gather(-1, best.unsqueeze(-1)).squeeze(-1), best_pos, best_rot
