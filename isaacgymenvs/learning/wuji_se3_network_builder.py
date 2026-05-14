"""SE(3)-equivariant policy wrapper for WujiHand-style observations.

Wuji has 22 DOFs (2 wrist + 20 fingers) instead of ShadowHand's 24, so the base
observation is 207-dim (vs ShadowHand's 211). With ``appendHandBasePose: True``,
the env passes 214 dims; this preprocess rotates all world-frame slices into the
hand-base frame and emits a 207-dim tensor for the actor MLP.

Expected input: ``obs.shape == (N, 214)``
    obs[:, :207]    - original 207-dim full_state observation (world frame)
    obs[:, 207:210] - hand root position (world frame)
    obs[:, 210:214] - hand root quaternion (world frame, xyzw)

After preprocessing: ``(N, 207)``, consumed verbatim by the unchanged actor_mlp.

Obs layout (Wuji full_state, 207 dims):
  [0:22]    DOF positions       (hand-local — leave)
  [22:44]   DOF velocities      (hand-local — leave)
  [44:66]   DOF forces          (hand-local — leave)
  [66:69]   Object pos          (world → hand)
  [69:73]   Object quat         (world → hand)
  [73:76]   Object linvel       (world → hand)
  [76:79]   Object angvel       (world → hand)
  [79:82]   Goal pos            (world → hand)
  [82:86]   Goal quat           (world → hand)
  [86:90]   quat_mul(obj, conj(goal)) — relative rotation, world-frame
  [90:155]  5 fingertips × (pos3 + quat4 + linvel3 + angvel3)
  [155:185] 5 force-torque sensors (sensor-local — leave)
  [185:207] Last 22 actions     (hand-local — leave)
"""

import torch

from rl_games.algos_torch import models
from rl_games.algos_torch import network_builder

from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)


MLP_INPUT_DIM = 207
HAND_POSE_DIM = 7
EXT_OBS_DIM = MLP_INPUT_DIM + HAND_POSE_DIM  # 214


@torch.jit.script
def wuji_se3_preprocess(obs: torch.Tensor) -> torch.Tensor:
    hand_pos = obs[:, 207:210].contiguous()
    hand_quat = obs[:, 210:214].contiguous()
    hand_quat_inv = quat_conjugate(hand_quat)

    out = obs[:, :207].contiguous()

    # Object pos[66:69], quat[69:73]
    out[:, 66:69] = quat_rotate_inverse(hand_quat, (obs[:, 66:69] - hand_pos).contiguous())
    out[:, 69:73] = quat_mul(hand_quat_inv, obs[:, 69:73])

    # Object linvel[73:76], angvel[76:79]
    out[:, 73:76] = quat_rotate_inverse(hand_quat, obs[:, 73:76].contiguous())
    out[:, 76:79] = quat_rotate_inverse(hand_quat, obs[:, 76:79].contiguous())

    # Goal pos[79:82], quat[82:86]
    out[:, 79:82] = quat_rotate_inverse(hand_quat, (obs[:, 79:82] - hand_pos).contiguous())
    out[:, 82:86] = quat_mul(hand_quat_inv, obs[:, 82:86])

    # [86:90] is quat_mul(obj_rot_w, conj(goal_rot_w)): conjugate into hand frame
    out[:, 86:90] = quat_mul(quat_mul(hand_quat_inv, obs[:, 86:90]), hand_quat)

    # Fingertips: 5 tips x (pos3 + quat4 + linvel3 + angvel3) starting at 90.
    for i in range(5):
        base = 90 + i * 13
        out[:, base:base + 3] = quat_rotate_inverse(
            hand_quat, (obs[:, base:base + 3] - hand_pos).contiguous()
        )
        out[:, base + 3:base + 7] = quat_mul(hand_quat_inv, obs[:, base + 3:base + 7])
        out[:, base + 7:base + 10] = quat_rotate_inverse(
            hand_quat, obs[:, base + 7:base + 10].contiguous()
        )
        out[:, base + 10:base + 13] = quat_rotate_inverse(
            hand_quat, obs[:, base + 10:base + 13].contiguous()
        )

    # [155:185] fingertip force-torque sensors are in sensor-local frames — leave.
    # [185:207] last actions — hand-local control commands — leave.

    return out


class WujiSE3Builder(network_builder.A2CBuilder):
    def build(self, name, **kwargs):
        net = network_builder.A2CBuilder.Network(self.params, **kwargs)
        return net


class WujiSE3ModelA2CContinuousLogStd(models.ModelA2CContinuousLogStd):
    def build(self, config):
        config = dict(config)
        config['input_shape'] = (MLP_INPUT_DIM,)
        return super().build(config)

    class Network(models.ModelA2CContinuousLogStd.Network):
        def forward(self, input_dict):
            input_dict['obs'] = wuji_se3_preprocess(input_dict['obs'])
            return super().forward(input_dict)
