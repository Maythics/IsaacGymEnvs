"""SE(3)-equivariant policy wrapper for ShadowHand-style observations.

The inner actor-critic MLP is unchanged in shape and weight layout, so checkpoints
from the baseline ShadowHand/ShadowHandTilted training (211-dim input, 512-512-256-128
hidden, mu 20 / value 1) load cleanly. A math-only preprocess re-expresses all
world-frame quantities in the 211-dim observation vector in the hand-base frame,
using the 7-dim hand root pose appended by the task (see ``appendHandBasePose``).

Expected input: ``obs.shape == (N, 218)``
    obs[:, :211]    - original 211-dim full_state observation (world frame)
    obs[:, 211:214] - hand root position (world frame)
    obs[:, 214:218] - hand root quaternion (world frame, xyzw)

After preprocessing: ``(N, 211)``, consumed verbatim by the unchanged actor_mlp.
"""

import torch

from rl_games.algos_torch import models
from rl_games.algos_torch import network_builder

from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)


MLP_INPUT_DIM = 211
HAND_POSE_DIM = 7
EXT_OBS_DIM = MLP_INPUT_DIM + HAND_POSE_DIM  # 218


@torch.jit.script
def se3_preprocess(obs: torch.Tensor) -> torch.Tensor:
    """Transform world-frame slices of a 218-dim observation into the hand-base frame.

    Returns a new (N, 211) tensor. The original input is not modified.
    quat_rotate_inverse uses ``.view()`` internally, so vector/position slices must
    be passed as contiguous tensors.
    """
    hand_pos = obs[:, 211:214].contiguous()
    hand_quat = obs[:, 214:218].contiguous()
    hand_quat_inv = quat_conjugate(hand_quat)

    out = obs[:, :211].contiguous()  # writable contiguous copy

    # Object pos[72:75], quat[75:79]
    out[:, 72:75] = quat_rotate_inverse(hand_quat, (obs[:, 72:75] - hand_pos).contiguous())
    out[:, 75:79] = quat_mul(hand_quat_inv, obs[:, 75:79])

    # Object linvel[79:82], angvel[82:85] (pre-scaled; rotation preserves scale)
    out[:, 79:82] = quat_rotate_inverse(hand_quat, obs[:, 79:82].contiguous())
    out[:, 82:85] = quat_rotate_inverse(hand_quat, obs[:, 82:85].contiguous())

    # Goal pos[85:88], quat[88:92]
    out[:, 85:88] = quat_rotate_inverse(hand_quat, (obs[:, 85:88] - hand_pos).contiguous())
    out[:, 88:92] = quat_mul(hand_quat_inv, obs[:, 88:92])

    # [92:96] is quat_mul(obj_rot_w, conj(goal_rot_w)) — a relative rotation expressed with
    # world-frame quaternions. Under a global SE(3) rotation Q it becomes Q · q_rel · conj(Q),
    # so express it in the hand frame via conjugation: conj(q_hand) · q_rel · q_hand.
    out[:, 92:96] = quat_mul(quat_mul(hand_quat_inv, obs[:, 92:96]), hand_quat)

    # Fingertips: 5 tips x (pos3 + quat4 + linvel3 + angvel3) starting at 96.
    for i in range(5):
        base = 96 + i * 13
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

    # [161:191] fingertip force-torque sensors are in sensor-local frames — leave.
    # [191:211] last actions — hand-local control commands — leave.

    return out


class SE3Builder(network_builder.A2CBuilder):
    """Drop-in replacement for A2CBuilder.

    The model constructs this with ``input_shape=(211,)`` (see SE3ModelA2CContinuousLogStd.build),
    so the resulting A2CBuilder.Network has the same parameter shapes as the baseline
    and loads old checkpoints cleanly. All SE(3) preprocessing happens at the model
    level, not inside the network.
    """

    def build(self, name, **kwargs):
        net = network_builder.A2CBuilder.Network(self.params, **kwargs)
        return net


class SE3ModelA2CContinuousLogStd(models.ModelA2CContinuousLogStd):
    """Thin wrapper around ModelA2CContinuousLogStd that inserts the SE(3) preprocess.

    The outer ``Network.forward`` slices the 218-dim input down to 211 by expressing
    all world-frame quantities in the hand-base frame before normalization and the
    inner MLP. The running_mean_std stays 211-dim so old checkpoints' normalization
    statistics load unchanged.
    """

    def build(self, config):
        config = dict(config)
        config['input_shape'] = (MLP_INPUT_DIM,)
        return super().build(config)

    class Network(models.ModelA2CContinuousLogStd.Network):
        def forward(self, input_dict):
            input_dict['obs'] = se3_preprocess(input_dict['obs'])
            return super().forward(input_dict)
