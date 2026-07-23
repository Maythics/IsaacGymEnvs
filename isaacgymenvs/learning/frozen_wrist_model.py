"""Continuous A2C model that keeps legacy wrist outputs but does not explore them.

The WujiHand and XHandHand checkpoints were trained with two wrist action
entries. Removing those entries would change the actor head shape and prevent a
direct checkpoint resume. This wrapper keeps the original network and parameter
shapes, while making the first two action dimensions deterministic zeros and
excluding them from the policy likelihood and entropy terms.
"""

import numpy as np
import torch

from rl_games.algos_torch import models


WRIST_DIMS = 2


class FrozenWristModelA2CContinuousLogStd(models.ModelA2CContinuousLogStd):
    """Drop-in replacement for ``continuous_a2c_logstd``.

    The inner actor-critic network is unchanged, so its state-dict keys and
    tensor shapes remain compatible with old WujiHand/XHandHand checkpoints.
    Only the distribution-facing values are masked at runtime.
    """

    class Network(models.ModelA2CContinuousLogStd.Network):
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)

            # Keep full-width tensors for checkpoint-compatible actor outputs,
            # but make wrist distribution parameters constant and deterministic.
            # The constant sigma is only exposed for diagnostics; wrist terms
            # are omitted from likelihood and entropy below.
            masked_mu = mu.clone()
            masked_sigma = sigma.clone()
            masked_mu[..., :WRIST_DIMS] = 0.0
            masked_sigma[..., :WRIST_DIMS] = 1.0

            active_mu = mu[..., WRIST_DIMS:]
            active_logstd = logstd[..., WRIST_DIMS:]
            active_sigma = sigma[..., WRIST_DIMS:]
            distr = torch.distributions.Normal(active_mu, active_sigma,
                                               validate_args=False)

            if is_train:
                # The environment stores the full legacy action tensor. Only
                # finger entries participate in PPO's old/new likelihood.
                active_prev_actions = prev_actions[..., WRIST_DIMS:]
                prev_neglogp = self.neglogp(
                    active_prev_actions, active_mu, active_sigma, active_logstd)
                entropy = distr.entropy().sum(dim=-1)
                return {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': masked_mu,
                    'sigmas': masked_sigma,
                }

            active_action = distr.sample()
            selected_action = torch.zeros_like(mu)
            selected_action[..., WRIST_DIMS:] = active_action
            neglogp = self.neglogp(
                active_action, active_mu, active_sigma, active_logstd)
            return {
                'neglogpacs': torch.squeeze(neglogp),
                'values': self.denorm_value(value),
                'actions': selected_action,
                'rnn_states': states,
                'mus': masked_mu,
                'sigmas': masked_sigma,
            }

        @staticmethod
        def neglogp(x, mean, std, logstd):
            return 0.5 * (((x - mean) / std) ** 2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
