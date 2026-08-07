import unittest

import torch

from isaacgymenvs.scripts import evaluate_tilted_policy as evaluator


class TaskUnwrapTests(unittest.TestCase):
    def test_uses_direct_task_from_newer_rlgames(self):
        class Task:
            num_envs = 128
            device = "cuda:0"
            object_type_pool = ["block"]

        task = Task()
        self.assertIs(task, evaluator._task_from_vec_env(task))

    def test_unwraps_legacy_rlgames_environment(self):
        class Task:
            num_envs = 128
            device = "cuda:0"
            object_type_pool = ["block"]

        class LegacyRLGPUEnv:
            def __init__(self, env):
                self.env = env

        task = Task()
        self.assertIs(task, evaluator._task_from_vec_env(LegacyRLGPUEnv(task)))


class PlayerBatchTests(unittest.TestCase):
    def test_initializes_vectorized_observation_batch_before_first_action(self):
        class Player:
            is_rnn = False

            def __init__(self):
                self.calls = []

            def get_batch_size(self, observation, fallback):
                self.calls.append((tuple(observation.shape), fallback))
                return observation.shape[0]

        player = Player()
        evaluator._initialize_player_batch(player, {"obs": torch.zeros(128, 209)})
        self.assertEqual([((128, 209), 1)], player.calls)

    def test_initializes_rnn_when_needed(self):
        class Player:
            is_rnn = True

            def __init__(self):
                self.initialized = False

            def get_batch_size(self, observation, fallback):
                return observation.shape[0]

            def init_rnn(self):
                self.initialized = True

        player = Player()
        evaluator._initialize_player_batch(player, {"obs": torch.zeros(2, 209)})
        self.assertTrue(player.initialized)


if __name__ == "__main__":
    unittest.main()
