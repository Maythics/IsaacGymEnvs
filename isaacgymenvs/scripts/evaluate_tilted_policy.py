#!/usr/bin/env python3
"""Evaluate a fixed-tilt checkpoint with retained-success semantics.

This uses the normal registered task, including its ordinary viewer path and
object-gravity configuration.  The JSON result is consumed by the automated
curriculum but the script is also useful as a direct checkpoint smoke test.
"""

from __future__ import print_function

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np


def _vector(value):
    values = [float(item) for item in value.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("Shadowhand18Tilted", "WujiHandFixedTilt"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--angle-deg", type=float, required=True)
    parser.add_argument("--axis", type=_vector, required=True)
    parser.add_argument("--base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--offset", type=_vector, required=True)
    parser.add_argument("--gravity-hold-seconds", type=float, default=0.2)
    parser.add_argument("--gravity-ramp-seconds", type=float, default=0.1)
    parser.add_argument("--object-type", default="block")
    parser.add_argument(
        "--object-type-pool",
        default="",
        help="Comma-separated pool; empty keeps the single --object-type",
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.num_envs <= 0 or args.episode_length <= 0:
        parser.error("episodes, num-envs, and episode-length must be positive")
    if args.gravity_hold_seconds < 0 or args.gravity_ramp_seconds < 0:
        parser.error("gravity hold/ramp durations must be non-negative")
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist: {}".format(args.checkpoint))
    return args


def _format_vector(values):
    return "[{}]".format(",".join("{:.9g}".format(float(v)) for v in values))


def _atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _task_from_vec_env(environment):
    """Return the underlying task for both supported RL-Games wrappers.

    Older RL-Games releases expose ``player.env`` as an RLGPUEnv wrapper whose
    ``.env`` attribute is the Isaac task.  In newer releases (including the
    version used by the curriculum launcher), ``player.env`` is already the
    Isaac task.  Looking for the task interface instead of blindly accessing
    ``.env`` keeps physical certification independent of that version detail.
    """
    current = environment
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        if all(hasattr(current, name) for name in ("num_envs", "device", "object_type_pool")):
            return current
        nested = getattr(current, "env", None)
        if nested is None:
            break
        current = nested
    raise RuntimeError(
        "could not find the Isaac task under the RL-Games environment "
        "(expected num_envs, device, and object_type_pool)"
    )


def _initialize_player_batch(player, obs_dict):
    """Tell RL-Games that the evaluator observation already has a batch axis.

    ``CommonPlayer.run`` normally calls ``get_batch_size`` immediately after
    reset.  The retained-success loop owns its episode accounting, so it must
    reproduce that small but essential step.  Without it RL-Games regards a
    ``[num_envs, obs_dim]`` observation as one unbatched observation and
    flattens it to ``[1, num_envs * obs_dim]`` before the first MLP layer.
    """
    if not isinstance(obs_dict, dict) or "obs" not in obs_dict:
        raise RuntimeError("RL-Games reset did not return an observation dictionary with key 'obs'")
    player.get_batch_size(obs_dict["obs"], 1)
    if player.is_rnn:
        player.init_rnn()


def main(argv=None):
    args = parse_args(argv)

    # Isaac Gym must be imported before torch.
    import isaacgym  # noqa: F401
    import torch
    import isaacgymenvs
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from isaacgymenvs.learning import (
        amp_continuous,
        amp_models,
        amp_network_builder,
        amp_players,
        frozen_wrist_model,
        se3_network_builder,
        wuji_se3_network_builder,
    )
    from isaacgymenvs.learning.common_player import CommonPlayer
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class RetainedSuccessPlayer(CommonPlayer):
        def run(self):
            obs_dict = self.env_reset(self.env)
            _initialize_player_batch(self, obs_dict)
            task = _task_from_vec_env(self.env)
            completed = 0
            retained = 0
            dropped = 0
            timed_out = 0
            timeout_without_success = 0
            goal_hit = 0
            reward_sum = 0.0
            step_sum = 0
            object_names = list(task.object_type_pool)
            per_object = {
                name: {"episodes": 0, "retained_successes": 0}
                for name in object_names
            }
            episode_rewards = torch.zeros(task.num_envs, device=task.device)
            episode_steps = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)

            while completed < args.episodes:
                action = self.get_action(obs_dict, is_determenistic=True)
                obs_dict, reward, done, info = self.env_step(self.env, action)
                episode_rewards += reward.to(task.device)
                episode_steps += 1
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() == 0:
                    continue
                remaining = args.episodes - completed
                done_ids = done_ids[:remaining]
                n = int(done_ids.numel())
                if n == 0:
                    break

                retained_mask = info["tilt_retained_success"][done_ids].bool()
                dropped_mask = info["tilt_dropped"][done_ids].bool()
                timeout_mask = info["tilt_timed_out"][done_ids].bool()
                no_success_mask = info["tilt_timeout_without_success"][done_ids].bool()
                goal_mask = info["tilt_goal_hit"][done_ids].bool()
                type_indices = info["tilt_object_type_index"][done_ids].long()
                retained += int(retained_mask.sum().item())
                dropped += int(dropped_mask.sum().item())
                timed_out += int(timeout_mask.sum().item())
                timeout_without_success += int(no_success_mask.sum().item())
                goal_hit += int(goal_mask.sum().item())
                for local_index in range(n):
                    name = object_names[int(type_indices[local_index].item())]
                    per_object[name]["episodes"] += 1
                    per_object[name]["retained_successes"] += int(
                        retained_mask[local_index].item()
                    )
                reward_sum += float(episode_rewards[done_ids].sum().item())
                step_sum += int(episode_steps[done_ids].sum().item())
                episode_rewards[done_ids] = 0.0
                episode_steps[done_ids] = 0
                completed += n

            for values in per_object.values():
                values["retained_success_rate"] = (
                    values["retained_successes"] / float(values["episodes"])
                    if values["episodes"] else 0.0
                )
            minimum_object_success_rate = min(
                values["retained_success_rate"] for values in per_object.values()
            )
            result = {
                "version": 1,
                "task": args.task,
                "checkpoint": str(args.checkpoint.resolve()),
                "episodes": completed,
                "retained_successes": retained,
                "retained_success_rate": retained / float(completed),
                "goal_hit_episodes": goal_hit,
                "goal_hit_rate": goal_hit / float(completed),
                "drops": dropped,
                "drop_rate": dropped / float(completed),
                "timeouts": timed_out,
                "timeout_rate": timed_out / float(completed),
                "timeout_without_success": timeout_without_success,
                "timeout_without_success_rate": timeout_without_success / float(completed),
                "mean_episode_reward": reward_sum / float(completed),
                "mean_episode_steps": step_sum / float(completed),
                "seed": args.seed,
                "angle_deg": args.angle_deg,
                "axis": args.axis,
                "base_yaw_deg": args.base_yaw_deg,
                "offset": args.offset,
                "object_type": args.object_type,
                "object_type_pool": object_names,
                "per_object": per_object,
                "minimum_object_success_rate": minimum_object_success_rate,
                "object_gravity_compensation_seconds": args.gravity_hold_seconds,
                "object_gravity_ramp_seconds": args.gravity_ramp_seconds,
            }
            _atomic_json(args.result, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return result

    package_dir = Path(isaacgymenvs.__file__).resolve().parent
    GlobalHydra.instance().clear()
    object_type_pool = [
        value.strip() for value in args.object_type_pool.split(",") if value.strip()
    ]
    overrides = [
        "task={}".format(args.task),
        "num_envs={}".format(args.num_envs),
        "task.env.numEnvs={}".format(args.num_envs),
        "task.env.episodeLength={}".format(args.episode_length),
        "task.env.baseTiltAngleDeg={:.9g}".format(args.angle_deg),
        "task.env.baseTiltAxis={}".format(_format_vector(args.axis)),
        "task.env.baseYawDeg={:.9g}".format(args.base_yaw_deg),
        "task.env.objectPalmOffset={}".format(_format_vector(args.offset)),
        "task.env.objectGravityCompensationSeconds={:.9g}".format(args.gravity_hold_seconds),
        "task.env.objectGravityRampSeconds={:.9g}".format(args.gravity_ramp_seconds),
        "task.env.objectType={}".format(args.object_type),
        "task.env.objectTypePool=[{}]".format(",".join(object_type_pool)),
        "sim_device={}".format(args.device),
        "rl_device={}".format(args.device),
        "graphics_device_id=0",
        "headless={}".format(str(args.headless).lower()),
        "test=True",
        "multi_gpu=False",
        "capture_video=False",
        "force_render=False",
        "seed={}".format(args.seed),
    ]
    with initialize_config_dir(config_dir=str(package_dir / "cfg"), version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)
    cfg.seed = set_seed(cfg.seed, torch_deterministic=True)
    set_np_formatting()

    def create_env(**kwargs):
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.task.env.numEnvs,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
            cfg.headless, cfg.multi_gpu, cfg.capture_video,
            cfg.force_render, cfg, **kwargs
        )

    env_configurations.register(
        "rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: create_env(**kwargs)}
    )
    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    runner = Runner(RLGPUAlgoObserver())
    runner.algo_factory.register_builder("amp_continuous", lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
    runner.player_factory.register_builder("amp_continuous", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
    runner.player_factory.register_builder("a2c_continuous", lambda **kwargs: RetainedSuccessPlayer(**kwargs))
    model_builder.register_model("continuous_amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    model_builder.register_network("amp", lambda **kwargs: amp_network_builder.AMPBuilder())
    model_builder.register_model(
        "continuous_a2c_logstd_se3",
        lambda network, **kwargs: se3_network_builder.SE3ModelA2CContinuousLogStd(network),
    )
    model_builder.register_network("se3_actor_critic", lambda **kwargs: se3_network_builder.SE3Builder())
    model_builder.register_model(
        "continuous_a2c_logstd_se3_wuji",
        lambda network, **kwargs: wuji_se3_network_builder.WujiSE3ModelA2CContinuousLogStd(network),
    )
    model_builder.register_network("se3_actor_critic_wuji", lambda **kwargs: wuji_se3_network_builder.WujiSE3Builder())
    model_builder.register_model(
        "continuous_a2c_logstd_frozen_wrist",
        lambda network, **kwargs: frozen_wrist_model.FrozenWristModelA2CContinuousLogStd(network),
    )
    config = omegaconf_to_dict(cfg.train)
    config["params"]["config"]["device"] = cfg.rl_device
    config["params"]["config"]["full_experiment_name"] = "tilt_evaluation"
    runner.load(config)
    runner.reset()
    runner.run({"train": False, "play": True, "checkpoint": str(args.checkpoint.resolve()), "sigma": None})
    return 0


if __name__ == "__main__":
    sys.exit(main())
