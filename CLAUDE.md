# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

IsaacGymEnvs (v1.5.1) is NVIDIA's library of RL benchmark environments for Isaac Gym. It provides 20+ GPU-accelerated parallel environments for locomotion, manipulation, and assembly tasks. Training uses the [rl_games](https://github.com/Denys88/rl_games) library with PPO/SAC, configured via [Hydra](https://hydra.cc/).

## Commands

### Installation
```bash
# Isaac Gym Preview 4 must be installed first (from NVIDIA website)
pip install -e .
```

### Training
```bash
# Run from isaacgymenvs/ directory
cd isaacgymenvs
python train.py task=Cartpole
python train.py task=Ant headless=True
python train.py task=ShadowHand num_envs=8192 headless=True

# Inference with a checkpoint
python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64

# Multi-GPU
python train.py task=Ant multi_gpu=True

# With W&B logging
python train.py task=Ant wandb_activate=True wandb_project=my_project
```

Key CLI overrides (Hydra syntax): `task`, `num_envs`, `seed`, `sim_device`, `rl_device`, `graphics_device_id`, `headless`, `checkpoint`, `test`, `max_iterations`, `experiment`, `capture_video`.

Training outputs go to `runs/TASKNAME/`: checkpoints in `nn/`, configs in `config.yaml`, tensorboard in `summaries/`.

### Linting
```bash
pre-commit run --all-files  # runs codespell (spell checker)
```

## Architecture

### Configuration System (Hydra)

Three-level YAML hierarchy in `isaacgymenvs/cfg/`:
- `config.yaml` — top-level defaults and device settings
- `task/TASKNAME.yaml` — environment parameters (`env`, `sim`, `task` sections)
- `train/TASKNAMEPPO.yaml` — rl_games algorithm config

All values are overridable from CLI. Config is interpolated: e.g., `use_gpu_pipeline: ${eq:${...pipeline},"gpu"}`.

### Environment Base Classes (`tasks/base/vec_task.py`)

All environments inherit from `VecTask` which handles:
- Isaac Gym sim creation, viewer, and GPU pipeline setup
- PyTorch tensor buffers: `obs_buf`, `rew_buf`, `reset_buf`, `progress_buf`, `extras`
- Domain randomization application
- The physics step loop:
  ```
  pre_physics_step(actions) → gym.simulate() → post_physics_step()
  ```

Each environment subclasses `VecTask` and implements:
- `create_sim()` — load assets, create actors, set up environments
- `pre_physics_step(actions)` — apply actions to DOFs/forces
- `post_physics_step()` — compute observations and rewards, call `reset_idx()`
- `reset_idx(env_ids)` — reset specific environments to initial state

### Adding a New Environment

1. Create `isaacgymenvs/tasks/my_task.py` subclassing `VecTask`
2. Set `self.cfg["env"]["numObservations"]` and `numActions` before calling `super().__init__()`
3. Create `isaacgymenvs/cfg/task/MyTask.yaml` with `env`, `sim`, and `task` sections
4. Create `isaacgymenvs/cfg/train/MyTaskPPO.yaml` (copy from a similar task)
5. Register in `isaacgymenvs/tasks/__init__.py`: add import and entry to `isaacgym_task_map`

### RL Integration (`utils/rlgames_utils.py`)

`RLGPUEnv` wraps `VecTask` for rl_games. `get_rlgames_env_creator()` in `train.py` registers environments with rl_games' factory. Custom AMP (Adversarial Motion Priors) agents live in `learning/`.

### Programmatic API

```python
import isaacgym
import isaacgymenvs
envs = isaacgymenvs.make(seed=0, task="Ant", num_envs=2000,
                          sim_device="cuda:0", rl_device="cuda:0")
obs = envs.reset()
obs, rew, done, info = envs.step(actions)
```

### Key Utilities

- `utils/torch_jit_utils.py` — TorchScript quaternion/rotation math used throughout tasks
- `utils/dr_utils.py` — Domain randomization property getters/setters
- `learning/amp_*.py` — Adversarial Motion Priors algorithm implementation
- `pbt/` — Population-Based Training support

### Assets

Robot descriptions (URDF, MJCF) and 3D models (GLB) live in `assets/`. Factory and IndustReal tasks have their own subdirectories.
