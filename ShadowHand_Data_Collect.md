# ShadowHand Data Collection — Usage Manual

## Overview

This guide covers training and data collection with the ShadowHand dexterous manipulation environment, including YCB object support.

Available tasks:

| Task | Description |
|---|---|
| `ShadowHand` | Reorientation to a goal pose |
| `ShadowHandTilted` | Reorientation with continuously rotating base |
| `ShadowStableGrasp` | Stable grasp: keep object at rest in palm under gravity |

---

## Object Types

### Standard built-in objects

| `objectType` | Description |
|---|---|
| `block` | Simple cube |
| `egg` | Egg-shaped object |
| `pen` | Cylindrical pen |

### YCB objects (real meshes)

Located in `assets/urdf/ycb_balls/`. Each uses `textured.obj` for visual and `collision.ply` for collision.

| `objectType` | Mass | Description |
|---|---|---|
| `ycb_apple` | 68 g | YCB 013_apple |
| `ycb_banana` | 66 g | YCB 011_banana |
| `ycb_orange` | 131 g | YCB 017_orange |
| `ycb_peach` | 100 g | YCB 015_peach |
| `ycb_pear` | 166 g | YCB 016_pear |
| `ycb_plum` | 80 g | YCB 018_plum |
| `ycb_tomato_soup_can` | 349 g | YCB 005_tomato_soup_can |

---

## Training

All commands are run from the `isaacgymenvs/` directory.

```bash
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
```

### Single object type

```bash
# Train with a single YCB object
python train.py task=ShadowHand task.env.objectType=ycb_apple headless=True

# Train with a standard object
python train.py task=ShadowHand task.env.objectType=pen headless=True
```

### Multi-object pool (randomized per episode)

Set `objectTypePool` to a non-empty list to randomize object type each episode reset.

```bash
# All YCB fruits
python train.py task=ShadowHand \
  'task.env.objectTypePool=[ycb_apple,ycb_banana,ycb_orange,ycb_peach,ycb_pear,ycb_plum]' \
  headless=True

# Full YCB set including soup can
python train.py task=ShadowHand \
  'task.env.objectTypePool=[ycb_apple,ycb_banana,ycb_orange,ycb_peach,ycb_pear,ycb_plum,ycb_tomato_soup_can]' \
  headless=True

# Mixed standard + YCB
python train.py task=ShadowHand \
  'task.env.objectTypePool=[block,egg,pen,ycb_apple,ycb_orange,ycb_tomato_soup_can]' \
  headless=True
```

When `objectTypePool` is set, `objectType` is ignored.

---

## Key CLI Overrides

| Parameter | Default | Description |
|---|---|---|
| `num_envs` | 16384 | Number of parallel environments |
| `headless` | False | Disable rendering (required for servers) |
| `seed` | 42 | Random seed |
| `max_iterations` | — | Stop training after N policy updates |
| `checkpoint` | — | Resume from checkpoint path |
| `test` | False | Run inference only (no training) |
| `sim_device` | `cuda:0` | Physics simulation device |
| `rl_device` | `cuda:0` | RL algorithm device |
| `experiment` | — | Custom run name suffix |
| `capture_video` | False | Record video of inference |

### Batch size note

`batch_size = num_envs × horizon_length` (default horizon = 8) must be divisible by `minibatch_size` (default 32768). Use large `num_envs` (≥4096) or override `minibatch_size`:

```bash
# Small-scale test (256 envs)
python train.py task=ShadowHand task.env.objectType=ycb_apple \
  num_envs=256 'train.params.config.minibatch_size=512' \
  max_iterations=100 headless=True
```

---

## Resuming / Inference

```bash
# Resume training from checkpoint
python train.py task=ShadowHand \
  checkpoint=runs/ShadowHand/nn/ShadowHand.pth \
  headless=True

# Inference only
python train.py task=ShadowHand \
  checkpoint=runs/ShadowHand/nn/ShadowHand.pth \
  test=True num_envs=64
```

---

## Output Structure

```
runs/
└── ShadowHand_<timestamp>/
    ├── nn/
    │   ├── ShadowHand.pth          # best checkpoint
    │   └── last_ShadowHand_ep_N_rew_R.pth
    ├── summaries/                  # TensorBoard logs
    └── config.yaml                 # full resolved config
```

View training progress:
```bash
tensorboard --logdir runs/ShadowHand_<timestamp>/summaries
```

---

## YAML Configuration

Edit `isaacgymenvs/cfg/task/ShadowHand.yaml` to set defaults:

```yaml
env:
  objectType: "pen"         # default single object
  objectTypePool: []        # set non-empty to enable multi-object randomization
  episodeLength: 600        # steps per episode
  num_envs: 16384
```

Object scale randomization per type:
```yaml
objectScaleRanges:
  block: [0.5, 1.2]
  egg: [0.5, 1.2]
  pen: [0.5, 1.2]
  ycb_apple: [1.0, 1.0]    # YCB objects default to no scaling
```

---

---

## ShadowStableGrasp Task

### What it does

The object starts at rest inside the palm. The hand is randomly tilted (up to `maxInitOrientationDeg` degrees) at each episode reset, varying the effective gravity direction in palm space. The agent must hold the object in the palm without letting it fall.

Episode terminates when the object drifts more than `fallDistance` (0.3 m) from the palm center, or after `episodeLength` (300) steps.

### Reward structure

| Term | Sign | What it penalizes |
|---|---|---|
| `stable_reward` | + | `exp(-‖v_obj − v_palm‖ · stableGraspScale)` — 1.0 when object is stationary relative to palm |
| `palmDistPenalty` | − | Object distance from palm center |
| `objLinvelPenalty` | − | Object linear velocity (L2 squared) — quasi-static |
| `objAngvelPenalty` | − | Object angular velocity (L2 squared) — quasi-static |
| `dofVelPenalty` | − | Joint velocity squared sum — suppresses hand shaking |
| `dofForcePenalty` | − | Joint torque squared sum — reduces unnecessary force |
| `actionPenalty` | − | Action magnitude squared |
| `contactReward` | + | Bonus per active fingertip contact |

### Observation space (219-dim)

| Slice | Content |
|---|---|
| 0–23 | DOF positions |
| 24–47 | DOF velocities × 0.2 |
| 48–71 | DOF forces × 10 |
| 72–78 | Object pose (pos + quat) |
| 79–84 | Object linvel / angvel × 0.2 |
| 85–97 | Palm pose (pos + quat) + linvel + angvel × 0.2 |
| 98–103 | Relative pos and linvel (object − palm) |
| 104–168 | Fingertip states (13 × 5) |
| 169–198 | Fingertip FT sensors × 10 (6 × 5) |
| 199–218 | Last actions |

### Training

```bash
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs

# Basic run
python train.py task=ShadowStableGrasp headless=True

# Single YCB object
python train.py task=ShadowStableGrasp task.env.objectType=ycb_apple headless=True

# Multi-object pool
python train.py task=ShadowStableGrasp \
  'task.env.objectTypePool=[block,egg,ycb_apple,ycb_orange]' \
  headless=True

# Small-scale test (256 envs)
python train.py task=ShadowStableGrasp \
  num_envs=256 'train.params.config.minibatch_size=512' \
  max_iterations=100 headless=True
```

### Key config overrides (`task.env.*`)

| Parameter | Default | Description |
|---|---|---|
| `maxInitOrientationDeg` | 30 | Max random tilt of hand at episode start (degrees) |
| `stableGraspScale` | 5.0 | Exp decay rate for relative-velocity reward |
| `fallDistance` | 0.3 | Palm-to-object distance that triggers episode reset (m) |
| `palmDistPenaltyScale` | −1.0 | Weight for palm distance penalty |
| `objLinvelPenaltyScale` | −0.05 | Weight for object linear velocity penalty |
| `objAngvelPenaltyScale` | −0.02 | Weight for object angular velocity penalty |
| `dofVelPenaltyScale` | −0.001 | Weight for joint velocity penalty |
| `dofForcePenaltyScale` | −0.0002 | Weight for joint torque penalty |
| `contactRewardScale` | 0.001 | Bonus per active fingertip contact |
| `episodeLength` | 300 | Max steps per episode |

### Resuming / Inference

```bash
# Resume training
python train.py task=ShadowStableGrasp \
  checkpoint=runs/ShadowStableGrasp/nn/ShadowStableGrasp.pth \
  headless=True

# Inference only
python train.py task=ShadowStableGrasp \
  checkpoint=runs/ShadowStableGrasp/nn/ShadowStableGrasp.pth \
  test=True num_envs=64
```

---

## Conda Environment

All commands should use the `isaac` conda environment:

```bash
conda activate isaac
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
python train.py task=ShadowHand ...
```

Or with `conda run`:
```bash
conda run -n isaac --no-capture-output bash -c "
  cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
  python train.py task=ShadowHand task.env.objectType=ycb_apple headless=True
"
```
