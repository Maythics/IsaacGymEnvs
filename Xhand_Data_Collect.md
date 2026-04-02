# XHand Data Collection Guide

## Overview

Two tasks are available for the XHand robot:

| Task | Description | Obs dims | Action dims |
|---|---|---|---|
| `XHandHand` | In-hand reorientation (match goal rotation) | 175 | 14 |
| `XHandPush` | In-hand repositioning (push along trajectory) | 175 | 14 |

The XHand has **14 actuated DOFs**: 2 wrist (WRJ2 radial/ulnar + WRJ1 flexion/extension) + 3 thumb + 3 index + 2 middle + 2 ring + 2 pinky.

---

## Asset

The self-contained URDF and meshes live in:
```
assets/urdf/xhand/
├── xhand_right_with_wrist.urdf   # XHand + 2-DOF wrist
└── meshes/                       # 30 STL files (copied from source)
```

---

## Training

Run from the `isaacgymenvs/` directory:

```bash
cd isaacgymenvs

# Train XHandHand (reorientation)
python train.py task=XHandHand num_envs=8192 headless=True

# Train XHandPush (repositioning)
python train.py task=XHandPush num_envs=8192 headless=True

# With W&B logging
python train.py task=XHandHand num_envs=8192 headless=True \
    wandb_activate=True wandb_project=xhand

# Resume from checkpoint
python train.py task=XHandHand num_envs=8192 headless=True \
    checkpoint=runs/XHandHand/nn/XHandHand.pth
```

Checkpoints and configs are saved to `runs/XHandHand/` and `runs/XHandPush/`.

---

## Inference / Rollout

```bash
# Visual rollout with a trained policy
python train.py task=XHandHand num_envs=64 \
    checkpoint=runs/XHandHand/nn/XHandHand.pth test=True

python train.py task=XHandPush num_envs=64 \
    checkpoint=runs/XHandPush/nn/XHandPush.pth test=True
```

---

## Programmatic API

```python
import isaacgym
import isaacgymenvs

# Create XHandHand environment
envs = isaacgymenvs.make(
    seed=0,
    task="XHandHand",
    num_envs=64,
    sim_device="cuda:0",
    rl_device="cuda:0",
    headless=True,
)

obs = envs.reset()
print("Obs shape:", obs["obs"].shape)   # (64, 175)

for _ in range(100):
    actions = envs.action_space.sample()  # random actions, shape (64, 14)
    obs, rew, done, info = envs.step(actions)
```

---

## Observation Layout (full_state, 175 dims)

| Slice | Size | Content |
|---|---|---|
| `[0:14]` | 14 | DOF positions (unscaled to [-1, 1]) |
| `[14:28]` | 14 | DOF velocities × 0.2 |
| `[28:42]` | 14 | DOF forces × 10.0 |
| `[42:49]` | 7 | Object pose (xyz + xyzw quat) |
| `[49:52]` | 3 | Object linear velocity |
| `[52:55]` | 3 | Object angular velocity × 0.2 |
| `[55:62]` | 7 | Goal pose (xyz + xyzw quat) |
| `[62:66]` | 4 | `quat_mul(obj_rot, conj(goal_rot))` |
| `[66:131]` | 65 | 5 fingertips × 13 (pos + quat + linvel + angvel) |
| `[131:161]` | 30 | 5 fingertip force-torque sensors × 6 |
| `[161:175]` | 14 | Previous actions |

---

## Action Space

14-dimensional continuous, clipped to `[-1, 1]`.

By default (`useRelativeControl: False`), actions are mapped to absolute joint position targets via a moving average:
```
target = α × scale(action, lower, upper) + (1−α) × prev_target
```
where `α = actionsMovingAverage` (default 1.0).

DOF order in the action vector:
```
[0]  right_hand_WRJ2          (radial/ulnar,  range [-0.14, 0.49] rad)
[1]  right_hand_WRJ1          (flexion/ext,   range [-0.49, 0.14] rad)
[2]  right_hand_thumb_bend    (range [0, 1.83] rad)
[3]  right_hand_thumb_rota1   (range [-0.70, 1.57] rad)
[4]  right_hand_thumb_rota2   (range [0, 1.57] rad)
[5]  right_hand_index_bend    (range [-0.17, 0.17] rad)
[6]  right_hand_index_joint1  (range [0, 1.92] rad)
[7]  right_hand_index_joint2  (range [0, 1.92] rad)
[8]  right_hand_mid_joint1    (range [0, 1.92] rad)
[9]  right_hand_mid_joint2    (range [0, 1.92] rad)
[10] right_hand_ring_joint1   (range [0, 1.92] rad)
[11] right_hand_ring_joint2   (range [0, 1.92] rad)
[12] right_hand_pinky_joint1  (range [0, 1.92] rad)
[13] right_hand_pinky_joint2  (range [0, 1.92] rad)
```

---

## Key Config Parameters

All parameters can be overridden from the CLI using Hydra syntax (`param=value`).

### Shared (XHandHand & XHandPush)

| Parameter | Default | Description |
|---|---|---|
| `num_envs` | 8192 | Number of parallel environments |
| `episodeLength` | 600 / 200 | Max steps per episode |
| `objectType` | `"block"` | `block`, `egg`, `pen`, or `ycb_*` |
| `objectTypePool` | `[]` | Multi-object training pool (empty = single type) |
| `dofSpeedScale` | 20.0 | Joint speed multiplier |
| `useRelativeControl` | False | Relative (velocity-increment) vs absolute control |
| `distRewardScale` | -10.0 | Position distance penalty |
| `actionPenaltyScale` | -0.0002 | Action magnitude penalty |
| `palmDistPenaltyScale` | -0.5 | Palm-object distance penalty |
| `fallDistance` | 0.24 | Object drop threshold (m) |

### XHandHand only

| Parameter | Default | Description |
|---|---|---|
| `rotRewardScale` | 1.0 | Rotation alignment reward |
| `successTolerance` | 0.3 | Rotation success threshold (rad) |
| `reachGoalBonus` | 250 | Bonus reward on success |

### XHandPush only

| Parameter | Default | Description |
|---|---|---|
| `yawRewardScale` | 1.0 | Yaw alignment reward |
| `posSuccessTolerance` | 0.02 | Position success threshold (m) |
| `yawSuccessTolerance` | 1.6 | Yaw success threshold (rad) |
| `pushStepSize` | 0.01 | Goal advancement per step (m) |
| `flipAngle` | 1.5708 | Roll/pitch termination threshold (rad) |
| `uprightPenaltyScale` | -0.5 | Roll²+pitch² penalty |

---

## Multi-Object Training

To train on a mixture of object types simultaneously:

```bash
python train.py task=XHandHand num_envs=8192 headless=True \
    task.env.objectTypePool='["block","egg","pen"]'

# Include YCB objects
python train.py task=XHandHand num_envs=8192 headless=True \
    "task.env.objectTypePool=[block,egg,pen,ycb_apple,ycb_banana,ycb_orange]"
```

---

## Object Types

| Type | Asset |
|---|---|
| `block` | `urdf/objects/cube_multicolor.urdf` |
| `egg` | `mjcf/open_ai_assets/hand/egg.xml` |
| `pen` | `mjcf/open_ai_assets/hand/pen.xml` |
| `ycb_apple` | `urdf/ycb_balls/ycb_apple.urdf` |
| `ycb_banana` | `urdf/ycb_balls/ycb_banana.urdf` |
| `ycb_orange` | `urdf/ycb_balls/ycb_orange.urdf` |
| `ycb_peach` | `urdf/ycb_balls/ycb_peach.urdf` |
| `ycb_pear` | `urdf/ycb_balls/ycb_pear.urdf` |
| `ycb_plum` | `urdf/ycb_balls/ycb_plum.urdf` |
| `ycb_tomato_soup_can` | `urdf/ycb_balls/ycb_tomato_soup_can.urdf` |

---

## Troubleshooting

**Fingertip body not found at runtime**

If you see warnings about rigid body indices returning -1, it means `collapse_fixed_joints=True` merged tip links differently than expected. In `xhand_hand.py`, change `_XHAND_FINGERTIPS` to use the actual tip link names:
```python
_XHAND_FINGERTIPS = [
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
    "right_hand_thumb_rota_tip",
]
```
and set `asset_options.collapse_fixed_joints = False`.

**DOF count mismatch**

The URDF must have exactly 14 revolute joints (2 wrist + 12 fingers). Verify with:
```bash
python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('assets/urdf/xhand/xhand_right_with_wrist.urdf')
revolute = [j for j in tree.getroot().findall('joint') if j.get('type') == 'revolute']
print('Revolute joints:', len(revolute))
for j in revolute: print(' ', j.get('name'))
"
```
