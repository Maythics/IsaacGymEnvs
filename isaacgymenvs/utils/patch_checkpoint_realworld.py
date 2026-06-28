"""Reset running_mean_std for masked obs dims to (mean=0, var=1).

The masked dims correspond to inputs the policy will see as zeros once
realWorldObs=True is set in the task config: per-DOF joint torques,
per-fingertip linvel+angvel, and per-fingertip 6-axis F/T sensor values.

Run this once per source checkpoint before fine-tuning, otherwise the saved
input-normalization will warp the zeroed dims into large negative values for a
few thousand steps after restart.

Usage:
    python -m isaacgymenvs.utils.patch_checkpoint_realworld \
        --task ShadowHand \
        --in  runs/ShadowHand/nn/ShadowHand.pth \
        --out runs/ShadowHand/nn/ShadowHand_realworld.pth

Supported --task values: ShadowHand, XHandHand, WujiHand. If --out is omitted
the patched checkpoint is written next to the input with a "_realworld" suffix.
"""

import argparse
from pathlib import Path

import torch


# (num_dofs, num_fingertips, num_actions) per supported hand. obs layout is
# [dof_pos(n), dof_vel(n), dof_torque(n), obj_pose(7), obj_linvel(3),
#  obj_angvel(3), goal_pose(7), rel_quat(4), 5x fingertip(13 each),
#  5x fingertip_FT(6 each), prev_actions(num_actions), ...]
_HAND_SPECS = {
    "ShadowHand": (24, 5, 20),
    "XHandHand":  (14, 5, 14),
    "WujiHand":   (22, 5, 22),
}


def build_mask_idx(task: str) -> torch.Tensor:
    if task not in _HAND_SPECS:
        raise ValueError(
            f"Unsupported task '{task}'. Supported: {list(_HAND_SPECS)}")
    n, num_fingertips, _ = _HAND_SPECS[task]
    fingertip_obs_start = 3 * n + 13 + 11
    num_ft_states = 13 * num_fingertips
    num_ft_force_torques = 6 * num_fingertips

    torque_idx = list(range(2 * n, 3 * n))
    ft_vel_idx = []
    for i in range(num_fingertips):
        base = fingertip_obs_start + i * 13
        ft_vel_idx.extend(range(base + 7, base + 13))
    ft_force_idx = list(range(
        fingertip_obs_start + num_ft_states,
        fingertip_obs_start + num_ft_states + num_ft_force_torques))

    return torch.tensor(torque_idx + ft_vel_idx + ft_force_idx, dtype=torch.long)


def patch_state_dict(model_state: dict, mask_idx: torch.Tensor) -> None:
    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    if mean_key not in model_state or var_key not in model_state:
        raise KeyError(
            f"Checkpoint model state is missing {mean_key} / {var_key}. "
            f"Was the source policy trained with normalize_input=True?")
    rm = model_state[mean_key]
    rv = model_state[var_key]
    obs_dim = rm.shape[0]
    if int(mask_idx.max()) >= obs_dim:
        raise ValueError(
            f"Mask index {int(mask_idx.max())} exceeds obs dim {obs_dim} of "
            f"checkpoint. Task / checkpoint mismatch?")
    rm[mask_idx] = 0.0
    rv[mask_idx] = 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(_HAND_SPECS))
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", type=Path, default=None)
    args = p.parse_args()

    if args.out_path is None:
        args.out_path = args.in_path.with_name(
            args.in_path.stem + "_realworld" + args.in_path.suffix)

    state = torch.load(args.in_path, map_location="cpu")
    if "model" not in state:
        raise KeyError("Checkpoint has no 'model' key — not an rl_games checkpoint?")

    mask_idx = build_mask_idx(args.task)
    patch_state_dict(state["model"], mask_idx)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out_path)
    print(f"Patched {len(mask_idx)} obs dims (task={args.task}).")
    print(f"  in:  {args.in_path}")
    print(f"  out: {args.out_path}")


if __name__ == "__main__":
    main()
