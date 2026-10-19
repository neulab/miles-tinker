"""Weight every trajectory equally in the loss (PLATOON_LOSS_NORM=trajectory).

The train loop scales every advantage by 1 / (trained tokens in the
microbatch), and the CISPO loss sums over tokens. That makes the objective a
mean over all tokens in the step, so a trajectory pulls on the gradient in
proportion to advantage x length. Advantages are centred per group, so with
equal lengths the pushes up and down cancel; in the tmax-15k OpenHands runs
step-limit rollouts are ~2.5x as long as solved ones, and the net pull on
tokens every rollout shares (tool-call syntax, common commands) was -24% of
the total mass, every version (analysis/token_mass.py).

With PLATOON_LOSS_NORM=trajectory each datum is instead scaled by
1 / (n_trajectories x trained tokens of its trajectory): the mean over each
trajectory's tokens, then the mean over trajectories, as GRPO and GSPO reduce
(Dr. GRPO's critique applies the other way: a long failing trajectory spreads
its penalty thinner per token). A trajectory whose prompts stop being prefixes
is split into several datums, so the token count is the trajectory's, recorded
on every one of its datums as ``traj_tokens`` and stripped before
forward_backward. The number of trajectories is sum(datum tokens / traj_tokens).

The default (unset, or "token") keeps the token mean. Logged per step:
train/loss_norm_trajectories (trajectories in the microbatch; 0 under the token
mean).

Usage: python trajectory_loss_norm.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_LOSS_NORM"
DATA = "platoon/utils/tinker_data_processing.py"
LOOP = "platoon/train/tinker/rl.py"

DATA_EDITS = (
    (
        "    logger.debug(\n"
        '        f"Found {count_found} steps, produced {len(data)} datums for task {task_id} trajectory {trajectory_id}"\n'
        "    )\n",
        "    # PLATOON_LOSS_NORM=trajectory: every datum carries its whole trajectory's trained\n"
        "    # token count, so the train loop can average per trajectory across split datums.\n"
        '    traj_tokens = sum(float(d.loss_fn_inputs["mask"].to_torch().sum()) for d in data)\n'
        "    for d in data:\n"
        '        d.loss_fn_inputs["traj_tokens"] = TensorData.from_torch(torch.tensor([traj_tokens]))\n'
        "\n"
        "    logger.debug(\n"
        '        f"Found {count_found} steps, produced {len(data)} datums for task {task_id} trajectory {trajectory_id}"\n'
        "    )\n",
    ),
)

LOOP_EDITS = (
    (
        '                                if k not in ("mask", "checkpoint_version", "traj_depth", "traj_start")\n',
        '                                if k not in ("mask", "checkpoint_version", "traj_depth", "traj_start", "traj_tokens")\n',
    ),
    (
        "                    scale_factor = 0.0\n"
        "                    for datum in task_rollout_results:\n"
        '                        scale_factor += datum.loss_fn_inputs["mask"].to_torch().sum()\n'
        "                    scale_factor = 1.0 / (scale_factor + 1e-8)\n"
        "\n"
        "                    for datum in filtered_datums:\n"
        '                        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(\n'
        '                            datum.loss_fn_inputs["advantages"].to_torch() * scale_factor\n'
        "                        )\n",
        "                    if os.environ.get(\"PLATOON_LOSS_NORM\", \"token\") == \"trajectory\":\n"
        "                        # Mean over each trajectory's tokens, then over trajectories.\n"
        "                        scales = _trajectory_scales(task_rollout_results)\n"
        "                        shared_state.train_tracker.scalar(loss_norm_trajectories=1.0 / scales[1])\n"
        "                        scales = scales[0]\n"
        "                    else:\n"
        "                        scale_factor = 0.0\n"
        "                        for datum in task_rollout_results:\n"
        '                            scale_factor += datum.loss_fn_inputs["mask"].to_torch().sum()\n'
        "                        scales = [1.0 / (scale_factor + 1e-8)] * len(task_rollout_results)\n"
        "                        shared_state.train_tracker.scalar(loss_norm_trajectories=0.0)\n"
        "\n"
        "                    for datum, scale_factor in zip(filtered_datums, scales, strict=True):\n"
        '                        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(\n'
        '                            datum.loss_fn_inputs["advantages"].to_torch() * scale_factor\n'
        "                        )\n",
    ),
    (
        "\n\ndef apply_depth_level_weighting(",
        '''

def _trajectory_scales(datums: list[tinker.Datum]) -> tuple[list[float], float]:
    """Per-datum advantage scales for a per-trajectory mean (PLATOON_LOSS_NORM=trajectory).

    Returns the scales, 1 / (n_trajectories x trajectory tokens), and 1 / n_trajectories.
    """
    shares, totals = [], []
    for datum in datums:
        tokens = float(datum.loss_fn_inputs["mask"].to_torch().sum())
        total = float(datum.loss_fn_inputs["traj_tokens"].to_torch().item())
        totals.append(total)
        shares.append(tokens / total if total > 0 else 0.0)
    n_trajectories = max(sum(shares), 1e-8)
    scales = [1.0 / (n_trajectories * total) if total > 0 else 0.0 for total in totals]
    return scales, 1.0 / n_trajectories


def apply_depth_level_weighting(''',
    ),
)


def apply(path: Path, edits) -> str:
    source = path.read_text()
    for anchor, _ in edits:
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find {anchor.strip()!r} in {path}; Platoon changed upstream and this patch needs review.")
    for anchor, replacement in edits:
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    data, loop = Path(sys.argv[1]) / DATA, Path(sys.argv[1]) / LOOP
    for path in (data, loop):
        if not path.is_file():
            raise SystemExit(f"No {path.name} at {path}")
    if MARKER in loop.read_text():
        print(f"trajectory loss normalization already applied in {loop}")
        return
    new_data, new_loop = apply(data, DATA_EDITS), apply(loop, LOOP_EDITS)
    data.write_text(new_data)
    loop.write_text(new_loop)
    print(f"PLATOON_LOSS_NORM=trajectory available in {loop}")


if __name__ == "__main__":
    main()
