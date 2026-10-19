"""Send each datum's per-token loss normalization as ``weights`` (PLATOON_SEND_LOSS_WEIGHTS=1).

The miles gateway's KL penalty to the starting weights (--tinker-kl-coef)
weights every token's KL by the datum's ``loss_fn_inputs["weights"]``. Tinker
losses are raw sums over tokens, and the client alone knows how a token should
count: the train loop folds that into the advantages, scaling each datum by
1 / (trained tokens in the microbatch), or 1 / (n_trajectories x trajectory
tokens) under PLATOON_LOSS_NORM=trajectory. Sending the same scale, times the
action-token mask, as ``weights`` makes the penalty average exactly like the
policy-gradient term -- zero on prompt and tool-output tokens, and the same
per-trajectory mean. The advantage losses (cispo, ppo, importance_sampling) do
not otherwise read ``weights``, so with the penalty off this changes nothing but
the payload.

Must run after trajectory_loss_norm.py, whose ``scales`` it reuses.

Usage: python send_loss_weights.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_SEND_LOSS_WEIGHTS"
LOOP = "platoon/train/tinker/rl.py"

ANCHOR = (
    "                    for datum, scale_factor in zip(filtered_datums, scales, strict=True):\n"
    '                        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(\n'
    '                            datum.loss_fn_inputs["advantages"].to_torch() * scale_factor\n'
    "                        )\n"
)
REPLACEMENT = ANCHOR + (
    "\n"
    "                    # PLATOON_SEND_LOSS_WEIGHTS=1: the same per-token normalization, on action\n"
    "                    # tokens only, as `weights` for the gateway's KL penalty (--tinker-kl-coef).\n"
    '                    if os.environ.get("PLATOON_SEND_LOSS_WEIGHTS", "0") == "1":\n'
    "                        for datum, source, scale_factor in zip(\n"
    "                            filtered_datums, task_rollout_results, scales, strict=True\n"
    "                        ):\n"
    '                            mask = source.loss_fn_inputs["mask"].to_torch().float()\n'
    '                            datum.loss_fn_inputs["weights"] = TensorData.from_torch(mask * float(scale_factor))\n'
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    loop = Path(sys.argv[1]) / LOOP
    if not loop.is_file():
        raise SystemExit(f"No rl.py at {loop}")
    source = loop.read_text()
    if MARKER in source:
        print(f"loss weights already sent in {loop}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"Could not find the advantage scaling loop in {loop}; apply trajectory_loss_norm.py first, "
            "or Platoon changed upstream and this patch needs review."
        )
    loop.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"PLATOON_SEND_LOSS_WEIGHTS=1 available in {loop}")


if __name__ == "__main__":
    main()
