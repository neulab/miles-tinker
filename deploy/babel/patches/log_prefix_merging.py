"""Log how well each rollout's steps merge into training sequences.

The datum builder (tinker_data_processing.trajectory_to_data) extends one
sequence while step N's prompt + sampled tokens is a token prefix of step N+1's
prompt, and starts a new datum, re-encoding the whole history, whenever it is
not. How often that happens decides how many tokens the learner trains per step,
and it is what the chat-template and tool-parser fixes changed: before them
about half the steps of a rollout broke the prefix. TrajectoryStats already
carries num_datums per trajectory, but nothing logged it, and the aggregate
optim/num_datums cannot stand in for it because groups with equal rewards are
dropped before datums are counted.

This logs, per trajectory and before any group filtering:

* ``num_datums``: training sequences the trajectory became (1 = fully merged);
* ``prefix_merge_ratio``: (steps - datums) / (steps - 1), the fraction of step
  boundaries that merged; trajectories with one step are left out.

Both appear under the stats scope as train/num_datums/{avg,max,min} and
train/prefix_merge_ratio/{avg,max,min}.

Usage: python log_prefix_merging.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "prefix_merge_ratio"
TARGET = "platoon/train/tinker/workflows/group_rollout_workflow.py"
EDITS = (
    (
        "        num_steps_per_traj = torch.tensor([float(s.num_steps) for s in all_trajectory_stats])\n",
        "        num_steps_per_traj = torch.tensor([float(s.num_steps) for s in all_trajectory_stats])\n"
        "        # Training sequences per trajectory, and the share of step boundaries that\n"
        "        # extended the previous sequence instead of starting a new one.\n"
        "        num_datums_per_traj = torch.tensor([float(s.num_datums) for s in all_trajectory_stats])\n"
        "        prefix_merge_ratio = (num_steps_per_traj - num_datums_per_traj) / torch.clamp(num_steps_per_traj - 1, min=1.0)\n"
        "        merge_mask = num_steps_per_traj > 1\n",
    ),
    (
        "            num_steps_mask=trajectory_mask,\n",
        "            num_steps_mask=trajectory_mask,\n"
        "            num_datums_mask=trajectory_mask,\n"
        "            prefix_merge_ratio_mask=merge_mask,\n",
    ),
    (
        '        self.tracker.stat(num_steps=num_steps_per_traj, denominator="num_steps_mask")\n',
        '        self.tracker.stat(num_steps=num_steps_per_traj, denominator="num_steps_mask")\n'
        '        self.tracker.stat(num_datums=num_datums_per_traj, denominator="num_datums_mask")\n'
        '        self.tracker.stat(prefix_merge_ratio=prefix_merge_ratio, denominator="prefix_merge_ratio_mask")\n',
    ),
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1]) / TARGET
    if not path.is_file():
        raise SystemExit(f"No group rollout workflow at {path}")
    source = path.read_text()
    if MARKER in source:
        print(f"prefix-merging metrics already current in {path}")
        return
    for anchor, _ in EDITS:
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find {anchor.strip()!r} in {path}; Platoon changed upstream and this patch needs review.")
    for anchor, replacement in EDITS:
        source = source.replace(anchor, replacement, 1)
    path.write_text(source)
    print(f"logging num_datums and prefix_merge_ratio per trajectory in {path}")


if __name__ == "__main__":
    main()
