"""Fill every microbatch with groups that carry a gradient (active sampling).

GroupRolloutWorkflow returns None for a group whose rollouts all score the same
(their advantages are all zero), and for groups lost to infrastructure. The
train loop takes exactly ``tasks_per_microbatch`` results off the queue and drops
the empty ones, so a step trains on whatever survives. Measured on the batch-32
and batch-64 FLAME runs, 40-49% of groups were all 1.0 or all 0.2 early on, so a
"batch 32" step trained on roughly 17-20 tasks.

With PLATOON_ACTIVE_SAMPLING=1 the train loop keeps taking results until the
microbatch holds ``tasks_per_microbatch`` non-empty groups, as in Olmo 3's
active sampling (and DAPO's dynamic sampling, without oversampling up front).

The data loader has to keep up. It paces itself by batch index
(``i_batch - train_step <= max_staleness``), which feeds at most one batch ahead
of training; a trainer that needs more than ``batch_size`` results per step
would drain that and wait forever. With the flag on, the loader instead keeps
the same depth in tasks: it feeds while tasks handed out but not yet taken off
the result queue stay within ``(max_staleness + 1) * batch_size``, and it keeps
going past ``end_batch`` (get_batch wraps around the dataset). Rollouts that end
up too stale are still filtered by the existing staleness check.

A microbatch stops after PLATOON_ACTIVE_SAMPLING_MAX_DRAWS (default 4) times
``tasks_per_microbatch`` results even if short of groups, so a collapsed policy
whose groups are all equal cannot stall training.

Logged per step: train/active_sampling_draws (results taken per microbatch) and
train/active_sampling_groups (non-empty groups kept per microbatch).

Usage: python active_sampling.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_ACTIVE_SAMPLING"
TARGET = "platoon/train/tinker/rl.py"
HELPER = '''

class _MicrobatchSlots:
    """Result slots for one microbatch: a fixed count, or until enough groups carry a gradient."""

    def __init__(self, needed: int, active: bool, max_draws: int):
        self.needed, self.active, self.max_draws = needed, active, max_draws
        self.draws = 0
        self.groups = 0

    def __iter__(self):
        while (self.groups if self.active else self.draws) < self.needed:
            if self.draws >= self.max_draws:
                logger.warning(
                    f"Active sampling: {self.groups}/{self.needed} groups with a gradient after "
                    f"{self.draws} rollouts; training on what there is"
                )
                return
            self.draws += 1
            yield self.draws - 1

    def accept(self, has_gradient: bool) -> None:
        self.groups += int(has_gradient)


@dataclass
class TrainLoopSharedState:'''

EDITS = (
    # Helper class, ahead of the shared state dataclass.
    ("\n\n@dataclass\nclass TrainLoopSharedState:", HELPER),
    # Loader: pace by tasks in flight instead of batch index when active sampling is on.
    (
        "        i_batch = start_batch\n"
        "        max_staleness = self.config.train.max_staleness\n"
        "        while not shared_state.shutdown_event.is_set() and i_batch < end_batch:\n"
        "            # Control staleness by waiting for training to catch up.\n"
        "            if max_staleness is not None and i_batch - shared_state.train_step > max_staleness:\n"
        "                await asyncio.sleep(20.0)\n"
        "                continue\n"
        "\n"
        "            batch = shared_state.train_dataloader.get_batch(i_batch)\n"
        "            for data in batch:\n"
        "                await data_queue.put(data)\n"
        "            i_batch += 1\n",
        "        i_batch = start_batch\n"
        "        max_staleness = self.config.train.max_staleness\n"
        "        # Active sampling takes more than batch_size results per step, so pace the loader by\n"
        "        # tasks handed out but not yet consumed (same depth as max_staleness + 1 batches).\n"
        "        fed = 0\n"
        "        in_flight_limit = ((max_staleness or 0) + 1) * self.config.train.batch_size\n"
        "        while not shared_state.shutdown_event.is_set() and (i_batch < end_batch or self._active_sampling):\n"
        "            # Control staleness by waiting for training to catch up.\n"
        "            if self._active_sampling:\n"
        "                if fed + self.config.train.batch_size - self._results_consumed > in_flight_limit:\n"
        "                    await asyncio.sleep(5.0)\n"
        "                    continue\n"
        "            elif max_staleness is not None and i_batch - shared_state.train_step > max_staleness:\n"
        "                await asyncio.sleep(20.0)\n"
        "                continue\n"
        "\n"
        "            batch = shared_state.train_dataloader.get_batch(i_batch)\n"
        "            for data in batch:\n"
        "                await data_queue.put(data)\n"
        "            fed += len(batch)\n"
        "            i_batch += 1\n",
    ),
    # Flag and counter, set before the loader task starts.
    (
        "        # Initialize queues to pipeline data loading, rollout sampling and training.\n"
        "        task_data_queue = asyncio.Queue[dict](maxsize=self.config.train.batch_size)\n",
        "        # Initialize queues to pipeline data loading, rollout sampling and training.\n"
        '        self._active_sampling = os.getenv("PLATOON_ACTIVE_SAMPLING", "0") == "1"\n'
        '        self._active_sampling_max_draws = float(os.getenv("PLATOON_ACTIVE_SAMPLING_MAX_DRAWS", "4"))\n'
        "        self._results_consumed = 0\n"
        "        if self._active_sampling:\n"
        "            logger.info(\n"
        '                "Active sampling on: each microbatch waits for groups with a gradient "\n'
        '                f"(at most {self._active_sampling_max_draws:g}x tasks_per_microbatch rollouts)"\n'
        "            )\n"
        "        task_data_queue = asyncio.Queue[dict](maxsize=self.config.train.batch_size)\n",
    ),
    # Slots replace the fixed range.
    (
        "                    for task_num in range(tasks_per_microbatch):\n",
        "                    slots = _MicrobatchSlots(\n"
        "                        tasks_per_microbatch,\n"
        "                        self._active_sampling,\n"
        "                        max(tasks_per_microbatch, int(self._active_sampling_max_draws * tasks_per_microbatch)),\n"
        "                    )\n"
        "                    for task_num in slots:\n",
    ),
    (
        "                        rollout = await task_rollout_result_queue.get()\n",
        "                        rollout = await task_rollout_result_queue.get()\n"
        "                        self._results_consumed += 1\n",
    ),
    (
        "                        if rollout is not None:\n"
        "                            task_rollout_results.extend(rollout)\n"
        "                            total_datums += len(rollout)\n",
        "                        if rollout is not None:\n"
        "                            task_rollout_results.extend(rollout)\n"
        "                            total_datums += len(rollout)\n"
        "                        slots.accept(bool(rollout))\n"
        "\n"
        "                    shared_state.train_tracker.scalar(\n"
        "                        active_sampling_draws=float(slots.draws), active_sampling_groups=float(slots.groups)\n"
        "                    )\n",
    ),
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1]) / TARGET
    if not path.is_file():
        raise SystemExit(f"No Tinker RL trainer at {path}")
    source = path.read_text()
    if MARKER in source:
        print(f"active sampling already current in {path}")
        return
    for anchor, _ in EDITS:
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find {anchor.strip()[:80]!r} in {path}; Platoon changed upstream and this patch needs review.")
    for anchor, replacement in EDITS:
        source = source.replace(anchor, replacement, 1)
    path.write_text(source)
    print(f"active sampling (PLATOON_ACTIVE_SAMPLING=1) in {path}")


if __name__ == "__main__":
    main()
