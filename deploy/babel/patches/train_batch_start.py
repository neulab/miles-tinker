"""Start the training task loader at batch PLATOON_TRAIN_BATCH_START.

The train loop's task loader starts at the resumed step (``i_batch = start_batch``),
which assumes one dataloader batch per step. Active sampling breaks that: the
loader hands out extra batches whenever groups are rejected, so after 30 steps
156019 had consumed batches 0-61, not 0-29. A resume from its step-30 checkpoint
would hand out batches 30-61 a second time (~500 tasks) before reaching new ones.

With PLATOON_TRAIN_BATCH_START=N the loader starts at batch N instead; training
steps (and checkpoint names, eval schedule) still count from the checkpoint.
work/batch_progress.py finds N for a run: the first batch after the last one
any rollout of the checkpoint's version - 1 came from. Unset, nothing changes.

Usage: python train_batch_start.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_TRAIN_BATCH_START"
LOOP = "platoon/train/tinker/rl.py"
ANCHOR = "        i_batch = start_batch\n        max_staleness = self.config.train.max_staleness\n"
REPLACEMENT = (
    "        # PLATOON_TRAIN_BATCH_START: where the task loader resumes; with active sampling\n"
    "        # it runs ahead of the step count, so a resume must not restart it at start_batch.\n"
    '        i_batch = int(os.environ.get("PLATOON_TRAIN_BATCH_START") or start_batch)\n'
    '        logger.info(f"Task loader starts at batch {i_batch} (training step {start_batch})")\n'
    "        max_staleness = self.config.train.max_staleness\n"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    loop = Path(sys.argv[1]) / LOOP
    if not loop.is_file():
        raise SystemExit(f"No rl.py at {loop}")
    source = loop.read_text()
    if MARKER in source:
        print(f"train batch start already present in {loop}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(f"Could not find the task loader's start in {loop}; this patch needs review.")
    loop.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"PLATOON_TRAIN_BATCH_START available in {loop}")


if __name__ == "__main__":
    main()
