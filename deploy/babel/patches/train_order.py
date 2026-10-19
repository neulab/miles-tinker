"""Training task order: list order (PLATOON_TRAIN_SHUFFLE=0) or a new shuffle every pass (=epoch).

PlatoonTinkerDataloader shuffles the training set once (seed 42) before batching,
so a task list ordered easy -> hard (analysis/build_medium_pool.py writes
medium_pool_curriculum.txt) would reach the train loop scrambled. With
PLATOON_TRAIN_SHUFFLE=0 the training dataloader is built with shuffle_seed=None,
like the eval dataloader already is: step n trains on the list's n-th batch of
batch_size tasks, wrapping around at the end. Unset (default) keeps the shuffle.

The default shuffle is drawn once and replayed on every pass, and the batching
drops the last partial batch, so with 166 tasks and batch 16 the same 6 tasks
are never trained and every pass visits the same batches in the same order.
PLATOON_TRAIN_SHUFFLE=epoch draws a fresh permutation of the whole dataset for
each pass (seed 42 + pass number) and cuts its batches from that, so batch
composition and order change every pass and the left-over tasks differ each
time. get_batch(n) stays a pure function of n (pass = n // batches_per_pass).

Usage: python train_order.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_TRAIN_SHUFFLE"
LOOP = "platoon/train/tinker/rl.py"
ANCHOR = "        train_dataloader = PlatoonTinkerDataloader(self.train_dataset, self.config.train.batch_size)\n"
REPLACEMENT = (
    "        # PLATOON_TRAIN_SHUFFLE=0: keep the task list's order (curricula);\n"
    "        # =epoch: a fresh shuffle of the whole list on every pass.\n"
    "        train_dataloader = PlatoonTinkerDataloader(\n"
    "            self.train_dataset,\n"
    "            self.config.train.batch_size,\n"
    '            shuffle_seed=None if os.environ.get("PLATOON_TRAIN_SHUFFLE", "1") in ("0", "epoch") else 42,\n'
    "        )\n"
    '        if os.environ.get("PLATOON_TRAIN_SHUFFLE") == "epoch":\n'
    "            train_dataloader.reshuffle_every_pass(seed=42, batch_size=self.config.train.batch_size)\n"
)
CLASS_ANCHOR = (
    "    def get_batch(self, batch_num: int) -> list[dict]:\n"
    "        batch_num = batch_num % len(self.batched_dataset)\n"
)
CLASS_REPLACEMENT = (
    "    def reshuffle_every_pass(self, seed: int, batch_size: int) -> None:\n"
    '        """PLATOON_TRAIN_SHUFFLE=epoch: a fresh permutation of the dataset for every pass."""\n'
    "        self._pass_seed = seed\n"
    "        self._pass_batch_size = batch_size\n"
    "\n"
    "    def get_batch(self, batch_num: int) -> list[dict]:\n"
    '        if getattr(self, "_pass_seed", None) is not None:\n'
    "            import random\n"
    "\n"
    "            pass_num, index = divmod(batch_num, len(self.batched_dataset))\n"
    "            order = list(range(len(self.dataset)))\n"
    "            random.Random(self._pass_seed + pass_num).shuffle(order)\n"
    "            size = self._pass_batch_size\n"
    "            return [dict(row) for row in self.dataset.select(order[index * size : (index + 1) * size])]\n"
    "        batch_num = batch_num % len(self.batched_dataset)\n"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    loop = Path(sys.argv[1]) / LOOP
    if not loop.is_file():
        raise SystemExit(f"No rl.py at {loop}")
    source = loop.read_text()
    if MARKER in source:
        print(f"train order switch already present in {loop}")
        return
    if source.count(ANCHOR) != 1 or source.count(CLASS_ANCHOR) != 1:
        raise SystemExit(f"Could not find the train dataloader in {loop}; this patch needs review.")
    loop.write_text(source.replace(ANCHOR, REPLACEMENT, 1).replace(CLASS_ANCHOR, CLASS_REPLACEMENT, 1))
    print(f"PLATOON_TRAIN_SHUFFLE=0 available in {loop}")


if __name__ == "__main__":
    main()
