"""Let the eval loop run once before any training, at step 0.

``rl.py``'s eval loop gates every evaluation on::

    if shared_state.train_step % shared_state.eval_every == 0 and shared_state.train_step > 0:

The ``> 0`` is the only thing suppressing a step-0 eval. The loop's other
precondition is already satisfied at startup: ``sampling_client_updated_event``
is created *and set* where the shared state is built, so the loop's first
``await ... .wait()`` returns immediately, against the initial sampling client,
before the train loop has taken a step.

Why it is worth having. Every reward comparison so far has been against a
number measured after training had already moved the policy, which leaves the
run's own starting point unmeasured -- the Instruct run's four held-out evals
(15.0/15.0/16.2/18.8%) had no step-0 reading to anchor them, so a flat series
could not be distinguished from a series that had already risen before the
first eval landed. A graded reward makes this worse, not better: pass-rate
scoring changes the scale of eval/reward itself, so a number from a binary run
is not a baseline for a graded one. The run has to measure its own.

Cost is one eval cycle (20 tasks x group_size 4) before the first step.

Off unless PLATOON_EVAL_AT_START=1, because this patch list is shared with
train_openhands.sbatch and that run's eval schedule should not move.

Re-running an eval at a step already evaluated is possible on resume: a job
restarted at a step that is a multiple of eval.every evaluates once on startup
and again when it next reaches the interval. Harmless -- both land on the same
wandb step -- and it only bites a resumed run with the flag set.

Usage: python eval_at_step_zero.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_EVAL_AT_START"
TARGET = "platoon/train/tinker/rl.py"

ANCHOR = (
    "            if shared_state.train_step % shared_state.eval_every == 0 "
    "and shared_state.train_step > 0:\n"
)
REPLACEMENT = (
    "            _eval_at_start = os.getenv(\"PLATOON_EVAL_AT_START\", \"0\") == \"1\"\n"
    "            if shared_state.train_step % shared_state.eval_every == 0 and (\n"
    "                shared_state.train_step > 0 or _eval_at_start\n"
    "            ):\n"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No trainer at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"step-0 eval already configurable in {target}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"Expected exactly one step-0 eval guard in {target}; "
            "Platoon changed upstream and this patch needs review."
        )
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"made the step-0 eval configurable in {target}")


if __name__ == "__main__":
    main()
