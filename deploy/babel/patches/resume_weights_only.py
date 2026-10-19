"""Let a resume load the weights without the optimizer state.

``rl.py`` always resumes with ``create_training_client_from_state_with_optimizer``.
That cannot work when the checkpoint has no usable optimizer state: a server run
with --no-save-optim writes weights only, and full fine-tuning checkpoints reload
their optimizer state only into the parallel layout that saved them (miles
full/executor.py: "Resume needs the same parallel topology"), e.g. not from one
node (DP 1) into two (DP 2). Megatron's load then fails with KeyError: 'optimizer'.
(A server with --no-save-optim also falls back to weights only by itself.)

With PLATOON_RESUME_WEIGHTS_ONLY=1 the resume loads the weights alone and starts
Adam fresh (the server's weights-only load resets the optimizer state); the run
otherwise resumes as usual: step count, task-selection state and wandb run.
Unset, resumes are unchanged.

Usage: python resume_weights_only.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_RESUME_WEIGHTS_ONLY"
TARGET = "platoon/train/tinker/rl.py"

ANCHOR = """        if resume_info:
            # Resuming interrupted training - load optimizer state for proper continuation
            training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path
            )
"""
REPLACEMENT = """        if resume_info and os.getenv("PLATOON_RESUME_WEIGHTS_ONLY", "0") == "1":
            # Resuming into another parallel layout: the optimizer state cannot be reloaded
            # there, so take the weights and start Adam fresh.
            training_client = await service_client.create_training_client_from_state_async(
                resume_info.state_path
            )
            logger.warning("Resuming with weights only (fresh optimizer state)")
        elif resume_info:
            # Resuming interrupted training - load optimizer state for proper continuation
            training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path
            )
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No trainer at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"weights-only resume already available in {target}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"Expected exactly one optimizer-state resume in {target}; "
            "Platoon changed upstream and this patch needs review."
        )
    if "\nimport os\n" not in source:
        raise SystemExit(f"{target} no longer imports os; this patch needs review.")
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"weights-only resume (PLATOON_RESUME_WEIGHTS_ONLY) available in {target}")


if __name__ == "__main__":
    main()
