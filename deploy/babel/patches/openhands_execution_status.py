"""Point Platoon's openhands plugin at the SDK's current status field.

Platoon reads ``conversation_state.agent_status``. openhands-sdk renamed that
field to ``execution_status`` -- ``ConversationState.model_fields`` in 1.49.6
lists only the latter, and ``RemoteState`` exposes only the latter -- so every
openhands rollout dies in the episode loop with::

    AttributeError: 'RemoteState' object has no attribute 'agent_status'

It fails identically for a local conversation, which is why this path has never
produced a trajectory step on this SDK version. The comparison values are
unchanged: ConversationExecutionStatus already carries FINISHED, STUCK and ERROR.

Applied to the Platoon checkout at launch because the change belongs to Platoon
core, not to a plugin. Idempotent, and safe to re-run on a fresh clone.

Usage: python openhands_execution_status.py <platoon-checkout>
"""

import sys
from pathlib import Path

TARGETS = (
    "platoon/utils/openhands_utils.py",
    "plugins/openhands/platoon/openhands/env.py",
)
OLD = ".agent_status"
NEW = ".execution_status"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    touched = 0
    for relative in TARGETS:
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"No {relative} under {root}; Platoon's layout changed and this patch needs review.")
        source = path.read_text()
        if OLD not in source:
            continue
        path.write_text(source.replace(OLD, NEW))
        print(f"renamed agent_status -> execution_status in {path}")
        touched += 1
    if touched == 0:
        print("openhands status field already current")


if __name__ == "__main__":
    main()
