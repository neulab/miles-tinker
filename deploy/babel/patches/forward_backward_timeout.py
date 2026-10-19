"""Size the forward_backward watchdog to this workload instead of 900s.

``rl.py`` waits for each microbatch's ``forward_backward`` under a hardcoded
timeout, annotated "Timeout after 15 min to detect hung Tinker operations"::

    total_timeout=900.0,

Fifteen minutes is not a lot of headroom for agentic terminal episodes, whose
batch size swings with how long the rollouts survive rather than with any
configured size. Measured on this pool, step 0 carried 1059 trained steps in
run 260928-210415-252 and 1888 in 260928-232958-550, and the longest
forward_backward wait moved with it: 360s against 660s-and-climbing. Three
launches -- 10595807, 10597116 and 10597592 -- were killed at step 1 by this
limit rather than by anything being hung.

The asymmetry is what sets the new value. Firing early destroys the run and
costs an hour of rollouts; firing late only delays the diagnosis of a genuine
hang. So default to 45 minutes, roughly 4x the worst legitimate wait observed,
and make it an environment variable so a run can tune it without a new patch.

Usage: python forward_backward_timeout.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_FORWARD_BACKWARD_TIMEOUT"
TARGET = "platoon/train/tinker/rl.py"

ANCHOR = "                            total_timeout=900.0,\n"
REPLACEMENT = (
    "                            total_timeout=float(\n"
    '                                os.getenv("PLATOON_FORWARD_BACKWARD_TIMEOUT", "2700")\n'
    "                            ),\n"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No trainer at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"forward_backward timeout already configurable in {target}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"Expected exactly one total_timeout=900.0 in {target}; "
            "Platoon changed upstream and this patch needs review."
        )
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"made the forward_backward timeout configurable in {target}")


if __name__ == "__main__":
    main()
