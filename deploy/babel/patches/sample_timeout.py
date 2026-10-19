"""Make the per-call sampling timeout configurable instead of 600s.

``proxy.py`` bounds every ``sample_async`` under a hardcoded limit::

    SAMPLE_TIMEOUT_SECONDS = 600

The episode's own per-step limit (``rollout_config.step_timeout``, STEP_TIMEOUT
in the launcher) wraps the same wait, so raising STEP_TIMEOUT past 600s has no
effect on its own: a slow call fails here instead, as "sample_async timed out
after 600.2s". Run 261016-140723-019 (27B policy, ~190 concurrent rollouts)
hit both limits. Read PLATOON_SAMPLE_TIMEOUT so the two can be raised together
without a new patch; the default keeps the upstream 600s.

Usage: python sample_timeout.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "PLATOON_SAMPLE_TIMEOUT"
TARGET = "platoon/train/tinker/proxy.py"

ANCHOR = "        SAMPLE_TIMEOUT_SECONDS = 600\n"
REPLACEMENT = '        SAMPLE_TIMEOUT_SECONDS = float(__import__("os").getenv("PLATOON_SAMPLE_TIMEOUT", "600"))\n'


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No proxy at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"sample timeout already configurable in {target}")
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"Expected exactly one SAMPLE_TIMEOUT_SECONDS = 600 in {target}; "
            "Platoon changed upstream and this patch needs review."
        )
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"made the sample timeout configurable in {target}")


if __name__ == "__main__":
    main()
