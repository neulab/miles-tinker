"""Refuse to run heavy Tinker clients on an ABCI login node.

Import (or call ``check``) at the top of a client launcher. Rollout clients pin a
core for hours; run them on the gateway job's head node through
``qsub -v CLIENT_SCRIPT=...`` (deploy/abci/tinker_gateway.pbs). Set
``MILES_ALLOW_LOGIN_NODE=1`` for a short interactive smoke.
"""

import os
import socket
import sys


def check() -> None:
    host = socket.gethostname().split(".")[0]
    if host.startswith("login") and os.environ.get("MILES_ALLOW_LOGIN_NODE") != "1":
        sys.exit(
            f"refusing to run on login node {host}: submit through deploy/abci/tinker_gateway.pbs with "
            "CLIENT_SCRIPT=... (or set MILES_ALLOW_LOGIN_NODE=1 for a short smoke)"
        )


check()
