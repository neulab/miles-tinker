"""Run Platoon's batch client without its unused interactive-shell exit hook."""

import asyncio
import atexit
import gc
import sys

from IPython.core.interactiveshell import InteractiveShell
from platoon.number_search.train_tinker import main


if __name__ == "__main__":
    try:
        asyncio.run(main(sys.argv[1:]))
    finally:
        # Platoon closes its logger (including wandb.finish) in its async context.
        # IPython's display-cache flush hangs afterwards in this batch process.
        for shell in gc.get_objects():
            if issubclass(type(shell), InteractiveShell):
                atexit.unregister(shell.atexit_operations)
