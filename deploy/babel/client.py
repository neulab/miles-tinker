"""Run a Platoon training module with bounded shell lifetime and drained logging.

Usage: client.py <train-module> [args...]
The module must expose an async ``main(args)``.
"""

import asyncio
import atexit
import importlib
import logging
import sys

import litellm.utils
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

logger = logging.getLogger(__name__)
pending_logs = set()


def patch_number_search() -> None:
    """Bound the lifetime of number-search's IPython rollout shells."""
    from platoon.number_search import env

    class BatchCodeExecutor(env.IPythonCodeExecutor):
        def _create_shell(self):
            shell = super()._create_shell()
            # History is disabled by Platoon. These REPL exit hooks otherwise retain
            # every rollout shell until exit, even after its environment is closed.
            # ScriptMagics also cleans up its processes in its destructor.
            atexit.unregister(shell.atexit_operations)
            atexit.unregister(shell.magics_manager.registry["ScriptMagics"].kill_bg_processes)
            return shell

        async def close(self):
            self.shell = None

    env.IPythonCodeExecutor = BatchCodeExecutor


async def log_completion(logging_obj, result, start_time, end_time, is_completion_with_fallbacks):
    # LiteLLM already schedules this helper in a task with the request context.
    # Its additional queue (1.80.11) overflows into unbounded callback batches
    # and calls task_done for an item that was never enqueued. Run logging in
    # the existing task instead, and retain it so shutdown can drain it.
    task = asyncio.current_task()
    pending_logs.add(task)
    try:
        if not is_completion_with_fallbacks:
            async with asyncio.timeout(GLOBAL_LOGGING_WORKER.timeout):
                await logging_obj.async_success_handler(result=result, start_time=start_time, end_time=end_time)
            logging_obj.handle_sync_success_callbacks_for_async_calls(
                result=result, start_time=start_time, end_time=end_time
            )
    except Exception:
        logger.exception("LiteLLM completion logging failed")
    finally:
        pending_logs.discard(task)


async def run(main, args):
    litellm.utils._client_async_logging_helper = log_completion
    try:
        await main(args)
    finally:
        # Let helpers already scheduled by completed requests enter the set.
        await asyncio.sleep(0)
        if pending_logs:
            await asyncio.gather(*pending_logs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    module_name, arguments = sys.argv[1], sys.argv[2:]
    if module_name.startswith("platoon.number_search"):
        patch_number_search()
    module = importlib.import_module(module_name)
    asyncio.run(run(module.main, arguments))
