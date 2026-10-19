"""Keep the OpenHands SDK's blocking HTTP off the asyncio event loop.

``OpenHandsEnv.reset`` builds the conversation and sends the task goal with two
synchronous SDK calls, straight from a coroutine::

    self._conversation = Conversation(agent=..., workspace=..., ...)
    self._conversation.send_message(self._task.goal)
    # NOTE: Run the conversation in a separate thread to avoid blocking the main thread.
    threading.Thread(target=self._conversation.run, daemon=True).start()

The comment shows the hazard was known -- ``run`` was moved to a thread -- but
the constructor and ``send_message`` were left behind, and the constructor is the
expensive one: ``RemoteConversation`` creates the conversation over HTTP, opens a
WebSocket and waits on ``wait_until_ready`` against a server that has only just
started. ``close`` has the same shape.

While the loop's thread sits in one of those calls nothing else in the process
runs, and that includes the timer machinery. ``asyncio.wait_for`` is serviced by
the loop, so ``step_timeout`` cannot fire either -- the rollouts' own safety net
is suspended by the same block that stalls them.

Job 10624988 shows the cost. At 16:26:13 six sandboxes reported ready in one
second and six rollouts entered reset(); the eight ``run() triggered`` responses
that followed took 17 seconds to arrive over loopback. In that same minute 75 of
the 79 rollouts that later died took their last step, and the gateway fell to
zero in-flight requests. They did not fail independently: one wave of new
rollouts blocked the loop and stalled everything already running. Only 3 of 79
produced a stall probe, because the probe lives inside act()'s poll loop and
those coroutines were never scheduled to reach it -- when the loop resumed, every
overdue timer fired at once and cancelled them.

Moving both calls onto a worker thread restores the property the loop depends on:
one slow or wedged sandbox costs one thread, not every rollout in the batch. The
open is bounded by its own timeout because the SDK sets none on that path (it
does set timeout=30 on the run trigger); close is already wrapped by the caller's
CLOSE_TEARDOWN_TIMEOUT, which starts working again once the call is off the loop.

Cancelling an ``await`` on a thread does not kill the thread, so a genuinely
wedged sandbox still holds one worker until its own I/O gives up. The pool is
sized for that, and the loop stays responsive either way.

Usage: python conversation_io_off_the_loop.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_run_blocking_sdk_call"
TARGET = "plugins/openhands/platoon/openhands/env.py"

IMPORT_ANCHOR = """from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
"""

IMPORT = """from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
"""

HELPER_ANCHOR = "class OpenHandsEnv:"

HELPER = '''# The SDK's conversation open and close are synchronous HTTP. Run them on a
# dedicated pool rather than the default executor, so they cannot be starved by
# -- or starve -- anything else the process hands to asyncio.to_thread.
_SDK_IO_WORKERS = int(os.getenv("TERMINAL_RL_SDK_IO_WORKERS", "64"))
_CONVERSATION_OPEN_TIMEOUT = float(os.getenv("TERMINAL_RL_CONVERSATION_OPEN_TIMEOUT", "300"))
_sdk_io_pool: ThreadPoolExecutor | None = None


def _sdk_io_executor() -> ThreadPoolExecutor:
    global _sdk_io_pool
    if _sdk_io_pool is None:
        _sdk_io_pool = ThreadPoolExecutor(
            max_workers=_SDK_IO_WORKERS, thread_name_prefix="openhands-sdk-io"
        )
    return _sdk_io_pool


async def _run_blocking_sdk_call(func, timeout: float | None = None):
    """Run a synchronous SDK call without parking the event loop thread.

    A timeout here only abandons the await; the worker thread keeps running until
    its own I/O returns. That is the intended trade: one wedged sandbox costs one
    thread instead of every rollout sharing this loop.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_sdk_io_executor(), func)
    if timeout is None:
        return await future
    return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)


class OpenHandsEnv:'''

RESET_ANCHOR = """    async def reset(self) -> OpenHandsObservation:
        self._conversation: BaseConversation = Conversation(
            agent=self._agent,
            workspace=self._workspace,
            visualizer=None,
            max_iteration_per_run=self._task.max_steps,
        )
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
        self._conversation.send_message(self._task.goal)
"""

RESET_REPLACEMENT = '''    async def reset(self) -> OpenHandsObservation:
        def _open() -> BaseConversation:
            # Both calls are synchronous SDK HTTP; keeping them together means
            # one hop onto the worker thread instead of two.
            conversation: BaseConversation = Conversation(
                agent=self._agent,
                workspace=self._workspace,
                visualizer=None,
                max_iteration_per_run=self._task.max_steps,
            )
            conversation.send_message(self._task.goal)
            return conversation

        self._conversation = await _run_blocking_sdk_call(
            _open, timeout=_CONVERSATION_OPEN_TIMEOUT
        )
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
'''

CLOSE_ANCHOR = """    async def close(self) -> None:
        if self._conversation is not None:
            self._conversation.close()
        self._conversation = None
"""

CLOSE_REPLACEMENT = '''    async def close(self) -> None:
        if self._conversation is not None:
            # Also synchronous SDK HTTP. On the loop it made the caller's
            # CLOSE_TEARDOWN_TIMEOUT unenforceable, since the timer that would
            # fire it is serviced by the loop this call was blocking.
            conversation = self._conversation
            await _run_blocking_sdk_call(conversation.close)
        self._conversation = None
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No openhands env at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"SDK conversation i/o already off the loop in {target}")
        return
    anchors = (
        ("import", IMPORT_ANCHOR),
        ("helper", HELPER_ANCHOR),
        ("reset", RESET_ANCHOR),
        ("close", CLOSE_ANCHOR),
    )
    for name, anchor in anchors:
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Could not find a unique {name} anchor in {target}; "
                "apply this after openhands_execution_status.py, or review it "
                "against upstream."
            )
    source = source.replace(IMPORT_ANCHOR, IMPORT, 1)
    source = source.replace(HELPER_ANCHOR, HELPER, 1)
    source = source.replace(RESET_ANCHOR, RESET_REPLACEMENT, 1)
    source = source.replace(CLOSE_ANCHOR, CLOSE_REPLACEMENT, 1)
    target.write_text(source)
    print(f"moved the SDK's blocking conversation i/o off the event loop in {target}")


if __name__ == "__main__":
    main()
