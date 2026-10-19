"""The LiteCoder terminal environment.

``CodeActEnv`` already runs the action, records the trajectory step and calls
``evaluate`` after each one; all this adds is the reward, which exists only once
the agent has submitted and the verifier has run inside the sandbox.
"""

from __future__ import annotations

from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv

from platoon.terminal_rl.executor import ApptainerBashExecutor


class LiteCoderTerminalEnv(CodeActEnv):
    """A terminal task scored by its own verifier at submission time."""

    def __init__(self, task: Task, code_executor: ApptainerBashExecutor | None = None, **kwargs):
        super().__init__(task=task, code_executor=code_executor or ApptainerBashExecutor(task), **kwargs)

    async def evaluate(self) -> tuple[float, dict]:
        executor = self.code_executor
        assert isinstance(executor, ApptainerBashExecutor)
        return executor.take_reward(), {"submitted": executor.submitted}
