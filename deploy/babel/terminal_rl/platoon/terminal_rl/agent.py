"""The Vanillux agent: one `<bash>` command per turn, after a `<thought>`.

Platoon's ``CodeActAgent`` hardcodes the ``<python>`` tag in its stop sequence,
so this is a focused agent rather than a subclass. The conversation keeps
``CodeActPromptBuilder``'s sequence-extension shape -- each step's prompt is a
prefix of the next -- so Tinker can merge consecutive steps into fewer Datums.
"""

from __future__ import annotations

import re
from typing import cast

from openai.types.chat import ChatCompletionMessageParam
from platoon.config_defs import InferenceParams
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActAction, CodeActObservation
from platoon.utils.llm_client import ChatMessage, Conversation, LLMClient

from platoon.terminal_rl.prompts import LAST_STEP_WARNING, SYSTEM_TEMPLATE

_BOTH = re.compile(r"<thought>(.*?)</thought>\s*<bash>(.*?)</bash>", re.DOTALL)
_THOUGHT = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
_BASH = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
STOP_SEQUENCE = "</bash>"


def extract_command_and_thought(raw_action: str) -> tuple[str, str]:
    """Pull the command and the reasoning out of a response, in either order."""
    match = _BOTH.search(raw_action)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    thought = _THOUGHT.search(raw_action)
    command = _BASH.search(raw_action)
    return (command.group(1).strip() if command else ""), (thought.group(1).strip() if thought else "")


class VanilluxPromptBuilder:
    """Builds the sequence-extension conversation for the Vanillux harness."""

    def build_messages(self, obs: CodeActObservation) -> Conversation:
        messages: Conversation = [
            ChatMessage(role="system", content=SYSTEM_TEMPLATE),
            # obs.action_space is the rendered instance template: task plus workflow.
            ChatMessage(role="user", content=str(obs.action_space)),
        ]
        for index, step in enumerate(obs.history):
            messages.append(ChatMessage(role="assistant", content=self._format_action(step)))
            messages.append(ChatMessage(role="user", content=self._format_observation(obs, step, index)))
        return messages

    def _format_action(self, step) -> str:
        parts = []
        if step.thought:
            parts.append(f"<thought>\n{step.thought}\n</thought>")
        parts.append(f"<bash>\n{step.code or ''}\n</bash>")
        return "\n".join(parts)

    def _format_observation(self, obs: CodeActObservation, step, index: int) -> str:
        observation = step.output or step.error or "(no output)"
        max_steps = obs.task.max_steps if obs.task is not None else None
        if max_steps is not None and index == max_steps - 2:
            observation = f"{observation}\n\n{LAST_STEP_WARNING}"
        return observation


class VanilluxAgent:
    """Emits exactly one bash command per turn."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        inference_params: InferenceParams | None = None,
        prompt_builder: VanilluxPromptBuilder | None = None,
        request_timeout: int = 1800,
    ):
        self.llm_client = llm_client or LLMClient()
        self.inference_params = inference_params or InferenceParams()
        self.prompt_builder = prompt_builder or VanilluxPromptBuilder()
        self.request_timeout = request_timeout

    async def act(self, obs: CodeActObservation) -> CodeActAction:
        prompt = cast(list[ChatCompletionMessageParam], self.prompt_builder.build_messages(obs))
        request_kwargs = {
            "stop": [STOP_SEQUENCE],
            "max_completion_tokens": self.inference_params.max_completion_tokens,
            "timeout": self.request_timeout,
        }
        if self.inference_params.temperature is not None:
            request_kwargs["temperature"] = self.inference_params.temperature
        if self.inference_params.top_p is not None:
            request_kwargs["top_p"] = self.inference_params.top_p

        response = await self.llm_client.async_chat_completion(prompt, **request_kwargs)
        text = response.choices[0].message.content or ""
        # The stop sequence is trimmed from the response; put it back so the
        # action parses. Backends without stop-sequence support already have it.
        if STOP_SEQUENCE not in text:
            text += STOP_SEQUENCE
        command, thought = extract_command_and_thought(text)
        action = CodeActAction(action=text, parsed_code=command, parsed_thought=thought)
        action.misc["usage"] = response.usage.to_dict() if response.usage is not None else {}
        action.misc["model"] = response.model
        action.misc["completion_id"] = response.id
        return action

    async def reset(self) -> None:
        pass

    async def fork(self, task: Task) -> "VanilluxAgent":
        return type(self)(
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            prompt_builder=self.prompt_builder,
            request_timeout=self.request_timeout,
        )

    async def close(self) -> None:
        await self.llm_client.aclose()
