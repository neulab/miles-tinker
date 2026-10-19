"""Render the proxy's prompts with the model's own chat template.

The Tinker proxy turns openhands' LiteLLM messages into tokens with
tinker-cookbook's ``qwen3_5`` renderer, a hand port of Qwen3.5's
``chat_template.jinja`` that drifts from it wherever a rollout actually goes:

* An assistant turn with reasoning and a tool call but no visible text renders
  as ``</think>\\n\\n\\n\\n<tool_call>``. The template writes ``</think>\\n\\n``
  and adds ``\\n\\n`` before the first call only when there is text, which is
  also what the model generates. Every such turn reached the next prompt with two
  newlines the model never wrote; run 261009-013359-001 shows 524 of 525 in
  history and the policy drifting toward copying them (0% -> 10% of turns).
* Message content is not trimmed; the template trims every message.
* Consecutive tool results become one user turn per result; the template groups
  them into a single user turn.
* The tool list is dumped as the inner function object with ``json.dumps``; the
  template dumps the tool as given (``{"type": "function", "function": ...}``)
  with ``ensure_ascii=False``.
* Prior reasoning is kept for every assistant turn; the template drops it for
  turns before the last real user query.

Rather than correct the port, render with the template itself: build the
messages an OpenAI-compatible server (SGLang, vLLM) would receive for the same
conversation and call ``apply_chat_template``, then tokenize the text in one
piece as those servers do. Training then sees exactly the prompt the evaluation
serves, by construction.

Images are dropped from the messages. The proxy samples from token ids only, so
the template's ``<|vision_start|><|image_pad|><|vision_end|>`` placeholders would
reach the sampler with no pixels behind them. The renderer path dropped them too:
openhands sends ``image_url`` parts (its LLM is configured as gpt-4o, which it
treats as vision-capable) and the renderer only recognises ``image``. Only the
text parts of a message, such as openhands' image-output notice, remain.

Terminal escape sequences are stripped from tool results. The task images have
no tmux, so openhands falls back to its PTY terminal and returns the raw byte
stream: bracketed-paste toggles (``\\x1b[?2004l``/``h``) in 94-95% of terminal
outputs, plus colour and line-editing codes. terminal-bench drives its
containers through tmux, where openhands reads the rendered pane and none of
these appear. ``TERMINAL_RL_STRIP_ANSI=0`` keeps them.

Boolean tool-call arguments are written as the model writes them. The parser
turns ``<parameter=is_input>\ntrue`` into a JSON ``true``, and the template formats
a top-level value with ``string``, so history showed Python's ``True``: the first
history mismatch in 306 of 473 rollouts that had one (job 156019), each forcing
a new datum and showing the model a turn it did not write. Top-level booleans
are passed as ``"true"``/``"false"`` instead (nested ones already go through
``tojson``). ``TERMINAL_RL_JSON_BOOLEANS=0`` keeps the template's ``True``.

``TERMINAL_RL_HF_CHAT_TEMPLATE=0`` restores the renderer path. A conversation
this module cannot express (for example tool-call arguments that are a JSON list
or string) also falls back to the renderer rather than failing the rollout. That
call is then in the renderer's format, so every fallback is counted and logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import tinker

logger = logging.getLogger(__name__)

_fallbacks: dict[str, int] = {}
_announced = False

# CSI (colours, cursor moves, mode toggles), OSC (window titles), and the
# two-byte escapes (``\x1bc`` reset, ``\x1b>`` keypad mode).
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[0-Z\\-~])")


JSON_BOOLEANS = os.getenv("TERMINAL_RL_JSON_BOOLEANS", "1") != "0"
# TERMINAL_RL_REASONING_EFFORT: passed to the template as ``reasoning_effort``. Qwen3.8's
# template defaults to xhigh, which prepends "think carefully ... validate key
# assumptions, consider plausible alternatives" to the system prompt; medium adds no
# instruction, low asks for brief thinking. Unset keeps the template's default.
REASONING_EFFORT = os.getenv("TERMINAL_RL_REASONING_EFFORT", "")


class UnsupportedConversation(ValueError):
    """The conversation has a shape the chat template cannot render."""


def enabled() -> bool:
    return os.environ.get("TERMINAL_RL_HF_CHAT_TEMPLATE", "1") != "0"


def _plain(value: Any) -> Any:
    """Pydantic objects (LiteLLM, tinker-cookbook) -> plain dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _is_media(part: Any) -> bool:
    """The parts the template renders as vision placeholders."""
    if not isinstance(part, dict):
        return False
    return bool({"image", "image_url", "video"} & set(part)) or part.get("type") in ("image", "image_url", "video")


def _content(content: Any) -> Any:
    """Message content with image and video parts removed; strings unchanged."""
    if not isinstance(content, list):
        return content
    return [p for p in (_plain(p) for p in content) if not _is_media(p)]


def _strip_ansi(content: Any) -> Any:
    """Tool-result content without terminal escape sequences, as a tmux pane shows it."""
    if os.environ.get("TERMINAL_RL_STRIP_ANSI", "1") == "0":
        return content
    if isinstance(content, str):
        return _ANSI.sub("", content)
    if isinstance(content, list):
        return [
            {**p, "text": _ANSI.sub("", p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str) else p
            for p in content
        ]
    return content


def _thinking_text(part: Any) -> str | None:
    if isinstance(part, dict) and part.get("type") == "thinking":
        return str(part.get("thinking", ""))
    return None


def _arguments(raw: Any) -> dict:
    """Tool-call arguments as the mapping the template iterates.

    OpenAI-compatible servers parse the JSON string before templating, and the
    template formats each value itself (``tojson`` for objects and lists, plain
    ``string`` otherwise).
    """
    if isinstance(raw, dict):
        return _json_booleans(raw)
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise UnsupportedConversation(f"tool-call arguments are not JSON: {raw!r:.200}") from error
    if parsed is None:  # "null": a call without arguments
        return {}
    if not isinstance(parsed, dict):
        raise UnsupportedConversation(f"tool-call arguments are not a JSON object: {raw!r:.200}")
    return _json_booleans(parsed)


def _json_booleans(arguments: dict) -> dict:
    """Top-level booleans as the model writes them ("true"), not the template's str(True)."""
    if not JSON_BOOLEANS:
        return arguments
    return {key: ("true" if value else "false") if isinstance(value, bool) else value for key, value in arguments.items()}


def _tool_call(call: Any) -> dict:
    call = _plain(call)
    if not isinstance(call, dict):
        raise UnsupportedConversation(f"unexpected tool call {type(call).__name__}")
    function = _plain(call.get("function")) or {}
    return {
        "type": "function",
        "id": call.get("id"),
        "function": {"name": function.get("name", ""), "arguments": _arguments(function.get("arguments"))},
    }


def _assistant(message: dict) -> dict:
    content = _content(message.get("content"))
    reasoning = message.get("reasoning_content")
    # Reasoning that arrived folded into the content as thinking parts, or as
    # Anthropic-style thinking blocks, moves to the field the template reads.
    if isinstance(content, list):
        thinking = [t for t in (_thinking_text(p) for p in content) if t is not None]
        content = [p for p in content if _thinking_text(p) is None]
        if thinking and not isinstance(reasoning, str):
            reasoning = "".join(thinking)
    if not isinstance(reasoning, str):
        blocks = [t for t in (_thinking_text(_plain(b)) for b in message.get("thinking_blocks") or []) if t is not None]
        if blocks:
            reasoning = "".join(blocks)
    converted: dict[str, Any] = {"role": "assistant", "content": content}
    if isinstance(reasoning, str):
        converted["reasoning_content"] = reasoning
    if message.get("tool_calls"):
        converted["tool_calls"] = [_tool_call(call) for call in message["tool_calls"]]
    return converted


def to_template_messages(messages: Any) -> list[dict]:
    """LiteLLM / openhands messages -> the messages the chat template expects."""
    if not isinstance(messages, list):
        raise UnsupportedConversation(f"expected a list of messages, got {type(messages).__name__}")
    converted = []
    for message in messages:
        message = _plain(message)
        if not isinstance(message, dict):
            raise UnsupportedConversation(f"unexpected message {type(message).__name__}")
        role = message.get("role")
        if role == "assistant":
            converted.append(_assistant(message))
        elif role in ("tool", "function"):
            # The template knows no "function" role; it is the legacy name for a
            # tool result. Keys the template does not read (name, tool_call_id,
            # cache_control) are dropped, as a server's request schema would.
            converted.append({"role": "tool", "content": _strip_ansi(_content(message.get("content")))})
        elif role in ("system", "user"):
            converted.append({"role": role, "content": _content(message.get("content"))})
        else:
            raise UnsupportedConversation(f"unexpected role {role!r}")
    return converted


def to_template_tools(tools: Any) -> list[dict] | None:
    """Tool specs in the OpenAI request form, which the template dumps verbatim."""
    wrapped = []
    for tool in tools or []:
        tool = _plain(tool)
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            wrapped.append({"type": tool.get("type", "function"), "function": _plain(tool["function"])})
        elif "name" in tool:
            wrapped.append({"type": "function", "function": tool})
    return wrapped or None


def render_text(tokenizer: Any, messages: Any, tools: Any = None) -> str:
    """The generation prompt as text, exactly as the chat template renders it."""
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    return tokenizer.apply_chat_template(
        to_template_messages(messages),
        tools=to_template_tools(tools),
        add_generation_prompt=True,
        tokenize=False,
        **extra,
    )


def _fall_back(reason: str, detail: str) -> None:
    """Count a call that goes to the renderer; log the first and every 100th."""
    count = _fallbacks[reason] = _fallbacks.get(reason, 0) + 1
    if count == 1 or count % 100 == 0:
        logger.warning(
            "chat template could not render a prompt (%s, %d so far in this process): %s; "
            "this call uses the tinker-cookbook renderer",
            reason, count, detail,
        )


def prompt_from_chat_template(tokenizer: Any, messages: Any, tools: Any = None) -> tinker.ModelInput | None:
    """The generation prompt as a ModelInput, or None to use the renderer instead."""
    if not enabled():
        return None
    if getattr(tokenizer, "chat_template", None) is None:
        _fall_back("no_template", "the tokenizer has no chat template")
        return None
    try:
        text = render_text(tokenizer, messages, tools)
    except Exception as error:  # noqa: BLE001 -- any template failure falls back
        _fall_back(type(error).__name__, str(error)[:300])
        return None
    global _announced
    if not _announced:
        _announced = True
        logger.info("[render] prompts are rendered with the model's chat template (TERMINAL_RL_HF_CHAT_TEMPLATE=0 to disable)")
    # One encode over the whole text, as an OpenAI-compatible server does.
    return tinker.ModelInput.from_ints(tokenizer.encode(text, add_special_tokens=False))
