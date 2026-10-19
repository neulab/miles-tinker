"""Return a thinking model's reasoning as ``reasoning_content``, not as reply text.

tinker-cookbook's thinking renderers (qwen3_5, qwen3, ...) parse a response into
``thinking`` and ``text`` parts. Platoon's proxy flattens both into one plain
``content`` string with the ``<think>`` tags dropped, so the agent receives the
reasoning as if it were the visible reply. openhands then echoes it back as the
assistant turn in history, and the Qwen3.5 renderer, finding no ``<think>`` in
that string, prepends an empty think block. Every earlier turn therefore reaches
the model as::

    <think>\\n\\n</think>\\n\\n{reasoning}{text}<tool_call>...

That is a history layout the model never sees at evaluation. There vLLM runs
with ``--reasoning-parser qwen3``, the reasoning arrives in
``reasoning_content``, openhands does not resend it for an ``openai/...`` model
(it is not in SEND_REASONING_CONTENT_MODELS), and the HF template renders the
earlier turn as ``<think>\\n\\n</think>\\n\\n{text}``.

This patch makes the proxy behave like that server: thinking parts go to
``reasoning_content``, only text parts go to ``content``. openhands records the
reasoning on its events, drops it from the next request, and training sees the
same history format as evaluation.

The visible text is also trimmed, as the HF template trims it; otherwise the
newlines before a tool call are doubled in history.

A response that fails to parse (typically truncated at max_tokens before the
turn closes) keeps the upstream behaviour: raw text in ``content``.

Usage: python separate_reasoning_content.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_split_reasoning"

# Upstream success branch of _parse_response, content through the Choices call.
ANCHOR = '''                # FIXME: The content should not be still there if tool call has been parsed.
                content = self._normalize_message_content(parsed_response["content"])
                # NOTE(yuge): I thought about adding this to make it more robust to empty responses,
                # but later I found it's a configuration error in my renderer. So I think it's better
                # to just log a warning and go with the default path.
                # if not content:
                #     raise ValueError("Parsed content is empty. Original response: " + str(response))
                if not content:
                    logger.warning("Parsed content is empty. Original response: " + str(response))
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]
                choices.append(
                    Choices(
                        message=LitellmMessage(role=role, content=content, tool_calls=tool_calls),
'''

REPLACEMENT = '''                # Thinking parts go to reasoning_content, as an OpenAI-compatible
                # server with a reasoning parser returns them; flattening them into
                # content would feed the reasoning back as visible reply text.
                content, reasoning = self._split_reasoning(parsed_response["content"])
                tool_calls = parsed_response.get("tool_calls", None)
                if not content and not reasoning and not tool_calls:
                    logger.warning("Parsed content is empty. Original response: " + str(response))
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]
                choices.append(
                    Choices(
                        message=LitellmMessage(
                            role=role, content=content, tool_calls=tool_calls, reasoning_content=reasoning
                        ),
'''

# Helper inserted ahead of _normalize_message_content.
HELPER_ANCHOR = "    def _normalize_message_content(self, content: Any) -> str:\n"
HELPER = '''    def _split_reasoning(self, content: Any) -> tuple[str, str | None]:
        """Parsed content -> (visible text, reasoning or None).

        The text is trimmed, as the HF template trims assistant content: the
        parser keeps the newlines that preceded a tool call, and the renderer
        adds its own two when it formats the call back into history.
        """
        if not isinstance(content, list):
            return self._normalize_message_content(content).strip(), None
        thinking = [p for p in content if isinstance(p, dict) and p.get("type") == "thinking"]
        rest = [p for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
        reasoning = "".join(str(p.get("thinking", "")) for p in thinking)
        return self._normalize_message_content(rest).strip(), (reasoning or None)

'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()

    if MARKER in source:
        if REPLACEMENT in source and HELPER in source:
            print(f"reasoning separation already current in {proxy}")
            return
        raise SystemExit(f"{proxy} has an unrecognised version of this patch; review it by hand.")

    if ANCHOR not in source or HELPER_ANCHOR not in source:
        raise SystemExit(
            f"Could not find _parse_response's content handling in {proxy}; "
            "Platoon changed upstream and this patch needs review."
        )
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    source = source.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    proxy.write_text(source)
    print(f"separated reasoning_content in {proxy}")


if __name__ == "__main__":
    main()
