"""Let Platoon's Tinker proxy declare tools to the model, and render them back.

Platoon renders a generation prompt with ``build_generation_prompt(messages)``
and never passes the request's tools, so a tool schema never reaches the model
even though the proxy parses tool calls back out of the response. Agents that
rely on tool calling -- openhands-sdk, among others -- therefore see no tools at
all and answer in prose.

tinker-cookbook already supports this: ``Renderer.create_conversation_prefix_with_tools``
turns ToolSpecs into the messages a conversation should start with, and
Qwen3Renderer, Qwen3_5Renderer, GptOssRenderer and DeepSeekV3ThinkingRenderer
implement it. This patch wires the two together, and fixes three things that
only surface once tools actually flow:

* LiteLLM nests the caller's ``tools`` under ``optional_params``; reading
  ``kwargs["tools"]`` silently yields None and the patch does nothing.
* openhands sends message content as a list of typed parts, while the renderer
  concatenates the system prompt as a string.
* LiteLLM returns history as JSON, so a previous assistant turn's ``tool_calls``
  arrive as dicts, while the renderer formats them by attribute access.

It is applied to the Platoon checkout at launch because the change belongs to
Platoon core, not to a plugin. It is idempotent: re-running replaces whatever
version is already in place, so an older patched checkout is upgraded rather
than skipped. Upstream marks the same line with "TODO: Needs to be updated for
latest tinker cookbook version".

Usage: python enable_tool_calls.py <platoon-checkout>
"""

import sys
from pathlib import Path

# The upstream method, replaced wholesale.
ANCHOR = '''    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
        canonical_messages = self._canonicalize_messages(messages)
        # TODO: Needs to be updated for latest tinker cookbook version.
        return self.renderer.build_generation_prompt(canonical_messages)
'''

REPLACEMENT = '''    @staticmethod
    def _message_text(content: Any) -> str:
        """A message's text, whether it is a plain string or content parts."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return "" if content is None else str(content)

    @staticmethod
    def _canonicalize_tool_calls(messages: Any) -> Any:
        """Make a LiteLLM message history renderable by tinker-cookbook.

        Two mismatches, both only reachable once tool calling is on:

        * The renderer formats a past tool call by attribute access
          (``tool_call.function.name``), but LiteLLM returns history as JSON, so
          tool_calls arrive as plain dicts.
        * The renderer reads ``message["content"]`` directly, while an assistant
          turn that consists solely of tool calls carries no content key at all.
        """
        from tinker_cookbook.renderers import ToolCall as _ToolCall

        canonical = []
        for message in messages:
            if isinstance(message, dict) and message.get("content") is None:
                message = {**message, "content": ""}
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if not calls:
                canonical.append(message)
                continue
            converted = []
            for call in calls:
                if not isinstance(call, dict):
                    converted.append(call)
                    continue
                body = call.get("function") or {}
                converted.append(
                    _ToolCall(
                        type="function",
                        id=call.get("id"),
                        function=_ToolCall.FunctionBody(
                            name=body.get("name", ""),
                            arguments=body.get("arguments") or "{}",
                        ),
                    )
                )
            canonical.append({**message, "tool_calls": converted})
        return canonical

    @staticmethod
    def _tool_specs(tools: Any) -> list[dict]:
        """OpenAI tool dicts -> tinker-cookbook ToolSpec.

        ToolSpec is the inner ``function`` object: name, description, parameters.
        Accepts both the wrapped ``{"type": "function", "function": {...}}`` form
        that LiteLLM passes and an already-flat spec.
        """
        specs = []
        for tool in tools or []:
            spec = tool.get("function", tool) if isinstance(tool, dict) else tool
            if not isinstance(spec, dict) or "name" not in spec:
                continue
            specs.append(
                {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                }
            )
        return specs

    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
        canonical_messages = self._canonicalize_tool_calls(self._canonicalize_messages(messages))
        # LiteLLM does not hand a custom provider the caller's `tools` as a
        # top-level kwarg: tools, max_tokens, temperature and stop all arrive
        # nested under `optional_params`.
        optional_params = kwargs.get("optional_params") or {}
        specs = self._tool_specs(optional_params.get("tools"))
        if specs:
            # The renderer builds its own system message from the tools, so hand
            # it the caller's system prompt and drop the original.
            system_prompt = ""
            body = canonical_messages
            if body and body[0].get("role") == "system":
                system_prompt = self._message_text(body[0].get("content"))
                body = body[1:]
            try:
                prefix = self.renderer.create_conversation_prefix_with_tools(specs, system_prompt)
                canonical_messages = list(prefix) + list(body)
            except NotImplementedError:
                logger.warning(
                    "Renderer %s does not support tool calling; sending %d tools as prose context.",
                    type(self.renderer).__name__,
                    len(specs),
                )
        return self.renderer.build_generation_prompt(canonical_messages)
'''

# Where a previously applied version of this patch starts and ends. Any earlier
# revision begins with one of these and ends at build_generation_prompt.
PATCHED_STARTS = (
    "    @staticmethod\n    def _message_text(content: Any) -> str:",
    "    @staticmethod\n    def _canonicalize_tool_calls(",
    "    @staticmethod\n    def _tool_specs(tools: Any) -> list[dict]:",
)
PATCHED_END = "        return self.renderer.build_generation_prompt(canonical_messages)\n"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()

    if "_tool_specs" in source:
        starts = [source.index(marker) for marker in PATCHED_STARTS if marker in source]
        if not starts:
            raise SystemExit(f"{proxy} looks patched but has no recognisable start marker; review it by hand.")
        start = min(starts)
        end = source.index(PATCHED_END, start) + len(PATCHED_END)
        if source[start:end] == REPLACEMENT:
            print(f"tool calling already current in {proxy}")
            return
        proxy.write_text(source[:start] + REPLACEMENT + source[end:])
        print(f"refreshed tool calling in {proxy}")
        return

    if ANCHOR not in source:
        raise SystemExit(
            f"Could not find _prepare_model_input in {proxy}; Platoon changed upstream and this patch needs review."
        )
    proxy.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"enabled tool calling in {proxy}")


if __name__ == "__main__":
    main()
