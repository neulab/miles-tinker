"""Build the proxy's prompts with the model's own chat template.

``TinkerLLM._prepare_model_input`` renders openhands' messages with
tinker-cookbook's ``qwen3_5`` renderer, a port of Qwen3.5's chat template that
disagrees with it on real agentic turns. The visible one: an assistant turn with
reasoning and a tool call but no text renders as ``</think>\\n\\n\\n\\n<tool_call>``
in history although the model generated ``</think>\\n\\n<tool_call>``, so the policy
was trained on prompts the evaluation never shows and drifted toward copying the
extra newlines. ``platoon.terminal_rl.hf_chat_template`` lists the other
differences and renders with ``apply_chat_template`` instead; this patch routes
the prompt through it. Parsing the sampled response is unchanged.

Checked with repro/chat_template_check.py on 53 conversations from run
261009-013359-001: the new prompts equal ``apply_chat_template`` token for token;
the old ones differed by 610 tool-call separators, the tool declaration JSON in
every prompt, 51 untrimmed tool outputs and 3 ungrouped parallel tool results.
A turn the model generates in the template's own format now extends into the
next prompt 186/186 times, against 93/186 before, so far fewer datums restart
and re-encode the history.

One behaviour follows the template rather than the old renderer: a user message
in mid-episode (openhands' "Your last response did not include a function call"
nudge, or its image-output notice) is a new query to the template, which then
drops the reasoning of every earlier turn. The evaluation serves the same
template, so training now matches it there too, at the cost of one prefix break
at that turn.

The module returns None when ``TERMINAL_RL_HF_CHAT_TEMPLATE=0`` or when a
conversation cannot be expressed in the template's terms, and the original
renderer path runs.

Order: this must follow enable_tool_calls.py, which replaces the head of
_prepare_model_input that the anchor below matches. run.py restores the
checkout and applies the recipe's patches in list order on every launch, so
keep this one after it; listed earlier, enable_tool_calls cannot find its
anchor and the launch stops.

Usage: python hf_chat_template_prompt.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "prompt_from_chat_template"
ANCHOR = '''    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
'''
REPLACEMENT = ANCHOR + '''        # Render with the model's own chat template, so training sees the prompt an
        # OpenAI-compatible server would build for the same conversation; see
        # platoon.terminal_rl.hf_chat_template for where the renderer disagrees.
        from platoon.terminal_rl.hf_chat_template import prompt_from_chat_template

        templated = prompt_from_chat_template(
            self.tokenizer, messages, (kwargs.get("optional_params") or {}).get("tools")
        )
        if templated is not None:
            return templated
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()
    if MARKER in source:
        print(f"chat-template prompts already current in {proxy}")
        return
    if ANCHOR not in source:
        raise SystemExit(
            f"Could not find _prepare_model_input in {proxy}; "
            "Platoon changed upstream and this patch needs review."
        )
    proxy.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"prompts now rendered with the chat template in {proxy}")


if __name__ == "__main__":
    main()
