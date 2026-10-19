"""Parse the model's tool calls with SGLang's Qwen3Coder semantics.

tinker-cookbook's Qwen3.5 parser strips all leading and trailing whitespace from
every tool-call parameter, so openhands executed edits the model did not write
(first-line indentation of ``new_str`` gone, ``file_text`` without its trailing
newline) and history showed the altered call. ``platoon.terminal_rl.
qwen3_tool_parser`` reproduces SGLang's parser instead -- one newline trimmed on
each side, values typed by the tool schema -- and this patch installs it:

* on the renderer register_tinker_llm builds, replacing
  ``_parse_qwen3_5_tool_call_xml``;
* in ``_prepare_model_input``, recording the request's tools so the response of
  the same call is typed against them.

Independent of hf_chat_template_prompt.py, but order-sensitive in the same way:
it must follow enable_tool_calls.py, whose rewrite of _prepare_model_input keeps
the ``messages = kwargs.pop`` line this anchors on.

Usage: python sglang_tool_call_parser.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "qwen3_tool_parser"
REGISTER_ANCHOR = "    for key, value in (renderer_kwargs or {}).items():\n"
REGISTER_REPLACEMENT = '''    # Parse tool calls as SGLang does; see platoon.terminal_rl.qwen3_tool_parser.
    from platoon.terminal_rl import qwen3_tool_parser

    qwen3_tool_parser.install(renderer)
''' + REGISTER_ANCHOR
PROMPT_ANCHOR = '        messages = kwargs.pop("messages", None)\n'
PROMPT_REPLACEMENT = PROMPT_ANCHOR + '''        # The response of this call is typed against the tools it was offered.
        from platoon.terminal_rl import qwen3_tool_parser

        qwen3_tool_parser.set_tools((kwargs.get("optional_params") or {}).get("tools"))
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()
    if MARKER in source:
        print(f"SGLang tool-call parsing already current in {proxy}")
        return
    for anchor, what in ((REGISTER_ANCHOR, "the renderer_kwargs loop"), (PROMPT_ANCHOR, "_prepare_model_input")):
        if source.count(anchor) != 1:
            raise SystemExit(f"Could not find {what} in {proxy}; Platoon changed upstream and this patch needs review.")
    source = source.replace(REGISTER_ANCHOR, REGISTER_REPLACEMENT, 1).replace(PROMPT_ANCHOR, PROMPT_REPLACEMENT, 1)
    proxy.write_text(source)
    print(f"tool calls now parsed with SGLang's Qwen3Coder semantics in {proxy}")


if __name__ == "__main__":
    main()
