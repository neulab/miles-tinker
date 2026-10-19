"""Cap the tool calls per assistant turn (platoon.terminal_rl.single_tool_call).

Under TERMINAL_RL_MAX_TOOL_CALLS=K (or TERMINAL_RL_SINGLE_TOOL_CALL=1, K = 1) the
sampler stops on ``</tool_call>`` as well as on ``<|im_end|>``, and a turn with
fewer than K calls is sampled on from where it stopped (see the module for why).
Unset, both hooks are no-ops. This patch:

* installs the stop token and the parser on the renderer register_tinker_llm
  builds;
* routes TinkerLLM.acompletion's sampling through single_tool_call.sample.

Order-sensitive: it must follow sglang_tool_call_parser.py, whose install it
anchors on.

Usage: python single_tool_call.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "single_tool_call"
INSTALL_ANCHOR = "    qwen3_tool_parser.install(renderer)\n"
INSTALL_REPLACEMENT = INSTALL_ANCHOR + '''    # TERMINAL_RL_MAX_TOOL_CALLS: at most K tool calls per turn.
    from platoon.terminal_rl import single_tool_call

    single_tool_call.install(renderer)
'''
SAMPLE_ANCHOR = "                self.sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1),\n"
SAMPLE_REPLACEMENT = (
    "                # TERMINAL_RL_MAX_TOOL_CALLS: sampled on until the turn holds K calls or ends.\n"
    "                __import__(\"platoon.terminal_rl.single_tool_call\", fromlist=[\"sample\"]).sample(\n"
    "                    self.sampling_client, model_input, params\n"
    "                ),\n"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()
    if MARKER in source:
        print(f"tool-call cap already present in {proxy}")
        return
    if source.count(INSTALL_ANCHOR) != 1:
        raise SystemExit(f"Could not find qwen3_tool_parser.install in {proxy}; apply sglang_tool_call_parser.py first.")
    if source.count(SAMPLE_ANCHOR) != 1:
        raise SystemExit(f"Could not find acompletion's sample_async call in {proxy}; this patch needs review.")
    source = source.replace(INSTALL_ANCHOR, INSTALL_REPLACEMENT, 1).replace(SAMPLE_ANCHOR, SAMPLE_REPLACEMENT, 1)
    proxy.write_text(source)
    print(f"TERMINAL_RL_MAX_TOOL_CALLS available in {proxy}")


if __name__ == "__main__":
    main()
