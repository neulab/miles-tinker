"""Keep a thinking model's prior-turn reasoning in the rendered history.

tinker-cookbook's Qwen3 renderers default to ``strip_thinking_from_history=True``,
which matches the HF template for a chat turn: once a new *user* query arrives,
earlier reasoning is dropped. Their own docstring says to invert it for this
case -- "Use strip_thinking_from_history=False for multi-turn RL to get the
extension property" -- and an agentic rollout is precisely that case, because
the official Qwen3.5 template deliberately does not let a ``<tool_response>``
advance ``last_query_index``: every assistant turn after the human's one message
is supposed to keep its reasoning.

Without this, turn N's generated sequence is not a prefix of turn N+1's prompt.
Measured on a two-turn agentic exchange: 59% of the tokens shared with stripping
on, 100% with it off. The break costs throughput rather than correctness --
``_is_prefix`` in tinker_data_processing then fails, so every step becomes its
own datum re-encoding the whole history instead of extending one sequence, and
the optimiser is exactly linear in tokens.

``get_renderer`` builds the renderer positionally and offers no way to pass the
flag, so this sets it on the instance instead, which is what the renderer reads.

Usage: python retain_thinking_in_history.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "strip_thinking_from_history = False"
ANCHOR = """    renderer = get_renderer(renderer_name, tokenizer, image_processor=image_processor, model_name=model_name)
"""
REPLACEMENT = '''    renderer = get_renderer(renderer_name, tokenizer, image_processor=image_processor, model_name=model_name)
    # Multi-turn RL wants the extension property: prior-turn reasoning stays in
    # history, so what the model generated on turn N is a literal prefix of what
    # it is shown on turn N+1. The renderers that have no thinking to strip do
    # not define the attribute, hence the guard.
    if hasattr(renderer, "strip_thinking_from_history"):
        renderer.strip_thinking_from_history = False
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()
    if MARKER in source:
        print(f"thinking retention already current in {proxy}")
        return
    if ANCHOR not in source:
        raise SystemExit(
            f"Could not find the get_renderer call in {proxy}; "
            "Platoon changed upstream and this patch needs review."
        )
    proxy.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"retained thinking in history in {proxy}")


if __name__ == "__main__":
    main()
