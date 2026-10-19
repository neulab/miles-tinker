"""Turn an inbound ``reasoning_content`` field back into a thinking part.

The two halves of the round trip do not meet. Going out, the proxy hands
openhands the reasoning in ``reasoning_content`` (separate_reasoning_content),
and openhands echoes it back on the next request once the model is listed as a
thinking model. Coming in, though, tinker-cookbook's ``render_message`` only
looks at ``message["content"]`` -- a list with a ``thinking`` part, or a string
containing a literal ``<think>``. Its ``to_openai_message`` writes
``reasoning_content`` but nothing on the inbound path ever reads it.

So the reasoning arrives and is dropped, ``_assistant_header_suffix`` finds no
thinking and inserts ``<think>\\n\\n</think>``, and turn N's generated tokens
stop being a prefix of turn N+1's prompt -- measured as a one-token seam,
``[198]`` for ``\\n`` against ``[271]`` for ``\\n\\n``.

Folding the field back into a thinking part closes the loop: the renderer then
keeps it (with strip_thinking_from_history off) and renders the turn exactly as
the HF template does, ``<think>\\n{reasoning}\\n</think>\\n\\n{text}``.

Usage: python reasoning_content_to_thinking.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_fold_reasoning_content"
ANCHOR = """    def _canonicalize_messages(self, messages: Any) -> List[TinkerMessage]:
"""
HELPER = '''    @staticmethod
    def _fold_reasoning_content(messages: Any) -> Any:
        """Rewrite assistant ``reasoning_content`` as a thinking content part.

        The renderer reads thinking from ``content`` only, so a message that
        carries its reasoning in the separate field would render with an empty
        think block and break the prefix the next turn extends.
        """
        folded = []
        seen = False
        for message in messages:
            if not isinstance(message, dict):
                folded.append(message)
                continue
            reasoning = message.get("reasoning_content")
            if message.get("role") != "assistant" or not reasoning:
                folded.append(message)
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts = [{"type": "text", "text": content}] if content else []
            elif isinstance(content, list):
                parts = [p for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
            else:
                parts = []
            message = {k: v for k, v in message.items() if k != "reasoning_content"}
            message["content"] = [{"type": "thinking", "thinking": str(reasoning)}, *parts]
            folded.append(message)
            seen = True
        # Whether the field survives LiteLLM between openhands and this proxy is
        # the one link in the chain that cannot be checked from the source, so
        # say so once per process rather than leaving it to be inferred from a
        # prompt dump.
        if seen and not getattr(TinkerLLM, "_reasoning_fold_logged", False):
            TinkerLLM._reasoning_fold_logged = True
            logger.info("reasoning_content is arriving from openhands and is being folded into <think>")
        return folded

'''
OLD_CALL = "        return cast(List[TinkerMessage], messages)\n"
NEW_CALL = "        return cast(List[TinkerMessage], self._fold_reasoning_content(messages))\n"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    proxy = Path(sys.argv[1]) / "platoon/train/tinker/proxy.py"
    if not proxy.is_file():
        raise SystemExit(f"No Platoon proxy at {proxy}")
    source = proxy.read_text()
    if MARKER in source:
        print(f"reasoning folding already current in {proxy}")
        return
    if ANCHOR not in source or OLD_CALL not in source:
        raise SystemExit(
            f"Could not find _canonicalize_messages in {proxy}; "
            "Platoon changed upstream and this patch needs review."
        )
    source = source.replace(ANCHOR, HELPER + ANCHOR, 1)
    source = source.replace(OLD_CALL, NEW_CALL, 1)
    proxy.write_text(source)
    print(f"folded reasoning_content into thinking parts in {proxy}")


if __name__ == "__main__":
    main()
