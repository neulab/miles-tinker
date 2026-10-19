"""Write out the exact prompt and response of a rollout's sampling calls.

Nothing in the pipeline records what the model was actually shown. The proxy
keeps each call in ``proxy_interactions`` -- ``obs`` is the ModelInput it was
given, ``action`` the tokens it produced -- but that dict is consumed to build
the training datums and then dropped, so a question like "is the history
rendered the way evaluation renders it" can only be answered by reasoning about
the code rather than by looking.

This dumps the decoded text for the first few calls of a rollout, which answers
three things a reader usually wants:

* whether assistant turns in history carry ``<think>\\n\\n</think>`` as the HF
  template writes them, and whether reasoning leaks into the visible text;
* whether tool results arrive as ``role: "tool"`` (rendered ``<|im_start|>user``
  with a ``<tool_response>`` wrapper) rather than folded into a user message;
* whether the model closes its think block at all -- ``</think>`` appeared in
  only 34 of 2114 recorded thoughts, and if it is rarely closed then everything
  the model emits is reasoning and nothing is output.

Off unless TERMINAL_RL_DUMP_PROMPTS is set to the number of calls to write, so
it costs nothing in a normal run.
"""

import logging
import os

logger = logging.getLogger("platoon.terminal_rl.prompt_dump")

DUMP_LIMIT = int(os.getenv("TERMINAL_RL_DUMP_PROMPTS", "0"))
# How much of the prompt to show. The whole thing is ~10k tokens of mostly
# tool schemas; every question about format lives in the last turns, so the
# tail is kept in full and the head only far enough to see the system turn.
HEAD_CHARS = int(os.getenv("TERMINAL_RL_DUMP_HEAD_CHARS", "300"))
TAIL_CHARS = int(os.getenv("TERMINAL_RL_DUMP_TAIL_CHARS", "1800"))
TAIL_TOKENS = int(os.getenv("TERMINAL_RL_DUMP_TAIL_TOKENS", "48"))


def _tokenizer():
    """The TinkerLLM's tokenizer, via the LiteLLM provider map it registers in."""
    try:
        import litellm

        for entry in getattr(litellm, "custom_provider_map", []) or []:
            handler = entry.get("custom_handler") if isinstance(entry, dict) else None
            tokenizer = getattr(handler, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
    except Exception as error:
        logger.debug("no tokenizer available for the prompt dump: %s", error)
    return None


def dump(interactions: dict, path: str, limit: int | None = None) -> None:
    """Write what the model received, what it generated, and what is trained on."""
    limit = DUMP_LIMIT if limit is None else limit
    if not limit or not interactions:
        return
    tokenizer = _tokenizer()
    if tokenizer is None:
        logger.warning("prompt dump skipped: no tokenizer reachable")
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        previous: list[int] | None = None
        with open(path, "w") as handle:
            for index, (completion_id, interaction) in enumerate(list(interactions.items())[:limit]):
                prompt = interaction.obs.to_ints()
                response = list(interaction.action.tokens)
                extends = previous is not None and prompt[: len(previous)] == previous
                handle.write(f"{'=' * 78}\n")
                handle.write(
                    f"call {index + 1}  id={completion_id}  prompt={len(prompt)} tok  "
                    f"response={len(response)} tok  extends_previous={extends}\n"
                )
                # Where a prompt stops extending the last one the text on both
                # sides usually reads the same and only the tokenisation differs
                # -- "<think>\n" then "\n</think>" is [198, ...] while
                # "<think>\n\n</think>" merges the newlines into [271, ...].
                # Decoded text cannot show that, so print the ids at the seam.
                if previous is not None and not extends:
                    seam = next(
                        (j for j, (a, b) in enumerate(zip(previous, prompt)) if a != b),
                        min(len(previous), len(prompt)),
                    )
                    handle.write(
                        f"--- SEAM: diverges at token {seam} of {len(previous)} ---\n"
                        f"  previous: {previous[seam:seam + 8]} -> {tokenizer.decode(previous[seam:seam + 8])!r}\n"
                        f"  current : {prompt[seam:seam + 8]} -> {tokenizer.decode(prompt[seam:seam + 8])!r}\n"
                    )
                text = tokenizer.decode(prompt)
                handle.write(f"--- RECEIVED, head ({HEAD_CHARS} chars) ---\n{text[:HEAD_CHARS]}\n")
                handle.write(
                    f"--- RECEIVED, tail ({TAIL_CHARS} chars: the recent turns) ---\n"
                    f"{text[-TAIL_CHARS:]}\n"
                )
                handle.write(
                    f"--- RECEIVED, last {TAIL_TOKENS} token ids (mask=0.0, not trained) ---\n"
                    f"{prompt[-TAIL_TOKENS:]}\n"
                )
                handle.write(
                    f"--- GENERATED = TRAINED ON (mask=1.0, carries the trajectory advantage) ---\n"
                    f"{tokenizer.decode(response)}\n"
                )
                handle.write(f"--- GENERATED token ids ---\n{response}\n")
                previous = prompt
        logger.info("wrote %d prompt/response pairs to %s", min(limit, len(interactions)), path)
    except Exception as error:  # diagnostics must never fail a rollout
        logger.warning("prompt dump failed: %s", error)
