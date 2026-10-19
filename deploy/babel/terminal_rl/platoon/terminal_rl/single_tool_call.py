"""Cap the tool calls in an assistant turn at TERMINAL_RL_MAX_TOOL_CALLS.

Openhands executes every ``<tool_call>`` block in a response, one after another,
and the whole batch costs one step of the conversation's iteration limit. In
156019 (step-limit rollouts dropped) the policy learned to write dozens of calls
per turn without waiting for their output: trajectories with a multi-call turn
went from 5% (v30) to 85% (v33) of a version, the responses ran into the
16384-token cap and were scored malformed (truncation was 42 of the 48 malformed
rollouts of v31-v33), and the run fell to ~10% solved before it pulled itself
back. See analysis/hypotheses.md, 2026-10-13 13:10.

With ``TERMINAL_RL_MAX_TOOL_CALLS=K`` (K >= 1) the sampler also stops on
``</tool_call>`` (a single token in the Qwen3.5 vocabulary). A turn that stopped
there with fewer than K calls is sampled further from where it stopped, with the
same sampling client and the remaining token budget, and the pieces are joined:
tokens and logprobs are exactly what the policy sampled, so the turn is an
ordinary sample that happens to end after its K-th call. Calls before that are
still written blind (their output comes back together), as they are now; only
their number is bounded. ``TERMINAL_RL_SINGLE_TOOL_CALL=1`` is K = 1.

The stop token is kept in the sampled tokens, as ``<|im_end|>`` is, so what the
policy generated is still a literal prefix of the next prompt (the chat template
closes the turn with ``<|im_end|>``, which lands in the observation and is not
trained on). The renderer only parses a response that ends on ``<|im_end|>``; one
that stopped on ``</tool_call>`` is parsed as if ``<|im_end|>`` followed.

Train and eval rollouts share the sampler, so both run under the cap.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("platoon.terminal_rl.single_tool_call")

CLOSE_TAG = "</tool_call>"
END_TAG = "<|im_end|>"

#: Token id of </tool_call>, set by install(); None while the cap is off.
_close: int | None = None


def max_calls() -> int:
    """K, the most tool calls a turn may carry; 0 means no cap."""
    if os.getenv("TERMINAL_RL_SINGLE_TOOL_CALL", "0") == "1":
        return 1
    return max(0, int(os.getenv("TERMINAL_RL_MAX_TOOL_CALLS", "0") or 0))


def enabled() -> bool:
    return max_calls() > 0


def _single_token(tokenizer: Any, text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"{text!r} is {len(tokens)} tokens for this tokenizer; capping tool calls needs one")
    return tokens[0]


def install(renderer: Any) -> None:
    """Add ``</tool_call>`` to the renderer's stop tokens and parse responses ending on it."""
    global _close
    if not enabled():
        return
    close = _single_token(renderer.tokenizer, CLOSE_TAG)
    end = _single_token(renderer.tokenizer, END_TAG)
    stops = list(renderer.get_stop_sequences())
    if not all(isinstance(stop, int) for stop in stops):
        raise ValueError(f"capping tool calls needs token-id stop sequences, the renderer has {stops!r}")
    if close not in stops:
        stops.append(close)
    parse_response = renderer.parse_response

    def parse_capped(response: list[int]):
        if response and response[-1] == close:
            return parse_response(list(response) + [end])
        return parse_response(response)

    renderer.get_stop_sequences = lambda: list(stops)
    renderer.parse_response = parse_capped
    _close = close
    logger.info("at most %d tool call(s) per turn: sampling stops on %s (token %d)", max_calls(), CLOSE_TAG, close)


async def sample(client: Any, prompt: Any, params: Any) -> Any:
    """``client.sample_async`` for one sequence, continued until the turn holds K calls or ends."""
    result = await client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
    k = max_calls()
    if _close is None or k <= 1:
        return result
    from tinker.types import ModelInput

    first = result.sequences[0]
    tokens = list(first.tokens)
    logprobs = None if first.logprobs is None else list(first.logprobs)
    stop_reason = first.stop_reason
    pieces = 1
    while tokens and tokens[-1] == _close and tokens.count(_close) < k:
        remaining = params.max_tokens - len(tokens)
        if remaining <= 0:
            break
        more = await client.sample_async(
            prompt=ModelInput.from_ints(prompt.to_ints() + tokens),
            sampling_params=params.model_copy(update={"max_tokens": remaining}),
            num_samples=1,
        )
        piece = more.sequences[0]
        tokens += piece.tokens
        if logprobs is not None:
            logprobs += list(piece.logprobs or [])
        stop_reason = piece.stop_reason
        pieces += 1
    if pieces == 1:
        return result
    joined = first.model_copy(update={"tokens": tokens, "logprobs": logprobs, "stop_reason": stop_reason})
    return result.model_copy(update={"sequences": [joined]})
