"""Check, per rollout, that each prompt extends what the model actually generated.

The proxy records every sampling call: ``obs`` is the exact prompt and
``action`` the exact tokens sampled. When history is rendered faithfully, call
N+1's prompt begins with call N's prompt followed by everything call N generated
-- reasoning, ``</think>``, the separator, any text, the tool call. Any
difference there is a turn the model is shown differently from how it wrote it,
which is what the cookbook renderer did to every reasoning-then-tool-call turn
(``</think>\\n\\n<tool_call>`` came back as ``</think>\\n\\n\\n\\n<tool_call>``).

At the end of each rollout this logs one line under ``[render]``:

* how many calls extend the previous call's prompt + response;
* where any mismatch sits inside the generated turn: in the reasoning, at
  ``</think>``, in the separator after it, in the visible text, in the tool
  call, or after the response, with the first one shown token by token;
* how many differ only in tokenisation (same text, different split), which
  every server's re-encoding of history produces and is not a fault;
* how many calls rewrote earlier history instead -- a condensation summary, the
  condenser's own call, or a mid-episode user message, after which the template
  drops all earlier reasoning;
* in the last prompt, the separator after every ``</think>`` and how many think
  blocks in history are empty, which is what history looks like if openhands
  stops echoing ``reasoning_content``;
* how many responses never closed ``</think>`` (usually cut at the token limit;
  their re-rendered turn cannot match).

Every 50 rollouts it also logs the process-wide totals. Set
``TERMINAL_RL_RENDER_CHECK=0`` to turn it off. It never raises into a rollout.
"""

from __future__ import annotations

import collections
import logging
import os

logger = logging.getLogger("platoon.terminal_rl.render_check")

ENABLED = os.getenv("TERMINAL_RL_RENDER_CHECK", "1") != "0"
SUMMARY_EVERY = int(os.getenv("TERMINAL_RL_RENDER_CHECK_SUMMARY_EVERY", "50"))
# Earlier calls a prompt is compared against: the condenser's calls interleave
# with the agent's, so the right predecessor is not always the one just before.
LOOKBACK = 4

_totals: collections.Counter = collections.Counter()
_rollouts = 0


class _Ids:
    def __init__(self, tokenizer):
        one = lambda text: tokenizer.encode(text, add_special_tokens=False)
        self.think, self.end_think = one("<think>")[0], one("</think>")[0]
        self.tool_call = one("<tool_call>")[0]
        self.nn, self.nnnn = one("\n\n"), one("\n\n\n\n")
        self.decode = lambda ids: tokenizer.decode(ids)


def _common_prefix(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix, by bisection over C-level slice compares."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _where_in_response(offset: int, response: list[int], ids: _Ids) -> str:
    """Name the part of a generated turn at which history stopped matching it."""
    if offset >= len(response):
        return "after the response (turn end / next header)"
    if ids.end_think not in response:
        return "in a response that never closed </think>"
    close = response.index(ids.end_think)
    if offset < close:
        return "inside the reasoning" if offset > 0 else "at the start of the reasoning"
    if offset == close:
        return "at </think> (reasoning whitespace or empty reasoning)"
    after = close + 1
    sep_end = after
    while sep_end < len(response) and ids.decode(response[sep_end : sep_end + 1]).strip() == "":
        sep_end += 1
    if offset < sep_end or offset == after:
        return "in the separator after </think>"
    if ids.tool_call in response[after:] and offset >= response.index(ids.tool_call, after):
        return "inside the tool call"
    return "in the visible text"


def _history_shape(prompt: list[int], ids: _Ids) -> collections.Counter:
    """Separators after </think> and empty think blocks in a prompt's history."""
    shape = collections.Counter()
    for i, token in enumerate(prompt):
        if token == ids.end_think:
            rest = prompt[i + 1 : i + 1 + len(ids.nnnn) + 1]
            if rest[: len(ids.nnnn)] == ids.nnnn:
                shape["</think>\\n\\n\\n\\n"] += 1
            elif rest[: len(ids.nn)] == ids.nn:
                follow = prompt[i + 1 + len(ids.nn) : i + 2 + len(ids.nn)]
                shape["</think>\\n\\n<tool_call>" if follow == [ids.tool_call] else "</think>\\n\\n+text"] += 1
            else:
                shape["</think>+other"] += 1
        elif token == ids.think:
            shape["think blocks"] += 1
            if prompt[i + 1 : i + 1 + len(ids.nn)] == ids.nn and prompt[i + 1 + len(ids.nn) : i + 2 + len(ids.nn)] == [ids.end_think]:
                shape["empty think blocks"] += 1
    # The prompt ends with the generation prefix "<think>\n", which is not history.
    shape["think blocks"] -= 1
    return shape


def check(interactions: dict, task_id: str, tokenizer) -> None:
    """Log how faithfully each call's history reproduces the model's own output."""
    global _rollouts
    if not ENABLED or not interactions or tokenizer is None:
        return
    try:
        ids = _Ids(tokenizer)
        calls = [(i.obs.to_ints(), list(i.action.tokens)) for i in interactions.values()]
        stats = collections.Counter()
        first_mismatch = None
        for n, (prompt, response) in enumerate(calls):
            stats["calls"] += 1
            if ids.end_think not in response:
                stats["unclosed"] += 1
            if n == 0:
                continue
            best = max(range(max(0, n - LOOKBACK), n), key=lambda j: _common_prefix(calls[j][0] + calls[j][1], prompt))
            prev_prompt, prev_response = calls[best]
            expected = prev_prompt + prev_response
            k = _common_prefix(expected, prompt)
            # The prompt ends with the generation prefix "<think>\n". History that
            # differs from the response at its very first token can retokenise
            # that newline with what follows ("<think>\n\n</think>" is <think>,
            # "\n\n", </think>), so a divergence inside the prefix belongs to the
            # response, not to earlier history.
            prefix_start = len(prev_prompt) - prev_prompt[::-1].index(ids.think) - 1 if ids.think in prev_prompt else len(prev_prompt)
            if k == len(expected):
                stats["extends"] += 1
            elif k <= prefix_start:
                stats["history rewritten"] += 1
            elif ids.decode(prompt[prefix_start:]).startswith(ids.decode(expected[prefix_start:])):
                # Same text, split into different tokens: re-encoding the turn
                # does not reproduce a sampled split ("etc" + ".)" vs "etc.)").
                # Every server re-tokenises history this way; nothing to fix.
                stats["retokenized"] += 1
            else:
                where = _where_in_response(max(0, k - len(prev_prompt)), prev_response, ids)
                stats[f"mismatch {where}"] += 1
                if first_mismatch is None:
                    first_mismatch = (
                        n + 1, where,
                        ids.decode(expected[max(prefix_start, k - 12) : k + 8]),
                        ids.decode(prompt[max(prefix_start, k - 12) : k + 8]),
                    )
        shape = _history_shape(calls[-1][0], ids)
        mismatches = {k[len("mismatch "):]: v for k, v in stats.items() if k.startswith("mismatch ")}
        line = (
            f"[render] {task_id}: {stats['extends']}/{stats['calls'] - 1} calls extend the previous prompt+response; "
            f"{sum(mismatches.values())} mismatch in a generated turn {dict(mismatches) or ''}; "
            f"{stats['retokenized']} same text re-tokenised; {stats['history rewritten']} rewrote history; "
            f"{stats['unclosed']} responses without </think>; "
            f"last prompt: {shape['think blocks']} think blocks, {shape['empty think blocks']} empty, separators "
            + str({k: v for k, v in shape.items() if k.startswith("</think>")})
        )
        if first_mismatch:
            call, where, generated, shown = first_mismatch
            line += f"; first mismatch call {call} {where}: generated {generated!r} history {shown!r}"
        suspicious = mismatches or shape["</think>\\n\\n\\n\\n"] or shape["empty think blocks"]
        (logger.warning if suspicious else logger.info)(line)
        _totals.update(stats)
        _totals["empty think blocks"] += shape["empty think blocks"]
        _totals["think blocks"] += shape["think blocks"]
        _rollouts += 1
        if SUMMARY_EVERY and _rollouts % SUMMARY_EVERY == 0:
            compared = _totals["calls"] - _rollouts
            mism = sum(v for k, v in _totals.items() if k.startswith("mismatch "))
            logger.info(
                "[render] totals over %d rollouts in this process: %d/%d calls extend (%.1f%%), %d mismatch in a "
                "generated turn, %d re-tokenised only, %d rewrote history, %d unclosed responses, %d/%d empty think "
                "blocks in last prompts",
                _rollouts, _totals["extends"], compared, 100.0 * _totals["extends"] / max(1, compared), mism,
                _totals["retokenized"], _totals["history rewritten"], _totals["unclosed"], _totals["empty think blocks"], _totals["think blocks"],
            )
    except Exception as error:  # diagnostics must never fail a rollout
        logger.warning("render check failed for %s: %s", task_id, error)
