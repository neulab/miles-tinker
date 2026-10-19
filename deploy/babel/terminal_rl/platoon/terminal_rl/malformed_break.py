"""Where a malformed final response broke, for targeted malformed credit.

Under TERMINAL_RL_MALFORMED_CREDIT=targeted a malformed final response no longer
costs its whole trajectory -1 (that -1, spread over ~20k action tokens, left the
token that broke the call with 1/20,000 of the signal and pushed down the normal
work before it). Instead the trajectory is scored on the task, and a fixed
penalty goes on the tokens where the response broke. This module finds them.

``penalty_span`` takes the final response as per-token text (thinking, then
``</think>``, then the content, then ``<|im_end|>`` when the turn ended on the
stop token) and returns the failure kind and the token range [a, b) to penalise:

* in_thinking   the call was opened inside the reasoning: the ``<tool_call>`` (or
                ``<function=``) before ``</think>`` -> THINKING_SPAN tokens from it
* fake_output   tool output written inside the open call (``</tool_response>``,
                ``[The command completed``, openhands' bracketed notices, ANSI
                escapes, ``<|im_start|>``, a stray ``</think>``, a literal
                ``<|endoftext|>``) -> SPAN tokens from the first marker
* loop          the content ends in repetition (period detection with digits
                normalised, so counting loops match) -> SPAN tokens from the onset
* unclosed_stop a call opened and never closed and the turn ended: only the stop
                token -- the decision to end the turn with the call open
* other         none of the above (e.g. cut at the token cap) -> the last
                FALLBACK_SPAN tokens

The kinds and spans were checked by hand on 155806's 35 malformed endings
(analysis/malformed_break.py).
"""

from __future__ import annotations

import os
import re

STOP_TOKEN = "<|im_end|>"
SPAN = int(os.getenv("TERMINAL_RL_MALFORMED_SPAN", "64"))
THINKING_SPAN = int(os.getenv("TERMINAL_RL_MALFORMED_THINKING_SPAN", "4"))
FALLBACK_SPAN = int(os.getenv("TERMINAL_RL_MALFORMED_FALLBACK_SPAN", "32"))

_MARKERS = re.compile(
    "|".join(
        [
            r"</?tool_response>",
            r"\[The command completed",
            r"\[Command finished",
            r"\[Current working directory",
            r"<\|im_start\|>",
            r"\x1b\[",
            r"</think>",
            r"<response clipped>",
            r"\[An error occurred",
            r"<\|endoftext\|>",
            r"\n\[[A-Z][a-z]+ [^\]\n]{20,}\]\n",  # "[Your command ... timed out ...]"
        ]
    )
)


def loop_onset(text: str, min_run: int = 120, max_period: int = 160) -> int | None:
    """Character offset where a trailing repetition starts (digits normalised), or None."""
    s = re.sub(r"\d", "0", text)
    best: tuple[int, int] | None = None
    for period in range(1, max_period + 1):
        run, i = 0, len(s) - 1
        while i - period >= 0 and s[i] == s[i - period]:
            run += 1
            i -= 1
        if run >= max(min_run, 4 * period) and (best is None or run > best[1]):
            best = (period, run)
    if best is None:
        return None
    period, run = best
    return len(s) - run - period


def _token_at(starts: list[int], char: int) -> int:
    """Index of the token whose text contains character offset ``char``."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= char:
            lo = mid
        else:
            hi = mid - 1
    return lo


def penalty_span(pieces: list[str]) -> tuple[str, int, int]:
    """(kind, a, b): the failure kind and the token range [a, b) to penalise."""
    n = len(pieces)
    if n == 0:
        return "other", 0, 0
    starts, offset = [], 0
    for piece in pieces:
        starts.append(offset)
        offset += len(piece)
    text = "".join(pieces)
    ends_with_stop = pieces[-1] == STOP_TOKEN
    body_end = len(text) - (len(STOP_TOKEN) if ends_with_stop else 0)

    def from_char(kind: str, char: int, width: int) -> tuple[str, int, int]:
        a = _token_at(starts, char)
        return kind, a, min(n, a + width)

    think_end = text.find("</think>")
    reasoning_end = think_end if think_end >= 0 else body_end
    opener = min((i for i in (text.find("<tool_call>", 0, reasoning_end), text.find("<function=", 0, reasoning_end)) if i >= 0), default=-1)
    if opener >= 0:
        return from_char("in_thinking", opener, THINKING_SPAN)
    if think_end < 0:  # all reasoning, no call: nothing more specific to point at
        return "other", max(0, n - FALLBACK_SPAN), n

    content_start = think_end + len("</think>")
    content = text[content_start:body_end]
    opened = content.rfind("<tool_call>")
    region = content_start + max(opened, 0)
    marker = _MARKERS.search(text, region, body_end)
    if marker:
        return from_char("fake_output", marker.start(), SPAN)
    onset = loop_onset(content)
    if onset is not None and content_start + onset >= region:
        return from_char("loop", content_start + onset, SPAN)
    if ends_with_stop and content.count("<tool_call>") > content.count("</tool_call>"):
        return "unclosed_stop", n - 1, n
    return "other", max(0, n - FALLBACK_SPAN), n
