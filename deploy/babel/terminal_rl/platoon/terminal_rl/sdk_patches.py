"""Two fixes to the openhands SDK client, installed together by ``apply_sdk_patches``.

1. Make the status poller give up on a server that is gone for good.

``RemoteConversation._handle_poll_exception`` classifies every
``httpx.RequestError`` as retryable and returns, so the run loop polls again a
half second later. That is right for a transient blip and wrong for a dead
process: when an in-sandbox agent-server exits, the loop retries forever. One
such rollout in job 10589983 logged "Error polling status (will retry): [Errno
111] Connection refused" at a steady two per second for twenty-five minutes,
straight through the optimiser phase, because ``conversation.run()`` drives
that loop on a worker thread that outlives the rollout that abandoned it.

Connection refused is not a blip. Nothing is listening on the port, and since
the server only ever binds once, nothing will start listening later. Escalating
after a few consecutive refusals turns the hang into a ``ConversationRunError``,
which ``run_episode`` records as the episode's error before returning the
trajectory -- so the rollout is still scored on what the agent achieved, and the
thread exits instead of leaking.

The threshold is consecutive, reset by any successful poll, so a refusal during
the server's own startup window cannot trip it.

2. Create every conversation with ``autotitle`` off.

``StartConversationRequest.autotitle`` defaults to True, and the agent-server
then titles the conversation from the first user message with the agent's own
LLM when no ``title_llm_profile`` is set. Here that LLM is the policy behind the
Tinker proxy, so the title request ("generate a concise title ... include ONE
most relevant emoji") is sampled from the policy, recorded as a rollout
interaction, and trained with the trajectory's advantage. Run 261007-133957-006
had one in 398 of its 2,427 prompt dumps, the reasoning alone running to ~500
tokens. A title is a UI label that nothing scores.

``RemoteConversation.__init__`` builds its creation payload inline and has no
``autotitle`` argument, so the flag is added where that payload is sent: the
module's ``_send_request``, for a POST to ``CONVERSATIONS_PATH`` that carries an
``agent``. Only titling changes. The task goal still reaches the agent as the
first user message via ``conversation.send_message``; the title subscriber only
ever read it.
"""

import logging

import httpx
from openhands.sdk.conversation.impl import remote_conversation
from openhands.sdk.conversation.impl.remote_conversation import (
    CONVERSATIONS_PATH,
    ConversationRunError,
    RemoteConversation,
)

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_REFUSALS = 5
_ATTRIBUTE = "_terminal_rl_refusals"
_ORIGINAL = RemoteConversation._handle_poll_exception
_ORIGINAL_SEND_REQUEST = remote_conversation._send_request


def _handle_poll_exception(self, exc: Exception) -> None:
    if isinstance(exc, httpx.ConnectError):
        refusals = getattr(self, _ATTRIBUTE, 0) + 1
        setattr(self, _ATTRIBUTE, refusals)
        if refusals >= MAX_CONSECUTIVE_REFUSALS:
            raise ConversationRunError(
                self._id,
                RuntimeError(
                    f"Agent server unreachable for {refusals} consecutive polls "
                    f"({exc}); treating it as gone."
                ),
            ) from exc
    else:
        setattr(self, _ATTRIBUTE, 0)
    _ORIGINAL(self, exc)


def _without_autotitle(payload):
    """The creation payload with autotitle off; any other request unchanged."""
    if isinstance(payload, dict) and "agent" in payload:
        return {**payload, "autotitle": False}
    return payload


def _send_request(client, method, url, *args, **kwargs):
    if method == "POST" and url == CONVERSATIONS_PATH and "json" in kwargs:
        kwargs["json"] = _without_autotitle(kwargs["json"])
    return _ORIGINAL_SEND_REQUEST(client, method, url, *args, **kwargs)


def _patch_poller() -> None:
    if getattr(RemoteConversation._handle_poll_exception, "_terminal_rl_patched", False):
        return
    _handle_poll_exception._terminal_rl_patched = True
    RemoteConversation._handle_poll_exception = _handle_poll_exception
    logger.info(
        "patched poller to give up after %d consecutive connection refusals",
        MAX_CONSECUTIVE_REFUSALS,
    )


def _patch_autotitle() -> None:
    if getattr(remote_conversation._send_request, "_terminal_rl_patched", False):
        return
    _send_request._terminal_rl_patched = True
    remote_conversation._send_request = _send_request
    logger.info("patched conversation creation to send autotitle=False")


def apply_sdk_patches() -> None:
    """Install both patches. Idempotent."""
    _patch_poller()
    _patch_autotitle()
