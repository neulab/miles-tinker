"""Expose Platoon's in-process Tinker proxy over HTTP, per rollout.

Platoon samples through ``TinkerLLM``, a LiteLLM ``CustomLLM`` registered in
``litellm.custom_provider_map``. It records each sampled sequence into
``proxy_interactions`` -- a ContextVar dict that
``group_rollout_workflow`` enters once per rollout -- and the training step
later joins those interactions to trajectory steps by completion id.

Both halves of that design assume the agent runs in the client process. It
cannot: openhands' TerminalTool spawns a local shell rather than routing
through the workspace, so an agent driving a sandboxed task has to run inside
the sandbox, where the only way to reach a model is HTTP.

This bridges the two. It serves an OpenAI-compatible endpoint in the client
process; a rollout leases a token, hands it to the in-container agent as an API
key, and every request bearing that token is answered by ``TinkerLLM`` with
``proxy_interactions`` pointed at *that rollout's* dict. The recording is
therefore identical to an in-process call, and the sampled tokens and logprobs
land where the trainer expects them.

The server binds loopback: Apptainer shares the host's network namespace, so
the sandbox reaches 127.0.0.1 without opening anything to the cluster.

Every completion's ending is also classified here, where the sampler's stop
reason and token ids are still at hand (the rollout files keep neither): why
sampling stopped ("stop" or "length"), the last token, and whether a tool call
was left open. A completion that leaves a call open or hits the length limit is
logged with its rollout's label, and a tally of all endings every 500
completions, so a malformed ending can be told apart as an end-of-turn token
sampled mid-call, a stop string, or the token cap. With ``TERMINAL_RL_STOP_LOG``
set, every completion's ending is also appended there as one JSON line.

A lease can also collect the number of tokens in each of its rollout's
completions (``response_tokens``), which the reward's output-length penalty
reads (openhands_env.py, TERMINAL_RL_LENGTH_PENALTY_N).
"""

# No `from __future__ import annotations` here on purpose: FastAPI resolves a
# route's annotations against module globals, and `Request` is imported inside
# _build_app so that the Vanillux recipes never need fastapi. Under PEP 563 the
# annotation would be the unresolvable string "Request" and FastAPI would treat
# the parameter as a query field, answering every completion with a 422.

import asyncio
import collections
import contextlib
import json
import logging
import os
import secrets
import socket
import threading
import time
from typing import Any, Iterator

logger = logging.getLogger("platoon.terminal_rl.proxy_bridge")

# Parameters LiteLLM would reject or that belong to the caller's own transport
# rather than to sampling. The model is replaced with Platoon's provider string.
_DROPPED_REQUEST_KEYS = frozenset({"model", "stream", "stream_options", "user", "api_key", "base_url"})

_STOP_TALLY_EVERY = 500


def _tokenizer() -> Any:
    """TinkerLLM's tokenizer, or None when it cannot be reached."""
    import litellm

    for provider in litellm.custom_provider_map or []:
        renderer = getattr(provider.get("custom_handler"), "renderer", None)
        if getattr(renderer, "tokenizer", None) is not None:
            return renderer.tokenizer
    return None


class StopTally:
    """How completions end: stop reason x last token x open tool call."""

    def __init__(self) -> None:
        self.counts: collections.Counter = collections.Counter()
        self.total = 0
        path = os.environ.get("TERMINAL_RL_STOP_LOG")
        self._file = open(path, "a", buffering=1) if path else None

    def observe(self, response: dict, label: str) -> None:
        choice = (response.get("choices") or [{}])[0]
        reason = choice.get("finish_reason")
        tokens = choice.get("token_ids") or []
        tokenizer = _tokenizer()
        if tokenizer is not None and tokens:
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            last = tokenizer.decode(tokens[-1:], skip_special_tokens=False)
        else:
            message = choice.get("message") or {}
            text, last = f"{message.get('reasoning_content') or ''}{message.get('content') or ''}", str(tokens[-1:])
        open_call = text.count("<tool_call>") > text.count("</tool_call>")
        self.total += 1
        self.counts[(reason, last if last.startswith("<|") else "text", "open call" if open_call else "closed")] += 1
        if self._file is not None:
            self._file.write(json.dumps({
                "time": time.time(), "rollout": label, "reason": reason, "last_token": last,
                "tokens": len(tokens), "open_tool_call": open_call, "tail": text[-200:],
            }) + "\n")
        if open_call or reason == "length":
            logger.warning(
                "[stop] %s: reason=%s last_token=%r tokens=%d open_tool_call=%s tail=%r",
                label, reason, last, len(tokens), open_call, text[-160:],
            )
        if self.total % _STOP_TALLY_EVERY == 0:
            tally = ", ".join(f"{r}/{t}/{c}={n}" for (r, t, c), n in self.counts.most_common())
            logger.info("[stop] %d completions: %s", self.total, tally)


def completion_tokens(response: dict) -> int:
    """Tokens the model sampled for one completion, reasoning included."""
    choice = (response.get("choices") or [{}])[0]
    tokens = choice.get("token_ids")
    if tokens:
        return len(tokens)
    return int((response.get("usage") or {}).get("completion_tokens") or 0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TinkerProxyBridge:
    """One HTTP front-end for the process's TinkerLLM, shared by all rollouts."""

    def __init__(self, model_name: str, host: str = "127.0.0.1", port: int | None = None):
        #: Platoon's provider-qualified name, e.g. "platoon-tinker/Qwen/Qwen3.5-4B".
        self.model_name = model_name
        self.host = host
        self.port = port or _free_port()
        # Leases are read from the bridge's thread and written from the
        # rollouts', so the dict must not be rebuilt -- only mutated in place.
        self._leases: dict[str, dict] = {}
        self._labels: dict[str, str] = {}
        self._response_tokens: dict[str, list[int]] = {}
        self._stops = StopTally()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _build_app(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(docs_url=None, redoc_url=None)

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "leases": len(self._leases)}

        @app.get("/v1/models")
        async def models() -> dict:
            # Some clients probe this before their first completion.
            return {"object": "list", "data": [{"id": self.model_name, "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            token = self._token_from_headers(dict(request.headers))
            interactions = self._leases.get(token or "")
            if interactions is None:
                # Never echo the token: it is the rollout's only credential.
                return JSONResponse({"error": {"message": "unknown or expired rollout lease"}}, status_code=401)
            body = await request.json()
            try:
                response = await self._sample(body, interactions)
                lengths = self._response_tokens.get(token or "")
                if lengths is not None:
                    lengths.append(completion_tokens(response))
                with contextlib.suppress(Exception):  # diagnostics never fail a completion
                    self._stops.observe(response, self._labels.get(token or "", "?"))
            except Exception as error:
                logger.exception("bridge sampling failed")
                return JSONResponse(
                    {"error": {"message": str(error), "type": type(error).__name__}}, status_code=502
                )
            return JSONResponse(response)

        return app

    @staticmethod
    def _token_from_headers(headers: dict[str, str]) -> str | None:
        authorization = headers.get("authorization") or headers.get("Authorization") or ""
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return headers.get("x-rollout-token")

    async def _sample(self, body: dict, interactions: dict) -> dict:
        """Run one completion with the recording pointed at this rollout.

        ``proxy_interactions`` is a ContextVar, so setting it here affects only
        the task serving this request; TinkerLLM then writes the interaction
        into the very dict the rollout's session will read.
        """
        import litellm
        from platoon.train.tinker.proxy import proxy_interactions

        proxy_interactions.set(interactions)
        arguments = {key: value for key, value in body.items() if key not in _DROPPED_REQUEST_KEYS}
        response = await litellm.acompletion(model=self.model_name, **arguments)
        return response.model_dump()

    async def start(self) -> "TinkerProxyBridge":
        """Serve on a private event loop in its own thread.

        The bridge must answer while the rollouts' loop is busy, and it is not:
        a rollout creates its remote conversation and sends its first message
        with blocking HTTP calls made straight on the episode loop, and with
        several rollouts starting together the loop stalls for seconds at a
        time. Sharing that loop deadlocks the whole system -- the container's
        completion request waits on a bridge that cannot be scheduled, the agent
        never acts, and the host then polls for actions that will never come,
        on the same blocked loop. Giving the bridge its own loop and thread
        decouples the two entirely.
        """
        if self._thread is not None:
            return self
        import uvicorn

        config = uvicorn.Config(
            self._build_app(), host=self.host, port=self.port, log_level="warning", access_log=False
        )
        self._server = uvicorn.Server(config)
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()

        def serve() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.call_later(0.05, ready.set)
            self._loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=serve, name="tinker-proxy-bridge", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                logger.info("tinker proxy bridge listening on %s", self.base_url)
                return self
            await asyncio.sleep(0.1)
        await self.close()
        raise RuntimeError(f"proxy bridge did not start on {self.base_url}")

    async def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 15)
        if self._loop is not None:
            with contextlib.suppress(Exception):
                self._loop.close()
        self._server = self._thread = self._loop = None

    @contextlib.contextmanager
    def lease(
        self, interactions: dict | None = None, label: str = "?", response_tokens: list[int] | None = None
    ) -> Iterator[str]:
        """Route one rollout's requests into its own interactions dict.

        With no argument the lease binds whatever ``TinkerLLMProxySession`` is
        active in the caller's context, which is what a rollout wants: the dict
        object is shared, so the session sees every recorded interaction.
        ``label`` names the rollout in the bridge's stop-reason log lines.
        ``response_tokens``, if given, receives each completion's token count.
        """
        if interactions is None:
            from platoon.train.tinker.proxy import proxy_interactions

            try:
                interactions = proxy_interactions.get()
            except LookupError as error:
                raise RuntimeError(
                    "no TinkerLLMProxySession is active; a rollout must lease inside one, "
                    "or pass an explicit dict for tests"
                ) from error
        token = secrets.token_urlsafe(24)
        self._leases[token] = interactions
        self._labels[token] = label
        if response_tokens is not None:
            self._response_tokens[token] = response_tokens
        try:
            yield token
        finally:
            self._leases.pop(token, None)
            self._labels.pop(token, None)
            self._response_tokens.pop(token, None)


_BRIDGE: TinkerProxyBridge | None = None
_BRIDGE_LOCK = asyncio.Lock()


async def get_bridge(model_name: str) -> TinkerProxyBridge:
    """The process-wide bridge, started on first use.

    Rollouts run concurrently on one event loop and all share a TinkerLLM, so
    one server serves them all; the lease token is what keeps their recordings
    apart.
    """
    global _BRIDGE
    async with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = await TinkerProxyBridge(model_name).start()
        elif _BRIDGE.model_name != model_name:
            raise RuntimeError(f"bridge already serving {_BRIDGE.model_name}, cannot also serve {model_name}")
    return _BRIDGE
