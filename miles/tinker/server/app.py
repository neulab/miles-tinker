"""HTTP skin over TinkerService: the endpoints the Tinker SDK calls.

All wire translation happens here (encoding.py for JSON, proto_codec.py for
protobuf); the service only ever sees decoded commands and returns internal
results. Auth is the SDK's X-API-Key (bearer token as a fallback) used as the
tenant identity; authorization (ownership of models, promises, checkpoints)
is enforced in the service.
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from miles.tinker.core.promise import FAILED, PENDING
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import OwnershipError, UserInputError
from miles.tinker.server.encoding import decode_command, decode_forward_request, decode_sample_request, render_result
from miles.tinker.server.proto_codec import (
    PROTO_CONTENT_TYPE,
    PROTO_ENCODERS,
    decode_forward_backward_request,
    maybe_decompress,
)

logger = logging.getLogger(__name__)

COMMAND_ROUTES = {
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_state",
    "/api/v1/load_weights": "load_state",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
}


def _tenant(request: Request) -> str:
    """Read straight from the headers: FastAPI Header() dependencies cost a
    tenth of a request's event-loop time at load."""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    authorization = request.headers.get("authorization")
    if authorization:
        return authorization.removeprefix("Bearer ").strip()
    return "anonymous"


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def render_training_run(run: dict) -> dict:
    return {
        "training_run_id": run["training_run_id"],
        "base_model": run["base_model"],
        "model_owner": run["model_owner"],
        "is_lora": run["is_lora"],
        "lora_rank": run["lora_rank"],
        "user_metadata": run["user_metadata"] or None,
        "last_request_time": _iso(run["last_request_at"]),
        "corrupted": run["corrupted"],
    }


def build_app(service: TinkerService) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(UserInputError)
    async def _user_error(request: Request, error: UserInputError):
        return JSONResponse(status_code=400, content={"error": str(error)})

    @app.exception_handler(OwnershipError)
    async def _ownership_error(request: Request, error: OwnershipError):
        return JSONResponse(status_code=403, content={"error": str(error)})

    @app.get("/api/v1/healthz")
    @app.get("/health")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    @app.get("/api/v1/client/config")
    async def client_config():
        return {}

    @app.post("/api/v1/client/dynamic_config")
    @app.get("/api/v1/client/dynamic_config")
    async def client_dynamic_config():
        return {}

    @app.post("/api/v1/telemetry")
    async def telemetry(request: Request):
        payload = await _json_or_empty(request)
        if _announces_session_end(payload):
            await service.end_session(payload["session_id"])
        return {"status": "accepted"}

    @app.post("/api/v1/create_session")
    async def create_session(request: Request):
        payload = await request.json()
        session_id = service.create_session(_tenant(request), payload)
        return {"type": "create_session", "session_id": session_id}

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(request: Request):
        payload = await request.json()
        service.heartbeat(payload["session_id"])
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/get_server_capabilities")
    @app.get("/api/v1/get_server_capabilities")
    async def get_server_capabilities():
        return {"supported_models": [{"model_name": service.config.base_model, "trainable": True, "sampleable": True}]}

    @app.post("/api/v1/create_model")
    async def create_model(request: Request):
        payload = await request.json()
        request_id, model_id = service.create_model(_tenant(request), payload)
        return {"request_id": request_id, "model_id": model_id}

    @app.post("/api/v1/get_info")
    async def get_info(request: Request):
        payload = await request.json()
        record = service.get_model(_tenant(request), payload["model_id"])
        lora_rank = record.lora_rank if record.is_lora else None
        return {
            "type": "get_info",
            "model_id": record.model_id,
            "is_lora": record.is_lora,
            "lora_rank": lora_rank,
            "model_name": record.base_model,
            "model_data": {
                "model_name": record.base_model,
                "tokenizer_id": record.base_model,
                "lora_rank": lora_rank,
                "is_lora": record.is_lora,
            },
        }

    @app.post("/api/v1/unload_model")
    async def unload_model(request: Request):
        payload = await request.json()
        request_id = service.unload_model(_tenant(request), payload["model_id"])
        return {"request_id": request_id, "model_id": payload["model_id"]}

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request):
        if PROTO_CONTENT_TYPE in request.headers.get("content-type", ""):
            body = maybe_decompress(await request.body(), request.headers.get("content-encoding"))
            kind, payload = decode_forward_backward_request(body)
        else:
            kind, payload = decode_command("forward_backward", await request.json())
        request_id = service.submit(_tenant(request), kind, payload)
        return {"request_id": request_id, "model_id": payload["model_id"]}

    @app.post("/api/v1/forward")
    async def forward(request: Request):
        kind, payload = decode_forward_request(await request.json())
        request_id = service.submit(_tenant(request), kind, payload)
        return {"request_id": request_id, "model_id": payload["model_id"]}

    for route, route_kind in COMMAND_ROUTES.items():

        def _make(route_kind: str):
            async def command(request: Request):
                kind, payload = decode_command(route_kind, await request.json())
                request_id = service.submit(_tenant(request), kind, payload)
                return {"request_id": request_id, "model_id": payload["model_id"]}

            return command

        app.post(route)(_make(route_kind))

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request):
        payload = await request.json()
        request_id = payload["request_id"]
        promise = service.retrieve(_tenant(request), request_id)
        if promise is None:
            return JSONResponse(status_code=410, content={"error": "unknown or expired promise"})
        if promise.state == PENDING:
            # long-poll: hold the request until the promise settles or the cap
            # elapses; 408 is the SDK's quiet "poll again" (try_again logs a warning)
            try:
                await asyncio.wait_for(promise.settled.wait(), service.config.retrieve_poll_timeout_s)
            except TimeoutError:
                return JSONResponse(
                    status_code=408, content={"type": "try_again", "request_id": request_id, "queue_state": "active"}
                )
        if promise.state == FAILED:
            return JSONResponse({"error": promise.error, "category": promise.error_category})
        encoder = PROTO_ENCODERS.get(promise.result["kind"])
        if encoder is not None and PROTO_CONTENT_TYPE in request.headers.get("accept", ""):
            return Response(content=encoder(promise.result), media_type=PROTO_CONTENT_TYPE)
        # an explicit JSONResponse: returning the dict would first send every token and
        # logprob through FastAPI's jsonable_encoder
        return JSONResponse(render_result(promise.result))

    @app.post("/api/v1/cancel_future")
    async def cancel_future(request: Request):
        payload = await request.json()
        service.cancel(_tenant(request), payload["request_id"])
        return {"status": "ok"}

    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: Request):
        payload = await request.json()
        sampling_session_id = service.create_sampling_session(_tenant(request), payload)
        return {"type": "create_sampling_session", "sampling_session_id": sampling_session_id}

    @app.post("/api/v1/asample")
    async def asample(request: Request):
        payload = decode_sample_request(await request.json())
        request_id, sequence_ids = service.submit_sample(_tenant(request), payload)
        return {"request_id": request_id, "sample_sequence_ids": sequence_ids}

    @app.get("/api/v1/samplers/{sampling_session_id}")
    async def get_sampler(request: Request, sampling_session_id: str):
        try:
            return service.get_sampler(_tenant(request), sampling_session_id)
        except UserInputError as error:
            return JSONResponse(status_code=404, content={"error": str(error)})

    @app.post("/api/v1/weights_info")
    async def weights_info(request: Request):
        payload = await request.json()
        try:
            return service.weights_info(_tenant(request), payload["tinker_path"])
        except UserInputError as error:
            return JSONResponse(status_code=404, content={"error": str(error)})

    @app.get("/api/v1/training_runs/{model_id}")
    async def training_run(request: Request, model_id: str):
        try:
            return render_training_run(service.training_run(_tenant(request), model_id))
        except UserInputError as error:
            return JSONResponse(status_code=404, content={"error": str(error)})

    @app.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id:path}")
    async def delete_checkpoint(request: Request, model_id: str, checkpoint_id: str):
        service.delete_checkpoint(_tenant(request), model_id, checkpoint_id)
        return Response(status_code=204)

    return app


async def _json_or_empty(request: Request) -> dict:
    try:
        payload = await request.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _announces_session_end(payload: dict) -> bool:
    """Both SDKs flush a SESSION_END event when the client shuts down."""
    if not payload.get("session_id"):
        return False
    return any(
        isinstance(event, dict) and event.get("event", event.get("event_type")) == "SESSION_END"
        for event in payload.get("events") or []
    )
