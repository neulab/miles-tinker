"""Host OpenHands on Babel; execute isolated learner/verifier containers on Sail.

Run with SCIENCE_TASK_ROOT, SCIENCE_TRIAL_ROOT, SCIENCE_CREDENTIAL_FILE and
SCIENCE_AGENT_AUTH_FILE set. Credentials stay on the host outside learner boxes.
Each process handles exactly one trial to isolate SDK registration and state.
"""

import asyncio
import hmac
import json
import multiprocessing
import os
import re
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
import httpx
import paper2rlenv.harbor as harbor_runtime
import paper2rlenv.runtime as solver_runtime
from pydantic import BaseModel

from examples.science_rl.policy_llm import SciencePolicyLLM


class TrialRequest(BaseModel):
    task_id: str
    base_url: str
    sampling: dict


def _trial(request):
    # Miles cancels surplus sampled groups after filling the batch. Avoid
    # provisioning a Sailbox for a queued request whose TITO session is gone.
    response = httpx.get(request["base_url"].removesuffix("/v1"), timeout=30, trust_env=False)
    if response.status_code == 404:
        return {"science_status": "canceled", "science_reward": None}
    response.raise_for_status()
    credentials = json.loads(Path(os.environ["SCIENCE_CREDENTIAL_FILE"]).read_text())
    os.environ.update(credentials)
    os.environ["SCIENCE_LOCAL_MODEL_KEY"] = "local-miles"
    solver_runtime.ProxyLLM = SciencePolicyLLM
    # An exhausted action budget ends the policy rollout. Grade its actual
    # artifacts with the hidden verifier rather than labeling it infrastructure.
    harbor_runtime._sdk_run = partial(harbor_runtime._sdk_run, allow_iteration_limit=True)
    # One trial per process: scope native Harbor resources to this allocation,
    # allowing the external watchdog to clean up even if the host is killed.
    app_name = os.environ.get("SCIENCE_SAIL_APP", "science-rl-" + os.environ.get("SLURM_JOB_ID", "local"))
    harbor_runtime._environment_config = partial(harbor_runtime._environment_config, app_name=app_name)
    trial_id = str(uuid.uuid4())
    output = Path(os.environ["SCIENCE_TRIAL_ROOT"]) / trial_id
    sampling = dict(request["sampling"])
    temperature = sampling.pop("temperature", 1.0)
    result = harbor_runtime.run_solver(
        Path(os.environ["SCIENCE_TASK_ROOT"]) / request["task_id"],
        "Qwen3.5-4B",
        request["base_url"],
        output,
        backend="sail",
        api_key_env="SCIENCE_LOCAL_MODEL_KEY",
        temperature=temperature,
        extra_body=sampling,
    )
    return {
        "science_status": result["status"],
        "science_reward": int(result["passed"]) if result.get("passed") is not None else None,
        "science_trial_id": trial_id,
        "science_error": result.get("error"),
    }


@asynccontextmanager
async def _lifespan(app):
    app.state.pool = ProcessPoolExecutor(
        max_workers=int(os.environ.get("SCIENCE_AGENT_WORKERS", "64")),
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    )
    app.state.counts = Counter()
    app.state.active = 0
    yield
    app.state.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "active": app.state.active, "completed": dict(app.state.counts)}


@app.post("/trial")
async def trial(request: TrialRequest, authorization: str = Header(default="")):
    expected = "Bearer " + Path(os.environ["SCIENCE_AGENT_AUTH_FILE"]).read_text().strip()
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", request.task_id):
        raise HTTPException(status_code=400, detail="Invalid task id")
    task = Path(os.environ["SCIENCE_TASK_ROOT"]) / request.task_id
    if not (task / "task.toml").is_file():
        raise HTTPException(status_code=404, detail="Unknown task")
    app.state.active += 1
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            app.state.pool, _trial, request.model_dump()
        )
        app.state.counts[result["science_status"]] += 1
        print(json.dumps({"task_id": request.task_id, **result}), flush=True)
        return result
    finally:
        app.state.active -= 1
