"""Miles TITO adapter; infrastructure errors invalidate the complete GRPO group."""

import os
from pathlib import Path

import httpx

from miles.rollout.agentic.session import resolve_session_url
from miles.utils.types import Sample


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    token = Path(os.environ["SCIENCE_AGENT_AUTH_FILE"]).read_text().strip()
    # Includes queue time for 256 trajectories, beyond the individual task deadline.
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=7200), trust_env=False) as client:
        response = await client.post(
            os.environ["SCIENCE_AGENT_SERVER_URL"] + "/trial",
            headers={"Authorization": "Bearer " + token},
            json={
                "task_id": metadata["task_id"],
                "base_url": resolve_session_url(base_url),
                "sampling": request_kwargs,
                "max_seq_len": metadata.get("max_seq_len"),
            },
        )
        response.raise_for_status()
        return response.json()


async def reward_func(args, sample, **kwargs):
    result = sample.metadata
    reward = result.get("science_reward")
    if result.get("science_status") != "completed" or reward not in (0, 1):
        sample.status = Sample.Status.ABORTED
        return 0.0  # Group filter discards this placeholder before optimization.
    return float(reward)
