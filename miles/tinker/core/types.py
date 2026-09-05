"""Shared types of the Tinker gateway.

Core speaks only the gateway's internal language (commands, rows, results).
server/ translates the SDK wire (JSON and proto); runtime.py translates
miles (trainer batches). Each foreign language lives only at its boundary.
"""

import time
from dataclasses import dataclass, field


class UserInputError(Exception):
    """Rejected request content; fails the promise with category User."""


class OwnershipError(Exception):
    """model/checkpoint does not belong to the caller's tenant."""


@dataclass
class GatewayConfig:
    base_model: str
    n_slots: int
    checkpoint_root: str
    max_datums_per_request: int = 1024
    max_tokens_per_datum: int = 32768
    max_tokens_per_request: int = 4_000_000
    lora_alpha: float | None = None  # None: 2 * rank
    lease_timeout_s: float = 600.0  # SDK heartbeats every 10 s from a background thread; silence means the client is gone
    unit_token_budget: int = 262_144  # packing bound per work unit
    train_mode: str = "lora"  # "lora" | "full"
    retrieve_poll_timeout_s: float = 20.0  # long-poll cap on /retrieve_future (< the SDK's 45 s)
    sampler_versions_to_keep: int = 3  # LoRA: adapter versions kept loaded per model
    coalesce_wait_s: float = 0.3  # planner: wait for more chunks of an open window
    coalesce_max_wait_s: float = 2.0
    max_concurrent_samples: int = 512  # in-flight /generate posts per gateway

    @property
    def is_lora(self) -> bool:
        return self.train_mode != "full"


@dataclass
class Command:
    model_id: str
    seq_id: int
    kind: str
    payload: dict
    request_id: str
    arrival: int  # global submit order, the planner's FCFS key
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class ModelRecord:
    model_id: str
    tenant: str
    slot: int
    base_model: str
    lora_rank: int  # 0 for full fine-tuning
    lora_alpha: float
    session_id: str
    is_lora: bool = True
    sdk_version: str = ""
    sampler_version: int = 0
    user_metadata: dict = field(default_factory=dict)
    sampler_names: dict[str, int] = field(default_factory=dict)  # sampler path name -> version
    loaded_versions: set[int] = field(default_factory=set)  # LoRA versions resident in the engines
    created_at: float = field(default_factory=time.time)
    last_request_at: float = field(default_factory=time.time)
