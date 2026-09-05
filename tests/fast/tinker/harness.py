"""Shared fakes for the tinker gateway suite."""

import asyncio
import tempfile

from miles.tinker.core.promise import PENDING, Promise
from miles.tinker.core.service import ExecutorBackend, TinkerService
from miles.tinker.core.types import Command, GatewayConfig

ADAM = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


class FakeBackend(ExecutorBackend):
    """Records every call; returns deterministic shapes. Set ``fail_next`` to
    make the next backend call raise it. ``full`` mimics the FFT executor:
    push_slot returns a growing engine weight version."""

    def __init__(self, full: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_next: Exception | None = None
        self.full = full
        self.weight_version = 0
        self.warmed_up = False

    def _record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error

    def named(self, name: str) -> list[dict]:
        return [kwargs for called, kwargs in self.calls if called == name]

    async def load_slot(self, slot, rank, alpha, ckpt_path=None, load_optimizer=True):
        self._record(
            "load_slot", slot=slot, rank=rank, alpha=alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )

    async def unload_slot(self, slot):
        self._record("unload_slot", slot=slot)

    async def forward_backward(self, unit_id, slot_rows, loss_fn, loss_fn_config):
        self._record(
            "forward_backward", unit_id=unit_id, slot_rows=slot_rows, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )
        return [{"loss": 1.0, "logprobs": [0.0] * row["target_len"]} for _, row in slot_rows]

    async def forward_only(self, unit_id, slot_rows, loss_fn, loss_fn_config):
        self._record(
            "forward_only", unit_id=unit_id, slot_rows=slot_rows, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )
        return [{"loss": 0.0, "logprobs": [0.0] * row["target_len"]} for _, row in slot_rows]

    async def optim_step(self, adam_params_by_slot):
        self._record("optim_step", adam_params_by_slot=adam_params_by_slot)
        return {
            slot: {"grad_norm": 0.5 + slot, "skipped": 0, "lr": params["learning_rate"], "num_units": 1}
            for slot, params in adam_params_by_slot.items()
        }

    async def discard_grads(self, slot):
        self._record("discard_grads", slot=slot)

    async def save_slot(self, slot, path):
        self._record("save_slot", slot=slot, path=path)

    async def push_slot(self, slot, lora_name, rank, alpha):
        self._record("push_slot", slot=slot, lora_name=lora_name, rank=rank, alpha=alpha)
        if not self.full:
            return None
        self.weight_version += 1
        return self.weight_version

    async def unload_adapter(self, lora_name):
        self._record("unload_adapter", lora_name=lora_name)

    async def warmup(self):
        self._record("warmup")
        self.warmed_up = True

    async def sample(self, payload, lora_name):
        self._record("sample", payload=payload, lora_name=lora_name)
        return {
            "sequences": [
                {"sequence_id": f"seq-{i}", "tokens": [1, 2], "logprobs": [0.0, 0.0], "stop_reason": "stop"}
                for i in range(payload["num_samples"])
            ]
        }


def make_config(**overrides) -> GatewayConfig:
    defaults = dict(base_model="base", n_slots=2, checkpoint_root=tempfile.mkdtemp(prefix="tinker-test-"))
    return GatewayConfig(**{**defaults, **overrides})


def make_service(**config_overrides) -> TinkerService:
    config = make_config(**config_overrides)
    return TinkerService(FakeBackend(full=not config.is_lora), config)


def row(tokens: int = 3) -> dict:
    return {"tokens": list(range(tokens + 1)), "target_len": tokens}


def fb_payload(model_id: str, seq_id: int, rows: list[dict], loss_fn: str = "cross_entropy") -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "rows": rows, "loss_fn": loss_fn, "loss_fn_config": {}}


def command(model_id: str, seq_id: int, kind: str, payload: dict, arrival: int) -> Command:
    return Command(
        model_id=model_id, seq_id=seq_id, kind=kind, payload=payload, request_id=f"req-{seq_id}", arrival=arrival
    )


def create_model_payload(service: TinkerService, **extra) -> dict:
    return {"base_model": service.config.base_model, "lora_config": {"rank": 8}, **extra}


async def created_model(service: TinkerService, tenant: str = "tenant", **extra) -> str:
    _, model_id = service.create_model(tenant, create_model_payload(service, **extra))
    await asyncio.sleep(0.01)  # let the slot-init task run against the fake backend
    return model_id


async def await_settled(service: TinkerService, tenant: str, request_id: str, timeout: float = 2.0) -> Promise:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        promise = service.retrieve(tenant, request_id)
        assert promise is not None, f"promise {request_id} expired"
        if promise.state != PENDING:
            return promise
        await asyncio.sleep(0.005)
    raise AssertionError(f"promise {request_id} still pending after {timeout}s")
