"""Route full-model slots to independent trainers, each exposing local slot 0.

The service serializes training commands on the shared actor/critic GPUs.
Optional offloading switches trainers only after their gradients are consumed.
Sampling uses dedicated inference GPUs and never waits for these transitions.
"""

from miles.tinker.core.service import ExecutorBackend
from miles.tinker.core.types import UserInputError


class FullModelBackend(ExecutorBackend):
    def __init__(self, policy: ExecutorBackend, critic: ExecutorBackend, *, offload: bool = False) -> None:
        self.backends = {0: policy, 1: critic}
        self._offload_enabled = offload
        # With offload enabled, both workers finish initialization asleep.
        self._resident_slot: int | None = None
        self._pending_slots: set[int] = set()

    async def _activate(self, slot: int) -> None:
        if not self._offload_enabled or slot == self._resident_slot:
            return
        if self._resident_slot is not None:
            if self._resident_slot in self._pending_slots:
                raise UserInputError(
                    f"complete optim_step for slot {self._resident_slot} before switching trainers with offload enabled"
                )
            await self.backends[self._resident_slot].offload()
            self._resident_slot = None
        await self.backends[slot].onload()
        self._resident_slot = slot

    async def load_slot(self, slot, rank, alpha, ckpt_path=None, load_optimizer=True):
        await self._activate(slot)
        await self.backends[slot].load_slot(0, rank, alpha, ckpt_path, load_optimizer)
        self._pending_slots.discard(slot)

    async def unload_slot(self, slot):
        if self._offload_enabled and slot != self._resident_slot:
            return  # Already asleep and clean; do not evict another model to unload it.
        await self.backends[slot].unload_slot(0)
        self._pending_slots.discard(slot)
        if self._offload_enabled:
            await self.backends[slot].offload()
            self._resident_slot = None

    async def _forward(self, kind, unit_id, slot_rows, loss_fn, loss_fn_config):
        slots = {slot for slot, _ in slot_rows}
        if len(slots) != 1:
            raise UserInputError("a full-model work unit must address exactly one model role")
        slot = next(iter(slots))
        await self._activate(slot)
        if kind == "forward_backward":
            # Even a failed backward may leave partial gradients. The service
            # clears a poisoned window through discard_grads before proceeding.
            self._pending_slots.add(slot)
        return await getattr(self.backends[slot], kind)(
            unit_id, [(0, row) for _, row in slot_rows], loss_fn, loss_fn_config
        )

    async def forward_backward(self, unit_id, slot_rows, loss_fn, loss_fn_config):
        return await self._forward("forward_backward", unit_id, slot_rows, loss_fn, loss_fn_config)

    async def forward_only(self, unit_id, slot_rows, loss_fn, loss_fn_config):
        return await self._forward("forward_only", unit_id, slot_rows, loss_fn, loss_fn_config)

    async def optim_step(self, adam_params_by_slot):
        results = {}
        # A merged barrier must consume resident gradients before switching GPUs.
        slots = sorted(adam_params_by_slot, key=lambda slot: slot != self._resident_slot)
        for slot in slots:
            await self._activate(slot)
            result = await self.backends[slot].optim_step({0: adam_params_by_slot[slot]})
            results[slot] = result[0]
            self._pending_slots.discard(slot)
        return results

    async def discard_grads(self, slot):
        if self._offload_enabled and slot not in self._pending_slots:
            return  # A request rejected before activation never touched gradients.
        await self._activate(slot)
        await self.backends[slot].discard_grads(0)
        self._pending_slots.discard(slot)

    async def save_slot(self, slot, path):
        await self._activate(slot)
        await self.backends[slot].save_slot(0, path)

    async def push_slot(self, slot, lora_name, rank, alpha):
        if slot != 0:
            raise UserInputError("only the policy can publish sampler weights")
        await self._activate(slot)
        return await self.backends[0].push_slot(0, lora_name, rank, alpha)

    async def warmup(self):
        await self._activate(0)
        await self.backends[0].warmup()

    async def sample(self, payload, lora_name):
        return await self.backends[0].sample(payload, lora_name)

    async def unload_adapter(self, lora_name):
        raise UserInputError("full models do not have sampler adapters")
