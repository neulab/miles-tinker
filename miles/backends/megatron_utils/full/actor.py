"""Tinker full fine-tuning trainer: one model (slot 0) trains the base weights.

The command surface mirrors ``lora.actor.MultiLoRATrainRayActor``; sampler
weights reach the engines through the base actor's ``update_weights``, which
the gateway brackets with the inference controller.
"""

from contextlib import ExitStack

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.full import model as full_model
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs


class FullTinkerTrainRayActor(MegatronTrainRayActor):
    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        return self._run_batch(batch_id, rollout_data_ref, forward_only=False)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        """Same loss pass as forward_backward, without the backward."""
        return self._run_batch(batch_id, rollout_data_ref, forward_only=True)

    def _run_batch(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        self._heartbeat.bump()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref)
            stack.enter_context(store_get_result)
            return full_model.forward_backward(
                self.args, batch_id, self.model, self.optimizer, rollout_data, forward_only=forward_only
            )

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        self._heartbeat.bump()
        ((slot, adam_params),) = adam_params_by_slot.items()  # full fine-tuning serves a single model
        return {slot: full_model.optim_step(self.args, self.model, self.optimizer, adam_params)}

    @with_logs
    def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        """A new model starts from the base weights; a load_state restores a full checkpoint. rank/alpha unused."""
        self._heartbeat.bump()
        if ckpt_path is None:
            full_model.reset_to_base(self.args, self.model, self.optimizer)
        else:
            full_model.load_state(self.args, self.model, self.optimizer, ckpt_path, load_optimizer)

    @with_logs
    def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> None:
        self._heartbeat.bump()
        full_model.save_state(self.args, self.model, self.optimizer, path, metadata=metadata)

    @with_logs
    def export_slot(self, slot: int, rank: int, alpha: float, path: str, metadata: dict | None = None) -> None:
        raise NotImplementedError("full fine-tuning publishes sampler weights with update_weights, not export_slot")

    @with_logs
    def unload_slot(self, slot: int) -> dict | None:
        """The model's pending gradients die with it; its weights stay until the next model resets them."""
        full_model.discard_grads(self.model, self.optimizer)
        return None
