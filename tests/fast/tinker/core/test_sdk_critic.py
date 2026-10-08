"""SDK critic request validation, independent of GPU execution."""

import asyncio
from copy import deepcopy

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, created_model, fb_payload, make_service

from miles.tinker.core.critic import validate_model_pass
from miles.tinker.core.types import UserInputError


def sdk_payload(**overrides):
    row = {
        "tokens": [10, 20, 30, 40],
        "target_len": 2,
        "values": [0.1, 0.2],
        "returns": [1.0, 2.0],
        "weights": [0.0, 0.5],
        **overrides,
    }
    return fb_payload("critic", 1, [row], "ppo")


@pytest.mark.parametrize("fields", [{"model_role": "other"}, {"model_role": None}])
def test_invalid_roles_fail_before_slot_allocation(fields):
    service = make_service(train_mode="full", enable_critic=True, n_slots=2)
    with pytest.raises(UserInputError):
        service.create_model("tenant", {"base_model": "base", **fields})
    assert service.free_slots == {0, 1}
    assert not service.models


def test_critic_ppo_preserves_weights_and_does_not_mutate_inputs():
    payload = sdk_payload()
    before = deepcopy(payload)
    validate_model_pass("critic", "forward_backward", payload)
    assert payload == before


@pytest.mark.parametrize("field", ["values", "returns", "weights"])
def test_backward_requires_each_critic_field(field):
    payload = sdk_payload()
    del payload["rows"][0][field]
    with pytest.raises(UserInputError, match="missing"):
        validate_model_pass("critic", "forward_backward", payload)


@pytest.mark.parametrize(
    "fields",
    [
        {"values": [0.1]},
        {"returns": [float("nan"), 0.0]},
        {"weights": [-1.0, 1.0]},
        {"weights": [float("inf"), 0.0]},
        {"weights": [1.0]},
        {"old_values": [0.0, 0.0]},
        {"loss_mask": [1, 1]},
    ],
)
def test_invalid_or_mixed_dialects_fail(fields):
    with pytest.raises(UserInputError):
        validate_model_pass("critic", "forward_backward", sdk_payload(**fields))


def test_forward_needs_no_regression_targets():
    payload = fb_payload("critic", 1, [{"tokens": [10, 20, 30], "target_len": 1}], "ppo")
    before = deepcopy(payload)
    validate_model_pass("critic", "forward_only", payload)
    assert payload == before


def test_policy_ppo_is_unchanged_and_rejects_value_inputs():
    payload = fb_payload("policy", 1, [{"tokens": [1, 2], "target_len": 1}], "ppo")
    before = deepcopy(payload)
    validate_model_pass("policy", "forward_backward", payload)
    assert payload == before
    with pytest.raises(UserInputError, match="critic"):
        validate_model_pass("policy", "forward_backward", sdk_payload())


@pytest.mark.parametrize("config", [{"value_clip": -1}, {"value_clip": float("nan")}, {"unknown": 1}])
def test_sdk_value_clip_is_validated(config):
    payload = sdk_payload()
    payload["loss_fn_config"] = config
    with pytest.raises(UserInputError):
        validate_model_pass("critic", "forward_backward", payload)


@pytest.mark.parametrize("loss_fn", ["ppo_critic", "ppo_value"])
@pytest.mark.parametrize("role", ["policy", "critic"])
@pytest.mark.parametrize("kind", ["forward_only", "forward_backward"])
def test_removed_loss_aliases_are_rejected(role, kind, loss_fn):
    payload = sdk_payload()
    payload["loss_fn"] = loss_fn
    with pytest.raises(UserInputError, match="unsupported loss_fn"):
        validate_model_pass(role, kind, payload)


def test_critic_creation_is_disabled_by_default():
    service = make_service(train_mode="full", n_slots=1)
    with pytest.raises(UserInputError, match="does not support critic"):
        service.create_model("tenant", {"base_model": "base", "model_role": "critic"})
    assert service.free_slots == {0}


@pytest.fixture
async def pair(tmp_path):
    service = make_service(train_mode="full", enable_critic=True, n_slots=2, checkpoint_root=str(tmp_path))
    task = asyncio.create_task(service.run())
    policy = await created_model(service)
    critic = await created_model(service, model_role="critic")
    yield service, policy, critic
    tasks = [task, service._sweep_task, *service._background_tasks]
    for pending in tasks:
        pending.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_roles_have_independent_slots_and_checkpoint_lineages(pair):
    service, policy, critic = pair
    assert (service.models[policy].slot, service.models[critic].slot) == (0, 1)
    request = service.submit("tenant", "optim_step", {"model_id": critic, "seq_id": 1, "adam_params": ADAM})
    assert (await await_settled(service, "tenant", request)).error is None
    request = service.submit("tenant", "save_state", {"model_id": critic, "seq_id": 2, "name": "critic"})
    saved = await await_settled(service, "tenant", request)
    assert saved.error is None
    path = saved.result["path"]
    _, meta = service._resolve_checkpoint(service.models[critic], path)
    assert meta["model_role"] == "critic"
    assert meta["checkpoint_lineage"] == critic
    with pytest.raises(UserInputError, match="role"):
        service._resolve_checkpoint(service.models[policy], path)
    await service._load_state(service.models[critic], {"path": path, "optimizer": True})
    assert service.models[critic].checkpoint_lineage == critic
    with pytest.raises(UserInputError, match="publish"):
        await service._publish_sampler(service.models[critic], {})


@pytest.mark.parametrize("load_optimizer", [True, False])
async def test_pair_checkpoints_resolve_after_gateway_restart(pair, load_optimizer):
    """Real metadata files and dispatch; the fake backend does not serialize tensors."""
    service, policy, critic = pair
    saved_paths = {}
    for role, model_id in (("policy", policy), ("critic", critic)):
        request = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "step-1"})
        saved = await await_settled(service, "tenant", request)
        assert saved.error is None
        saved_paths[role] = saved.result["path"]
    assert saved_paths["policy"] != saved_paths["critic"]

    # A separate instance has no registry entries for the checkpoint's original models.
    restarted = make_service(
        train_mode="full", enable_critic=True, n_slots=2, checkpoint_root=service.config.checkpoint_root
    )
    task = asyncio.create_task(restarted.run())
    try:
        for role, old_id, slot in (("policy", policy, 0), ("critic", critic, 1)):
            new_id = await created_model(restarted, model_role=role)
            assert old_id not in restarted.models
            request = restarted.submit(
                "tenant",
                "load_state",
                {"model_id": new_id, "seq_id": 1, "path": saved_paths[role], "optimizer": load_optimizer},
            )
            loaded = await await_settled(restarted, "tenant", request)
            assert loaded.error is None
            call = restarted.backend.named("load_slot")[-1]
            assert call["slot"] == slot and call["load_optimizer"] is load_optimizer
            assert call["ckpt_path"] == restarted._checkpoint_dir(old_id, "weights", "step-1")
            assert restarted.models[new_id].checkpoint_lineage == old_id

            # The opposite role must fail before invoking the trainer.
            wrong_role = "critic" if role == "policy" else "policy"
            calls_before = len(restarted.backend.named("load_slot"))
            request = restarted.submit(
                "tenant",
                "load_state",
                {"model_id": new_id, "seq_id": 2, "path": saved_paths[wrong_role], "optimizer": load_optimizer},
            )
            rejected = await await_settled(restarted, "tenant", request)
            assert rejected.error_category == "user" and "role" in rejected.error
            assert len(restarted.backend.named("load_slot")) == calls_before
    finally:
        tasks = [task, restarted._sweep_task, *restarted._background_tasks]
        for pending in tasks:
            pending.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_pair_cannot_be_split_between_tenants(pair):
    service, _, _ = pair
    with pytest.raises(UserInputError, match="tenant"):
        service.create_model("other", {"base_model": "base", "model_role": "critic"})


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo"])
async def test_omitted_role_preserves_policy_training(pair, loss_fn):
    service, policy, _ = pair
    assert service.models[policy].model_role == "policy"
    row = {"tokens": [10, 20, 30], "target_len": 2, "sampling_logprobs": [-0.5, -0.2], "advantages": [1.0, 1.0]}
    request = service.submit("tenant", "forward_backward", fb_payload(policy, 1, [row], loss_fn))
    result = await await_settled(service, "tenant", request)
    assert result.error is None
    assert "logprobs" in result.result["outputs"][0]
    assert "values" not in result.result["outputs"][0]
