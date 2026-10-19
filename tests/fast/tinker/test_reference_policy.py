"""The trainer-side KL reference (full/reference.py): the starting weights are
parked once, swapped in only for the scoring pass, and the live weights always
come back -- re-parked only after they change."""

import pytest
import torch

from miles.backends.megatron_utils.full.reference import ReferencePolicy, sampled_kl_to_reference


class FakeBackuper:
    """A one-tensor 'model' whose weights are copied to and from named tags."""

    def __init__(self, value: float) -> None:
        self.weight = torch.tensor([value])
        self.tags: dict[str, torch.Tensor] = {}
        self.calls: list[str] = []

    @property
    def backup_tags(self):
        return list(self.tags)

    def backup(self, tag: str) -> None:
        self.calls.append(f"backup:{tag}")
        self.tags[tag] = self.weight.clone()

    def restore(self, tag: str) -> None:
        self.calls.append(f"restore:{tag}")
        self.weight.copy_(self.tags[tag])


def test_reference_scores_with_the_starting_weights_and_restores_the_live_ones():
    model = FakeBackuper(1.0)
    active = []
    reference = ReferencePolicy(model, set_active_tag=active.append)
    model.weight.fill_(2.0)  # an optimizer step
    reference.weights_changed()

    seen = reference.score(lambda: model.weight.item())

    assert seen == 1.0, "scored with the starting weights"
    assert model.weight.item() == 2.0, "live weights restored"
    assert active == ["ref", "actor"]


def test_live_weights_are_reparked_only_after_they_change():
    model = FakeBackuper(1.0)
    reference = ReferencePolicy(model)
    reference.score(lambda: None)
    reference.score(lambda: None)
    assert model.calls.count("backup:actor") == 1, "same optim window: one park"
    model.weight.fill_(3.0)
    reference.weights_changed()
    reference.score(lambda: None)
    assert model.calls.count("backup:actor") == 2
    assert model.weight.item() == 3.0


def test_live_weights_come_back_when_scoring_fails():
    model = FakeBackuper(1.0)
    reference = ReferencePolicy(model)
    model.weight.fill_(5.0)
    reference.weights_changed()

    def boom():
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError):
        reference.score(boom)
    assert model.weight.item() == 5.0


def test_sampled_kl_counts_only_trained_tokens():
    sampled = [[-1.0, -2.0, -0.5]]
    ref = [torch.tensor([-1.0, -3.0, -9.0])]
    kl = sampled_kl_to_reference(sampled, ref, weights=[[0.1, 0.1, 0.0]])
    assert kl["kl_ref_tokens"] == 2.0
    assert kl["kl_ref_k1"] == pytest.approx((0.0 + 1.0) / 2)
    k3 = (torch.exp(torch.tensor(-1.0)) - 1 + 1.0).item()
    assert kl["kl_ref_k3"] == pytest.approx(k3 / 2, rel=1e-5)
    assert sampled_kl_to_reference(sampled, ref, weights=[[0.0, 0.0, 0.0]])["kl_ref_tokens"] == 0.0


def test_sampled_kl_accepts_tensors_and_lists_of_any_device_mix():
    """The trainer batch holds sampled logprobs and weights as tensors (on the GPU in a
    real run) and the reference scores elsewhere; job 155795 died on that mix."""
    sampled = [torch.tensor([-1.0, -2.0]), [-0.5]]
    ref = [torch.tensor([-1.0, -3.0]), [-0.5]]
    weights = [torch.tensor([0.1, 0.1]), [0.2]]
    kl = sampled_kl_to_reference(sampled, ref, weights)
    assert kl["kl_ref_tokens"] == 3.0
    assert kl["kl_ref_k1"] == pytest.approx(1.0 / 3)
    if torch.cuda.is_available():
        on_gpu = sampled_kl_to_reference([t.cuda() if isinstance(t, torch.Tensor) else t for t in sampled], ref, weights)
        assert on_gpu == pytest.approx(kl)
