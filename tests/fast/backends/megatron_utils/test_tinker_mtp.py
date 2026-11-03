"""MTP losses must be independent of Tinker's work-unit and microbatch partitioning."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from miles.backends.megatron_utils import tinker_mtp
from miles.backends.training_utils.loss_hub import fused_lm_head


def _model_with_grad_buffers():
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2, bias=False)
    model.mtp = torch.nn.Linear(2, 2, bias=False)
    model.decoder.weight.main_grad = torch.full_like(model.decoder.weight, 7.0)
    model.mtp.weight.main_grad = torch.zeros_like(model.mtp.weight)
    return model


def test_window_normalizes_only_mtp_gradients_and_accumulates_record_counts():
    model = _model_with_grad_buffers()
    model.mtp.weight.main_grad.fill_(12.0)
    window = tinker_mtp.MTPWindow()
    window.record(torch.tensor(3.0), torch.tensor(2.0))
    window.record(torch.tensor(9.0), torch.tensor(4.0))

    window.normalize_gradients([model])

    torch.testing.assert_close(model.mtp.weight.main_grad, torch.full_like(model.mtp.weight, 2.0))
    torch.testing.assert_close(model.decoder.weight.main_grad, torch.full_like(model.decoder.weight, 7.0))


def test_window_reset_discards_prior_token_count():
    model = _model_with_grad_buffers()
    window = tinker_mtp.MTPWindow()
    window.record(torch.tensor(30.0), torch.tensor(100.0))
    window.reset()
    window.record(torch.tensor(4.0), torch.tensor(2.0))
    model.mtp.weight.main_grad.fill_(6.0)

    window.normalize_gradients([model])

    torch.testing.assert_close(model.mtp.weight.main_grad, torch.full_like(model.mtp.weight, 3.0))


def test_zero_token_window_has_finite_gradients():
    model = _model_with_grad_buffers()
    window = tinker_mtp.MTPWindow()
    window.record(torch.tensor(0.0), torch.tensor(0.0))

    window.normalize_gradients([model])

    assert torch.count_nonzero(model.mtp.weight.main_grad) == 0
    assert torch.isfinite(model.decoder.weight.main_grad).all()


def _partitioned_update(partitions, policy_loss_scale):
    """Simulate different microbatches for the same masked examples and DDP grad buffers."""
    model = _model_with_grad_buffers()
    with torch.no_grad():
        model.mtp.weight.copy_(torch.tensor([[0.2, 0.1], [-0.3, 0.4]]))
    inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0], [3.0, 1.0], [-2.0, 1.0]])
    targets = torch.tensor([[0.5, -0.5], [0.0, 0.5], [-0.5, 0.0], [0.5, 0.5], [0.1, -0.1]])
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
    window = tinker_mtp.MTPWindow()
    offset = 0
    for size in partitions:
        rows = slice(offset, offset + size)
        predictions = model.mtp(inputs[rows])
        losses = (predictions - targets[rows]).square().sum(-1)
        loss_sum = (losses * mask[rows]).sum()
        window.record(loss_sum, mask[rows].sum())
        # The attached MTP loss must not inherit scheduling scales applied to the
        # policy loss, including a zero policy coefficient in uniform-reward data.
        policy_hidden = torch.ones(size, 2, requires_grad=True)
        output = tinker_mtp.attach_loss(policy_hidden, loss_sum)
        (output.sum() * policy_loss_scale).backward()
        model.mtp.weight.main_grad.add_(model.mtp.weight.grad)
        model.mtp.weight.grad = None
        offset += size
    assert offset == len(inputs)
    window.normalize_gradients([model])
    return model.mtp.weight.main_grad


def test_mtp_gradient_is_invariant_to_partition_and_policy_loss_scaling():
    expected = _partitioned_update([5], policy_loss_scale=1.0)
    for partitions, scale in [([1, 3, 1], 0.125), ([2, 3], 17.0), ([3, 2], 0.0)]:
        torch.testing.assert_close(_partitioned_update(partitions, scale), expected)


def test_window_normalization_uses_global_token_count_before_dp_average(monkeypatch):
    group = object()
    monkeypatch.setattr(tinker_mtp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tinker_mtp.dist, "get_world_size", lambda received: 2 if received is group else pytest.fail())
    models = [_model_with_grad_buffers(), _model_with_grad_buffers()]
    losses, tokens, gradient_sums = [4.0, 15.0], [2.0, 3.0], [6.0, 9.0]
    for rank, model in enumerate(models):
        other = 1 - rank

        def all_reduce(tensor, *, group, peer=other):
            tensor.add_(torch.tensor([losses[peer], tokens[peer]], dtype=tensor.dtype))

        monkeypatch.setattr(tinker_mtp.dist, "all_reduce", all_reduce)
        window = tinker_mtp.MTPWindow()
        window.record(torch.tensor(losses[rank]), torch.tensor(tokens[rank]))
        model.mtp.weight.main_grad.fill_(gradient_sums[rank])
        metrics = window.normalize_gradients([model], dp_group=group)
        assert metrics == {"mtp_loss": 19.0 / 5, "mtp_tokens": 5.0}
    averaged = (models[0].mtp.weight.main_grad + models[1].mtp.weight.main_grad) / 2
    torch.testing.assert_close(averaged, torch.full_like(averaged, sum(gradient_sums) / sum(tokens)))


def test_window_rejects_missing_records_and_missing_mtp_parameters():
    window = tinker_mtp.MTPWindow()
    with pytest.raises(RuntimeError, match="no draft loss"):
        window.normalize_gradients([_model_with_grad_buffers()])
    window.record(torch.tensor(1.0), torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="no MTP parameters"):
        window.normalize_gradients([torch.nn.Linear(2, 2)])


def test_window_normalizes_fallback_grad_once_for_shared_model_references():
    model = _model_with_grad_buffers()
    del model.mtp.weight.main_grad
    model.mtp.weight.grad = torch.full_like(model.mtp.weight, 6.0)
    window = tinker_mtp.MTPWindow()
    window.record(torch.tensor(1.0), torch.tensor(2.0))
    window.normalize_gradients([model, model])
    torch.testing.assert_close(model.mtp.weight.grad, torch.full_like(model.mtp.weight, 3.0))


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("objective", ["loss_weights", "advantages"])
def test_target_mask_excludes_context_zero_weights_and_padding(qkv_format, objective):
    batch = {
        "loss_weights": None,
        "advantages": None,
        "loss_masks": [torch.tensor([1, 1, 0, 1]), torch.tensor([1, 1])],
        "total_lengths": [6, 4],
        "response_lengths": [4, 2],
        "full_loss_masks": torch.ones(1, 12) if qkv_format == "thd" else torch.ones(2, 6),
    }
    batch[objective] = [torch.tensor([1.0, 0.0, 2.0, -1.0]), torch.tensor([0.0, 4.0])]
    mask = tinker_mtp.target_mask(batch, qkv_format=qkv_format)
    expected = (
        [[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]] if qkv_format == "thd" else [[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]]
    )
    torch.testing.assert_close(mask, torch.tensor(expected))


def _roll_reference(tensors, *, packed_seq_params=None, sequence_fields=None, **kwargs):
    """Independent per-sequence reference for the CP1 rolling contract."""
    output = []
    for tensor in tensors:
        result = torch.zeros_like(tensor)
        edges = [0, tensor.shape[-1]] if packed_seq_params is None else packed_seq_params.cu_seqlens_q.tolist()
        for begin, end in zip(edges[:-1], edges[1:], strict=True):
            result[..., begin : end - 1] = tensor[..., begin + 1 : end]
        output.append(result)
    return output


class _Head(torch.nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, hidden) * 0.1)

    def forward(self, hidden, *, weight=None, runtime_gather_output=None):
        return F.linear(hidden, self.weight if weight is None else weight), None


def _cross_entropy(labels, logits):
    return -torch.log_softmax(logits, -1).gather(-1, labels.transpose(0, 1).unsqueeze(-1)).squeeze(-1).transpose(0, 1)


def _loss_arguments(hidden, head, *, layers=1):
    return {
        "hidden_states": hidden,
        "labels": None,
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        # [P,P,A,A] + [P,A,A]. This mask is already aligned to next-token targets.
        "loss_mask": torch.tensor([[0, 1, 1, 0, 1, 1, 0]]),
        "packed_seq_params": SimpleNamespace(cu_seqlens_q=torch.tensor([0, 4, 7])),
        "output_layer": head,
        "compute_language_model_loss": _cross_entropy,
        "config": SimpleNamespace(mtp_num_layers=layers, mtp_detach_heads=True, mtp_loss_scaling_factor=0.2),
    }


def test_packed_mtp_targets_preserve_alignment_and_do_not_cross_sequences():
    torch.manual_seed(8)
    hidden = torch.randn(14, 1, 3, requires_grad=True)
    head = _Head(11, 3)
    arguments = _loss_arguments(hidden, head)
    window = tinker_mtp.MTPWindow()
    output = tinker_mtp._process_window_loss(arguments, window=window, roll_tensor=_roll_reference)
    # One MTP layer predicts two tokens ahead. The first sequence contributes
    # token3+token4; the second contributes token7. The next sequence's first
    # token must never become a target for the preceding sequence.
    expected_labels = torch.tensor([[3, 4, 0, 0, 7, 0, 0]])
    expected_mask = torch.tensor([[1, 1, 0, 0, 1, 0, 0]])
    expected_loss = (_cross_entropy(expected_labels, F.linear(hidden[7:], head.weight)) * expected_mask).sum()
    torch.testing.assert_close(window.totals, torch.stack((expected_loss.detach(), torch.tensor(3.0))).double())
    (output.sum() * 0.0).backward()
    assert head.weight.grad is None
    assert torch.count_nonzero(hidden.grad[:7]) == 0
    assert torch.count_nonzero(hidden.grad[7:][expected_mask.transpose(0, 1) == 0]) == 0
    assert torch.count_nonzero(hidden.grad[7:][expected_mask.transpose(0, 1) != 0]) > 0


def _native_cpu_roll():
    """Load Megatron's pure CPU rolling functions without importing CUDA-only TE.

    Execute the installed source unchanged. This keeps the test usable on CPU
    hosts even when importing the full Megatron module loads libcuda through TE.
    """
    spec = importlib.util.find_spec("megatron")
    if spec is None:
        pytest.skip("Megatron source is not installed")
    candidates = [
        Path(directory) / "core/transformer/multi_token_prediction.py"
        for directory in spec.submodule_search_locations or []
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        pytest.skip("Megatron MTP source is not installed")
    names = {
        "roll_tensor",
        "_roll_tensors_unpacked",
        "_roll_tensors_packed_seq",
        "_roll_tensor_packed_seq_cp1",
        "_get_packed_roll_cu_seqlens",
        "_get_packed_seq_end_indices",
    }
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names, "Megatron's rolling API changed; update the compatibility test"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["roll_tensor"]


def test_native_packed_roll_matches_independent_reference():
    """Exercise actual Megatron packing when that optional runtime is installed."""
    native_roll = _native_cpu_roll()
    head = _Head(11, 3)
    hidden = torch.randn(14, 1, 3, requires_grad=True)
    arguments = _loss_arguments(hidden, head)
    edges = torch.tensor([0, 4, 7], dtype=torch.int32)
    arguments["packed_seq_params"] = SimpleNamespace(cu_seqlens_q=edges, cu_seqlens_kv=edges, qkv_format="thd")
    native_window, reference_window = tinker_mtp.MTPWindow(), tinker_mtp.MTPWindow()
    native_output = tinker_mtp._process_window_loss(arguments, window=native_window, roll_tensor=native_roll)
    reference_output = tinker_mtp._process_window_loss(arguments, window=reference_window, roll_tensor=_roll_reference)
    torch.testing.assert_close(native_output, reference_output)
    torch.testing.assert_close(native_window.totals, reference_window.totals)


def test_loss_context_copies_config_nests_and_restores_after_failure(monkeypatch):
    seen = []

    def original(hidden_states, config, *, input_ids=None):
        seen.append(config)
        return hidden_states

    gpt_model = SimpleNamespace(process_mtp_loss=original)
    monkeypatch.setitem(sys.modules, "megatron.core.models.gpt", SimpleNamespace(gpt_model=gpt_model))
    monkeypatch.setitem(
        sys.modules, "megatron.core.transformer.multi_token_prediction", SimpleNamespace(roll_tensor=_roll_reference)
    )
    config = SimpleNamespace(cross_entropy_loss_fusion=False, cross_entropy_fusion_impl="native")
    head_state = fused_lm_head.HeadState(output_layer=None, original_forward=None, tp_group=None)
    hidden = torch.ones(2, 1, 3)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        with tinker_mtp.loss_context(True), fused_lm_head.activate(head_state):
            wrapped = gpt_model.process_mtp_loss
            with tinker_mtp.loss_context(True):
                assert gpt_model.process_mtp_loss is wrapped
                assert gpt_model.process_mtp_loss(hidden, config) is hidden
            assert gpt_model.process_mtp_loss is wrapped
            raise RuntimeError("deliberate failure")
    assert gpt_model.process_mtp_loss is original
    assert config.cross_entropy_loss_fusion is False and config.cross_entropy_fusion_impl == "native"
    assert seen[0] is not config
    assert seen[0].cross_entropy_loss_fusion is True and seen[0].cross_entropy_fusion_impl == "linear"


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 3)
        self.decoder = torch.nn.Linear(3, 3, bias=False)
        self.mtp = torch.nn.Linear(6, 3, bias=False)
        self.output_layer = _Head(11, 3)
        self.config = SimpleNamespace(mtp_detach_heads=True, mtp_num_layers=1, mtp_loss_scaling_factor=0.2)
        self.post_process = self.mtp_process = True
        self.output_layer.tp_group = None
        self.output_layer.sequence_parallel = False


def _tiny_training_window(partition, enable_mtp):
    torch.manual_seed(29)
    model = _TinyModel()
    window = tinker_mtp.MTPWindow()
    state = fused_lm_head.install(
        [model], copy_to_tp_region=lambda value, **kw: value, gather_from_sp_region=lambda value, **kw: value
    )
    sequences = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8, 9])]
    masks = [torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 1, 1, 0])]
    for indices in partition:
        tokens = torch.cat([sequences[i] for i in indices]).unsqueeze(0)
        mask = torch.cat([masks[i] for i in indices]).unsqueeze(0)
        edges = torch.tensor([0] + list(torch.tensor([len(sequences[i]) for i in indices]).cumsum(0)))
        packed = SimpleNamespace(cu_seqlens_q=edges)
        next_tokens = _roll_reference([tokens], packed_seq_params=packed)[0]
        hidden = model.decoder(model.embedding(tokens)).transpose(0, 1)
        with fused_lm_head.activate(state):
            output = hidden
            if enable_mtp:
                draft_input = torch.cat(
                    [hidden.detach(), model.embedding(next_tokens).transpose(0, 1).detach()], dim=-1
                )
                draft = model.mtp(draft_input)
                output = tinker_mtp._process_window_loss(
                    {
                        "hidden_states": torch.cat([hidden, draft], dim=0),
                        "input_ids": tokens,
                        "loss_mask": mask,
                        "packed_seq_params": packed,
                        "output_layer": model.output_layer,
                        "config": model.config,
                    },
                    window=window,
                    roll_tensor=_roll_reference,
                )
            mapped, _ = model.output_layer(output)
            logprobs = fused_lm_head.target_logprobs(
                mapped.squeeze(1), next_tokens.squeeze(0), None, temperature=1.0, chunk_size=3
            )
            policy_loss = -(logprobs * mask.squeeze(0)).sum()
        policy_loss.backward()
    metrics = window.normalize_gradients([model]) if enable_mtp else {}
    gradients = {name: None if param.grad is None else param.grad.clone() for name, param in model.named_parameters()}
    return gradients, metrics


def test_fused_policy_plus_mtp_matches_work_unit_partitions_and_isolates_policy_gradients():
    baseline, _ = _tiny_training_window([[0], [1]], enable_mtp=False)
    split, metrics = _tiny_training_window([[0], [1]], enable_mtp=True)
    packed, packed_metrics = _tiny_training_window([[0, 1]], enable_mtp=True)
    assert metrics["mtp_tokens"] == packed_metrics["mtp_tokens"] == 5
    assert metrics["mtp_loss"] == pytest.approx(packed_metrics["mtp_loss"])
    for name in baseline:
        if name.startswith("mtp."):
            assert baseline[name] is None
            assert torch.count_nonzero(split[name]) > 0
        else:
            torch.testing.assert_close(split[name], baseline[name])
        torch.testing.assert_close(split[name], packed[name])
