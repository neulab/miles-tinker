"""CPU stand-in for Megatron's fused vocab-parallel cross entropy (TP=1 only).

``math_utils.compute_log_probs`` imports ``megatron.core.fusions.fused_cross_entropy``
lazily, and the CPU fast-test venv has no Megatron. A test that scores logprobs
through the production path calls :func:`install` (in every spawned worker) so the
import resolves to this module; a real Megatron, when importable, is left alone.
"""

import importlib
import sys
import types

import torch
import torch.distributed as dist

MODULE = "megatron.core.fusions.fused_cross_entropy"


def fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group=None):
    """``-log_softmax(logits)[target]`` over the full vocabulary: valid for one TP rank."""
    if tp_group is not None and dist.is_initialized():
        assert dist.get_world_size(tp_group) == 1, "the CPU cross-entropy stand-in supports TP=1 only"
    log_probs = torch.log_softmax(vocab_parallel_logits.float(), dim=-1)
    return -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


def install() -> None:
    try:
        importlib.import_module(MODULE)
        return
    except ImportError:
        pass
    for name in ("megatron", "megatron.core", "megatron.core.fusions"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    module = types.ModuleType(MODULE)
    module.fused_vocab_parallel_cross_entropy = fused_vocab_parallel_cross_entropy
    sys.modules[MODULE] = module
