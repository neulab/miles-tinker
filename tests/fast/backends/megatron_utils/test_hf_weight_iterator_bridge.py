"""The bridge weight iterator takes both export shapes of megatron-bridge.

radixark's megatron-bridge (the serving image's) yields ``(hf_name, weight, megatron_name)``. Newer ones (8cd3466d,
the Muse Glimmer overlay's) yield ``(hf_name, weight)`` unless the export is called with ``with_megatron_names``,
and then ``(hf_name, weight, (megatron_name, ...))``. The iterator asks for the names where it can, and unpads,
quantizes and groups by them either way.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as bridge_iterator


def _iterator(vocab_size: int):
    iterator = object.__new__(bridge_iterator.HfWeightIteratorBridge)
    iterator.args = Namespace(vocab_size=vocab_size)
    iterator.quantization_config = None
    return iterator


def test_source_names_of_every_export_shape():
    weight = torch.zeros(1)
    assert bridge_iterator._source_names(("w", weight, "decoder.layers.0.mlp.linear_fc2.weight")) == (
        "decoder.layers.0.mlp.linear_fc2.weight",
    )
    assert bridge_iterator._source_names(("w", weight, ("expert.0", "expert.1"))) == ("expert.0", "expert.1")
    assert bridge_iterator._source_names(("w", weight, ())) == ()
    assert bridge_iterator._source_names(("w", weight, None)) == ()
    assert bridge_iterator._source_names(("w", weight)) == ()


def test_the_names_are_asked_for_only_where_the_export_takes_the_flag():
    def radixark_export(model, cpu=False, conversion_tasks=None, merge_adapter_weights=True):
        return []

    def newer_export(model, cpu=False, conversion_tasks=None, merge_adapter_weights=True, with_megatron_names=False):
        return []

    assert bridge_iterator._source_name_kwargs(radixark_export) == {}
    assert bridge_iterator._source_name_kwargs(newer_export) == {"with_megatron_names": True}


EMBED, HEAD = "embedding.word_embeddings.weight", "output_layer.weight"


@pytest.mark.parametrize("shape", ["radixark", "with_megatron_names"])
def test_both_shapes_unpad_the_vocabulary_and_carry_name_tuples(shape):
    padded = torch.arange(12.0).reshape(6, 2)  # a padded vocabulary of 6 rows, 4 of them real
    norm = torch.ones(2)
    source = (lambda name: name) if shape == "radixark" else (lambda name: (name,))
    exported = [
        ("model.embed_tokens.weight", padded, source(EMBED)),
        ("lm_head.weight", padded, source(HEAD)),
        ("model.norm.weight", norm, source("decoder.final_norm.weight")),
    ]
    out = list(_iterator(vocab_size=4)._postprocess_and_quantize(exported, "base"))
    assert [name for name, _, _ in out] == ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    assert [names for _, _, names in out] == [(EMBED,), (HEAD,), ("decoder.final_norm.weight",)]
    assert out[0][1].shape == (4, 2) and out[1][1].shape == (4, 2) and torch.equal(out[2][1], norm)


def test_a_tensor_without_a_megatron_source_passes_through():
    padded = torch.arange(12.0).reshape(6, 2)
    out = list(_iterator(vocab_size=4)._postprocess_and_quantize([("model.extra.weight", padded)], "base"))
    assert out[0][0] == "model.extra.weight" and out[0][2] == () and torch.equal(out[0][1], padded)


def test_lora_base_layer_names_are_folded():
    weight = torch.ones(2, 2)
    exported = [("model.layers.0.q_proj.base_layer.weight", weight, ("decoder.layers.0.q.weight",))]
    out = list(_iterator(vocab_size=4)._postprocess_and_quantize(exported, "lora"))
    assert out[0][0] == "model.layers.0.q_proj.weight"
