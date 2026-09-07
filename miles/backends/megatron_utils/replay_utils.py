import logging

from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.replay_base import ReplayError

logger = logging.getLogger(__name__)


def moe_layer_indices(models) -> list[int]:
    """Global layer ids of the MoE layers this rank built, in construction order.

    One entry per registered router: ``register_replay_list_moe`` maps entry ``i``
    to ``manager.replays[i]``, so the two lists must have the same length.
    """
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)
    return layer_indices


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs) -> int:
    """Map replay streams to Megatron MoE layers using the local model layout.

    Returns the number of streams registered (= the number of local MoE layers).
    """
    layer_indices = moe_layer_indices(models)
    if len(replay_list) != len(layer_indices):
        raise ReplayError(
            f"routing replay: {len(replay_list)} registered routers != {len(layer_indices)} local MoE layers "
            f"(layer ids {layer_indices[:8]}{'...' if len(layer_indices) > 8 else ''}). "
            "A mismatch means the routers registered against a different model than the one being filled "
            "(a second model build in this process, or a layer layout the registration does not follow)."
        )
    num_streams = replay_data.shape[1]
    if layer_indices and layer_indices[-1] >= num_streams:
        raise ReplayError(
            f"routing replay: local MoE layer {layer_indices[-1]} is past the {num_streams} streams the "
            f"rollout tensor carries (shape {tuple(replay_data.shape)}); --num-layers and the model disagree"
        )
    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)
    return len(layer_indices)
