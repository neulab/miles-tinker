def model_args(**kwargs) -> str:
    # meta-models/Muse-Glimmer-30B, its text model (text_config). It only runs in bridge mode
    # (--megatron-to-hf-mode bridge): megatron-bridge's MuseGlimmerModelBuilder builds the real model from the
    # HF config (three sliding-window layers to one global layer, no RoPE on the global ones, the gated
    # attention, the four sandwich norms per layer, the logit multiplier and softcap, the vision tower), so
    # these flags describe only the shape that hf_validate_args checks against the HF config and that the
    # FLOPs model reads.
    return (
        "--num-layers 52 "
        "--hidden-size 6656 "
        "--ffn-hidden-size 19968 "
        "--num-attention-heads 32 "
        "--group-query-attention "
        "--num-query-groups 2 "
        "--kv-channels 128 "
        "--attention-output-gate "
        "--qk-layernorm "
        "--swiglu "
        "--disable-bias-linear "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--norm-epsilon 1e-5 "
        "--position-embedding-type rope "
        "--rotary-base 500000 "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 202048 "
        # 202048 = 2^6 * 3157: Megatron pads the vocabulary to a multiple of this times TP, so 1 keeps it at
        # the bridge's unpadded 202048 for every TP up to 64
        "--make-vocab-size-divisible-by 1 "
    )
