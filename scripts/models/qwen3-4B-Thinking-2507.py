from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Architecturally identical to Qwen3-4B-Instruct-2507 -- same 36 layers,
    # 2560 hidden, 9728 ffn, 32 heads / 8 kv groups, 128 head_dim, 151936 vocab
    # and rope_theta 5000000. Only the chat template and the weights differ, and
    # neither is Megatron's concern. The base qwen3-4B uses rope_theta 1000000
    # and a 40960 context, so it is the wrong sibling to borrow from.
    return load_sibling_model_args(__file__, "qwen3-4B", rotary_base=5000000)
