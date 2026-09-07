import argparse

from sglang.srt.server_args import ServerArgs

from miles.utils.workers.argv_utils import render_cli_argv

_ALWAYS_RENDER_FIELDS = ("trust_remote_code", "model_path", "host", "port", "device")


def server_args_to_argv(server_args_dict: dict) -> list[str]:
    destinations = {action.dest for action in _make_cli_parser()._actions}
    legacy_names = {
        "tp_size": "tensor_parallel_size",
        "pp_size": "pipeline_parallel_size",
        "attn_cp_size": "attention_context_parallel_size",
        "moe_dp_size": "moe_data_parallel_size",
        "dp_size": "data_parallel_size",
        "ep_size": "expert_parallel_size",
    }
    return render_cli_argv(
        server_args_dict,
        expected_obj=ServerArgs(**server_args_dict),
        make_parser=_make_cli_parser,
        from_parsed=ServerArgs.from_cli_args,
        always_render_fields=_ALWAYS_RENDER_FIELDS,
        field_to_dest={name: dest for name, dest in legacy_names.items() if name not in destinations},
    )


def parse_server_args_argv(argv: list[str]) -> ServerArgs:
    return ServerArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser
