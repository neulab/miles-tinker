"""Print the field names of SGLang's ``ServerArgs`` dataclass, one per line.

The miles ``--sglang-<name>`` passthrough (``miles/backends/sglang_utils/arguments.py``)
prefixes ``ServerArgs.add_cli_args``, so a launcher may only emit ``--sglang-*`` flags
that correspond to a real ``ServerArgs`` field of the image it runs in. The
output of this tool is checked in as ``tests/fast/fork/data/sglang_server_args_fields.txt``
and ``tests/fast/fork/test_qwen36_gateway_launcher.py`` checks the launcher against it,
so the fast tests (which have no sglang) catch a stale flag such as
``--sglang-mamba-scheduler-strategy``.

Regenerate inside the serving image whenever it changes::

    singularity exec --env PYTHONNOUSERSITE=1 $SIF \\
        python tools/fork/dump_sglang_server_args.py > tests/fast/fork/data/sglang_server_args_fields.txt

Deprecated CLI aliases (``DeprecatedAliasStoreAction`` entries whose ``dest`` is another
field) are deliberately not listed: the snapshot names canonical fields only.
"""

import dataclasses
import sys


def field_names() -> list[str]:
    from sglang.srt.server_args import ServerArgs  # noqa: PLC0415  only importable in the serving image

    return sorted(field.name for field in dataclasses.fields(ServerArgs))


def main() -> int:
    import sglang  # noqa: PLC0415

    print(f"# sglang {sglang.__version__}: dataclasses.fields(ServerArgs), sorted")
    print("# regenerate: singularity exec --env PYTHONNOUSERSITE=1 $SIF python tools/fork/dump_sglang_server_args.py")
    for name in field_names():
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
