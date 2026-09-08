"""Run setup, then a repository command in the Miles container on Babel.

Usage: python deploy/babel/run.py -- python scripts/run_qwen3_dense.py --help
With no command, only prepare the environment.
"""

import argparse
import os
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOOTSTRAP = """set -e
python deploy/babel/setup.py --venv /run/miles/venv
source /run/miles/venv/bin/activate
export CUDNN_PATH="${CUDNN_PATH:-$(python -c 'import nvidia; print(next(iter(nvidia.__path__)))')}"
export LD_LIBRARY_PATH="$(python -c 'import nvidia, pathlib; print(":".join(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob("**/lib")))'):${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/root/miles:/root/Megatron-LM:$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
exec "$@"
"""


def container_command(work: Path, image: str, command: list[str]) -> tuple[list[str], dict[str, str]]:
    argv = [
        "apptainer", "exec", "--nv", "--contain", "--pid", "--ipc", "--no-home", "--no-mount", "cwd", "--no-eval",
        "--bind", f"{REPO}:/root/miles", "--bind", f"{work}:/run/miles",
        "--bind", f"{work / 'tmp'}:/tmp", "--pwd", "/root/miles",
    ]
    for name, destination in (("models", "/root/models"), ("data", "/root/datasets"), ("output", "/root/shared_data")):
        argv += ["--bind", f"{work / name}:{destination}"]
    if Path("/dev/infiniband").exists():
        argv += ["--bind", "/dev/infiniband"]
    if "://" not in image:
        image = str(Path(image).resolve())
    settings = {
        "PYTHONPATH": "/root/miles:/root/Megatron-LM",
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "UV_CACHE_DIR": "/run/miles/.cache/uv", "HF_HOME": "/run/miles/hf",
        "XDG_CACHE_HOME": "/run/miles/.cache", "TMPDIR": "/tmp",
        "NCCL_CUMEM_HOST_ENABLE": "0",  # Babel's Apptainer shared-memory compatibility.
    }
    # Inherit the launch environment, including GPU selection and multi-node settings.
    env = os.environ.copy()
    env.setdefault("APPTAINER_CACHEDIR", str(work / "apptainer-cache"))
    for key, value in settings.items():
        env.setdefault(f"APPTAINERENV_{key}", os.environ.get(key, value))
    return [*argv, image, "bash", "-c", BOOTSTRAP, "bash", *command], env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="docker://radixark/miles:latest", help="Miles image URI or local SIF path.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp") / f"miles-{os.getuid()}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    work = args.work_dir.resolve()
    argv, env = container_command(work, args.image, command)
    print(shlex.join(argv), flush=True)
    if not args.dry_run:
        for name in ("tmp", "models", "data", "output"):
            (work / name).mkdir(parents=True, exist_ok=True)
        os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
