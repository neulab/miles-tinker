"""Start one node of the Tinker gateway in the Miles Apptainer image.

Runs on the host, once per node: ``server.sbatch`` starts it through ``srun``,
and outside Slurm it serves from this node alone. The first Slurm task is the
head: it builds the server venv (``container/setup_env.py``), then runs the Ray
head and the gateway (``container/serve.py``). The other tasks wait for that
venv and join as Ray workers. Once the gateway is healthy its URL is written to
``<work-dir>/endpoints/<name>.endpoint``; clients only need that URL.

Requires ``apptainer`` on PATH, a local SIF image (``--image``; cache the pinned
image with ``image.py`` first) and a shared ``--work-dir`` for multi-node jobs.

Usage: python deploy/babel/server.py --image <sif> [options] [gateway options]
Unrecognised options go to ``container/serve.py`` and ``container/gateway.py``,
e.g. ``--mode full --model-name Qwen/Qwen3-1.7B --megatron-model-type qwen3-1.7B``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTAINER_REPO = "/root/miles"
CONTAINER_WORK = "/run/miles"
CONTAINER_VENV = f"{CONTAINER_WORK}/venv"
CONTAINER_SCRIPTS = f"{CONTAINER_REPO}/deploy/babel/container"
WORK_SUBDIRS = ("home", "models", "data", "output", "endpoints")


def main() -> None:
    parser = _parser()
    args, serve_args = parser.parse_known_args()
    if not Path(args.image).is_file():
        parser.error(f"--image must be a local SIF file; {args.image} is not. Cache one with deploy/babel/image.py.")
    work = args.work_dir.resolve()
    node_tmp = _node_tmp()
    apptainer, env = _apptainer_prefix(work=work, node_tmp=node_tmp, image=args.image)
    setup = [*apptainer, "python", f"{CONTAINER_SCRIPTS}/setup_env.py", f"--venv={CONTAINER_VENV}"]
    if args.use_fa2:
        setup.append("--use-fa2")
    serve = [
        *apptainer,
        f"{CONTAINER_VENV}/bin/python",
        f"{CONTAINER_SCRIPTS}/serve.py",
        *_serve_args(args),
        *serve_args,
    ]
    if args.dry_run:
        print(shlex.join(setup) if _is_head() else "# setup: waits for the head's venv", flush=True)
        print(shlex.join(serve), flush=True)
        return
    for name in WORK_SUBDIRS:
        (work / name).mkdir(parents=True, exist_ok=True)
    node_tmp.mkdir(parents=True, exist_ok=True)
    _prepare_venv(setup, env=env, ready=work / "venv/.ready")
    if _is_head():
        print(f"Endpoint: {work / 'endpoints' / f'{args.name}.endpoint'}; outputs: {work / 'output' / args.name}")
    os.execvpe(serve[0], serve, env)


def _parser() -> argparse.ArgumentParser:
    job = os.environ.get("SLURM_JOB_ID")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Local Miles SIF (deploy/babel/image.py).")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp") / f"miles-{os.getuid()}")
    parser.add_argument("--name", default=f"gateway-{job or 'local'}", help="Endpoint file and output directory name.")
    parser.add_argument("--port", type=int, help="Gateway port; defaults to 20000 + job id %% 20000, or a free port.")
    parser.add_argument("--use-fa2", action="store_true", help="Use FlashAttention 2 in Transformer Engine on SM8x.")
    parser.add_argument("--dry-run", action="store_true", help="Print the container commands and exit.")
    return parser


def _serve_args(args) -> list[str]:
    hosts = _slurm_hosts()
    return [
        f"--role={'head' if _is_head() else 'worker'}",
        f"--head-addr={hosts[0] if hosts else '127.0.0.1'}",
        f"--num-nodes={len(hosts) or 1}",
        f"--name={args.name}",
        f"--port={args.port or _default_port()}",
    ]


def _apptainer_prefix(*, work: Path, node_tmp: Path, image: str) -> tuple[list[str], dict[str, str]]:
    argv = [
        "apptainer",
        "exec",
        "--nv",
        "--contain",
        "--pid",
        "--ipc",
        "--home",
        f"{work / 'home'}:{Path.home()}",
        "--no-mount",
        "cwd",
        "--no-eval",
        "--bind",
        f"{REPO}:{CONTAINER_REPO}",
        "--bind",
        f"{work}:{CONTAINER_WORK}",
        "--bind",
        f"{node_tmp}:/tmp",
        "--pwd",
        CONTAINER_REPO,
    ]
    for name, destination in (("models", "/root/models"), ("data", "/root/datasets"), ("output", "/root/shared_data")):
        argv += ["--bind", f"{work / name}:{destination}"]
    if Path("/dev/infiniband").exists():
        argv += ["--bind", "/dev/infiniband"]
    settings = {
        "PYTHONPATH": f"{CONTAINER_REPO}:/root/Megatron-LM",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "UV_CACHE_DIR": f"{CONTAINER_WORK}/.cache/uv",
        "HF_HOME": f"{CONTAINER_WORK}/hf",
        "XDG_CACHE_HOME": f"{CONTAINER_WORK}/.cache",
        "TMPDIR": "/tmp",
        "NCCL_CUMEM_HOST_ENABLE": "0",  # Apptainer shared-memory compatibility.
    }
    # Inherit the launch environment, including GPU selection and network settings.
    env = os.environ.copy()
    for key, value in settings.items():
        env.setdefault(f"APPTAINERENV_{key}", os.environ.get(key, value))
    return [*argv, str(Path(image).resolve())], env


def _prepare_venv(setup: list[str], *, env: dict[str, str], ready: Path, timeout: float = 3600) -> None:
    """The head builds the shared venv; workers wait until it is ready for this job."""
    token = os.environ.get("SLURM_JOB_ID", "local")
    if _is_head():
        ready.unlink(missing_ok=True)
        subprocess.run(setup, env=env, check=True)
        ready.write_text(token)
        return
    deadline = time.monotonic() + timeout
    while not (ready.is_file() and ready.read_text() == token):
        if time.monotonic() >= deadline:
            sys.exit(f"The head did not finish setting up {ready.parent}.")
        time.sleep(5)


def _node_tmp() -> Path:
    # Ray keeps sockets and session files under /tmp; give every node its own local directory.
    job = os.environ.get("SLURM_JOB_ID", "local")
    return Path(os.environ.get("TMPDIR", "/tmp")) / f"miles-{os.getuid()}-{job}-{socket.gethostname()}"


def _is_head() -> bool:
    return os.environ.get("SLURM_PROCID", "0") == "0"


def _slurm_hosts() -> list[str]:
    if "SLURM_JOB_NODELIST" not in os.environ:
        return []
    output = subprocess.run(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], capture_output=True, text=True, check=True
    ).stdout
    return output.split()


def _default_port() -> int:
    # Stable per job, so every task and a client computing it agree; outside Slurm, any free port.
    if job := os.environ.get("SLURM_JOB_ID"):
        return 20000 + int(job) % 20000
    with socket.socket() as listener:
        listener.bind(("", 0))
        return listener.getsockname()[1]


if __name__ == "__main__":
    main()
