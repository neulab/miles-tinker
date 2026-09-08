"""Run Qwen3-4B number-search training on Babel, including setup and the server.

Usage: python deploy/babel/run.py
An explicit command after -- runs that command instead of number-search training.
"""

import argparse
import contextlib
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
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
        "apptainer",
        "exec",
        "--nv",
        "--contain",
        "--pid",
        "--ipc",
        "--no-home",
        "--no-mount",
        "cwd",
        "--no-eval",
        "--bind",
        f"{REPO}:/root/miles",
        "--bind",
        f"{work}:/run/miles",
        "--bind",
        f"{work / 'tmp'}:/tmp",
        "--pwd",
        "/root/miles",
    ]
    for name, destination in (("models", "/root/models"), ("data", "/root/datasets"), ("output", "/root/shared_data")):
        argv += ["--bind", f"{work / name}:{destination}"]
    if Path("/dev/infiniband").exists():
        argv += ["--bind", "/dev/infiniband"]
    netrc = Path(os.environ.get("NETRC", str(Path.home() / ".netrc"))).expanduser()
    if netrc.is_file():
        argv += ["--bind", f"{netrc}:/run/miles/wandb.netrc:ro"]
    if "://" not in image:
        image = str(Path(image).resolve())
    settings = {
        "PYTHONPATH": "/root/miles:/root/Megatron-LM",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "UV_CACHE_DIR": "/run/miles/.cache/uv",
        "HF_HOME": "/run/miles/hf",
        "XDG_CACHE_HOME": "/run/miles/.cache",
        "TMPDIR": "/tmp",
        "NCCL_CUMEM_HOST_ENABLE": "0",  # Babel's Apptainer shared-memory compatibility.
        "NETRC": "/run/miles/wandb.netrc",
    }
    # Inherit the launch environment, including GPU selection and multi-node settings.
    env = os.environ.copy()
    env.setdefault("APPTAINER_CACHEDIR", str(work / "apptainer-cache"))
    for key, value in settings.items():
        env.setdefault(f"APPTAINERENV_{key}", os.environ.get(key, value))
    env["APPTAINERENV_NETRC"] = settings["NETRC"]
    return [*argv, image, "bash", "-c", BOOTSTRAP, "bash", *command], env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="docker://radixark/miles:latest", help="Miles image URI or local SIF path.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp") / f"miles-{os.getuid()}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--learner-gpus", type=int, help="Defaults to half the visible GPUs.")
    parser.add_argument("--rollout-gpus", type=int, help="Defaults to the remaining visible GPUs.")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=1024)
    parser.add_argument("--in-container", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _training_commands(args, gpu_count: int, run_dir: Path, port: int) -> tuple[list[str], list[str]]:
    learners = args.learner_gpus
    if learners is None:
        learners = gpu_count - args.rollout_gpus if args.rollout_gpus is not None else gpu_count // 2
    rollouts = args.rollout_gpus if args.rollout_gpus is not None else gpu_count - learners
    if learners < 1 or rollouts < 1 or learners + rollouts > gpu_count:
        raise ValueError("This separate-server recipe needs at least one learner GPU and one rollout GPU.")
    gateway = [
        sys.executable,
        "examples/tinker/run_qwen3_4b_gateway.py",
        "serve",
        "--mode=full",
        f"--num-gpus-per-node={gpu_count}",
        "--actor-num-nodes=1",
        f"--actor-num-gpus-per-node={learners}",
        f"--rollout-num-gpus={rollouts}",
        "--tp=1",
        f"--max-tokens-per-gpu={args.max_tokens_per_gpu}",
        f"--tinker-port={port}",
        f"--output-dir={run_dir}",
        f"--checkpoint-root={run_dir / 'checkpoints'}",
        "--extra-args=--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 64",
    ]
    client = [
        "/run/miles/client/bin/python",
        str(REPO / "deploy/babel/client.py"),
        f"--tinker_base_url=http://127.0.0.1:{port}",
        "--train.model_name=Qwen/Qwen3-4B-Instruct-2507",
        "--train.renderer_name=qwen3_instruct",
        f"--train.max_training_steps={args.steps}",
        "--train.num_epochs=1000",
        f"--train.optimizer.learning_rate={args.learning_rate}",
        "--train.batch_size=32",
        "--train.workflow_config.group_size=8",
        "--checkpoint.strategy=step",
        "--checkpoint.every=25",
        f"--log_path={run_dir}",
        f"--stats.trial_name={run_dir.name}",
        f"--stats.wandb.mode={args.wandb_mode}",
    ]
    return gateway, client


def _prepare_client() -> tuple[Path, dict[str, str]]:
    checkout = Path("/run/miles/platoon")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/ApGa/platoon.git", str(checkout)], check=True)
    plugin = checkout / "plugins/number-search"
    env = os.environ.copy()
    # The client has its own SDK and CPU dependencies; do not import the server venv.
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CUDNN_PATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(UV_PROJECT_ENVIRONMENT="/run/miles/client", CUDA_VISIBLE_DEVICES="")
    env.setdefault("TINKER_API_KEY", "tml-local")
    subprocess.run(["uv", "sync", "--extra=tinker", "--extra=wandb", "--no-dev"], cwd=plugin, env=env, check=True)
    return plugin, env


def _wait_for_gateway(server, url: str, timeout: float = 1800) -> None:
    deadline = time.monotonic() + timeout
    while server.poll() is None:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Server startup timed out; see gateway.log.")
        time.sleep(1)
    raise RuntimeError(f"Server exited with status {server.returncode}; see gateway.log.")


def _stop_gateway(server) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def _run_training(gateway, client, plugin: Path, client_env, run_dir: Path, port: int) -> None:
    # Platoon's inference proxy constructs ServiceClient without the config URL.
    client_env = {**client_env, "TINKER_BASE_URL": f"http://127.0.0.1:{port}"}
    print(f"Starting the server; log: {run_dir / 'gateway.log'}", flush=True)
    with (run_dir / "gateway.log").open("w") as log:
        server = subprocess.Popen(gateway, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _wait_for_gateway(server, f"http://127.0.0.1:{port}")
            print(f"Starting training; output: {run_dir}", flush=True)
            subprocess.run(client, cwd=plugin, env=client_env, check=True)
        finally:
            _stop_gateway(server)


def _train(args) -> None:
    # Only the container supplies Torch and Miles' launcher dependencies.
    import torch

    from miles.utils.external_utils.command_utils import create_run_id

    run_dir = Path("/root/shared_data") / "number-search" / create_run_id()
    run_dir.mkdir(parents=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    gateway, client = _training_commands(args, torch.cuda.device_count(), run_dir, port)
    print("Setting up the number-search client and dataset...", flush=True)
    plugin, client_env = _prepare_client()
    if args.wandb_mode == "online":
        subprocess.run(
            [client[0], "-c", "import wandb; wandb.login()"],
            cwd=plugin,
            env=client_env,
            check=True,
        )
    print(f"Preparing Qwen3-4B-Instruct-2507; log: {run_dir / 'prepare.log'}", flush=True)
    with (run_dir / "prepare.log").open("w") as log:
        subprocess.run(
            [sys.executable, "examples/tinker/run_qwen3_4b_gateway.py", "prepare"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    _run_training(gateway, client, plugin, client_env, run_dir, port)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0 or args.max_tokens_per_gpu < 1:
        parser.error("Steps, learning rate and token target must be positive.")
    if args.in_container:
        _train(args)
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        command = [
            "python",
            "deploy/babel/run.py",
            "--in-container",
            f"--steps={args.steps}",
            f"--learning-rate={args.learning_rate}",
            f"--wandb-mode={args.wandb_mode}",
            f"--max-tokens-per-gpu={args.max_tokens_per_gpu}",
        ]
        for name in ("learner_gpus", "rollout_gpus"):
            if (value := getattr(args, name)) is not None:
                command.append(f"--{name.replace('_', '-')}={value}")
    work = args.work_dir.resolve()
    argv, env = container_command(work, args.image, command)
    if args.dry_run:
        print(shlex.join(argv), flush=True)
        return
    print(f"Runtime: {work}; training outputs: {work / 'output'}", flush=True)
    for name in ("tmp", "models", "data", "output"):
        (work / name).mkdir(parents=True, exist_ok=True)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
