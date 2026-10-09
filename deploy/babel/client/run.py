"""Train Platoon number-search against a running Tinker gateway.

Runs anywhere with Python 3 and ``uv``: no container, no GPU, and nothing from
the rest of this repository. Platoon is cloned and synced under ``--work-dir``
on first use. Point it at a gateway with ``--endpoint`` or ``--endpoint-file``
(written by ``deploy/babel/server.py``); ``--wait`` polls until the file exists
and the gateway is healthy.

Usage:
  python deploy/babel/client/run.py --endpoint-file .cache/babel/endpoints/<name>.endpoint
  python deploy/babel/client/run.py --endpoint http://127.0.0.1:20613 --steps 2
Arguments after ``--`` are passed to Platoon unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATOON_URL = "https://github.com/ApGa/platoon.git"


def main() -> None:
    parser = _parser()
    args, platoon_args = parser.parse_known_args()
    platoon_args = platoon_args[1:] if platoon_args[:1] == ["--"] else platoon_args
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error("Steps and learning rate must be positive.")
    if (args.endpoint is None) == (args.endpoint_file is None):
        parser.error("Pass exactly one of --endpoint and --endpoint-file.")
    work = args.work_dir.expanduser().resolve()
    run_dir = (args.log_dir or work / "output").expanduser().resolve() / "number-search" / _run_id()
    command = _platoon_command(args, venv=work / "venv", run_dir=run_dir, platoon_args=platoon_args)
    if args.dry_run:
        print(shlex.join(command), flush=True)
        return
    endpoint = args.endpoint or _read_endpoint(args.endpoint_file, wait=args.wait)
    _wait_until_healthy(endpoint, wait=args.wait)
    plugin, env = _prepare_platoon(work=work, ref=args.platoon_ref)
    # Platoon's inference proxy constructs ServiceClient from the environment, not its config.
    env.update(TINKER_BASE_URL=endpoint, TINKER_API_KEY=args.api_key)
    if args.wandb_mode == "online":
        subprocess.run([command[0], "-c", "import wandb; wandb.login()"], cwd=plugin, env=env, check=True)
    run_dir.mkdir(parents=True)
    print(f"Training against {endpoint}; output: {run_dir}", flush=True)
    sys.exit(subprocess.run(command, cwd=plugin, env=env).returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", help="Gateway URL, e.g. http://babel-1-23:20613.")
    parser.add_argument("--endpoint-file", type=Path, help="File containing the gateway URL.")
    parser.add_argument("--wait", type=float, default=0, help="Seconds to wait for the endpoint and its health.")
    parser.add_argument("--work-dir", type=Path, default=Path("~/.cache/miles-babel-client"))
    parser.add_argument("--log-dir", type=Path, help="Defaults to <work-dir>/output.")
    parser.add_argument("--platoon-ref", help="Platoon branch, tag or commit; defaults to the existing checkout.")
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-local"), help="Tenant label.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507", help="Must match the gateway.")
    parser.add_argument("--renderer", default="qwen3_instruct", help="Platoon renderer for --model-name.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true", help="Print the Platoon command and exit.")
    return parser


def _platoon_command(args, *, venv: Path, run_dir: Path, platoon_args: list[str]) -> list[str]:
    return [
        str(venv / "bin/python"),
        str(HERE / "number_search.py"),
        f"--train.model_name={args.model_name}",
        f"--train.renderer_name={args.renderer}",
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
        *platoon_args,
    ]


def _prepare_platoon(*, work: Path, ref: str | None) -> tuple[Path, dict[str, str]]:
    checkout = work / "platoon"
    if not checkout.exists():
        subprocess.run(["git", "clone", PLATOON_URL, str(checkout)], check=True)
    if ref is not None:
        subprocess.run(["git", "fetch", "origin", ref], cwd=checkout, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout, check=True)
    plugin = checkout / "plugins/number-search"
    env = os.environ.copy()
    # The client has its own SDK and CPU dependencies; ignore any active environment.
    for key in ("PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(UV_PROJECT_ENVIRONMENT=str(work / "venv"), CUDA_VISIBLE_DEVICES="")
    subprocess.run(["uv", "sync", "--extra=tinker", "--extra=wandb", "--no-dev"], cwd=plugin, env=env, check=True)
    return plugin, env


def _read_endpoint(path: Path, *, wait: float) -> str:
    deadline = time.monotonic() + wait
    while not path.is_file():
        if time.monotonic() >= deadline:
            sys.exit(f"No endpoint file at {path}.")
        time.sleep(5)
    return path.read_text().strip()


def _wait_until_healthy(endpoint: str, *, wait: float) -> None:
    deadline = time.monotonic() + wait
    while True:
        try:
            with urllib.request.urlopen(f"{endpoint}/api/v1/healthz", timeout=5) as response:
                status = f"HTTP {response.status}"
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            status = str(error)
        if time.monotonic() >= deadline:
            sys.exit(f"The gateway at {endpoint} is not healthy: {status}")
        time.sleep(5)


def _run_id() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d-%H%M%S") + f"-{os.getpid() % 1000:03d}"


if __name__ == "__main__":
    main()
