"""Run one node of the Babel Tinker gateway; runs inside the container with the server venv.

``--role worker`` joins the head's Ray cluster and blocks. ``--role head`` starts
the Ray head, waits for every node's GPUs, downloads the model, starts the
gateway (``gateway.py serve``), publishes ``<work>/endpoints/<name>.endpoint``
once it is healthy, and serves until the gateway fails or the job is stopped.
On exit the endpoint file is removed and ``ray stop`` takes down the gateway's
Ray job, engines and actors on every node.

Arguments not listed below are passed to ``gateway.py`` (``--mode``,
``--model-name``, ``--megatron-model-type``, ``--tp``, ``--extra-args``, ...).
"""

import argparse
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from pathlib import Path

import nvidia
import ray
import torch

REPO = Path(__file__).resolve().parents[3]
GATEWAY = Path(__file__).resolve().parent / "gateway.py"
WORK = Path("/run/miles")
OUTPUT = Path("/root/shared_data")
RAY_PORT = 6379
HEALTH_INTERVAL_S = 10
HEALTH_FAILURES_BEFORE_EXIT = 6


def main() -> None:
    args, gateway_args = _parser().parse_known_args()
    env = _runtime_env()
    gpus_per_node = torch.cuda.device_count()
    head_ip = socket.gethostbyname(args.head_addr)
    if args.role == "worker":
        _run_worker(head_ip=head_ip, gpus_per_node=gpus_per_node, env=env)
    else:
        sys.exit(_run_head(args, gateway_args=gateway_args, head_ip=head_ip, gpus_per_node=gpus_per_node, env=env))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("head", "worker"), required=True)
    parser.add_argument("--head-addr", required=True, help="Head node hostname or IP.")
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--name", required=True, help="Endpoint and output name.")
    parser.add_argument("--port", type=int, required=True, help="Gateway HTTP port.")
    parser.add_argument("--actor-num-nodes", type=int, help="Defaults to 1 on one node, else half the nodes.")
    parser.add_argument("--actor-num-gpus-per-node", type=int, help="Defaults to half the GPUs on one node, else all.")
    parser.add_argument("--rollout-num-gpus", type=int, help="Defaults to every GPU not used by the actor.")
    parser.add_argument("--startup-timeout", type=float, default=3600, help="Seconds until the gateway is healthy.")
    return parser


def _runtime_env() -> dict[str, str]:
    """The environment for Ray and the gateway: this venv first, then the image."""
    venv_bin = str(Path(sys.prefix) / "bin")
    nvidia_roots = list(nvidia.__path__)
    nvidia_libs = [str(path) for root in nvidia_roots for path in Path(root).glob("**/lib")]
    env = os.environ.copy()
    env.setdefault("CUDNN_PATH", nvidia_roots[0])
    env.update(
        VIRTUAL_ENV=sys.prefix,
        PATH=":".join([venv_bin, env.get("PATH", "")]),
        LD_LIBRARY_PATH=":".join(filter(None, [*nvidia_libs, env.get("LD_LIBRARY_PATH", "")])),
        PYTHONPATH=":".join([str(REPO), "/root/Megatron-LM", sysconfig.get_path("purelib")]),
    )
    return env


def _run_worker(*, head_ip: str, gpus_per_node: int, env: dict[str, str]) -> None:
    command = [
        "ray",
        "start",
        f"--address={head_ip}:{RAY_PORT}",
        f"--num-gpus={gpus_per_node}",
        "--disable-usage-stats",
        "--block",
    ]
    # The head may still be starting; --block returns once the head is gone.
    for attempt in range(60):
        if subprocess.run(command, env=env).returncode == 0:
            return
        print(f"Ray worker could not join {head_ip}:{RAY_PORT} (attempt {attempt + 1}/60)", flush=True)
        time.sleep(10)
    sys.exit("Ray worker never joined the head.")


def _run_head(args, *, gateway_args: list[str], head_ip: str, gpus_per_node: int, env: dict[str, str]) -> int:
    run_dir = OUTPUT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    endpoint = WORK / "endpoints" / f"{args.name}.endpoint"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    gateway = None
    try:
        _start_ray_head(head_ip=head_ip, gpus_per_node=gpus_per_node, num_nodes=args.num_nodes, env=env)
        _log_step(f"Preparing the model; log: {run_dir / 'prepare.log'}")
        with (run_dir / "prepare.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(GATEWAY), "prepare", *gateway_args], stdout=log, stderr=log, env=env, check=True
            )
        command = _gateway_command(
            args, gateway_args=gateway_args, run_dir=run_dir, head_ip=head_ip, gpus_per_node=gpus_per_node
        )
        _log_step(f"Starting the gateway; log: {run_dir / 'gateway.log'}")
        gateway_env = {**env, "MILES_SCRIPT_EXTERNAL_RAY": "1", "MASTER_ADDR": head_ip}
        with (run_dir / "gateway.log").open("w") as log:
            gateway = subprocess.Popen(command, stdout=log, stderr=log, env=gateway_env)
        url = f"http://127.0.0.1:{args.port}"
        _wait_until_healthy(gateway, url=url, timeout=args.startup_timeout)
        _publish(endpoint, f"http://{socket.gethostname()}:{args.port}")
        return _serve_while_healthy(gateway, url=url)
    finally:
        _log_step("Stopping the gateway")
        endpoint.unlink(missing_ok=True)
        subprocess.run(["ray", "stop", "--force"], env=env)
        if gateway is not None:
            _stop(gateway)


def _start_ray_head(*, head_ip: str, gpus_per_node: int, num_nodes: int, env: dict[str, str]) -> None:
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={head_ip}",
            f"--port={RAY_PORT}",
            f"--num-gpus={gpus_per_node}",
            "--disable-usage-stats",
        ],
        env=env,
        check=True,
    )
    expected = gpus_per_node * num_nodes
    ray.init(address=f"{head_ip}:{RAY_PORT}", logging_level=logging.ERROR)
    try:
        for _ in range(180):
            if (gpus := ray.cluster_resources().get("GPU", 0)) >= expected:
                break
            _log_step(f"Waiting for the Ray cluster: {gpus:.0f}/{expected} GPUs")
            time.sleep(10)
        else:
            raise TimeoutError(f"Only {gpus:.0f}/{expected} GPUs joined the Ray cluster.")
    finally:
        ray.shutdown()


def _gateway_command(args, *, gateway_args: list[str], run_dir: Path, head_ip: str, gpus_per_node: int) -> list[str]:
    actor_nodes, actor_gpus, rollout_gpus = _topology(args, gpus_per_node)
    _log_step(
        f"Topology: {actor_nodes}x{actor_gpus} actor GPUs, {rollout_gpus} rollout GPUs "
        f"across {args.num_nodes}x{gpus_per_node}"
    )
    return [
        sys.executable,
        str(GATEWAY),
        "serve",
        f"--num-gpus-per-node={gpus_per_node}",
        f"--actor-num-nodes={actor_nodes}",
        f"--actor-num-gpus-per-node={actor_gpus}",
        f"--rollout-num-gpus={rollout_gpus}",
        f"--tinker-port={args.port}",
        f"--output-dir={run_dir}",
        f"--checkpoint-root={run_dir / 'checkpoints'}",
        *gateway_args,
    ]


def _topology(args, gpus_per_node: int) -> tuple[int, int, int]:
    single_node = args.num_nodes == 1
    actor_nodes = args.actor_num_nodes or (1 if single_node else args.num_nodes // 2)
    actor_gpus = args.actor_num_gpus_per_node or (gpus_per_node // 2 if single_node else gpus_per_node)
    rollout_gpus = args.rollout_num_gpus or args.num_nodes * gpus_per_node - actor_nodes * actor_gpus
    if actor_gpus < 1 or rollout_gpus < 1 or actor_nodes * actor_gpus + rollout_gpus > args.num_nodes * gpus_per_node:
        raise ValueError(
            f"Cannot fit {actor_nodes}x{actor_gpus} actor and {rollout_gpus} rollout GPUs "
            f"on {args.num_nodes}x{gpus_per_node}; at least one of each is required."
        )
    return actor_nodes, actor_gpus, rollout_gpus


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/v1/healthz", timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _wait_until_healthy(gateway: subprocess.Popen, *, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not _healthy(url):
        if gateway.poll() is not None:
            raise RuntimeError(f"The gateway exited with status {gateway.returncode}; see gateway.log.")
        if time.monotonic() >= deadline:
            raise TimeoutError("The gateway did not become healthy; see gateway.log.")
        time.sleep(2)


def _serve_while_healthy(gateway: subprocess.Popen, *, url: str) -> int:
    """The child is only the Ray job submission; the gateway's health is the server's state."""
    failures = 0
    while gateway.poll() is None and failures < HEALTH_FAILURES_BEFORE_EXIT:
        time.sleep(HEALTH_INTERVAL_S)
        failures = 0 if _healthy(url) else failures + 1
    if failures:
        _log_step(f"The gateway failed {failures} consecutive health checks.")
        return 1
    _log_step(f"The gateway job exited with status {gateway.returncode}.")
    return gateway.returncode or 0


def _publish(endpoint: Path, url: str) -> None:
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    pending = endpoint.with_suffix(".tmp")
    pending.write_text(f"{url}\n")
    pending.replace(endpoint)
    _log_step(f"Gateway healthy at {url}; endpoint file: {endpoint}")


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)
        return
    process.kill()
    process.wait()


def _log_step(message: str) -> None:
    print(f"[serve] {message}", flush=True)


if __name__ == "__main__":
    main()
