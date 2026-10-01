"""Experimental host wrapper for an existing two-node AMD Slurm allocation.

Run ray-head and ray-worker in long-lived srun steps, then serve on the head.
Each node contributes one GPU. Uses the existing AMD image/setup and Tinker
server without modifying the training backend. All paths must be shared.
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

# Host Python has no installed Miles package; reuse the adjacent stdlib launcher.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.amd_tinker import (
    _SERVE_CODE,
    BackendConfig,
    _container_command,
    _download_model,
    _host_env,
    _prepare_dirs,
    _serve_with_modal,
    _server_args,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ray-head", "ray-worker", "serve"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--head-ip", required=True)
    parser.add_argument("--ray-port", type=int, default=16379)
    parser.add_argument("--port", type=int, default=11613)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--model-type", default="qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    config = BackendConfig(
        root=args.root,
        gpu="mi350",
        model_id=args.model_id,
        model_type=args.model_type,
        checkpoint_root=args.root / "checkpoints",
        train_gpus=1,
        rollout_gpus=1,
        context_length=66000,
        max_tokens_per_gpu=50000,
        port=args.port,
        modal_relay=True,
    )
    _prepare_dirs(config)
    node_ip = socket.gethostbyname(socket.gethostname())
    address = f"{args.head_ip}:{args.ray_port}"
    environment = [f"MILES_HOST_IP={node_ip}", "NCCL_SOCKET_IFNAME=eth0", "GLOO_SOCKET_IFNAME=eth0"]
    if args.action.startswith("ray-"):
        topology = (
            ["--head", f"--port={args.ray_port}", "--dashboard-port=18265", "--dashboard-host=127.0.0.1"]
            if args.action == "ray-head"
            else [f"--address={address}"]
        )
        command = _container_command(
            config,
            [
                "env",
                *environment,
                config.python,
                "-m",
                "ray.scripts.scripts",
                "start",
                *topology,
                f"--node-ip-address={node_ip}",
                "--num-gpus=1",
                "--num-cpus=20",
                "--object-store-memory=2147483648",
                "--min-worker-port=40000",
                "--max-worker-port=40999",
                "--disable-usage-stats",
                "--block",
            ],
        )
        os.execvpe(command[0], command, _host_env(config))
    _download_model(config)
    extra = [
        "--num-gpus-per-node",
        "1",
        "--bf16",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--adam-eps",
        "1e-8",
        "--weight-decay",
        "0.017",
        "--clip-grad",
        "1.0",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--balance-data",
    ]
    command = _container_command(
        config,
        [
            "env",
            *environment,
            f"RAY_ADDRESS={address}",
            config.python,
            "-c",
            _SERVE_CODE,
            config.root,
            config.repo,
            config.model_type,
            json.dumps(_server_args(config, extra)),
        ],
    )
    _serve_with_modal(config, command)


if __name__ == "__main__":
    main()
