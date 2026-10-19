"""SSH/tmux launcher for an existing allocation of MI350 nodes (one or more GPUs each).

Host Python 3.9+, standard library only. See docs/platforms/amd-tinker-cluster.md.
This does not allocate nodes or change the training backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The host has no installed Miles package; the adjacent launcher is stdlib-only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.amd_tinker import BackendConfig, _container_command, _host_env, _prepare_dirs

_RESULT = "MILES_CLUSTER_RESULT="
_MARKER = "MILES_TINKER_CLUSTER"
_REPO = Path(__file__).resolve().parents[1]
_RESERVED = {
    "--root", "--checkpoint-root", "--gpu", "--train-gpus", "--rollout-gpus", "--port",
    "--rollout-gpus-per-engine", "--num-gpus-per-node", "--actor-num-nodes",
    "--actor-num-gpus-per-node", "--rollout-num-gpus", "--rollout-num-gpus-per-engine",
    "--tinker-checkpoint-root", "--tinker-server-port", "--dry-run", "--colocate",
    "--rollout-external", "--debug-train-only", "--debug-rollout-only", "--eval-num-gpus",
}
# --roce: RCCL over the AMD HPC Fund mi3508x RoCE NICs (its docs: Running Jobs > Multi-node RoCE
# Networking). Without them the weight push between nodes can hang or fall back to TCP sockets.
_ROCE_ENV = {
    "NCCL_IB_HCA": ",".join(f"bnxt_re{i}" for i in range(8)),
    # GPU-direct RDMA breaks the lopsided weight-push group (1 trainer GPU on one node, 8 engine GPUs
    # on the other): the image's RCCL 2.27.7 bootstrap fails with "Message truncated: received 8192
    # bytes instead of 4096". Staging through host memory works and only slows that push.
    "NCCL_NET_GDR_LEVEL": "0",
    "NCCL_IB_GID_INDEX": "3",
}
# Every node starts from a fresh SSH shell, so settings exported on the submitting host would
# otherwise reach only the head; start records these and each node re-applies them.
_PASSTHROUGH_ENV = ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")


def _config(spec):
    root = Path(spec["root"])
    saved = json.loads((root / "setup.json").read_text())
    return BackendConfig(
        root=root, repo=Path(spec["repo"]), gpu="mi350",
        checkpoint_root=Path(spec["checkpoint_root"]),
        image_dir=Path(saved["image_dir"]) if saved.get("image_dir") else None,
        venv_dir=Path(saved["venv_dir"]) if saved.get("venv_dir") else None,
    )


def _serve_command(spec):
    forwarded = spec["serve_args"]
    split = forwarded.index("--") if "--" in forwarded else len(forwarded)
    return [
        "python3", str(Path(spec["repo"]) / "tools/amd_tinker.py"), "serve",
        "--root", spec["root"], "--checkpoint-root", spec["checkpoint_root"], "--gpu", "mi350",
        "--train-gpus", str(spec["train_gpus"]), "--rollout-gpus", str(spec["rollout_gpus"]),
        "--rollout-gpus-per-engine", "1", "--port", str(spec["port"]),
        *forwarded[:split], "--", *forwarded[split + 1:],
        *_actor_layout(spec),
    ]


def _actor_layout(spec):
    # Miles hands out GPUs sorted by node IP, then GPU id: the trainer takes the first
    # train_gpus of them, so whole nodes go to training and its TP groups stay in-node.
    per_node = spec.get("gpus_per_node", 1)
    nodes = max(1, spec["train_gpus"] // per_node)
    return [
        "--num-gpus-per-node", str(per_node), "--actor-num-nodes", str(nodes),
        "--actor-num-gpus-per-node", str(spec["train_gpus"] // nodes),
    ]


def _tmux(spec, *args, check=True):
    return subprocess.run(
        ["tmux", "-L", f"miles-{spec['id']}", *args],
        text=True, capture_output=True, check=check, timeout=10,
    )


def _has_session(spec, name):
    return _tmux(spec, "has-session", "-t", f"={name}", check=False).returncode == 0


def _session_alive(spec, name):
    if not _has_session(spec, name):
        return False
    result = _tmux(spec, "list-panes", "-t", f"={name}:", "-F", "#{pane_dead}")
    return "0" in result.stdout.splitlines()


def _processes():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            env = (entry / "environ").read_bytes().split(b"\0")
            yield int(entry.name), argv, env
        except (OSError, ProcessLookupError):
            continue


def _tagged_processes(spec):
    marker = f"{_MARKER}={spec['id']}".encode()
    return [pid for pid, _, env in _processes() if marker in env and pid != os.getpid()]


def _foreign_processes(spec):
    marker = f"{_MARKER}={spec['id']}".encode()
    return [
        pid for pid, argv, env in _processes()
        if marker not in env and (
            Path(os.fsdecode(argv[0])).name in {"raylet", "gcs_server", "sglang::router"}
            or b"sglang.launch_server" in argv or any(a.endswith(b"/serve_tinker.py") for a in argv)
        )
    ]


def _probe(spec):
    for executable in ("tmux", "apptainer", "ip", "tee"):
        if shutil.which(executable) is None:
            raise ValueError(f"Missing {executable} on {socket.gethostname()}")
    config = _config(spec)
    for path in (config.image, config.root / "deps/mi350/aiter", config.repo / "serve_tinker.py"):
        if not path.exists():
            raise ValueError(f"Missing shared setup path: {path}")
    # The overlay's interpreter can point into the image (/opt/venv/bin/python),
    # so its target need not exist in the host filesystem.
    if not os.path.lexists(config.python):
        raise ValueError(f"Missing shared setup path: {config.python}")
    if len(os.sched_getaffinity(0)) < spec["cpus"]:
        raise ValueError(f"Fewer than {spec['cpus']} CPUs available to this SSH shell")
    devices = []
    for p in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        props = dict(line.split(maxsplit=1) for line in p.read_text().splitlines() if " " in line)
        if int(props.get("simd_count", "0")):
            devices.append(props.get("gfx_target_version"))
    per_node = spec.get("gpus_per_node", 1)
    if devices != ["90500"] * per_node:
        raise ValueError(f"Expected exactly {per_node} MI350 GPU(s); found gfx targets {devices}")
    foreign = _foreign_processes(spec)
    if foreign:
        raise ValueError(f"Unmanaged Ray/Tinker/SGLang processes {foreign}; stop the manual deployment first")
    if _has_session(spec, "serve"):
        raise ValueError("Serve session already exists; use status or stop before another start")
    addresses = json.loads(subprocess.check_output(
        ["ip", "-j", "-4", "addr", "show", "dev", spec["interface"]], text=True,
    ))
    ips = [a["local"] for device in addresses for a in device["addr_info"] if a["scope"] == "global"]
    if len(ips) != 1:
        raise ValueError(f"Expected one IPv4 address on {spec['interface']}; found {ips}")
    return {"ip": ips[0], "ray_session": _has_session(spec, "ray")}


def _node_environment(spec, index):
    passthrough = {**(_ROCE_ENV if spec.get("roce", False) else {}), **spec.get("env", {})}
    return {
        _MARKER: spec["id"],
        f"APPTAINERENV_{_MARKER}": spec["id"],
        "APPTAINERENV_MILES_HOST_IP": spec["ips"][index],
        "APPTAINERENV_NCCL_SOCKET_IFNAME": spec["interface"],
        "APPTAINERENV_GLOO_SOCKET_IFNAME": spec["interface"],
        "APPTAINERENV_RAY_ADDRESS": f"{spec['ips'][0]}:{spec['ray_port']}",
        "PYTHONUNBUFFERED": "1",
        **{f"APPTAINERENV_{key}": value for key, value in passthrough.items()},
    }


def _run_node(spec, index, role):
    # tools/amd_tinker.py reads the allocator settings from this process's environment when it
    # builds the container command, so apply the recorded ones here, not only in env below.
    os.environ.update(spec.get("env", {}))
    config = _config(spec)
    _prepare_dirs(config)
    env = {**_host_env(config), **_node_environment(spec, index)}
    if role == "serve":
        command = _serve_command(spec)
        os.chdir(config.repo)
        os.execvpe(command[0], command, env)
    else:
        env.pop("APPTAINERENV_RAY_ADDRESS", None)
        env.pop("RAY_ADDRESS", None)
        topology = (
            ["--head", f"--port={spec['ray_port']}", "--dashboard-host=127.0.0.1", "--dashboard-port=18265"]
            if index == 0 else [f"--address={spec['ips'][0]}:{spec['ray_port']}"]
        )
        command = _container_command(config, [
            config.python, "-m", "ray.scripts.scripts", "start", *topology,
            f"--node-ip-address={spec['ips'][index]}", f"--num-gpus={spec.get('gpus_per_node', 1)}",
            f"--num-cpus={spec['cpus']}", f"--object-store-memory={spec.get('object_store_gb', 2) << 30}",
            "--min-worker-port=40000", "--max-worker-port=40999",
            "--disable-usage-stats", "--block",
        ])
    # Server/relay logs are handled by amd_tinker.py; capture Ray startup separately.
    directory = Path(spec["checkpoint_root"]) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = directory / f"cluster-{role}-{spec['nodes'][index]}-{stamp}.log"
    log.touch(mode=0o600, exist_ok=False)
    pipeline = f"{shlex.join(command)} 2>&1 | tee --output-error=warn-nopipe --append {shlex.quote(str(log))}"
    os.chdir(config.repo)
    os.execvpe("bash", ["bash", "-o", "pipefail", "-c", pipeline], env)


def _launch_session(spec, index, role):
    if _has_session(spec, role):
        if role == "ray" and _session_alive(spec, role):
            return {"reused": True}
        raise ValueError(f"Session {role} already exists (possibly exited); inspect it and run stop")
    command = [
        "python3", str(Path(spec["repo"]) / "tools/amd_tinker_cluster.py"), "_node",
        f"run-{role}", str(index), json.dumps(spec),
    ]
    _tmux(spec, "new-session", "-d", "-s", role, shlex.join(command),
          ";", "set-option", "-t", f"={role}:", "remain-on-exit", "on")
    return {"started": role}


def _ray_nodes(spec):
    config = _config(spec)
    code = (
        "import json, ray; "
        f"ray.init(address={str(spec['ips'][0] + ':' + str(spec['ray_port']))!r}, logging_level='ERROR'); "
        f"print({_RESULT!r} + json.dumps(["
        "{'ip': n['NodeManagerAddress'], 'gpu': n['Resources'].get('GPU', 0), "
        "'cpu': n['Resources'].get('CPU', 0)} for n in ray.nodes() if n['Alive']])); "
        "ray.shutdown()"
    )
    result = subprocess.run(
        _container_command(config, [config.python, "-c", code]),
        env=_host_env(config), capture_output=True, text=True, timeout=25, check=True,
    )
    return _decode_result(result.stdout)


def _healthy(url, api_key=None):
    headers = {"X-API-Key": api_key} if api_key else {}
    request = urllib.request.Request(f"{url.rstrip('/')}/health", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200 and json.load(response).get("status") == "ok"
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _connection_info(spec):
    if not _session_alive(spec, "serve"):
        return {"state": "exited"}
    if not _healthy(f"http://127.0.0.1:{spec['port']}"):
        return {"state": "initializing gateway"}
    runtime = Path(spec["root"]) / "runtime"
    forwarded = spec["serve_args"]
    options = forwarded[:forwarded.index("--")] if "--" in forwarded else forwarded
    uses_relay = any(a.startswith("--") and "--modal-relay".startswith(a) for a in options)
    if not uses_relay:
        return {"state": "ready", "url": f"http://{spec['ips'][0]}:{spec['port']}", "api_key": "local"}
    try:
        metadata_file = runtime / "modal-relay.json"
        if metadata_file.stat().st_mtime < spec.get("started_at", 0):
            return {"state": "waiting for new relay"}
        metadata = json.loads(metadata_file.read_text())
        api_key = (runtime / "public-api-key").read_text().strip()
        if not api_key or metadata["ssh_pid"] not in _tagged_processes(spec):
            return {"state": "waiting for relay connection"}
        url = metadata["url"]
    except (OSError, ValueError, KeyError):
        return {"state": "waiting for relay credentials"}
    if not _healthy(url, api_key):
        return {"state": "waiting for relay health"}
    return {"state": "ready", "url": url, "api_key": api_key}


def _print_connection(info):
    print("Tinker is ready. Client settings:", flush=True)
    print(f"export TINKER_BASE_URL={shlex.quote(info['url'])}", flush=True)
    print(f"export TINKER_API_KEY={shlex.quote(info['api_key'])}", flush=True)


def _wait_for_server(spec):
    deadline = time.monotonic() + spec["startup_timeout"]
    while time.monotonic() < deadline:
        info = _remote(spec, 0, "connection")
        if info["state"] == "ready":
            _print_connection(info)
            return
        if info["state"] == "exited":
            raise RuntimeError("Serve exited during startup; inspect the serve tmux pane and persistent logs")
        print(f"Waiting for Tinker: {info['state']}...", flush=True)
        time.sleep(10)
    raise TimeoutError("Tinker startup timed out; sessions remain running. Use status to recheck or stop to clean up")


def _signal_owned(spec, sig):
    for pid in _tagged_processes(spec):
        # Recheck ownership immediately before signalling, including after the grace period.
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if f"{_MARKER}={spec['id']}".encode() in env:
                os.kill(pid, sig)
        except (FileNotFoundError, ProcessLookupError):
            pass


def _stop_node(spec):
    _signal_owned(spec, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while _tagged_processes(spec) and time.monotonic() < deadline:
        time.sleep(0.25)
    _signal_owned(spec, signal.SIGKILL)
    _tmux(spec, "kill-server", check=False)
    deadline = time.monotonic() + 3
    while _tagged_processes(spec) and time.monotonic() < deadline:
        time.sleep(0.25)
    if _tagged_processes(spec):
        raise RuntimeError("Some owned processes remain after SIGKILL")
    return {"stopped": True}


def _node_action(action, spec, index):
    if action == "probe":
        return _probe(spec)
    if action.startswith("run-"):
        return _run_node(spec, index, action.removeprefix("run-"))
    if action in {"ray", "serve"}:
        return _launch_session(spec, index, action)
    if action == "ray-nodes":
        return _ray_nodes(spec)
    if action == "connection":
        return _connection_info(spec)
    if action == "interrupt-serve":
        if _session_alive(spec, "serve"):
            _tmux(spec, "send-keys", "-t", "=serve:", "C-c")
        return {}
    if action == "stop":
        return _stop_node(spec)
    if action == "status":
        return {"ray": _session_alive(spec, "ray"), "serve": _session_alive(spec, "serve"),
                "owned_processes": len(_tagged_processes(spec))}
    raise ValueError(f"Unknown node action: {action}")


def _decode_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT):
            return json.loads(line[len(_RESULT):])
    raise ValueError(f"No result from node: {stdout[-1000:]}")


def _remote(spec, index, action):
    command = [
        "python3", str(Path(spec["repo"]) / "tools/amd_tinker_cluster.py"), "_node",
        action, str(index), json.dumps(spec),
    ]
    # sshd drops bursts of new connections ("Connection closed by ... port 22"); those fail before
    # the command runs, so retrying them cannot repeat an action.
    for attempt in range(4):
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
             spec["nodes"][index], shlex.join(command)],
            capture_output=True, text=True, timeout=45,
        )
        refused = result.returncode == 255 and re.search(r"Connection (closed|refused|timed out)", result.stderr)
        if not refused or attempt == 3:
            break
        time.sleep(5 * (attempt + 1))
    if result.returncode:
        raise RuntimeError(f"{spec['nodes'][index]} ({action}): {result.stderr[-2500:]}")
    return _decode_result(result.stdout)


def _wait_for_ray(spec, *, head_only=False):
    expected = set(spec["ips"][:1] if head_only else spec["ips"])
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            nodes = _remote(spec, 0, "ray-nodes")
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"Waiting for Ray: {str(error)[-200:]}", flush=True)
        else:
            actual = {n["ip"] for n in nodes}
            if actual - set(spec["ips"]):
                raise ValueError(f"Unexpected Ray nodes: {actual - set(spec['ips'])}")
            per_node = spec.get("gpus_per_node", 1)
            if expected <= actual and all(n["gpu"] == per_node and n["cpu"] == spec["cpus"] for n in nodes):
                return
            print(f"Waiting for Ray: {nodes}", flush=True)
        time.sleep(3)
    raise TimeoutError(f"Ray not ready; inspect {spec['checkpoint_root']}/logs/cluster-ray-*.log")


def _state_path(root):
    return root / "runtime/ssh-cluster.json"


def _start(spec):
    state = _state_path(Path(spec["root"]))
    if state.exists():
        old = json.loads(state.read_text())
        for key in ("nodes", "repo", "cpus", "interface", "ray_port", "checkpoint_root", "gpus_per_node"):
            if old.get(key, 1) != spec[key]:
                raise ValueError(f"Existing cluster differs in {key}; stop it before changing nodes/settings")
    print("Checking all nodes before starting processes...", flush=True)
    probes = [_remote(spec, i, "probe") for i in range(len(spec["nodes"]))]
    spec = {**spec, "ips": [p["ip"] for p in probes], "started_at": time.time()}
    if len(set(spec["ips"])) != len(spec["ips"]):
        raise ValueError("Node names resolve to duplicate interface IPs")
    state.parent.mkdir(parents=True, exist_ok=True)
    temporary = state.with_suffix(".tmp")
    with open(temporary, "w", opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
        json.dump(spec, stream, indent=2)
    temporary.replace(state)
    _remote(spec, 0, "ray")
    _wait_for_ray(spec, head_only=True)
    for index in range(1, len(spec["nodes"])):
        _remote(spec, index, "ray")
    _wait_for_ray(spec)
    _remote(spec, 0, "serve")
    print("Serve launched; model initialization is still in progress.")
    print(shlex.join(["ssh", "-t", spec["nodes"][0], "tmux", "-L", f"miles-{spec['id']}",
                      "attach-session", "-t", "serve"]))
    print(f"Persistent logs: {spec['checkpoint_root']}/logs/")
    _wait_for_server(spec)


def _stop(spec):
    # Let the original launcher's signal handler close its relay before stopping Ray.
    try:
        _remote(spec, 0, "interrupt-serve")
        deadline = time.monotonic() + 100
        while _remote(spec, 0, "status")["serve"] and time.monotonic() < deadline:
            print("Waiting for server and relay cleanup...", flush=True)
            time.sleep(3)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Could not finish graceful head shutdown; will still clean up workers: {error}", flush=True)
    errors = []
    for index in [*range(1, len(spec["nodes"])), 0]:
        try:
            print(spec["nodes"][index], _remote(spec, index, "stop"), flush=True)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("Some nodes could not be cleaned up; rerun stop:\n" + "\n".join(errors))
    _state_path(Path(spec["root"])).unlink(missing_ok=True)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--nodes", nargs="+")
    parser.add_argument("--train-gpus", type=int)
    parser.add_argument("--rollout-gpus", type=int)
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs on each node (all of them are used)")
    parser.add_argument("--cpus-per-node", type=int, default=24)
    parser.add_argument("--object-store-gb", type=int, default=2, help="Ray object store per node")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--roce", action="store_true", help="Add the mi3508x RoCE NCCL settings on every node")
    parser.add_argument("--ray-port", type=int, default=16379)
    parser.add_argument("--port", type=int, default=11613)
    parser.add_argument("--startup-timeout", type=int, default=900, help="Seconds to wait for Tinker/relay readiness")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without SSH or filesystem writes")
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    forwarded = argv[split + 1:]
    if args.action != "start":
        if forwarded or args.dry_run:
            parser.error("Serve arguments and --dry-run are only supported by start")
        return args, None
    if not args.nodes or not args.train_gpus or not args.rollout_gpus:
        parser.error("start requires --nodes, --train-gpus, and --rollout-gpus")
    if min(args.train_gpus, args.rollout_gpus, args.gpus_per_node, args.cpus_per_node, args.object_store_gb) < 1:
        parser.error("GPU counts, CPUs per node and the object store must be positive")
    if args.startup_timeout < 1:
        parser.error("--startup-timeout must be positive")
    if args.train_gpus + args.rollout_gpus != len(args.nodes) * args.gpus_per_node:
        parser.error("train-gpus + rollout-gpus must equal nodes x gpus-per-node")
    if args.train_gpus > args.gpus_per_node and args.train_gpus % args.gpus_per_node:
        parser.error("Training spanning nodes must take whole nodes (train-gpus a multiple of gpus-per-node)")
    if len(set(args.nodes)) != len(args.nodes) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", n) for n in args.nodes):
        parser.error("Use distinct hostnames or IPv4 addresses, without user@ prefixes")
    if not all(0 < p < 65536 for p in (args.ray_port, args.port)) or args.ray_port == args.port:
        parser.error("Use distinct valid Ray and Tinker ports")
    # The original launcher accepts abbreviated options; reject those too.
    options = {a.split("=", 1)[0] for a in forwarded if a.startswith("--") and a != "--"}
    conflicts = {option for option in options if any(flag.startswith(option) for flag in _RESERVED)}
    if conflicts:
        parser.error(f"Topology and deployment flags are managed by this wrapper: {sorted(conflicts)}")
    root = args.root.resolve()
    return args, {
        "id": hashlib.sha256(str(root).encode()).hexdigest()[:12], "repo": str(_REPO),
        "root": str(root), "checkpoint_root": str((args.checkpoint_root or root / "checkpoints").resolve()),
        "nodes": args.nodes, "train_gpus": args.train_gpus, "rollout_gpus": args.rollout_gpus,
        "gpus_per_node": args.gpus_per_node, "object_store_gb": args.object_store_gb,
        "cpus": args.cpus_per_node, "interface": args.interface, "ray_port": args.ray_port,
        "port": args.port, "serve_args": forwarded,
        "startup_timeout": args.startup_timeout, "roce": args.roce,
        "env": {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ},
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_node":
        action, index, encoded = sys.argv[2:]
        result = _node_action(action, json.loads(encoded), int(index))
        print(_RESULT + json.dumps(result), flush=True)
        return
    args, spec = _parse_args(sys.argv[1:])
    if args.action == "start":
        if args.dry_run:
            print("Nodes (head first):", ", ".join(spec["nodes"]))
            print(f"Ray: {spec['gpus_per_node']} GPU(s), {spec['cpus']} CPUs and {spec['object_store_gb']} GB "
                  f"object store per node, interface {spec['interface']}")
            print("Node environment:", " ".join(
                f"{k.removeprefix('APPTAINERENV_')}={v}" for k, v in
                _node_environment({**spec, "ips": ["<ip>"] * len(spec["nodes"])}, 0).items()
                if k.startswith("APPTAINERENV_NCCL") or k.removeprefix("APPTAINERENV_") in _PASSTHROUGH_ENV
            ))
            print(shlex.join(_serve_command(spec)))
        else:
            _start(spec)
    else:
        spec = json.loads(_state_path(args.root.resolve()).read_text())
        if args.action == "stop":
            _stop(spec)
        else:
            for index, node in enumerate(spec["nodes"]):
                print(node, _remote(spec, index, "status"), flush=True)
            print("Ray resources:", _remote(spec, 0, "ray-nodes"))
            info = _remote(spec, 0, "connection")
            if info["state"] == "ready":
                _print_connection(info)
            else:
                print(f"Tinker: {info['state']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
