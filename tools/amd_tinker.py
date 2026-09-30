"""Provision and operate the Tinker gateway in an AMD Apptainer image.

This stdlib-only host tool runs on Python 3.9+. It starts serve_tinker.py, whose
Ray lifecycle differs from execute_train's one-shot training jobs. GPU packages
and model definitions are imported only inside the image.

Example: python3 tools/amd_tinker.py serve --dry-run
See docs/platforms/amd-tinker.md for setup and client connection commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# rocm/sgl-dev:miles-rocm720-mi35x-20260914, validated with Qwen3.5-4B.
IMAGE_URI = "docker://rocm/sgl-dev@sha256:f37c0e4d76f64185fe61c2a4362ae96f121d299f113319098c21f2e298afe185"
MI300_IMAGE_URI = "docker://rlsys/miles@sha256:545241a0dbbf8543ef9a6c6d14313b45ba2e78b49ce1ff8370f2c7ab354180e3"
BRIDGE_REVISION = "3fd3768045422d0aa5c97e90a4e6c659aea9acb9"
SGLANG_REVISION = "a929eb72882a8c47478de4d1eefd4a8ebb0716ff"


def _default_root():
    if not os.environ.get("WORK"):
        raise ValueError("Set WORK or pass --root on a persistent filesystem")
    return Path(os.environ["WORK"]) / "miles-tinker"


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "main"
# Install missing gateway dependencies without replacing the image's native stack.
OVERLAY_PACKAGES = (
    "ray[default]",
    "blake3",
    "xxhash",
    "zstandard",
    "polars",
    "safetensors",
    "psycopg[binary]",
    "backports.strenum",
)


@dataclass(frozen=True)
class BackendConfig:
    root: Path = field(default_factory=_default_root)
    repo: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    model_id: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    model_type: str = "qwen3-4B-Instruct-2507"
    port: int = 10613
    train_gpus: int = 4
    rollout_gpus: int = 4
    rollout_gpus_per_engine: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    peak_tflops: float | None = None
    max_tokens_per_gpu: int = 16384
    context_length: int = 32768
    mem_fraction: float = 0.9
    gpu: str = "mi350"
    model_family: str = "qwen3"
    image_dir: Path | None = None
    venv_dir: Path | None = None
    keep_checkpoints: int = 3
    modal_relay: bool = False
    checkpoint_root: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "root", self.root.resolve())
        object.__setattr__(self, "repo", self.repo.resolve())
        object.__setattr__(self, "checkpoint_root", (self.checkpoint_root or self.root / "checkpoints").resolve())
        object.__setattr__(self, "image_dir", (self.image_dir or self.root / "images").resolve())
        object.__setattr__(self, "venv_dir", (self.venv_dir or self.root / "venvs").resolve())
        if self.gpu == "mi300" and self.model_family != "qwen3":
            raise ValueError("MI300 is supported only for Qwen3; use MI350 for Qwen3.5")
        if self.keep_checkpoints < 1:
            raise ValueError("--keep-checkpoints must be at least 1")
        if self.peak_tflops is None:
            object.__setattr__(self, "peak_tflops", 1307.4 if self.gpu == "mi300" else 2309.6)
        sizes = (self.train_gpus, self.rollout_gpus, self.rollout_gpus_per_engine, self.tp, self.pp, self.cp)
        if min(sizes) <= 0 or self.train_gpus % (self.tp * self.pp * self.cp):
            raise ValueError("Positive training GPU count must divide across TP/PP/CP")
        if self.rollout_gpus % self.rollout_gpus_per_engine:
            raise ValueError("Rollout GPU count must divide across inference engines")
        if not 0 < self.port < 65536 or not 0 < self.peak_tflops < float("inf"):
            raise ValueError("Provide a valid port and positive finite dense BF16 peak TFLOPS")
        if not 0 < self.mem_fraction < 1 or min(self.max_tokens_per_gpu, self.context_length) <= 0:
            raise ValueError("Invalid memory fraction or token limit")
        if self.model_id != DEFAULT_MODEL and self.model_type == "qwen3-4B-Instruct-2507":
            raise ValueError("Another model requires its own --model-type")

    @property
    def image_uri(self):
        return MI300_IMAGE_URI if self.gpu == "mi300" else IMAGE_URI

    @property
    def image_workspace(self):
        return "/app" if self.gpu == "mi300" else "/sgl-workspace"

    @property
    def scratch(self):
        runtime_id = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        return Path(os.environ.get("MILES_AMD_TMPDIR", "/tmp")) / f"miles-{os.getuid()}-{runtime_id}-{self.gpu}"

    @property
    def image(self):
        image_id = hashlib.sha256(self.image_uri.encode()).hexdigest()[:12]
        return self.image_dir / f"miles-{image_id}.sif"

    @property
    def python(self):
        return self.venv_dir / self.gpu / "bin/python"

    @property
    def model_path(self):
        return self.root / "models" / self.model_id.rsplit("/", 1)[-1]


def _host_env(config):
    return {
        **os.environ,
        "APPTAINER_CACHEDIR": str(config.scratch / "apptainer-cache"),
        "APPTAINER_TMPDIR": str(config.scratch / "apptainer-tmp"),
        "TMPDIR": str(config.scratch / "apptainer-tmp"),
        "PIP_CACHE_DIR": str(config.scratch / "pip-cache"),
    }


def _runtime_env(config):
    root = config.root
    # Ray appends its own timestamp/PID session name. Keep the base short enough
    # for Unix socket paths even with long deployment roots and large PIDs.
    runtime_id = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    return {
        "HF_HOME": str(root / "hf-cache"),
        "XDG_CACHE_HOME": str(config.scratch / "cache"),
        "TMPDIR": str(config.scratch / "runtime"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{config.repo}:{config.image_workspace}/Megatron-LM:/root/Megatron-LM",
        "PATH": f"{config.python.parent}:/opt/venv/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin",
        "RAY_TMPDIR": f"/tmp/miles-ray-{runtime_id}",
        "RAY_DEDUP_LOGS": "0",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MAX_JOBS": "16",
        "SGLANG_SET_CPU_AFFINITY": "0",
        "AITER_JIT_DIR": str(config.scratch / ("aiter-bf16-rne" if config.gpu == "mi300" else "aiter")),
        **(
            {"CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT": "0", "MILES_SGLANG_LEGACY_WEIGHT_UPDATE": "1"}
            if config.gpu == "mi300"
            else {}
        ),
    }


def _container_command(config, command):
    checkpoint_bind = (
        ["--bind", f"{config.checkpoint_root}:{config.checkpoint_root}"]
        if not config.checkpoint_root.is_relative_to(Path("/tmp"))
        else []
    )
    return [
        "apptainer",
        "exec",
        "--rocm",
        "--cleanenv",
        "--no-mount",
        "cwd",
        "--home",
        f"{config.root}/container-home:/home/miles",
        "--bind",
        "/tmp:/tmp",
        "--bind",
        f"{config.repo}:{config.repo}",
        "--bind",
        f"{config.root}/deps/{config.gpu}/aiter:{config.image_workspace}/aiter",
        *[
            part
            for path in (config.root, config.venv_dir, config.image_dir, config.scratch)
            for part in ("--bind", f"{path}:{path}")
        ],
        *checkpoint_bind,
        "--env",
        ",".join(f"{k}={v}" for k, v in _runtime_env(config).items()),
        "--pwd",
        str(config.repo),
        str(config.image),
        *map(str, command),
    ]


def _run(config, command):
    subprocess.run(list(map(str, command)), env=_host_env(config), check=True)


def _detect_gpu():
    for path in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        props = dict(line.split(maxsplit=1) for line in path.read_text().splitlines() if " " in line)
        target = props.get("gfx_target_version", "")
        if target == "90402":
            return "mi300"
        if target == "90500":
            return "mi350"
    raise ValueError("Cannot detect MI300/MI350; run on a GPU node or pass --gpu")


def _prepare_dirs(config):
    for name in ("container-home", f"deps/{config.gpu}", "logs", "runtime", "models"):
        (config.root / name).mkdir(parents=True, exist_ok=True)
    for name in (
        "apptainer-cache",
        "apptainer-tmp",
        "pip-cache",
        "runtime",
        "cache",
        "aiter-bf16-rne" if config.gpu == "mi300" else "aiter",
    ):
        (config.scratch / name).mkdir(parents=True, exist_ok=True)
    for path in (config.checkpoint_root, config.image_dir, config.venv_dir):
        path.mkdir(parents=True, exist_ok=True)


def _setup_modal(config):
    python = config.venv_dir / "modal/bin/python"
    if not python.exists():
        _run(config, [sys.executable, "-m", "venv", python.parent.parent])
        _run(config, [python, "-m", "pip", "install", "modal==1.2.6"])
    # Credentials stay in the host's normal Modal config, outside the container.
    if not Path(os.environ.get("MODAL_CONFIG_PATH", str(Path.home() / ".modal.toml"))).exists() and not (
        os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")
    ):
        _run(config, [python, "-m", "modal", "token", "new"])


def _setup(config):
    _prepare_dirs(config)
    if not config.image.exists():
        _run(config, ["apptainer", "pull", config.image, config.image_uri])
        _run(config, ["apptainer", "cache", "clean", "--force"])
    if not (config.root / "deps" / config.gpu / "aiter").exists():
        _run(
            config,
            [
                "apptainer",
                "exec",
                "--cleanenv",
                "--no-mount",
                "cwd",
                "--bind",
                "/tmp:/tmp",
                "--bind",
                f"{config.root}:{config.root}",
                "--home",
                f"{config.root}/container-home:/home/miles",
                "--pwd",
                "/tmp",
                config.image,
                "cp",
                "-a",
                f"{config.image_workspace}/aiter",
                config.root / "deps" / config.gpu / "aiter",
            ],
        )
    ready = config.python.parent.parent / ".miles-ready"
    if not ready.exists():
        _run(
            config,
            _container_command(
                config, ["/opt/venv/bin/python", "-m", "venv", "--system-site-packages", config.python.parent.parent]
            ),
        )
        pth_code = """import sys; from pathlib import Path
venv = Path(sys.argv[1]); version = f'{sys.version_info.major}.{sys.version_info.minor}'
pth = venv / f'lib/python{version}/site-packages/image-deps.pth'
pth.write_text(f'import site; site.addsitedir("/opt/venv/lib/python{version}/site-packages")\\n')
    """
        _run(config, _container_command(config, [config.python, "-c", pth_code, config.python.parent.parent]))
        pip = [config.python, "-m", "pip", "install"]
        _run(config, _container_command(config, [*pip, "--no-deps", "--no-build-isolation", "-e", config.repo]))
        if config.gpu == "mi300":
            # Preserve the previously validated MI300 overlay recipe.
            packages = (
                "ray[default]==2.56.0",
                "blake3==1.0.9",
                "xxhash==3.7.1",
                "zstandard==0.25.0",
                "polars==1.42.1",
                "safetensors==0.8.0",
                "psycopg[binary]==3.3.5",
                "backports.strenum==1.3.1",
            )
            _run(config, _container_command(config, [*pip, *packages]))
        else:
            _run(config, _container_command(config, [*pip, "--no-deps", *OVERLAY_PACKAGES]))
        # The image uses Python 3.10; only the SDK's generated protobuf schema is loaded.
        _run(config, _container_command(config, [*pip, "--ignore-requires-python", "--no-deps", "tinker==0.27.1"]))
        if config.gpu == "mi300":
            _run(
                config,
                _container_command(
                    config,
                    [*pip, "--no-deps", "nvidia-modelopt==0.44.0"],
                ),
            )
            _run(config, _container_command(config, [*pip, "pylatexenc==2.10", "apache-tvm-ffi==0.1.13.post3"]))
            _ensure_bridge(config)
            _ensure_sglang(config)
        ready.write_text(config.image_uri + "\n")
    _setup_modal(config)
    _doctor(config)
    print(f"Setup ready: {config.root} ({config.gpu}, {config.model_family})")


def _download_model(config):
    marker = config.model_path / ".miles-download"
    if marker.exists() and marker.read_text() == config.revision:
        return
    download = "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
    _run(
        config,
        _container_command(
            config, [config.python, "-c", download, config.model_id, config.revision, config.model_path]
        ),
    )
    marker.write_text(config.revision)


def _convert(config):
    _prepare_dirs(config)
    _download_model(config)
    output = config.model_path.with_name(f"{config.model_path.name}_torch_dist")
    marker = output / "latest_checkpointed_iteration.txt"
    if marker.exists() and marker.read_text().strip() == "release":
        print(f"Reusing converted checkpoint: {output}")
        return
    if output.exists():
        raise ValueError(f"Incomplete conversion at {output}; move it aside before retrying")
    code = """import runpy, shlex, sys
from miles.utils.external_utils.model_args_utils import load_model_args
model_type, checkpoint, output, context = sys.argv[1:]
sys.argv = [
    'tools/convert_hf_to_torch_dist.py', *shlex.split(load_model_args(model_type)),
    '--hf-checkpoint', checkpoint, '--save', output,
    '--bf16', '--seq-length', context, '--max-position-embeddings', '262144',
    '--attention-backend', 'flash',
]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    _run(
        config,
        _container_command(
            config,
            [
                "env",
                "HIP_VISIBLE_DEVICES=0",
                "WORLD_SIZE=1",
                "RANK=0",
                "LOCAL_RANK=0",
                config.python,
                "-c",
                code,
                config.model_type,
                config.model_path,
                output,
                config.context_length,
            ],
        ),
    )
    if not marker.exists() or marker.read_text().strip() != "release":
        raise RuntimeError(f"Conversion did not finish: {output}")


def _ensure_bridge(config):
    probe = "from megatron.bridge.models.conversion.model_bridge import HFWeightTuple; assert 'megatron_param_name' in HFWeightTuple._fields"
    result = subprocess.run(_container_command(config, [config.python, "-c", probe]), env=_host_env(config))
    if result.returncode == 0:
        return
    bridge = config.root / "deps/mi300/Megatron-Bridge"
    if not bridge.exists():
        _run(config, ["git", "clone", "--no-checkout", "https://github.com/radixark/Megatron-Bridge.git", bridge])
        # This compatible revision predates the fork's bridge-branch rebase.
        _run(config, ["git", "-C", bridge, "fetch", "origin", BRIDGE_REVISION])
        _run(config, ["git", "-C", bridge, "checkout", "--detach", BRIDGE_REVISION])
    else:
        revision = subprocess.check_output(["git", "-C", str(bridge), "rev-parse", "HEAD"], text=True).strip()
        if revision != BRIDGE_REVISION:
            raise ValueError(f"Existing Bridge checkout {bridge} is not at the required {BRIDGE_REVISION}")
    pip = [config.python, "-m", "pip", "install"]
    _run(config, _container_command(config, [*pip, "setuptools>=77,<81", "wheel", "hydra-core==1.3.2"]))
    _run(config, _container_command(config, [*pip, "--no-deps", "--no-build-isolation", bridge]))


def _ensure_sglang(config):
    # Older MI300 images predate the dumper API imported by this Miles branch.
    probe = """import importlib.util, sys
from pathlib import Path
package = Path(importlib.util.find_spec('sglang').origin).parent
sys.exit('class DumperConfig' not in (package / 'srt/debug_utils/dumper.py').read_text())
"""
    result = subprocess.run(_container_command(config, [config.python, "-c", probe]), env=_host_env(config))
    if result.returncode == 0:
        return
    source = config.root / "deps/mi300/sglang"
    if not source.exists():
        _run(
            config,
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/sgl-project/sglang.git",
                source,
            ],
        )
        _run(config, ["git", "-C", source, "checkout", "--detach", SGLANG_REVISION])
    else:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != SGLANG_REVISION:
            raise ValueError(f"Existing SGLang checkout {source} is not at the required {SGLANG_REVISION}")
    # Build against the image's Torch/HIP and detected GPU. Reusing an MI350
    # wheel here would reintroduce the gfx950-only kernel crash on MI300X.
    build = "import os, sys; os.chdir(sys.argv[1]); os.execv(sys.executable, [sys.executable, 'setup_rocm.py', 'bdist_wheel'])"
    _run(config, _container_command(config, [config.python, "-c", build, source / "sgl-kernel"]))
    wheels = list((source / "sgl-kernel/dist").glob("*.whl"))
    wheel = max(wheels, key=lambda path: path.stat().st_mtime)
    pip = [config.python, "-m", "pip", "install", "--no-deps", "--force-reinstall"]
    _run(config, _container_command(config, [*pip, wheel]))
    # A normal wheel takes precedence over the image's existing editable
    # /app/sglang entry; a second editable install does not reliably do so.
    _run(config, _container_command(config, [*pip, source / "python"]))


def _server_args(config, extra=()):
    model_args = (
        [
            "--megatron-to-hf-mode",
            "raw",
            "--ref-load",
            config.model_path.with_name(f"{config.model_path.name}_torch_dist"),
        ]
        if config.model_family == "qwen3.5"
        else ["--megatron-to-hf-mode", "bridge"]
    )
    return list(
        map(
            str,
            [
                "--hf-checkpoint",
                config.model_path,
                *model_args,
                "--tinker-train-mode",
                "full",
                "--tinker-server-port",
                config.port,
                "--tinker-checkpoint-root",
                config.checkpoint_root,
                "--tinker-checkpoints-to-keep",
                config.keep_checkpoints,
                "--balance-data",
                "--tinker-base-model",
                config.model_id,
                "--optimizer",
                "adam",
                "--lr",
                "1e-6",
                "--mfu-peak-tflops",
                config.peak_tflops,
                "--tensor-model-parallel-size",
                config.tp,
                "--pipeline-model-parallel-size",
                config.pp,
                "--context-parallel-size",
                config.cp,
                "--use-dynamic-batch-size",
                "--max-tokens-per-gpu",
                config.max_tokens_per_gpu,
                "--num-gpus-per-node",
                config.train_gpus + config.rollout_gpus,
                "--actor-num-nodes",
                "1",
                "--actor-num-gpus-per-node",
                config.train_gpus,
                "--rollout-num-gpus",
                config.rollout_gpus,
                "--rollout-num-gpus-per-engine",
                config.rollout_gpus_per_engine,
                "--sglang-device",
                "cuda",
                "--sglang-mem-fraction-static",
                config.mem_fraction,
                "--sglang-context-length",
                config.context_length,
                "--sglang-chunked-prefill-size",
                "4096",
                "--sglang-max-running-requests",
                "128",
                "--pause-generation-mode",
                "retract",
                "--update-weight-buffer-size",
                "1073741824",
                "--attention-dropout",
                "0.0",
                "--hidden-dropout",
                "0.0",
                "--accumulate-allreduce-grads-in-fp32",
                "--attention-softmax-in-fp32",
                "--attention-backend",
                "flash",
                *extra,
            ],
        )
    )


_SERVE_CODE = """import json, os, shlex, sys
from pathlib import Path
from miles.utils.external_utils.model_args_utils import load_model_args
root, repo, model_type, extra = sys.argv[1:]
Path(root, 'runtime/server.pid').write_text(str(os.getpid()))
args = [sys.executable, str(Path(repo, 'serve_tinker.py')), *shlex.split(load_model_args(model_type)), *json.loads(extra)]
os.execv(sys.executable, args)
"""


def _serve(config, extra, dry_run):
    command = _container_command(
        config,
        [
            config.python,
            "-c",
            _SERVE_CODE,
            config.root,
            config.repo,
            config.model_type,
            json.dumps(_server_args(config, extra)),
        ],
    )
    if dry_run:
        print(shlex.join(command))
        return
    try:
        with socket.create_connection(("127.0.0.1", config.port), timeout=1):
            raise RuntimeError(f"Port {config.port} is already occupied; reuse or explicitly stop that backend")
    except OSError:
        pass
    _prepare_dirs(config)
    _download_model(config)
    if (
        config.model_family == "qwen3.5"
        and not config.model_path.with_name(f"{config.model_path.name}_torch_dist")
        .joinpath("latest_checkpointed_iteration.txt")
        .exists()
    ):
        raise ValueError("Run convert once for this Qwen3.5 model before serve")
    _serve_with_modal(config, command)


def _handle_serve_sigterm(signum, _frame):
    # Unwind through the supervisor's finally block instead of exiting directly.
    raise SystemExit(128 + signum)


def _serve_with_modal(config, command):
    python = config.venv_dir / "modal/bin/python"
    if config.modal_relay and not python.exists():
        raise ValueError("Run setup first to install Modal")
    relay = None
    server = None
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_serve_sigterm)
    try:
        if config.modal_relay:
            relay = subprocess.Popen(
                [
                    str(python),
                    str(config.repo / "tools/amd_tinker_modal.py"),
                    "--root",
                    str(config.root),
                    "--port",
                    str(config.port),
                ],
                env=_host_env(config),
            )
        server = subprocess.Popen(command, env=_host_env(config))
        while True:
            try:
                result = server.wait(timeout=1)
                if result:
                    raise subprocess.CalledProcessError(result, command)
                return
            except subprocess.TimeoutExpired:
                if relay is not None and relay.poll() is not None:
                    raise RuntimeError("Modal relay exited; stopping the server. Check the relay output.") from None
    finally:
        # Repeated stop requests must not interrupt relay/sandbox cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                if server is not None:
                    _stop(config)
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait()
            finally:
                if relay is not None and relay.poll() is None:
                    relay.terminate()
                    try:
                        relay.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        relay.kill()
                        relay.wait()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)


def _doctor(config):
    _prepare_dirs(config)
    code = """import importlib.util, json, math, sys, torch, miles
from pathlib import Path
from miles.backends.megatron_utils.tinker_perf import perf_metrics
from miles.utils.device_flops import local_peak_bf16_tflops
assert Path(miles.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert torch.version.hip and torch.cuda.is_available()
gpu_arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
te_spec = importlib.util.find_spec('transformer_engine')
te_build = Path(te_spec.origin).parent / 'build_info.txt'
te_targets = None
if te_build.exists():
    for line in te_build.read_text().splitlines():
        if line.startswith('GPU_TARGETS:'):
            te_targets = line.split(':', 1)[1].strip()
            assert gpu_arch in te_targets, f'Transformer Engine targets {te_targets}, but GPU is {gpu_arch}; select the matching image'
x = torch.ones((64, 64), device='cuda', dtype=torch.bfloat16)
assert (x @ x).float().mean().item() == 64
peak = float(sys.argv[2])
m = perf_metrics(actor_train_time=1, total_fwd_tflops=peak/6, num_tokens=128, peak_tflops=peak)
assert math.isclose(m['actor_train_mfu'], .5)
print(json.dumps(dict(miles_import=miles.__file__, torch=torch.__version__, hip=torch.version.hip,
    gpu_count=torch.cuda.device_count(), device=torch.cuda.get_device_name(), gpu_arch=gpu_arch, te_targets=te_targets,
    detected_peak=local_peak_bf16_tflops(), configured_peak=peak, gpu_matmul='passed', mfu_formula='passed'), indent=2))
"""
    _run(config, _container_command(config, [config.python, "-c", code, config.repo, config.peak_tflops]))


def _stop(config):
    groups = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            engine = argv[0] == str(config.python).encode() and b"sglang.launch_server" in argv
            router = argv[0] == b"sglang::router" and (entry / "cwd").resolve() == config.repo.resolve()
            if engine or router:
                groups.add(os.getpgid(int(entry.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    for group in groups:
        if group != os.getpgrp():
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        driver = int((config.root / "runtime/server.pid").read_text())
        entry = Path("/proc") / str(driver)
        if (entry / "cwd").resolve() == config.repo.resolve() and b"serve_tinker.py" in (
            entry / "cmdline"
        ).read_bytes():
            os.kill(driver, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "convert", "serve", "doctor", "stop", "tunnel"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--gpu", choices=("mi300", "mi350"), help="Detected on setup; saved with --root")
    parser.add_argument("--model-family", choices=("qwen3", "qwen3.5"))
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--venv-dir", type=Path)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Checkpoint and performance-log directory, bound into Apptainer (default: <root>/checkpoints)",
    )
    parser.add_argument(
        "--modal-relay",
        action="store_true",
        help="Expose the server through an authenticated ephemeral Modal HTTPS relay",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Optional HF branch, tag, or commit to download during setup (default: main)",
    )
    parser.add_argument("--model-type", default="qwen3-4B-Instruct-2507")
    for name, default in (
        ("port", 10613),
        ("train-gpus", 4),
        ("rollout-gpus", 4),
        ("rollout-gpus-per-engine", 1),
        ("tp", 1),
        ("pp", 1),
        ("cp", 1),
        ("max-tokens-per-gpu", 16384),
        ("context-length", 32768),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--mem-fraction", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-port", type=int)
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if extra and args.action != "serve":
        parser.error(f"Unexpected arguments: {extra}")
    args.root = args.root or _default_root()
    settings = args.root / "setup.json"
    saved = json.loads(settings.read_text()) if settings.exists() else {}
    args.gpu = args.gpu or _detect_gpu()
    args.model_family = args.model_family or ("qwen3.5" if args.model_type.startswith("qwen3.5") else "qwen3")
    for name in ("image_dir", "venv_dir"):
        if getattr(args, name) is None and saved.get(name):
            setattr(args, name, Path(saved[name]))
    fields = BackendConfig.__dataclass_fields__.keys() - {"repo"}
    config = BackendConfig(**{k: getattr(args, k) for k in fields})
    if args.action == "serve":
        _serve(config, extra, args.dry_run)
    elif args.action == "tunnel":
        if not args.remote_host:
            parser.error("tunnel requires --remote-host")
        _run(
            config,
            [
                "ssh",
                "-M",
                "-S",
                config.root / "runtime/tinker-tunnel.sock",
                "-fNT",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-R",
                f"127.0.0.1:{args.remote_port or config.port}:127.0.0.1:{config.port}",
                args.remote_host,
            ],
        )
    else:
        {"setup": _setup, "convert": _convert, "doctor": _doctor, "stop": _stop}[args.action](config)
        if args.action == "setup":
            settings.write_text(
                json.dumps(
                    {name: str(getattr(config, name)) for name in ("gpu", "model_family", "image_dir", "venv_dir")},
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
