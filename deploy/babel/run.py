"""Run Qwen3-4B Platoon training, including setup and the server.

Usage: python deploy/babel/run.py [--task number-search|litecoder-terminal]
The Miles server runs inside the Apptainer container; the Platoon client runs on
the host, so rollouts can start their own containers. The container shares the
host network namespace, so the client reaches the server on localhost.
An explicit command after -- runs that command in the container instead.
"""

import argparse
import contextlib
import datetime
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
PLATOON_REPO = "https://github.com/ApGa/platoon.git"
# The Platoon commit deploy/babel/patches were written against. A fresh checkout
# is pinned to it, so a new upstream commit cannot silently break the patches.
PLATOON_REV = os.environ.get("PLATOON_REV", "ca0d3adea65a731660c609e969e9a7900c1b2a4f")
PREPARE_COMMAND = ["python", "examples/tinker/run_qwen3_4b_gateway.py", "prepare"]
# Transformer Engine prefers FlashAttention "FA4" (flash_attn.cute) on SM120,
# but its CuTe kernels fail to compile on Blackwell ("unable to compute crd2idx"
# in pack_gqa) while the compiled FA2 kernels in the same image run GQA fine.
# TE gates FA4 on the package metadata, and only imports flash_attn.cute when
# that lookup succeeds:
#     try: fa_utils.fa4_version = PkgVersion(get_pkg_version("flash-attn-4"))
#     except PackageNotFoundError: flash_attn_func_v4 = None
#     else: from flash_attn.cute.interface import ...   # unguarded
# So mask the dist-info, not the package: hiding flash_attn/cute would leave the
# metadata in place and make that unguarded import raise instead.
# BOOTSTRAP puts <work-dir>/venv first on PYTHONPATH, and that venv carries its
# own Transformer Engine, so patching the image copy under /opt/sglang has no
# effect: the venv copy is the one imported. It lives on the host, so it is
# edited in place rather than bind-mounted.
TE_BACKENDS_RELATIVE = (
    "venv/lib/python3.12/site-packages/transformer_engine/pytorch"
    "/attention/dot_product_attention/backends.py"
)
FA4_GUARD = 'try:\n    fa_utils.fa4_version = PkgVersion(get_pkg_version("flash-attn-4"))'
# FA4's CuTe kernels fail to compile on Blackwell ("unable to compute crd2idx"
# in pack_gqa) while the image's compiled FA2 kernels run GQA correctly.
FA4_DISABLED = (
    'try:\n'
    '    raise PackageNotFoundError("flash-attn-4 disabled by miles-tinker on sm_120")\n'
    '    fa_utils.fa4_version = PkgVersion(get_pkg_version("flash-attn-4"))'
)
TERMINAL_PLUGIN = "plugins/terminal-rl"
TERMINAL_MODULE = "platoon.terminal_rl.train_tinker"
OPENHANDS_MODULE = "platoon.terminal_rl.train_openhands"
TERMINAL_SOURCE = "deploy/babel/terminal_rl"
# Tracked trees inside the Platoon checkout that deploy/babel/patches edits.
PATCHED_TREES = ("platoon", "plugins/openhands")


@dataclass(frozen=True)
class Recipe:
    """A training task: where its Platoon plugin lives and how to launch it."""

    plugin: str  # Path of the plugin inside the Platoon checkout.
    module: str  # Training module, run by deploy/babel/client.py.
    client_args: tuple[str, ...] = ()
    source: Path | None = None  # Local plugin sources copied into the checkout.
    env: tuple[tuple[str, str], ...] = ()  # Extra client environment for this task.
    # Pool of task images under --data-dir. The openhands recipes need images
    # carrying /openhands-venv, which live beside the originals rather than
    # replacing them, so the Vanillux recipes keep working unchanged.
    sif_subdir: str = "sifs"
    extras: tuple[str, ...] = ("tinker", "wandb")  # uv extras for the plugin.
    patches: tuple[str, ...] = ()  # deploy/babel/patches scripts to apply to Platoon.


RECIPES = {
    "number-search": Recipe(
        plugin="plugins/number-search",
        module="platoon.number_search.train_tinker",
        client_args=("--train.batch_size=32", "--train.workflow_config.group_size=8"),
    ),
    # Both terminal recipes share one plugin; they differ only in the staged pool
    # and its verifier budget (LiteCoder task.toml allows 600 s, tmax uses 120 s).
    "litecoder-terminal": Recipe(
        plugin=TERMINAL_PLUGIN,
        module=TERMINAL_MODULE,
        source=REPO / TERMINAL_SOURCE,
    ),
    "tmax-15k": Recipe(
        plugin=TERMINAL_PLUGIN,
        module=TERMINAL_MODULE,
        source=REPO / TERMINAL_SOURCE,
        env=(("TERMINAL_RL_TEST_TIMEOUT", "120"),),
    ),
    # Same pool and verifier as tmax-15k, but the model gets the tool-calling
    # interface the terminal-bench evaluation uses instead of <bash> blocks.
    "tmax-15k-openhands": Recipe(
        plugin=TERMINAL_PLUGIN,
        module=OPENHANDS_MODULE,
        source=REPO / TERMINAL_SOURCE,
        env=(("TERMINAL_RL_TEST_TIMEOUT", "120"),),
        extras=("tinker", "wandb", "openhands"),
        patches=(
            "enable_tool_calls.py",
            "openhands_execution_status.py",
            "openhands_finish_detection.py",
            "separate_reasoning_content.py",
            "retain_thinking_in_history.py",
            "reasoning_content_to_thinking.py",
            "forward_backward_timeout.py",
            "act_stall_probe.py",
            "state_updates_are_transparent.py",
            "release_observed_action.py",
            "actions_grouped_by_llm_call.py",
            "drop_infra_killed_rollouts.py",
            "conversation_io_off_the_loop.py",
            "eval_at_step_zero.py",
            "dump_token_logprob_gap.py",
            "hf_chat_template_prompt.py",
            "sglang_tool_call_parser.py",
            "single_tool_call.py",
            "log_prefix_merging.py",
            "active_sampling.py",
            "trajectory_loss_norm.py",
            "send_loss_weights.py",
            "targeted_malformed_credit.py",
            "train_order.py",
            "train_batch_start.py",
            "adaptive_task_selection.py",
            "drop_step_limit_rollouts.py",
            "drop_long_datums.py",
            "targeted_length_credit.py",
            "sample_timeout.py",
            "resume_weights_only.py",
        ),
        sif_subdir="sifs-oh",
    ),
}
BOOTSTRAP = """set -e
python deploy/babel/setup.py --venv /run/miles/venv
source /run/miles/venv/bin/activate
export CUDNN_PATH="${CUDNN_PATH:-$(python -c 'import nvidia; print(next(iter(nvidia.__path__)))')}"
export LD_LIBRARY_PATH="$(python -c 'import nvidia, pathlib; print(":".join(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob("**/lib")))'):${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/root/miles:/root/Megatron-LM:$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
exec "$@"
"""


def _apptainer(work: Path, env: dict[str, str]) -> str:
    if executable := shutil.which("apptainer"):
        return executable
    prefix = work / "apptainer"
    executable = prefix / "usr/bin/apptainer"
    if not executable.is_file():
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}[os.uname().machine]
        package = work / "apptainer.deb"
        print("Installing Apptainer in the runtime directory...", flush=True)
        urllib.request.urlretrieve(
            f"https://github.com/apptainer/apptainer/releases/download/v1.5.3/apptainer_1.5.3_{architecture}.deb",
            package,
        )
        subprocess.run(["dpkg-deb", "-x", str(package), str(prefix)], check=True)
    if not list(prefix.glob("usr/lib/*/libprotobuf-c.so.1")):
        subprocess.run(["apt-get", "download", "libprotobuf-c1"], cwd=prefix, check=True)
        package = next(prefix.glob("libprotobuf-c1_*.deb"))
        subprocess.run(["dpkg-deb", "-x", str(package), str(prefix)], check=True)
    libraries = [str(path) for path in prefix.glob("usr/lib/*-linux-gnu")]
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [*libraries, env.get("LD_LIBRARY_PATH", "")]))
    # The unprivileged binary resolves these directories relative to usr/bin.
    for name in ("etc", "var"):
        link = prefix / "usr" / name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(f"../{name}", target_is_directory=True)
    return str(executable)


def container_command(
    work: Path, image: str, command: list[str], mask_flash_cute: bool = False
) -> tuple[list[str], dict[str, str]]:
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
        "NCCL_CUMEM_HOST_ENABLE": "0",  # Apptainer shared-memory compatibility.
        "NETRC": "/run/miles/wandb.netrc",
    }
    # Inherit the launch environment, including GPU selection and multi-node settings.
    env = os.environ.copy()
    env.setdefault("APPTAINER_CACHEDIR", str(work / "apptainer-cache"))
    for key, value in settings.items():
        env.setdefault(f"APPTAINERENV_{key}", os.environ.get(key, value))
    env["APPTAINERENV_NETRC"] = settings["NETRC"]
    return [*argv, image, "bash", "-c", BOOTSTRAP, "bash", *command], env


def _container_argv(
    work: Path, image: str, command: list[str], mask_flash_cute: bool = False
) -> tuple[list[str], dict[str, str]]:
    argv, env = container_command(work, image, command, mask_flash_cute)
    argv[0] = _apptainer(work, env)
    return argv, env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="docker://radixark/miles:latest", help="Miles image URI or local SIF path.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp") / f"miles-{os.getuid()}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", help="Defaults to the experiment name in the Platoon config.")
    parser.add_argument("--wandb-entity", help="Defaults to your default W&B entity.")
    parser.add_argument("--run-name", help="W&B run name; defaults to the generated run id.")
    parser.add_argument(
        "--resume-run-id",
        help=(
            "Reuse an existing run directory instead of creating one, resuming from its "
            "last checkpoint. The run id feeds both the gateway's checkpoint root and the "
            "client's log path, so reusing it is what lets the trainer find its "
            "checkpoints.jsonl and restore the optimiser state rather than just the weights."
        ),
    )
    parser.add_argument("--learner-gpus", type=int, help="Defaults to half the visible GPUs.")
    parser.add_argument("--rollout-gpus", type=int, help="Defaults to the remaining visible GPUs.")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=1024)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help=(
            "Recompute activations in the backward pass instead of keeping them. "
            "Dynamic batching bins whole samples and never splits one, so a single long "
            "episode becomes a micro-batch of its own; the logits buffer for it is "
            "vocab_size x tokens x 4 bytes and OOMs on top of the resident activations. "
            "Trading compute for that memory is what makes long episodes fit."
        ),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model to train.")
    parser.add_argument(
        "--model-type",
        help="Megatron architecture from scripts/models/; defaults to the model name without its owner.",
    )
    parser.add_argument("--renderer", default="qwen3_instruct", help="tinker-cookbook renderer for the model.")
    parser.add_argument(
        "--mask-flash-cute",
        action="store_true",
        help="Bind a Transformer Engine backends.py that disables FA4, so TE uses the compiled "
        "FA2 kernels; FA4's CuTe kernels do not compile on Blackwell.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=32768,
        help="Sampling context. Terminal episodes append every action and observation, so long "
        "rollouts overrun the gateway default; raise it to the model's limit.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "lora"),
        default="full",
        help="Training mode. lora keeps a 4B learner inside a 48 GB card; full needs ~80 GB.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel size; >1 shards the learner.")
    parser.add_argument("--task", choices=tuple(RECIPES), default="number-search")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Staged task pool; required by the terminal recipes.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument for serve_tinker.py, e.g. --extra-arg=--log-probs-chunk-size=1024; repeatable.",
    )
    parser.add_argument(
        "--client-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument for the Platoon client, e.g. --client-arg=--train.batch_size=1; repeatable.",
    )
    parser.add_argument(
        "--tinker-url",
        help=(
            "Client only: train against a Tinker server that is already running at this URL "
            "(e.g. one started by tools/amd_tinker.py serve) instead of starting the gateway "
            "here. The API key comes from TINKER_API_KEY. Server-side settings (--image, GPU "
            "counts, --tp, --context-length, --extra-arg, ...) are then ignored."
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _model_type(args) -> str:
    """Resolve the Megatron architecture script for the model.

    scripts/models/ is named after the model, so "Qwen/Qwen3.5-4B" finds
    qwen3.5-4B.py, but the lookup is an exact filename match and the repository
    mixes cases. Match case-insensitively and fail here rather than inside the
    container, where the error is a Megatron assertion minutes later.
    """
    wanted = args.model_type or args.model_name.split("/")[-1]
    models = REPO / "scripts/models"
    if (models / f"{wanted}.py").is_file():
        return wanted
    matches = [path.stem for path in models.glob("*.py") if path.stem.lower() == wanted.lower()]
    if matches:
        return matches[0]
    raise SystemExit(
        f"No Megatron architecture at scripts/models/{wanted}.py. "
        f"Pass --model-type with one of: {', '.join(sorted(p.stem for p in models.glob('*.py')))}"
    )


def _disable_flash_attn_4(work: Path) -> None:
    """Make Transformer Engine fall back to FA2 in the launcher's venv.

    TE prefers FlashAttention "FA4" (flash_attn.cute) on SM120, but its CuTe
    kernels fail to compile on Blackwell while the compiled FA2 kernels in the
    same image run GQA correctly. TE gates FA4 on package metadata and then
    imports flash_attn.cute unguarded, so the only clean off switch is to force
    the not-installed branch.
    """
    backends = work / TE_BACKENDS_RELATIVE
    if not backends.is_file():
        raise SystemExit(f"--mask-flash-cute found no Transformer Engine at {backends}; run once without it first.")
    source = backends.read_text()
    if "flash-attn-4 disabled by miles-tinker" in source:
        return
    if FA4_GUARD not in source:
        raise SystemExit(f"Could not find the FA4 guard in {backends}; Transformer Engine changed and this needs review.")
    backends.write_text(source.replace(FA4_GUARD, FA4_DISABLED, 1))
    # A stale timestamp .pyc would still match if mtimes collide; drop it.
    cache = backends.parent / "__pycache__"
    for stale in cache.glob("backends.*.pyc"):
        stale.unlink(missing_ok=True)
    print(f"Disabled FlashAttention 4 in {backends}", flush=True)


def _client_venv(work: Path, task: str) -> Path:
    """One environment per task; their dependency sets differ."""
    return work / f"client-{task}"


def _create_run_id() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d-%H%M%S")
    return f"{stamp}-{random.Random().randint(0, 999):03d}"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _gpu_count(args) -> int:
    """Count the GPUs of the launch environment; the host has no Torch to ask."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([device for device in visible.split(",") if device.strip()])
    listing = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    if listing.returncode == 0:
        return len([line for line in listing.stdout.splitlines() if line.strip()])
    if args.learner_gpus is not None and args.rollout_gpus is not None:
        return args.learner_gpus + args.rollout_gpus
    raise RuntimeError("Cannot detect GPUs; pass both --learner-gpus and --rollout-gpus.")


def _training_commands(
    args,
    recipe: Recipe,
    gpu_count: int,
    run_dir: Path,
    container_run_dir: PurePosixPath,
    port: int,
    work: Path,
    base_url: str,
) -> tuple[list[str], list[str]]:
    learners = args.learner_gpus
    if learners is None:
        learners = gpu_count - args.rollout_gpus if args.rollout_gpus is not None else gpu_count // 2
    rollouts = args.rollout_gpus if args.rollout_gpus is not None else gpu_count - learners
    if learners < 1 or rollouts < 1 or learners + rollouts > gpu_count:
        raise ValueError("This separate-server recipe needs at least one learner GPU and one rollout GPU.")
    # The server writes through the container's view of the output bind mount.
    model_type = _model_type(args)
    gateway = [
        "python",
        "examples/tinker/run_qwen3_4b_gateway.py",
        "serve",
        f"--model-name={args.model_name}",
        f"--megatron-model-type={model_type}",
        f"--mode={args.mode}",
        f"--num-gpus-per-node={gpu_count}",
        "--actor-num-nodes=1",
        f"--actor-num-gpus-per-node={learners}",
        f"--rollout-num-gpus={rollouts}",
        f"--tp={args.tp}",
        f"--max-tokens-per-gpu={args.max_tokens_per_gpu}",
        *(
            ["--recompute-args=--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"]
            if args.recompute
            else []
        ),
        f"--sglang-context-length={args.context_length}",
        f"--tinker-port={port}",
        f"--output-dir={container_run_dir}",
        f"--checkpoint-root={container_run_dir / 'checkpoints'}",
        "--extra-args="
        + " ".join(
            [
                "--chunked-optimizer-state-offload",
                "--optimizer-state-offload-chunk-size-mb 64",
                "--sglang-device cuda",
                *args.extra_arg,
            ]
        ),
    ]
    return gateway, _client_command(args, recipe, run_dir, work, base_url)


def _client_command(args, recipe: Recipe, run_dir: Path, work: Path, base_url: str) -> list[str]:
    client = [
        str(_client_venv(work, args.task) / "bin/python"),
        str(REPO / "deploy/babel/client.py"),
        recipe.module,
        f"--tinker_base_url={base_url}",
        f"--train.model_name={args.model_name}",
        f"--train.renderer_name={args.renderer}",
        f"--train.max_training_steps={args.steps}",
        "--train.num_epochs=1000",
        f"--train.optimizer.learning_rate={args.learning_rate}",
        *recipe.client_args,
        "--checkpoint.strategy=step",
        "--checkpoint.every=25",
        f"--log_path={run_dir}",
        f"--stats.trial_name={run_dir.name}",
        f"--stats.wandb.mode={args.wandb_mode}",
    ]
    for name, value in (("project", args.wandb_project), ("entity", args.wandb_entity), ("name", args.run_name)):
        if value is not None:
            client.append(f"--stats.wandb.{name}={value}")
    # Appended last so an explicit override wins over the recipe's settings.
    client.extend(args.client_arg)
    return client


def _prepare_client(work: Path, task: str, recipe: Recipe, data_dir: Path | None) -> tuple[Path, dict[str, str]]:
    checkout = work / "platoon"
    if not checkout.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", PLATOON_REPO, str(checkout)], check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach", PLATOON_REV], cwd=checkout, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True).stdout.strip()
    if head != PLATOON_REV:
        print(f"Warning: Platoon checkout {checkout} is at {head}, not the pinned {PLATOON_REV}", flush=True)
    venv = _client_venv(work, task)
    if venv.exists() and not (venv / "bin/python").is_file():
        # uv refuses to sync into a directory that is not a valid environment,
        # so a half-written venv would fail every later launch. Rebuild it.
        print(f"Replacing the incomplete client environment at {venv}", flush=True)
        shutil.rmtree(venv)
    plugin = checkout / recipe.plugin
    if recipe.source is not None:
        # Plugins kept in this repository are copied in, so uv resolves the
        # editable Platoon dependency at ../.. exactly as an upstream plugin does.
        shutil.copytree(recipe.source, plugin, dirs_exist_ok=True)
    env = os.environ.copy()
    # The client has its own SDK and CPU dependencies; do not import the server venv.
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CUDNN_PATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(UV_PROJECT_ENVIRONMENT=str(_client_venv(work, task)), CUDA_VISIBLE_DEVICES="")
    env.setdefault("TINKER_API_KEY", "tml-local")
    env.setdefault("UV_CACHE_DIR", str(work / ".cache/uv"))
    # The same directory the container sees as HF_HOME=/run/miles/hf.
    env.setdefault("HF_HOME", str(work / "hf"))
    for key, value in recipe.env:
        env.setdefault(key, value)
    if data_dir is not None:
        env["TERMINAL_RL_DATA_DIR"] = str(data_dir)
        env["SWERL_APPTAINER_SIF_DIR"] = str(data_dir / recipe.sif_subdir)
        env.setdefault("APPTAINER_CACHEDIR", str(work / "apptainer-cache"))
        env.setdefault("APPTAINER_TMPDIR", str(work / "tmp"))
    extras = [f"--extra={name}" for name in recipe.extras]
    subprocess.run(["uv", "sync", *extras, "--no-dev"], cwd=plugin, env=env, check=True)
    if recipe.patches:
        # The checkout is reused between launches and every patch only ever adds
        # to it, so a patch dropped from the recipe would otherwise stay applied
        # for good. Restore the tracked Platoon sources and let the current list
        # rebuild them, which makes the recipe the single description of what is
        # patched. The plugins copied in above are untracked and are left alone.
        subprocess.run(["git", "checkout", "--", *PATCHED_TREES], cwd=checkout, check=True)
    for name in recipe.patches:
        # Patches belong to Platoon core, so they are reapplied after every sync.
        subprocess.run([sys.executable, str(REPO / "deploy/babel/patches" / name), str(checkout)], check=True)
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


def _run_client(client, plugin: Path, client_env, base_url: str) -> None:
    """Drive a Tinker server that something else started and owns (--tinker-url)."""
    client_env = {**client_env, "TINKER_BASE_URL": base_url}
    # The server may still be loading (model, engines, relay); wait rather than fail.
    deadline = time.monotonic() + float(os.environ.get("TINKER_WAIT_S", 3600))
    # A Modal relay answers 401 to any request without the key, /health included.
    health = urllib.request.Request(f"{base_url}/health", headers={"X-API-Key": client_env["TINKER_API_KEY"]})
    while True:
        try:
            with urllib.request.urlopen(health, timeout=30) as response:
                if response.status == 200:
                    break
                error = f"/health answered {response.status}"
        except (OSError, urllib.error.URLError) as exception:
            error = str(exception)
        if time.monotonic() >= deadline:
            raise SystemExit(f"No healthy Tinker server at {base_url}: {error}")
        print(f"Waiting for the Tinker server at {base_url} ({error})", flush=True)
        time.sleep(30)
    print(f"Starting training against {base_url}", flush=True)
    subprocess.run(client, cwd=plugin, env=client_env, check=True)


def _run_training(serve_argv, serve_env, client, plugin: Path, client_env, run_dir: Path, port: int, resume: bool = False) -> None:
    # Platoon's inference proxy constructs ServiceClient without the config URL.
    client_env = {**client_env, "TINKER_BASE_URL": f"http://127.0.0.1:{port}"}
    print(f"Starting the server; log: {run_dir / 'gateway.log'}", flush=True)
    with (run_dir / "gateway.log").open("a" if resume else "w") as log:
        server = subprocess.Popen(
            serve_argv, env=serve_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            _wait_for_gateway(server, f"http://127.0.0.1:{port}")
            print(f"Starting training; output: {run_dir}", flush=True)
            subprocess.run(client, cwd=plugin, env=client_env, check=True)
        finally:
            _stop_gateway(server)


def _train(args) -> None:
    recipe = RECIPES[args.task]
    data_dir = args.data_dir.resolve() if args.data_dir is not None else None
    work = args.work_dir.resolve()
    # The GPUs belong to the server; a client-only launch has none to count.
    gpu_count = 0 if args.tinker_url else _gpu_count(args)
    run_id = args.resume_run_id or _create_run_id()
    run_dir = work / "output" / args.task / run_id
    container_run_dir = PurePosixPath("/root/shared_data") / args.task / run_id
    port = _free_port()
    base_url = args.tinker_url.rstrip("/") if args.tinker_url else f"http://127.0.0.1:{port}"
    gateway, client = (
        _training_commands(args, recipe, gpu_count, run_dir, container_run_dir, port, work, base_url)
        if not args.tinker_url
        else ([], _client_command(args, recipe, run_dir, work, base_url))
    )
    # TINKER_CLIENT_SCRIPT=<script>: start the same gateway, but drive it with this
    # script instead of the Platoon client (e.g. deploy/clients/sign_test.py, which
    # checks that an update moves its own batch the way its advantages say).
    probe = os.environ.get("TINKER_CLIENT_SCRIPT")
    if probe:
        client = [client[0], str(Path(probe).resolve()), f"--base-url={base_url}",
                  f"--base-model={args.model_name}", f"--out={run_dir}"]
    if args.dry_run and args.tinker_url:
        print(shlex.join(client), flush=True)
        return
    if args.dry_run:
        print(shlex.join(container_command(work, args.image, PREPARE_COMMAND, args.mask_flash_cute)[0]), flush=True)
        print(shlex.join(container_command(work, args.image, gateway, args.mask_flash_cute)[0]), flush=True)
        print(shlex.join(client), flush=True)
        return
    print(f"Runtime: {work}; training outputs: {run_dir}", flush=True)
    for name in ("tmp", "home", "models", "data", "output"):
        (work / name).mkdir(parents=True, exist_ok=True)
    # A resumed run deliberately writes back into its own directory: that is
    # where its checkpoints.jsonl lives, and the trainer finds it by path.
    run_dir.mkdir(parents=True, exist_ok=bool(args.resume_run_id))
    print(f"Setting up the {args.task} client and dataset...", flush=True)
    plugin, client_env = _prepare_client(work, args.task, recipe, data_dir)
    if args.wandb_mode == "online":
        subprocess.run([client[0], "-c", "import wandb; wandb.login()"], cwd=plugin, env=client_env, check=True)
    if args.tinker_url:
        _run_client(client, plugin, client_env, base_url)
        return
    prepare_argv, prepare_env = _container_argv(
        work, args.image, [*PREPARE_COMMAND, f"--model-name={args.model_name}"], args.mask_flash_cute
    )
    print(f"Preparing {args.model_name}; log: {run_dir / 'prepare.log'}", flush=True)
    with (run_dir / "prepare.log").open("w") as log:
        subprocess.run(prepare_argv, env=prepare_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.mask_flash_cute:
        _disable_flash_attn_4(work)
    serve_argv, serve_env = _container_argv(work, args.image, gateway, args.mask_flash_cute)
    _run_training(serve_argv, serve_env, client, plugin, client_env, run_dir, port, bool(args.resume_run_id))


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0 or args.max_tokens_per_gpu < 1:
        parser.error("Steps, learning rate and token target must be positive.")
    if RECIPES[args.task].source is not None and args.data_dir is None:
        parser.error(f"--task {args.task} needs --data-dir; stage one with a deploy/babel/prepare_*.py script.")
    if args.data_dir is not None and not (args.data_dir / "selected.json").is_file():
        parser.error(f"{args.data_dir} is not a staged task pool (no selected.json).")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        _train(args)
        return
    work = args.work_dir.resolve()
    argv, env = container_command(work, args.image, command)
    if args.dry_run:
        print(shlex.join(argv), flush=True)
        return
    print(f"Runtime: {work}", flush=True)
    for name in ("tmp", "home", "models", "data", "output"):
        (work / name).mkdir(parents=True, exist_ok=True)
    argv[0] = _apptainer(work, env)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
