"""DRY_RUN coverage of the ABCI deployment scripts (design 10.1): no PBS, no GPU, no containers.

- `bash -n` on every deploy shell script;
- `tinker_gateway.pbs` DRY_RUN on 2 and 15 fake nodes: roles.tsv (trainers = TRAIN_NODES lowest IPoIB IPs by
  numeric per-octet sort), every pbsdsh/singularity line, `--actor-num-nodes/--rollout-num-gpus` ahead of
  LAUNCHER_ARGS, the C10 client exports, the 4-phase teardown sweep, and no secret name in the output;
- `toolathlon_service.sh` DRY_RUN: resolved SIF + sha256 (2fb0ee8b...), the exact guard argv, the container
  command with the release env, no secret name (skipped when the SIF is unreadable on this host);
- the copied memory guard self-test: the helper's argv against a fake cgroup dir prints `event=start`.
"""

import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy" / "abci"
PBS = DEPLOY / "tinker_gateway.pbs"
SERVICE = DEPLOY / "toolathlon_service.sh"
GUARD = DEPLOY / "openreward-node-memory-guard.py"
SECRET_NAMES = [
    "WANDB_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "LITELLM_API_KEY", "LITELLM_BASE_URL",
    "RUBRIC_DEFAULT_LLM", "HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
]
SECRET_RE = re.compile("|".join(SECRET_NAMES) + "|LEAKED_")
SIF_SHA = "2fb0ee8ba5377a79206af36c4645c6e32242300aad5cf08677addc8e25ba468d"

SHELL_SCRIPTS = sorted(
    list(DEPLOY.glob("*.sh")) + list(DEPLOY.glob("*.pbs")) + list(DEPLOY.glob("clients/*.sh"))
    + list(DEPLOY.glob("*.env")) + list((REPO / "deploy" / "clients").glob("*.sh"))
)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def run_pbs_dry(tmp_path: Path, hosts: list[str], ips: list[str], **env) -> tuple[str, Path]:
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("".join(f"{h}\n" * 8 for h in hosts))  # mpiprocs=8: eight lines per host
    project = tmp_path / "project"
    local = tmp_path / "local"
    local.mkdir()
    run_env = {
        "PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path)), "USER": os.environ.get("USER", "u"),
        "DRY_RUN": "1", "DRY_RUN_IPS": ",".join(ips), "PBS_NODEFILE": str(nodefile), "PBS_JOBID": "2245678.pbs1",
        "PBS_LOCALDIR": str(local), "PROJECT_ROOT": str(project), "MILES_SRC": str(REPO), "HF_SNAPSHOT": "/x/snap",
        "RUN_NAME": "dry", "WANDB_API_KEY": "LEAKED_WANDB", "HF_TOKEN": "LEAKED_HF", "OPENAI_API_KEY": "LEAKED_OAI",
    }
    run_env.update(env)
    proc = subprocess.run(["bash", str(PBS)], env=run_env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr, project / "jobs" / "dry" / "2245678"


FIFTEEN_HOSTS = [f"hnode{n:03d}" for n in (31, 48, 200, 12, 55, 7, 99, 100, 101, 102, 103, 104, 105, 106, 107)]
FIFTEEN_IPS = [
    "10.0.1.9", "10.0.1.10", "10.0.0.200", "10.0.1.100", "10.0.0.5", "10.0.2.1", "10.0.0.7", "10.0.0.99",
    "10.0.0.100", "10.0.0.101", "10.0.0.102", "10.0.0.103", "10.0.0.104", "10.0.0.105", "10.0.0.106",
]


def test_fifteen_nodes_roles_services_and_exports(tmp_path):
    out, jobdir = run_pbs_dry(
        tmp_path, FIFTEEN_HOSTS, FIFTEEN_IPS, MODE="full", SERVICES="1",
        GATEWAY_LAUNCHER="examples/tinker/run_qwen3_6_35b_a3b_gateway.py", LAUNCHER_ARGS="--check-weight-update-equal",
        CLIENT_SCRIPT=str(DEPLOY / "clients" / "smoke_q36.sh"),
    )
    roles = [line.split("\t") for line in (jobdir / "roles.tsv").read_text().splitlines()]
    assert [r[3] for r in roles] == ["train"] * 4 + ["engine"] * 11
    # numeric per-octet order (10.0.0.99 < 10.0.0.100 < 10.0.0.200 < 10.0.1.9 < 10.0.1.10 < 10.0.1.100 < 10.0.2.1)
    assert [r[2] for r in roles] == sorted(FIFTEEN_IPS, key=lambda ip: tuple(map(int, ip.split("."))))
    assert [r[0] for r in roles[:4]] == ["hnode055", "hnode099", "hnode100", "hnode101"]
    assert {r[1] for r in roles} == {str(8 * i) for i in range(15)}  # vnode index = first line of the host in PBS_NODEFILE
    engine_ips = [r[2] for r in roles[4:]]
    engine_hosts = [r[0] for r in roles[4:]]
    assert "nodes=15 train_nodes=4 rollout_gpus=88" in out
    assert "port=" in out and 30000 <= int(re.search(r" port=(\d+) ", out).group(1)) < 39000
    # services: one env -i pbsdsh per engine node, before any ray worker, with the C10 helper environment
    service_lines = [line for line in out.splitlines() if line.startswith("DRY_RUN bg[services-")]
    worker_lines = [line for line in out.splitlines() if line.startswith("DRY_RUN bg[ray-worker-")]
    assert len(service_lines) == 11 and len(worker_lines) == 14
    assert out.index(service_lines[-1]) < out.index(worker_lines[0])
    for host, line in zip(engine_hosts, service_lines):
        assert f"bg[services-{host}.log]: pbsdsh -n " in line and "/usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin" in line
        for key in ("PBS_JOBID=2245678.pbs1", "PBS_LOCALDIR=", "PLATOON_ROOT=", "TOOLATHLON_ENV=", "MILES_SRC=",
                    "OPENREWARD_REWARD_MODE=partial", "OPENREWARD_ENVIRONMENT_SET=toolathlon_only", "toolathlon_service.sh"):
            assert key in line, (key, line)
    head = next(line for line in out.splitlines() if line.startswith("DRY_RUN bg[head.log]:"))
    assert "/usr/bin/singularity exec --nv --cleanenv --containall" in head
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in head and "NUM_GPUS=120" in head
    serve = next(line for line in out.splitlines() if line.startswith("DRY_RUN serve"))
    assert re.search(r"--actor-num-nodes 4 --rollout-num-gpus 88 --check-weight-update-equal --extra-args", serve)
    head_inner = (jobdir / "head-inner.sh").read_text()
    assert 'grep -q "/${NUM_GPUS}.0 GPU"' in head_inner and "seq 1 180" in head_inner
    assert '--actor-num-nodes "$TRAIN_NODES" --rollout-num-gpus "$ROLLOUT_GPUS" $LAUNCHER_ARGS --extra-args "$EXTRA_ARGS"' in head_inner
    urls = ",".join(f"http://{ip}:8082" for ip in engine_ips)
    for export in (
        "export TINKER_BASE_URL=http://hnode031:", "export TINKER_API_KEY=tml-local", "export ENDPOINT_FILE=",
        "export JOBDIR=", "export RUN_NAME=dry", "export HF_HOME=", "export no_proxy=hnode031,10.0.1.9,",
        f"export MILES_SRC={REPO}", "export PLATOON_ROOT=", f"export SERVICE_NODES={','.join(engine_hosts)}",
        f"export OPENREWARD_SESSION_URLS_TOOLATHLON={urls}", f"export OPENREWARD_SESSION_URLS={urls}",
        "export PBS_LOCALDIR=", "export WORKERS=3",
    ):
        assert export in out, export
    # teardown: local sweep + one pbsdsh sweep per other node, from the repo copy of kill-leftovers.sh
    assert out.count("DRY_RUN pbsdsh[kill-leftovers ") == 14 and "DRY_RUN kill-leftovers[hnode031]" in out
    assert (jobdir / "kill-leftovers.sh").read_text() == (DEPLOY / "kill-leftovers.sh").read_text()
    assert not SECRET_RE.search(out), SECRET_RE.findall(out)


def test_two_nodes_defaults_to_one_trainer_and_no_services(tmp_path):
    out, jobdir = run_pbs_dry(tmp_path, ["hnode002", "hnode001"], ["10.0.0.2", "10.0.0.1"], MODE="lora", CLIENT_SCRIPT=str(DEPLOY / "clients" / "smoke.sh"))
    assert "nodes=2 train_nodes=1 rollout_gpus=8 services=0" in out
    roles = (jobdir / "roles.tsv").read_text().splitlines()
    assert roles == ["hnode001\t8\t10.0.0.1\ttrain", "hnode002\t0\t10.0.0.2\tengine"]
    assert "bg[services-" not in out and out.count("DRY_RUN bg[ray-worker-") == 1
    serve = next(line for line in out.splitlines() if line.startswith("DRY_RUN serve"))
    assert "run_qwen3_4b_gateway.py serve --mode lora" in serve and "--actor-num-nodes 1 --rollout-num-gpus 8" in serve
    assert "export OPENREWARD_SESSION_URLS_TOOLATHLON=\n" in out and "export SERVICE_NODES=\n" in out  # empty without services
    assert not SECRET_RE.search(out)


def test_launcher_alias_and_explicit_train_nodes(tmp_path):
    out, _ = run_pbs_dry(tmp_path, ["h1", "h2", "h3"], ["10.0.0.3", "10.0.0.2", "10.0.0.1"], LAUNCHER="examples/tinker/x.py", TRAIN_NODES="2", TINKER_PORT="31234")
    assert "launcher=examples/tinker/x.py" in out and "train_nodes=2 rollout_gpus=8" in out and "port=31234" in out


def test_rejects_train_nodes_out_of_range(tmp_path):
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("h1\nh2\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "DRY_RUN": "1", "PBS_NODEFILE": str(nodefile), "PBS_JOBID": "1.pbs",
           "PBS_LOCALDIR": str(tmp_path), "PROJECT_ROOT": str(tmp_path / "p"), "MILES_SRC": str(REPO), "HF_SNAPSHOT": "/x", "TRAIN_NODES": "2"}
    proc = subprocess.run(["bash", str(PBS)], env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and "TRAIN_NODES=2 must be in [1, 2)" in proc.stderr


def _toolathlon_env_values() -> dict[str, str]:
    values = {}
    for line in (DEPLOY / "toolathlon.env").read_text().splitlines():
        m = re.match(r"export ([A-Z0-9_]+)=(.*)", line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def test_toolathlon_env_is_the_c12_contract():
    values = _toolathlon_env_values()
    assert values["OPENREWARD_TOOLATHLON_SIF"] == "/groups/gai51740/aci18912it/platoon-next/toolathlon-audit/ghcr-c66ff0d/abci-sif/openreward-toolathlon-c66ff0d-pbs-7034e1a61a4a.sif"
    assert values["OPENREWARD_TOOLATHLON_SIF_SHA256"] == SIF_SHA
    assert values["PLATOON_CONTAINER_RUNTIME"] == "/usr/bin/singularity"
    assert values["OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH"] == "/opt/ms-playwright"
    assert values["OPENREWARD_NODE_MEMORY_LIMIT_BYTES"] == "2061584302080" and values["OPENREWARD_NODE_MEMORY_TRIGGER_PERCENT"] == "85"
    assert values["OPENREWARD_NODE_MEMORY_FALLBACK_AVAILABLE_BYTES"] == "274877906944"
    assert values["OPENREWARD_NODE_MEMORY_POLL_SECONDS"] == "2" and values["OPENREWARD_NODE_MEMORY_TELEMETRY_SECONDS"] == "60"
    assert "plugins/openreward/scripts" in values["TOOLATHLON_SCRIPTS_ROOT"]
    assert values["SINGULARITY_CACHEDIR"].startswith("${PBS_LOCALDIR")
    assert "abci.env" not in "".join(v for k, v in values.items())
    assert not any(name in values for name in SECRET_NAMES)


@pytest.mark.skipif(not os.access(_toolathlon_env_values()["OPENREWARD_TOOLATHLON_SIF"], os.R_OK) or shutil.which("singularity") is None,
                    reason="Toolathlon SIF or singularity not available on this host")
def test_toolathlon_service_dry_run(tmp_path):
    env = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": os.environ.get("HOME", str(tmp_path)), "DRY_RUN": "1",
           "MILES_SRC": str(REPO), "PLATOON_ROOT": "/home/aci18912it/source/platoon-tinker", "PBS_LOCALDIR": str(tmp_path),
           "WANDB_API_KEY": "LEAKED_WANDB", "OPENAI_API_KEY": "LEAKED_OAI"}
    proc = subprocess.run(["bash", str(SERVICE)], env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert f"DRY_RUN resolved_sif_sha256: {SIF_SHA}" in out
    guard = next(line for line in out.splitlines() if line.startswith("DRY_RUN guard:"))
    assert guard.endswith("--node-limit-bytes 2061584302080 --trigger-percent 85 --fallback-available-bytes 274877906944 --poll-seconds 2 --telemetry-seconds 60")
    container = next(line for line in out.splitlines() if line.startswith("DRY_RUN container:"))
    for token in ("/usr/bin/singularity exec --cleanenv --no-home --pid --overlay", "rootfs-overlay.img", ":/mnt/state",
                  "openreward-toolathlon-resilient-entrypoint.sh:ro", "PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright",
                  "UV_PYTHON_DOWNLOADS=never", "OPENREWARD_RUNTIME_MODE=user", "OPENREWARD_REWARD_MODE=partial",
                  "OPENREWARD_PG_PORT=5432", "OPENREWARD_PG_MAX_CONNECTIONS=512", "OPENREWARD_PORT=8082",
                  "OPENREWARD_WORKER_BASE_PORT=28100", "OPENREWARD_WORKERS=8", "OPENREWARD_WORKER_PID_ISOLATION=1"):
        assert token in container, token
    assert "DRY_RUN stage_pgdata: /usr/bin/singularity exec --fakeroot --cleanenv --no-home" in out
    assert "DRY_RUN mkfs: /usr/sbin/mkfs.ext3 -q -F -d" in out
    assert not (tmp_path / "toolathlon").exists()  # a dry run creates nothing
    assert not SECRET_RE.search(out), SECRET_RE.findall(out)


def _load_guard():
    spec = importlib.util.spec_from_file_location("openreward_node_memory_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve `from __future__` annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_guard_is_the_release_copy_and_accepts_only_the_helper_argv():
    guard = _load_guard()
    args = guard.parse_args(["--node-limit-bytes", "2061584302080", "--trigger-percent", "85", "--fallback-available-bytes", "274877906944",
                             "--poll-seconds", "2", "--telemetry-seconds", "60"])
    settings = guard.Settings(args.node_limit_bytes, args.trigger_percent, args.fallback_available_bytes, args.poll_seconds, args.telemetry_seconds)
    settings.validate_exact_contract()
    with pytest.raises(guard.GuardFailure):
        guard.Settings(2061584302080, 90, 274877906944, 2, 60).validate_exact_contract()
    release = Path("/home/aci18912it/rasp/releases/platoon-v8r36-07bebf9-057108f/pbs-scripts/openreward-node-memory-guard.py")
    if release.is_file():
        assert GUARD.read_bytes() == release.read_bytes()


def test_guard_self_test_against_a_fake_cgroup(tmp_path):
    guard = _load_guard()
    cgroup_root = tmp_path / "cgroup"
    job = cgroup_root / "pbs" / "job2245678"
    (job / "task").mkdir(parents=True)
    (job / "memory.max").write_text("2061584302080\n")
    (job / "memory.current").write_text(str(700 * 2**30) + "\n")
    (job / "task" / "memory.max").write_text("max\n")
    proc_self_cgroup = tmp_path / "proc_self_cgroup"
    proc_self_cgroup.write_text("0::/pbs/job2245678/task\n")
    settings = guard.Settings(2061584302080, 85, 274877906944, 2, 60)
    settings.validate_exact_contract()
    source = guard.resolve_memory_source(settings, proc_self_cgroup=proc_self_cgroup, cgroup_root=cgroup_root, proc_meminfo=tmp_path / "meminfo")
    assert isinstance(source, guard.CgroupMemorySource) and source.cgroup_dir == job
    out, err = io.StringIO(), io.StringIO()
    stop = threading.Event()
    status = guard.run_monitor(source, settings, stop, output=out, error=err, wait_for_stop=lambda _s: True)
    assert status == 0, err.getvalue()
    assert "event=start mode=cgroup" in out.getvalue() and "trigger_bytes=1752346656768" in out.getvalue()
    assert "event=stop" in out.getvalue()
    # over the 85 % trigger the guard fails closed with EXIT_THRESHOLD before monitoring
    (job / "memory.current").write_text(str(1700 * 2**30) + "\n")
    status = guard.run_monitor(source, settings, threading.Event(), output=io.StringIO(), error=err, wait_for_stop=lambda _s: True)
    assert status == guard.EXIT_THRESHOLD and "event=threshold" in err.getvalue()


def test_submit_presets_dry_run(tmp_path):
    submit = DEPLOY / "submit_toolathlon.sh"
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "MILES_SRC": str(REPO), "HF_SNAPSHOT": str(snapshot), "PLATOON_ROOT": str(tmp_path)}
    cfg = tmp_path / "plugins" / "openreward" / "platoon" / "openreward" / "configs" / "tinker"
    cfg.mkdir(parents=True)
    for name in ("abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml", "abci_toolathlon_15node_nonrecursive_qwen36_miles_tinker_r3_v1.yaml"):
        (cfg / name).write_text("x: 1\n")
    expected = {
        "smoke": ("select=5:mpiprocs=8", "walltime=06:00:00", "clients/smoke_q36.sh", "LAUNCHER_ARGS=--check-weight-update-equal"),
        "recursive": ("select=15:mpiprocs=8", "walltime=168:00:00", "RUN_NAME=ta15-rec-q36-r3-v1", "CONFIG=platoon/openreward/configs/tinker/abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml"),
        "nonrecursive": ("select=15:mpiprocs=8", "walltime=168:00:00", "RUN_NAME=ta15-nonrec-q36-r3-v1", "abci_toolathlon_15node_nonrecursive_qwen36_miles_tinker_r3_v1.yaml"),
    }
    for preset, tokens in expected.items():
        proc = subprocess.run(["bash", str(submit), preset, "--dry-run"], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        qsub = next(line for line in proc.stdout.splitlines() if line.startswith("qsub:"))
        for token in tokens + ("MODE=full", "SERVICES=1", "TRAIN_NODES=4", "GATEWAY_LAUNCHER=examples/tinker/run_qwen3_6_35b_a3b_gateway.py", "WORKERS=3", "tinker_gateway.pbs"):
            assert token in qsub.replace("\\,", ","), (preset, token, qsub)
        assert "(dry run: not submitted)" in proc.stdout and "reservation" in proc.stdout
    proc = subprocess.run(["bash", str(submit), "bogus", "--dry-run"], env=env, capture_output=True, text=True)
    assert proc.returncode == 2
    # a serve_tinker.py flag in LAUNCHER_ARGS would only be rejected by typer after the Ray cluster is up
    proc = subprocess.run(
        ["bash", str(submit), "smoke", "--dry-run"],
        env={**env, "LAUNCHER_ARGS": "--check-weight-update-equal --tinker-r3-check"}, capture_output=True, text=True,
    )
    assert proc.returncode == 2 and "--tinker-r3-check is a serve_tinker.py flag" in proc.stderr and "--r3-check" in proc.stderr
    proc = subprocess.run(
        ["bash", str(submit), "smoke", "--dry-run"],
        env={**env, "LAUNCHER_ARGS": "--check-weight-update-equal --r3-check --extra-env-vars MILES_TEST_R3_THRESHOLD=0.02",
             "EXTRA_ARGS": "--tinker-routing-replay-policy strict"},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    qsub = next(line for line in proc.stdout.splitlines() if line.startswith("qsub:"))
    assert "MILES_TEST_R3_THRESHOLD=0.02" in qsub and "EXTRA_ARGS=--tinker-routing-replay-policy" in qsub.replace("\\ ", " ")


def test_node_init_helper_finds_ip_without_usr_sbin_in_path(tmp_path):
    """pbsdsh runs helpers with a PATH that lacks /usr/sbin; node-init.sh must still find `ip` (job 2245383)."""
    import re
    import subprocess

    pbs = (Path(__file__).resolve().parents[3] / "deploy/abci/tinker_gateway.pbs").read_text()
    m = re.search(r'cat > "\$JOBDIR/node-init.sh" <<\'EOF\'\n(.*?)\nEOF\n', pbs, re.S)
    assert m, "node-init.sh heredoc not found"
    helper = tmp_path / "node-init.sh"
    helper.write_text(m.group(1) + "\n")
    probe = subprocess.run(
        ["/usr/bin/env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "bash", "-c", "ip -o -4 addr show | grep -c ' 10\\.0\\.'"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() == "0":
        pytest.skip("no 10.0.x.y IPoIB interface on this host")
    out = subprocess.run(
        ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "bash", str(helper), str(tmp_path / "local")],
        capture_output=True, text=True, check=False,
    )
    ok = [line for line in out.stdout.splitlines() if line.startswith("init_ok ")]
    assert ok, out.stdout + out.stderr
    assert len(ok[0].split()) == 3 and ok[0].split()[2].startswith("10.0."), ok[0]
