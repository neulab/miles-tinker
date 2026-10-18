"""DRY_RUN coverage of the ABCI deployment scripts (design 10.1): no PBS, no GPU, no containers.

- `bash -n` on every deploy shell script;
- `tinker_gateway.pbs` DRY_RUN on 2 and 15 fake nodes: roles.tsv (trainers = TRAIN_NODES lowest IPoIB IPs by
  numeric per-octet sort), every pbsdsh/singularity line, `--actor-num-nodes/--rollout-num-gpus` ahead of
  LAUNCHER_ARGS, the C10 client exports, the 4-phase teardown sweep, and no secret name in the output;
- gateway presets (`PRESET=`, `deploy/abci/presets/*.env`): read as bash would source them, rendered exactly like the
  same job variables, job variables win, the Qwen3.8-27B 128K preset is fcs-q38's gateway but for the advertised
  name, malformed presets fail before any node starts;
- `CLIENT_CMD`: written verbatim to `client-cmd.sh`, which runs as written (quotes, spaces, the client exports, the
  exit status) in `CLIENT_DIR` (default `PBS_O_WORKDIR`), and fails early when it cannot run;
- `toolathlon_service.sh` DRY_RUN: resolved SIF + sha256 (2fb0ee8b...), the exact guard argv, the container
  command with the release env, no secret name (skipped when the SIF is unreadable on this host);
- the copied memory guard self-test: the helper's argv against a fake cgroup dir prints `event=start`.
"""

import importlib.util
import io
import os
import re
import shlex
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
    + list(DEPLOY.glob("*.env")) + list(DEPLOY.glob("presets/*.env"))
    + list((REPO / "deploy" / "clients").glob("*.sh"))
)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def run_pbs(tmp_path: Path, hosts: list[str], ips: list[str], **env) -> subprocess.CompletedProcess:
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
    return subprocess.run(["bash", str(PBS)], env=run_env, capture_output=True, text=True, timeout=120)


def run_pbs_dry(tmp_path: Path, hosts: list[str], ips: list[str], **env) -> tuple[str, Path]:
    proc = run_pbs(tmp_path, hosts, ips, **env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr, tmp_path / "project" / "jobs" / "dry" / "2245678"


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
    # numeric per-octet order (10.0.0.99 < 10.0.0.100 < 10.0.0.200 < 10.0.1.10 < 10.0.1.100 < 10.0.2.1), the head
    # (hnode031, 10.0.1.9) last, as placement_group.sort_bundles_head_last orders the actor slots
    others = sorted(FIFTEEN_IPS[1:], key=lambda ip: tuple(map(int, ip.split("."))))
    assert [r[2] for r in roles] == others + [FIFTEEN_IPS[0]]
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


@pytest.mark.parametrize(
    "hosts, ips, train_nodes, trainers",
    [
        (["hnode001", "hnode002"], ["10.0.0.1", "10.0.0.2"], None, ["hnode002"]),
        # the failed job's layout: ten nodes, five trainers, and the head with the lowest IP
        (
            [f"hnode{n}" for n in (110, 401, 402, 403, 404, 601, 405, 406, 602, 603)],
            [
                "10.0.1.10",
                "10.0.4.1",
                "10.0.4.2",
                "10.0.4.3",
                "10.0.4.4",
                "10.0.6.1",
                "10.0.4.5",
                "10.0.4.6",
                "10.0.6.2",
                "10.0.6.3",
            ],
            "5",
            ["hnode401", "hnode402", "hnode403", "hnode404", "hnode405"],
        ),
    ],
    ids=["two-nodes", "ten-nodes"],
)
def test_the_head_node_is_never_a_trainer(tmp_path, hosts, ips, train_nodes, trainers):
    """The head runs the Ray head, the driver, the gateway and the client; trainer rank 0 there converted and sent
    weight-sync buckets 5 s apart on a saturated CPU and lost a keep-alive race with the engines."""
    env = {"TRAIN_NODES": train_nodes} if train_nodes else {}
    out, jobdir = run_pbs_dry(tmp_path, hosts, ips, **env)

    roles = [line.split("\t") for line in (jobdir / "roles.tsv").read_text().splitlines()]
    assert [r[0] for r in roles if r[3] == "train"] == trainers
    assert roles[-1] == [hosts[0], "0", ips[0], "engine"]
    engines = [r[0] for r in roles if r[3] == "engine"]
    assert (
        f"node roles: head {hosts[0]} ({ips[0]}) = ray head + driver + gateway + client + engine; "
        f"trainers (rank 0 first): {' '.join(trainers)}; engines: {' '.join(engines)}"
    ) in out


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


def _gateway_dead_function() -> str:
    match = re.search(r"^gateway_dead\(\) \{.*?^\}\n", PBS.read_text(), re.M | re.S)
    assert match, "tinker_gateway.pbs defines gateway_dead()"
    return match.group(0)


def _probe_gateway(tmp_path: Path, answers: list[str], probes: int = 4) -> tuple[list[str], str]:
    """Call gateway_dead once per answer against a fake curl replaying `answers` ("<http code> <body>")."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    answer_file = tmp_path / "answers"
    answer_file.write_text("".join(f"{answer}\n" for answer in answers))
    curl = bindir / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        "# the next answer: its body, a newline and its code, as --write-out '\\n%{http_code}' prints them\n"
        'line=$(head -n 1 "$ANSWERS"); tail -n +2 "$ANSWERS" > "$ANSWERS.next"; mv "$ANSWERS.next" "$ANSWERS"\n'
        "code=${line%% *}; body=${line#* }\n"
        'printf \'%s\\n%s\' "$body" "$code"\n'
        "[[ $code != 000 ]]\n"
    )
    curl.chmod(0o755)
    script = (
        "set -euo pipefail\n"
        f"HEAD_HOST=hnode001 TINKER_PORT=31234 GATEWAY_DEAD_PROBES={probes} GATEWAY_DEAD=0\n"
        f"{_gateway_dead_function()}"
        f'for _ in $(seq {len(answers)}); do if gateway_dead; then echo dead; else echo "alive $GATEWAY_DEAD"; fi; done\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": f"{bindir}:/usr/bin:/bin", "ANSWERS": str(answer_file)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines(), proc.stderr


DEAD = '503 {"status": "unavailable", "error": "the trainer is dead"}'
OK = '200 {"status": "ok"}'


def test_the_monitor_ends_the_job_after_four_consecutive_503s(tmp_path):
    """healthz answers 503 once the trainer is dead; the job used to idle on its nodes for an hour like that."""
    results, stderr = _probe_gateway(tmp_path, [DEAD] * 4)

    assert results == ["alive 1", "alive 2", "alive 3", "dead"]
    assert "monitor: gateway unavailable (4/4)" in stderr and "the trainer is dead" in stderr


def test_any_other_answer_resets_the_count(tmp_path):
    """A gateway that is busy (no answer within 15 s) or healthy again is not a dead one."""
    results, _ = _probe_gateway(tmp_path, [DEAD, DEAD, OK, DEAD, DEAD, "000 ", DEAD, DEAD, DEAD, DEAD])

    assert results == [
        "alive 1",
        "alive 2",
        "alive 0",
        "alive 1",
        "alive 2",
        "alive 0",
        "alive 1",
        "alive 2",
        "alive 3",
        "dead",
    ]


def test_the_monitor_loop_ends_the_job_on_a_dead_gateway():
    text = PBS.read_text()
    monitor = text[text.index("monitor_loop() {") :]
    assert (
        'gateway_dead && { echo "monitor: the gateway\'s trainer is dead; ending the job" >&2; kill -TERM $$; return; }'
        in monitor
    )
    assert "GATEWAY_DEAD_PROBES=${GATEWAY_DEAD_PROBES:-4}" in text


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


# ---------------------------------------------------------------- gateway presets (PRESET=)

PRESETS = DEPLOY / "presets"
QWEN38 = "qwen3.8-27b-128k"
# fcs-q38's gateway as rasp's launch/abci/submit.sh submits frontiercs-* (the same lines at rasp e3e0054, origin/main
# 157ea06 and apga/platoon-879b589) and as the production run fcs10-q38-nonrec-...-v25 recorded them.
FCS_Q38_SNAPSHOT = (
    "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/"
    "97c725615b8aa025b800da4ceed38811358b30f1"
)
FCS_Q38 = {
    "GATEWAY_LAUNCHER": "examples/tinker/run_qwen3_6_35b_a3b_gateway.py",
    "MODE": "full",
    "LAUNCHER_ARGS": (
        f"--extra-env-vars NCCL_DEBUG=INFO --model-variant 27b --model-name {FCS_Q38_SNAPSHOT} --no-routing-replay --tp 4 "
        "--context-len 131072 --max-tokens-per-gpu 131073 --max-tokens-per-datum 131073 --unit-token-budget 1310730"
    ),
    "EXTRA_ARGS": (
        "--sglang-attention-backend fa3 --lr 1e-6 --router-policy cache_aware --router-balance-abs-threshold 16 "
        "--router-balance-rel-threshold 1.5"
    ),
    "TINKER_SAMPLE_TIMEOUT_SECONDS": "1800",
}
# fcs-q38's ten hosts, with made-up unsorted IPoIB addresses
TEN_HOSTS = [f"hnode{n}" for n in (726, 725, 724, 720, 719, 717, 716, 715, 713, 723)]
TEN_IPS = [f"10.0.7.{n}" for n in (26, 25, 24, 20, 19, 17, 16, 15, 13, 23)]
TWO = (["hnode002", "hnode001"], ["10.0.0.2", "10.0.0.1"])
DATE_RE = re.compile(r"\w{3} \w{3} [ \d]\d \d\d:\d\d:\d\d \S+ \d{4}")


def preset_keys() -> list[str]:
    return re.search(r"^PRESET_KEYS=\((.*)\)$", PBS.read_text(), re.M).group(1).split()


def source_preset(path: Path) -> dict[str, str]:
    """The preset as bash reads it (`set -a; source`). The job parses the file instead; the two must agree."""
    script = 'set -a; source "$1"; shift; for k in "$@"; do [[ -n ${!k+set} ]] && printf "%s=%s\\0" "$k" "${!k}"; done; true'
    proc = subprocess.run(
        ["bash", "-c", script, "_", str(path), *preset_keys()],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(item.split("=", 1) for item in proc.stdout.split("\0") if item)


def fake_hf_home(root: Path, model: str, revision: str, refs_main: str | None = None) -> tuple[Path, Path]:
    repo = root / "hub" / f"models--{model.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    if refs_main is not None:
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(refs_main)
    return root, snapshot


def dry_argv(out: str, label: str) -> list[str]:
    """The argv of one `DRY_RUN <label>:` line (printf %q words, so shlex reads them back)."""
    line = next(line for line in out.splitlines() if line.startswith(f"DRY_RUN {label}:"))
    return shlex.split(line.split(":", 1)[1])


@pytest.mark.parametrize("preset", sorted(PRESETS.glob("*.env")), ids=lambda p: p.stem)
def test_presets_are_read_as_bash_sources_them(tmp_path, preset):
    values = source_preset(preset)
    assert set(values) <= set(preset_keys()) and {"GATEWAY_LAUNCHER", "MODE", "LAUNCHER_ARGS"} <= set(values)
    launcher_args = shlex.split(values["LAUNCHER_ARGS"])
    assert not [arg for arg in launcher_args if arg.startswith("--tinker-")]  # serve_tinker.py flags go in EXTRA_ARGS
    env = {"HF_SNAPSHOT": "/x/snap"}
    if "HF_MODEL" in values:  # the gateway advertises the id that trainer configs name
        assert re.fullmatch(r"[0-9a-f]{40}", values["HF_REVISION"])
        assert launcher_args[launcher_args.index("--model-name") + 1] == values["HF_MODEL"]
        hf_home, _ = fake_hf_home(tmp_path / "hf", values["HF_MODEL"], values["HF_REVISION"])
        env = {"HF_SNAPSHOT": "", "HF_HOME": str(hf_home)}
    out, _ = run_pbs_dry(tmp_path, *TWO, PRESET=preset.stem, **env)
    assert f"preset {preset.stem} ({preset}):" in out
    for key, value in values.items():
        assert f'\n  {key}="{value}"\n' in out, key


def test_the_qwen38_preset_is_the_fcs_q38_gateway_but_for_the_advertised_name(tmp_path):
    values = source_preset(PRESETS / f"{QWEN38}.env")
    for key in ("GATEWAY_LAUNCHER", "MODE", "EXTRA_ARGS", "TINKER_SAMPLE_TIMEOUT_SECONDS"):
        assert values[key] == FCS_Q38[key], key
    assert values["HF_MODEL"] == "apurvaga/Qwen3.8-27B-preserve-thinking"
    assert FCS_Q38_SNAPSHOT.endswith(
        f"/models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/{values['HF_REVISION']}"
    )
    assert values["LAUNCHER_ARGS"] == FCS_Q38["LAUNCHER_ARGS"].replace(FCS_Q38_SNAPSHOT, values["HF_MODEL"])
    # rendered on fcs-q38's ten nodes with TRAIN_NODES=5: the snapshot of HF_MODEL@HF_REVISION under HF_HOME
    hf_home, snapshot = fake_hf_home(
        tmp_path / "hf", values["HF_MODEL"], values["HF_REVISION"], refs_main=values["HF_REVISION"]
    )
    out, _ = run_pbs_dry(
        tmp_path,
        TEN_HOSTS,
        TEN_IPS,
        PRESET=QWEN38,
        HF_SNAPSHOT="",
        HF_HOME=str(hf_home),
        TRAIN_NODES="5",
        CLIENT_SCRIPT=str(DEPLOY / "clients" / "smoke.sh"),
    )
    assert (
        f"nodes=10 train_nodes=5 rollout_gpus=40 services=0 launcher={FCS_Q38['GATEWAY_LAUNCHER']} snapshot={snapshot} "
        in out
    )
    serve = dry_argv(out, "serve (inside the head container)")
    assert serve[:3] == ["python", FCS_Q38["GATEWAY_LAUNCHER"], "serve"]
    assert serve[3:7] == ["--mode", "full", "--hf-checkpoint", str(snapshot)]
    split = ["--actor-num-nodes", "5", "--rollout-num-gpus", "40"]
    expected = [*split, *shlex.split(values["LAUNCHER_ARGS"]), "--extra-args", FCS_Q38["EXTRA_ARGS"]]
    assert serve[serve.index("--actor-num-nodes") :] == expected
    assert f"HF_SNAPSHOT={snapshot}" in dry_argv(out, "bg[head.log]")
    assert "\n  export TINKER_SAMPLE_TIMEOUT_SECONDS=1800\n" in out
    assert "TINKER_SAMPLE_TIMEOUT_SECONDS=1800" in dry_argv(out, "client")
    assert "note:" not in out  # refs/main names the revision


def test_a_preset_renders_exactly_like_its_job_variables(tmp_path):
    values = source_preset(PRESETS / f"{QWEN38}.env")
    hf_home, snapshot = fake_hf_home(tmp_path / "hf", values["HF_MODEL"], values["HF_REVISION"])  # no refs/main
    common = {"TRAIN_NODES": "5", "HF_HOME": str(hf_home), "CLIENT_SCRIPT": str(DEPLOY / "clients" / "smoke.sh")}
    variables = {key: value for key, value in values.items() if key not in ("HF_MODEL", "HF_REVISION")}
    runs = {}
    for name, env in (
        ("preset", {"PRESET": QWEN38, "HF_SNAPSHOT": ""}),
        ("variables", {"HF_SNAPSHOT": str(snapshot), **variables}),
    ):
        root = tmp_path / name
        root.mkdir()
        out, jobdir = run_pbs_dry(root, TEN_HOSTS, TEN_IPS, **common, **env)
        files = {
            f.name: DATE_RE.sub("<date>", f.read_text().replace(str(root), "<tmp>"))
            for f in jobdir.iterdir()
            if f.is_file()
        }
        files.pop("launcher.log")  # the output itself
        runs[name] = DATE_RE.sub("<date>", out.replace(str(root), "<tmp>")).splitlines(), files
    (lines, files), (expected_lines, expected_files) = runs["preset"], runs["variables"]
    assert files == expected_files  # head-inner.sh, worker.sh, container_args.txt, roles.tsv, ...
    # the output adds the preset report and the client's sampling timeout, and nothing else
    start = lines.index(f"preset {QWEN38} ({PRESETS / QWEN38}.env):")
    assert sorted(lines[start + 1 : start + 1 + len(values)]) == sorted(
        f'  {key}="{value}"' for key, value in values.items()
    )
    note = lines[start + 1 + len(values)]
    assert note.startswith(
        f"note: {hf_home}/hub/models--apurvaga--Qwen3.8-27B-preserve-thinking/refs/main does not name "
    )
    del lines[start : start + 2 + len(values)]
    lines.remove("  export TINKER_SAMPLE_TIMEOUT_SECONDS=1800")
    client = next(i for i, line in enumerate(lines) if line.startswith("DRY_RUN client:"))
    lines[client] = lines[client].replace(" TINKER_SAMPLE_TIMEOUT_SECONDS=1800 bash ", " bash ")
    assert lines == expected_lines


def test_job_variables_replace_the_preset_values(tmp_path):
    out, _ = run_pbs_dry(
        tmp_path,
        *TWO,
        PRESET=QWEN38,
        HF_SNAPSHOT="/x/own-snapshot",
        LAUNCHER="examples/tinker/x.py",
        EXTRA_ARGS="--tinker-lease-timeout-s 7200",
        TINKER_SAMPLE_TIMEOUT_SECONDS="900",
        CLIENT_SCRIPT=str(DEPLOY / "clients" / "smoke.sh"),
    )
    assert "mode=full " in out and "launcher=examples/tinker/x.py " in out and "snapshot=/x/own-snapshot " in out
    assert (
        "\n  GATEWAY_LAUNCHER=\"examples/tinker/x.py\" (the job's LAUNCHER; the preset's value is not used)\n" in out
    )
    assert "\n  EXTRA_ARGS=\"--tinker-lease-timeout-s 7200\" (the job's value; the preset's is not used)\n" in out
    assert "\n  TINKER_SAMPLE_TIMEOUT_SECONDS=\"900\" (the job's value; the preset's is not used)\n" in out
    assert '\n  MODE="full"\n' in out  # from the preset
    serve = dry_argv(out, "serve (inside the head container)")
    assert serve[1] == "examples/tinker/x.py" and "--model-variant" in serve  # LAUNCHER_ARGS still the preset's
    assert serve[-2:] == ["--extra-args", "--tinker-lease-timeout-s 7200"]
    assert "\n  export TINKER_SAMPLE_TIMEOUT_SECONDS=900\n" in out and "note:" not in out


def test_a_preset_may_be_a_path(tmp_path):
    preset = tmp_path / "my preset.env"
    preset.write_text('# a project-local preset\nMODE=full\n\nLAUNCHER_ARGS="--check-weight-update-equal --tp 2"\n')
    out, _ = run_pbs_dry(tmp_path, *TWO, PRESET=str(preset))
    assert f'preset {preset} ({preset}):\n  MODE="full"\n  LAUNCHER_ARGS="--check-weight-update-equal --tp 2"\n' in out
    serve = dry_argv(out, "serve (inside the head container)")
    expected = ["--check-weight-update-equal", "--tp", "2", "--extra-args", ""]
    assert serve[serve.index("--rollout-num-gpus") + 2 :] == expected


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("MODE=full\nTRAIN_NODES=5\n", ":2: TRAIN_NODES is not a preset key"),
        ("LAUNCHER_ARGS=--tp 4\n", ":1: quote a value with spaces or symbols"),
        ('LAUNCHER_ARGS="--tp 4\n', ":1: quote a value with spaces or symbols"),
        ('EXTRA_ARGS="--lr $LR"\n', ":1: a value holds no quote, $, backquote or backslash"),
        ("MODE=full\n# lora next\nMODE=lora\n", ":3: MODE is set twice"),
        ("export MODE=full\n", ":1: expected KEY=value"),
    ],
)
def test_a_malformed_preset_fails_before_any_node_starts(tmp_path, content, message):
    preset = tmp_path / "bad.env"
    preset.write_text(content)
    proc = run_pbs(tmp_path, *TWO, PRESET=str(preset))
    assert proc.returncode == 2 and f"{preset}{message}" in proc.stderr, proc.stderr
    assert not (tmp_path / "project").exists()


def test_an_unknown_preset_or_a_missing_snapshot_fails_before_any_node_starts(tmp_path):
    unknown, missing = tmp_path / "unknown", tmp_path / "missing"
    unknown.mkdir(), missing.mkdir()
    proc = run_pbs(unknown, *TWO, PRESET="qwen9-1t")
    assert (
        proc.returncode == 2 and "presets/qwen9-1t.env is not a readable file" in proc.stderr and QWEN38 in proc.stderr
    )
    proc = run_pbs(missing, *TWO, PRESET=QWEN38, HF_SNAPSHOT="", HF_HOME=str(tmp_path / "empty-hf"))
    assert proc.returncode == 2, proc.stderr
    assert (
        "hf download apurvaga/Qwen3.8-27B-preserve-thinking --revision 97c725615b8aa025b800da4ceed38811358b30f1"
        in proc.stderr
    )
    assert not (unknown / "project").exists() and not (missing / "project").exists()


# ---------------------------------------------------------------- CLIENT_CMD

# quotes of both kinds, two spaces that only quotes keep, variables, a command substitution, a file name with a space
CLIENT_CMD = """printf '[%s]\\n' "two  words" 'it said "hi"' "$RUN_NAME" "$(basename "$PWD")" > "out file"; exit 3"""


def test_client_cmd_runs_as_written_in_the_submission_directory(tmp_path):
    submitted = tmp_path / "submit dir"
    submitted.mkdir()
    out, jobdir = run_pbs_dry(tmp_path, *TWO, CLIENT_CMD=CLIENT_CMD, PBS_O_WORKDIR=str(submitted))
    script = jobdir / "client-cmd.sh"
    assert script.read_text().splitlines()[2:] == [CLIENT_CMD]  # after the shebang and one comment line
    assert f"\nclient: CLIENT_CMD in {submitted} on " in out and f"): {CLIENT_CMD}\n" in out
    client = dry_argv(out, "client")
    assert client[0] == "env" and client[-2:] == ["bash", str(script)]
    assert "\n  export RUN_NAME=dry\n" in out and "RUN_NAME=dry" in client
    # the argv the job runs in CLIENT_DIR (there in its own session, with its output in client.log)
    proc = subprocess.run(client, cwd=submitted, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 3, proc.stderr  # the command's exit status is the client's
    assert (submitted / "out file").read_text() == '[two  words]\n[it said "hi"]\n[dry]\n[submit dir]\n'


def test_client_dir_overrides_the_submission_directory(tmp_path):
    out, _ = run_pbs_dry(tmp_path, *TWO, CLIENT_CMD="true", CLIENT_DIR=str(tmp_path), PBS_O_WORKDIR="/nonexistent")
    assert f"\nclient: CLIENT_CMD in {tmp_path} on " in out


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {"CLIENT_CMD": "true", "CLIENT_SCRIPT": str(DEPLOY / "clients" / "smoke.sh")},
            "set CLIENT_SCRIPT or CLIENT_CMD, not both",
        ),
        ({"CLIENT_CMD": "true", "CLIENT_DIR": "/nonexistent/dir"}, "CLIENT_DIR is not a directory on "),
        ({"CLIENT_CMD": "true", "PBS_O_WORKDIR": "/nonexistent/dir"}, "CLIENT_DIR is not a directory on "),
        (
            {"CLIENT_SCRIPT": str(DEPLOY / "clients" / "smoke.sh"), "CLIENT_DIR": "/nonexistent/dir"},
            "CLIENT_DIR is not a directory on ",
        ),
        ({"CLIENT_CMD": 'platoon train "configs/a.yaml'}, "CLIENT_CMD is not a bash command line"),
    ],
)
def test_a_client_that_cannot_start_fails_before_any_node_starts(tmp_path, env, message):
    proc = run_pbs(tmp_path, *TWO, **env)
    assert proc.returncode == 2 and message in proc.stderr, proc.stderr
    assert not (tmp_path / "project").exists()


def test_without_preset_or_client_cmd_a_client_script_launch_is_unchanged(tmp_path):
    script = DEPLOY / "clients" / "smoke.sh"
    # TINKER_SAMPLE_TIMEOUT_SECONDS as rasp's submit.sh passes it: the client inherits it from the job, unlisted
    out, jobdir = run_pbs_dry(tmp_path, *TWO, CLIENT_SCRIPT=str(script), TINKER_SAMPLE_TIMEOUT_SECONDS="1800")
    assert f"\nclient: {script} on " in out and dry_argv(out, "client")[-2:] == ["bash", str(script)]
    assert "TINKER_SAMPLE_TIMEOUT_SECONDS" not in out and "\npreset " not in out
    assert not (jobdir / "client-cmd.sh").exists()
