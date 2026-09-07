"""Step 9 of the 5-node smoke (design 10.2 step 9): platoon + OpenHands + Toolathlon through the gateway.

No PBS, no GPU, no gateway. Covers:

- `bash -n` on the three changed/added client scripts;
- `clients/platoon_env.sh` is sourced by BOTH clients (so the smoke runs the production environment) and sets
  exactly the C10/C11 client contract, with the secrets coming from `abci.env` only;
- every `--a.b.c` override of the step-9 argv names a real field of platoon's config dataclasses, and the argv
  resolves to the intended tiny run (`load_config` silently drops unknown keys, so a typo'd `--train.foo` would
  be a no-op); skipped when the platoon venv or the run yaml is not on this host;
- `PLATOON_STEP=0` skips the step;
- the step-9 block itself, run against a stub trainer: the facts block, all of its `expect_*` assertions, and the
  failure diagnostics, driven by artifacts written in the exact formats platoon/OpenHands/the gateway use.
"""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CLIENTS = REPO / "deploy" / "abci" / "clients"
SMOKE = CLIENTS / "smoke_q36.sh"
PLATOON_CLIENT = CLIENTS / "platoon_toolathlon.sh"
PLATOON_ENV = CLIENTS / "platoon_env.sh"
SMOKE_TEXT = SMOKE.read_text()

PLATOON_ROOT = Path(os.environ.get("PLATOON_ROOT", "/home/aci18912it/source/platoon-tinker"))
PLATOON_PY = PLATOON_ROOT / "plugins" / "openreward" / ".venv" / "bin" / "python"
RUN_YAML = (
    PLATOON_ROOT / "plugins" / "openreward" / "platoon" / "openreward" / "configs" / "tinker"
    / "abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml"
)

# C11 client env (design section 1) plus the C10 inputs the helper forwards.
REQUIRED_ENV = {
    "TINKER_API_KEY": "${TINKER_API_KEY:-tml-local}",
    "TINKER_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OPENREWARD_REWARD_MODE": "partial",
    "OPENREWARD_ENVIRONMENT_SET": "toolathlon_only",
    "OPENREWARD_DISABLE_UPDATE_CHECK": "1",
    "OPENHANDS_SUPPRESS_BANNER": "1",
    "OPENREWARD_MCP_TIMEOUT": "1800",
    "WANDB_MODE": "${WANDB_MODE:-online}",
    "OPENREWARD_SESSION_URLS": "$OPENREWARD_SESSION_URLS_TOOLATHLON",
}


def _slice(start_re: str, end_re: str) -> str:
    """The lines of smoke_q36.sh from the line matching start_re up to (excluding) the one matching end_re."""
    lines = SMOKE_TEXT.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if re.search(start_re, line))
    end = next(i for i, line in enumerate(lines) if i > start and re.search(end_re, line))
    return "".join(lines[start:end])


def _prelude() -> str:
    """The smoke's own knobs + pass/fail helpers (COOKBOOK_PY .. expect_files), verbatim."""
    return _slice(r"^COOKBOOK_PY=", r'^echo "smoke_q36 against ')


def _step9() -> str:
    """The step-9 block, verbatim, up to the final summary."""
    return _slice(r"^# -+ step 9$", r'^echo; echo "smoke_q36 done')


def _platoon_argv() -> list[str]:
    body = re.search(r"^  PLATOON_ARGV=\(\n(.*?)^  \)\n", SMOKE_TEXT, re.S | re.M).group(1)
    return shlex.split(body, comments=True)


# Every run-size value of the step-9 argv is `${KNOB:-default}` so the same block can be run at production
# rollout concurrency on the 5-node smoke.  The tests below check the *defaults* (the tiny run) and, separately,
# that the knob exists, is documented and is forwarded by submit_toolathlon.sh.
_KNOB = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*):-(.*)\}$")


def _default(token: str) -> str:
    """`${PLATOON_WORKERS:-2}` -> `2`; anything else unchanged."""
    match = _KNOB.match(token)
    return match.group(2) if match else token


def _knob_of(token: str) -> str | None:
    match = _KNOB.match(token)
    return match.group(1) if match else None


def _platoon_argv_defaults() -> list[str]:
    return [_default(tok) for tok in _platoon_argv()]


def _override_keys() -> list[str]:
    """The train_tinker overrides of the step-9 argv (everything after the bare `--`, minus `--config`)."""
    argv = _platoon_argv()
    tail = argv[argv.index("--") + 1:]
    return [tok[2:] for tok in tail if tok.startswith("--") and tok != "--config"]


def test_bash_syntax_of_the_changed_clients():
    for script in (SMOKE, PLATOON_CLIENT, PLATOON_ENV):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_both_clients_source_the_shared_platoon_env():
    """The smoke must test the production environment, not a copy of it."""
    source_line = 'source "$MILES_SRC/deploy/abci/clients/platoon_env.sh"'
    assert source_line in PLATOON_CLIENT.read_text()
    assert source_line in SMOKE_TEXT
    # and neither client re-implements the env: abci.env (the secrets) is sourced in exactly one place
    for script in (PLATOON_CLIENT, SMOKE):
        code = [line for line in script.read_text().splitlines() if not line.lstrip().startswith("#")]
        assert not [line for line in code if "abci.env" in line], (script.name, code)
    assert PLATOON_ENV.read_text().count("source /home/aci18912it/rasp/config/abci.env") == 1


def test_platoon_env_is_the_c10_c11_client_contract():
    text = PLATOON_ENV.read_text()
    exported = dict(re.findall(r"(?:^|\s)([A-Z_][A-Z0-9_]*)=(\S*)", text))
    for name, value in REQUIRED_ENV.items():
        assert exported.get(name) == value, (name, exported.get(name))
    # TINKER_BASE_URL / OPENREWARD_SESSION_URLS_TOOLATHLON are re-exported from the C10 job environment
    assert re.search(r"^export TINKER_BASE_URL ", text, re.M)
    assert re.search(r"^export OPENREWARD_SESSION_URLS_TOOLATHLON ", text, re.M)
    assert "HF_HOME=/groups/gai51740/aci18912it/rasp/cache/huggingface" in text
    # node-local scratch under PBS_LOCALDIR, and the 0600 gate on secrets.env
    assert "TMPDIR=$CLIENT_LOCAL/tmp XDG_CACHE_HOME=$CLIENT_LOCAL/xdg" in text
    assert "${PBS_LOCALDIR:?}" in text and "stat -c %a /home/aci18912it/rasp/config/secrets.env" in text
    # "sets everything and defines nothing else": no function, no launcher, only the mkdir of the scratch dirs
    assert not re.search(r"^\s*\w+\(\)\s*[({]", text, re.M)
    commands = [
        line.split()[0]
        for line in text.splitlines()
        if line and not line.startswith(("#", "[[", "export ", "set ", "CLIENT_LOCAL="))
    ]
    assert commands == ["source", "mkdir"], commands


def test_step9_runs_last_and_is_bounded_and_skippable():
    body = _step9()
    assert SMOKE_TEXT.index("step 8 ") < SMOKE_TEXT.index("step 9 ")  # after the resume gate
    assert 'if [[ ${PLATOON_STEP:-1} != 1 ]]; then' in body
    assert re.search(r'exec timeout -k 120 "\$\{PLATOON_TIMEOUT:-2400\}"', body)
    assert "${SMOKE_WANDB:-offline}" in body  # never the production W&B project by default
    assert '--stats.wandb.mode "$SMOKE_WANDB_MODE"' in body
    assert "smoke" in "toolathlon-smoke" and "SMOKE_EXPERIMENT:-toolathlon-smoke" in body
    assert "SMOKE_TRIAL:-$RUN_NAME-platoon-smoke" in body
    assert 'SMOKE_LOG_PATH=$JOBDIR/platoon-smoke' in body
    header = SMOKE_TEXT.split("\nset -uo pipefail")[0]  # the documented steps and knobs
    assert "steps 1-9" in header.splitlines()[1]
    for knob in ("PLATOON_STEP", "PLATOON_TIMEOUT", "SMOKE_WANDB"):
        assert knob in header, knob


def test_step9_argv_is_the_production_entry_point_with_a_tiny_run_size():
    argv = _platoon_argv_defaults()
    assert argv[:5] == ["$PLATOON_PY", "-m", "platoon.train.tinker.restart_wrapper", "--max-restarts", "0"]
    assert argv[argv.index("--") + 1:argv.index("--") + 3] == ["$PLATOON_PY", "-m"]
    assert argv[argv.index("--") + 3] == "platoon.openreward.train_scripts.tinker.train_tinker"
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--train.max_training_steps"] == "2" and pairs["--train.batch_size"] == "2"
    assert pairs["--train.workflow_config.group_size"] == "2"
    assert pairs["--train.num_concurrent_rollout_workflow_workers"] == "2"
    assert pairs["--checkpoint.recovery_every"] == "1" and pairs["--checkpoint.every"] == "1000"
    # group_size 2 makes the yaml's straggler_quorum 6 / min_successful_group_size 4 illegal
    assert pairs["--train.workflow_config.straggler_quorum"] == "2"
    assert pairs["--train.workflow_config.min_successful_group_size"] == "1"
    assert int(pairs["--train.workflow_config.rollout_config.max_steps"]) <= 16
    assert int(pairs["--train.workflow_config.rollout_config.timeout"]) <= 900
    assert int(pairs["--train.workflow_config.rollout_config.step_timeout"]) <= 300


RUN_SIZE_KNOBS = {
    "--train.max_training_steps": "PLATOON_TRAIN_STEPS",
    "--train.batch_size": "PLATOON_BATCH_SIZE",
    "--train.num_concurrent_rollout_workflow_workers": "PLATOON_WORKERS",
    "--train.workflow_config.group_size": "PLATOON_GROUP_SIZE",
    "--train.workflow_config.straggler_quorum": "PLATOON_QUORUM",
    "--train.workflow_config.min_successful_group_size": "PLATOON_MIN_SUCCESS",
    "--train.workflow_config.straggler_timeout_seconds": "PLATOON_STRAGGLER_TIMEOUT",
    "--train.workflow_config.rollout_config.max_steps": "PLATOON_MAX_STEPS",
    "--train.workflow_config.rollout_config.timeout": "PLATOON_ROLLOUT_TIMEOUT",
    "--train.workflow_config.rollout_config.step_timeout": "PLATOON_STEP_TIMEOUT",
    "--watchdog.timeout_seconds": "PLATOON_WATCHDOG",
}


def test_every_run_size_value_is_an_env_knob():
    """The 15-node client bugs only appear once rollouts overlap, so the smoke must be able to reach
    production concurrency (WORKERS x GROUP_SIZE = 24 trajectories) without editing this script."""
    argv = _platoon_argv()
    pairs = dict(zip(argv, argv[1:]))
    for flag, knob in RUN_SIZE_KNOBS.items():
        assert _knob_of(pairs[flag]) == knob, (flag, pairs[flag])


def test_the_run_size_knobs_are_documented_and_forwarded_by_the_submitter():
    header = SMOKE_TEXT.split("\nset -uo pipefail")[0]
    submitter = (REPO / "deploy" / "abci" / "submit_toolathlon.sh").read_text()
    forwarded = re.search(r"^for knob in (.*?); do$", submitter, re.S | re.M).group(1).split()
    for knob in RUN_SIZE_KNOBS.values():
        assert knob in header, knob
        assert knob in forwarded, knob


def test_step9_checks_the_mcp_bridge_connect_failure_rate():
    """Run ta15-rec-q36-r3-v1 lost 968 of 1060 trajectories to `MCPError: MCP Connection Failure` at step 0.
    The smoke has to fail on a recurrence, so it counts them from the rollout event records."""
    body = _step9()
    assert '"$SMOKE_RUNDIR/checkpoints.jsonl" "$ROLLOUTS"' in body
    assert 'record.get("type") != "trajectory_finished"' in body
    assert '"MCP Connection Failure" in message' in body
    assert re.search(r"expect_log \"MCP bridges connected.*'mcp_failure_rate_pct=\[0-9\]\$'", body)


@pytest.fixture(scope="module")
def platoon_config_probe(tmp_path_factory):
    """Resolve the step-9 overrides against the real platoon dataclasses, in the platoon venv."""
    if not (os.access(PLATOON_PY, os.X_OK) and RUN_YAML.is_file()):
        pytest.skip(f"platoon venv/config not on this host ({PLATOON_PY}, {RUN_YAML})")
    probe = tmp_path_factory.mktemp("probe") / "probe.py"
    probe.write_text(
        "import dataclasses, json, sys, typing\n"
        "from platoon.utils.config import load_config\n"
        "from platoon.openreward.tinker_config import OpenRewardTinkerTrainerConfig as C\n"
        "keys, argv = json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
        "def unwrap(t):\n"
        "    inner = [a for a in typing.get_args(t) if a is not type(None)]\n"
        "    return inner[0] if typing.get_origin(t) is not None and len(inner) == 1 else t\n"
        "def resolve(key):\n"
        "    cls = C\n"
        "    for part in key.split('.'):\n"
        "        cls = unwrap(cls)\n"
        "        if not dataclasses.is_dataclass(cls):\n"
        "            return False\n"
        "        if part not in {f.name for f in dataclasses.fields(cls)}:\n"
        "            return False\n"
        "        cls = typing.get_type_hints(cls).get(part, object)\n"
        "    return True\n"
        "unknown = [k for k in keys if not resolve(k)]\n"
        "cfg, _ = load_config(args=argv, config_class=C)\n"
        "def get(path):\n"
        "    obj = cfg\n"
        "    for part in path.split('.'):\n"
        "        obj = getattr(obj, part)\n"
        "    return obj\n"
        "print(json.dumps({'unknown': unknown, 'values': {k: get(k) for k in keys if k not in unknown}}))\n"
    )
    argv = _platoon_argv()
    tail = argv[argv.index("--") + 1:]
    substitutions = {
        "$PLATOON_CONFIG": str(RUN_YAML),
        "$TINKER_BASE_URL": "http://head:30001",
        "$SMOKE_LOG_PATH": "/jobdir/platoon-smoke",
        "$SMOKE_EXPERIMENT": "toolathlon-smoke",
        "$SMOKE_TRIAL": "run-platoon-smoke",
        "$SMOKE_WANDB_MODE": "offline",
    }
    train_argv = [substitutions.get(_default(tok), _default(tok)) for tok in tail[3:]]
    proc = subprocess.run(
        [str(PLATOON_PY), str(probe), json.dumps(_override_keys() + ["train.definitely_not_a_field"]),
         json.dumps(train_argv)],
        cwd=str(PLATOON_ROOT / "plugins" / "openreward"),
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def test_every_step9_override_names_a_real_platoon_config_field(platoon_config_probe):
    """load_config drops unknown keys silently: a typo'd override would be a no-op, not an error."""
    assert platoon_config_probe["unknown"] == ["train.definitely_not_a_field"]


def test_step9_overrides_resolve_to_the_intended_tiny_run(platoon_config_probe):
    values = platoon_config_probe["values"]
    assert values["train.max_training_steps"] == 2
    assert values["train.batch_size"] == 2
    assert values["train.num_concurrent_rollout_workflow_workers"] == 2
    assert values["train.workflow_config.group_size"] == 2
    assert values["train.workflow_config.straggler_quorum"] == 2
    assert values["train.workflow_config.min_successful_group_size"] == 1  # "1" parses as True == 1
    assert values["checkpoint.recovery_every"] == 1 and values["checkpoint.every"] == 1000
    assert values["train.workflow_config.rollout_config.max_steps"] == 12
    assert values["stats.wandb.mode"] == "offline"
    assert "smoke" in values["stats.experiment_name"] and "smoke" in values["stats.trial_name"]
    assert values["log_path"] == "/jobdir/platoon-smoke"
    assert values["tinker_base_url"] == "http://head:30001"


# --------------------------------------------------------------------------- running the block against a stub
PERF_OLD = [  # steps 4/5/8 of the smoke: cookbook clients train without a routing_handle
    {"time": 1.0, "slot": 0, "num_datums": 32, "num_tokens": 2 << 20, "routing_covered_datums": 0,
     "routing_uncovered_datums": 32, "routing_prefix_hits": 3, "actor_train_mfu": 0.2},
]
PERF_NEW = [  # the two optim windows of the platoon run: every datum carries a routing_handle
    {"time": 2.0, "slot": 0, "num_datums": 4, "num_tokens": 40000, "routing_covered_datums": 4,
     "routing_uncovered_datums": 0, "routing_prefix_hits": 9, "routing_cache_bytes": 12, "grad_norm": 0.7,
     "actor_train_mfu": 0.19},
    {"time": 3.0, "slot": 0, "num_datums": 4, "num_tokens": 41000, "routing_covered_datums": 4,
     "routing_uncovered_datums": 0, "routing_prefix_hits": 17, "routing_cache_bytes": 15, "grad_norm": 0.6,
     "actor_train_mfu": 0.2},
]
CHECKPOINTS = [
    {"name": "recovery-000000", "batch": 0, "state_path": "tinker://model-a/weights/recovery-000000",
     "sampler_path": "tinker://model-a/sampler_weights/recovery-000000"},
    {"name": "recovery-000001", "batch": 1, "state_path": "tinker://model-a/weights/recovery-000001",
     "sampler_path": "tinker://model-a/sampler_weights/recovery-000001"},
]
SESSION_URL = "http://10.0.0.42:8082"


MCP_FAILURE = (
    "Error in episode loop at step 0 (openhands/sdk/mcp/client.py:55 in connect)\n"
    "MCPError: MCP Connection Failure\n"
    "RuntimeError: Client failed to connect: fileno\n"
)


def _write_step9_artifacts(jobdir: Path, checkpoint_root: Path, *, good: bool, mcp_failures: int = 0) -> None:
    """Everything the assertions read, in the byte formats their producers use."""
    rundir = jobdir / "platoon-smoke" / "toolathlon-smoke" / "smoke-run-platoon-smoke"
    rollouts = rundir / "rollouts" / "train" / "0"
    (rollouts / "events").mkdir(parents=True)
    bridge = rollouts / "openreward" / "task-1" / "abcd1234" / "bridge"
    bridge.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "perf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in PERF_OLD))
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in CHECKPOINTS))
    # platoon.visualization.event_sinks.JsonlFileSink: json.dumps(record), default separators
    events = [{"type": "trajectory_created", "trajectory": {"id": "t1"}}]
    steps = 2 if good else 1
    for index in range(steps):
        events.append({"type": "trajectory_step_added", "trajectory_id": "t1", "step_index": index,
                       "step": {"action_events": [{"tool_name": "execute_bash", "tool_call_id": "c1"}]},
                       "reward": 0.0, "finish_message": None, "error_message": None})
    # platoon.episode: one terminal record per trajectory; error_message carries the masked MCP failure
    events.append({"type": "trajectory_finished", "trajectory_id": "t1", "reward": 1.0,
                   "finish_message": "done", "error_message": None, "misc": {}})
    (rollouts / "events" / "events_task-1_coll1.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    # 9 more healthy trajectories, of which `mcp_failures` died at step 0 without an MCP bridge
    siblings = []
    for index in range(9):
        failed = index < mcp_failures
        siblings.append({"type": "trajectory_finished", "trajectory_id": f"s{index}", "reward": 0.0,
                         "finish_message": None,
                         "error_message": MCP_FAILURE if failed else None, "misc": {}})
    (rollouts / "events" / "events_task-1_coll2.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in siblings)
    )
    # platoon.openreward.mcp_bridge._write_jsonl: json.dumps(record, sort_keys=True)
    bridge_events = [{"type": "session_started", "env": "toolathlongym", "session_url": SESSION_URL,
                      "time": 1.0, "tool_names": ["bash"]}]
    if good:
        bridge_events += [
            {"type": "tool_call", "turn": 1, "call_id": "call_0001_bash", "tool_name": "bash", "time": 2.0},
            {"type": "tool_result", "turn": 1, "call_id": "call_0001_bash", "tool_name": "bash",
             "result": {"finished": False}, "time": 3.0},
        ]
    (bridge / "bridge_events.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in bridge_events)
    )
    (jobdir / "services-hnode001.log").write_text(
        '[toolathlon-service] 2026-09-08T00:00:00 container up\n'
        'INFO:     10.0.0.42:39002 - "GET /health HTTP/1.1" 200 OK\n'
        'INFO:     10.0.0.42:39004 - "POST /toolathlongym/task HTTP/1.1" 200 OK\n'
    )


def _run_step9(tmp_path: Path, *, good: bool = True, mcp_failures: int = 0,
               env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the extracted step-9 block with a stub trainer instead of the real platoon client."""
    jobdir = tmp_path / "jobdir"
    jobdir.mkdir()
    checkpoint_root = tmp_path / "ckpt"
    miles_src = tmp_path / "miles"       # only used to source platoon_env.sh; the real one needs abci.env + secrets
    (miles_src / "deploy" / "abci" / "clients").mkdir(parents=True)
    (miles_src / "deploy" / "abci" / "clients" / "platoon_env.sh").write_text("export STUB_PLATOON_ENV=1\n")
    plugin = tmp_path / "platoon" / "plugins" / "openreward"
    (plugin / "cfg").mkdir(parents=True)
    (plugin / "cfg" / "run.yaml").write_text("x: 1\n")
    bin_dir = tmp_path / "bin"           # a no-op pkill: never signal anything on the test host
    bin_dir.mkdir()
    (bin_dir / "pkill").write_text("#!/bin/bash\nexit 1\n")
    (bin_dir / "pkill").chmod(0o755)
    stub = tmp_path / "stub-python"
    stub.write_text(
        "#!/bin/bash\n"
        'echo "[RESTART-WRAPPER] INFO: Starting training (attempt 1)"\n'
        'echo "argv: $*"\n'
        f'cat {tmp_path / "perf-new.jsonl"} >> {checkpoint_root / "perf.jsonl"}\n'
        + ('echo "[RESTART-WRAPPER] INFO: Training completed successfully (elapsed: 42s)"\n' if good else
           'echo "Traceback (most recent call last):"\necho "RuntimeError: boom"\nexit 1\n')
    )
    stub.chmod(0o755)
    (tmp_path / "perf-new.jsonl").write_text("".join(json.dumps(r) + "\n" for r in PERF_NEW))
    _write_step9_artifacts(jobdir, checkpoint_root, good=good, mcp_failures=mcp_failures)
    script = tmp_path / "step9.sh"
    script.write_text(
        "set -uo pipefail\n" + _prelude() + _step9() + '\necho "FAILED=$FAILED"\nexit $(( FAILED > 0 ))\n'
    )
    run_env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path),
        "JOBDIR": str(jobdir), "RUN_NAME": "smoke-run", "CHECKPOINT_ROOT": str(checkpoint_root),
        "MILES_SRC": str(miles_src), "PLATOON_ROOT": str(tmp_path / "platoon"),
        "PLATOON_PYTHON": str(stub), "CONFIG": "cfg/run.yaml",
        "TINKER_BASE_URL": "http://head:30001", "OPENREWARD_SESSION_URLS_TOOLATHLON": SESSION_URL,
        "PLATOON_TIMEOUT": "60",
    }
    run_env.update(env or {})
    return subprocess.run(["bash", str(script)], cwd=str(REPO), env=run_env,
                          capture_output=True, text=True, timeout=180)


def test_platoon_step_0_skips_the_step(tmp_path):
    proc = _run_step9(tmp_path, env={"PLATOON_STEP": "0"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIP (PLATOON_STEP=0)" in proc.stdout
    assert "+ timeout" not in proc.stdout and "PASS" not in proc.stdout
    assert proc.stdout.strip().endswith("FAILED=0")
    assert not (tmp_path / "jobdir" / "platoon-smoke" / "step9-facts.txt").exists()


def test_step9_passes_every_assertion_on_a_good_run(tmp_path):
    proc = _run_step9(tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert not re.search(r"^FAIL ", out, re.M), out
    assert proc.stdout.strip().endswith("FAILED=0"), out
    # the printed command line is the real argv (with the tiny-run overrides), not a paraphrase
    command = next(line for line in out.splitlines() if line.startswith("+ timeout -k 120"))
    for token in ("platoon.train.tinker.restart_wrapper", "--max-restarts 0",
                  "platoon.openreward.train_scripts.tinker.train_tinker", "--train.max_training_steps 2",
                  "--train.workflow_config.group_size 2", "--checkpoint.recovery_every 1",
                  "--stats.wandb.mode offline"):
        assert token in command, (token, command)
    for check in ("trainer completed >= 2 optim steps", "R3 covered the run's datums",
                  "R3 left no datum uncovered", "gateway extended a turn-1 prefix",
                  "a recovery checkpoint record has a state_path", "episodes ran >= 2 agent steps",
                  "episodes emitted tool calls", "Toolathlon executed a tool",
                  "the job's Toolathlon service served the run", "MCP bridges connected",
                  "restart_wrapper saw a clean trainer exit",
                  "no watchdog kill", "no unhandled exception at exit"):
        assert f"PASS {check}" in out, (check, out)
    facts = (tmp_path / "jobdir" / "platoon-smoke" / "step9-facts.txt").read_text()
    assert "optim_windows_with_datums=2" in facts and "routing_uncovered_datums=0" in facts
    assert "routing_covered_datums=8" in facts and "routing_prefix_hits_delta=14" in facts
    assert "recovery_checkpoint_with_state_path=recovery-000001" in facts
    assert "trajectories_finished=10 trajectories_errored=0 mcp_connection_failures=0" in facts
    assert "mcp_failure_rate_pct=0" in facts
    assert "services-hnode001.log: 2 request lines" in out and "POST /toolathlongym/task" in out


def test_step9_fails_loudly_and_prints_the_logs_when_the_run_breaks(tmp_path):
    """A crashed trainer: non-zero exit, a traceback at the tail, one-step episodes, no tool call."""
    proc = _run_step9(tmp_path, good=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "FAIL: exit 1" in out
    for check in ("episodes ran >= 2 agent steps", "Toolathlon executed a tool",
                  "restart_wrapper saw a clean trainer exit", "no unhandled exception at exit"):
        assert f"FAIL {check}" in out, (check, out)
    # the R3/checkpoint facts still hold (the stub appended its perf records before dying)
    assert "PASS R3 left no datum uncovered" in out
    assert "--- tail -40 " in out and "--- tail -30 " in out and "services-hnode001.log" in out
    assert "exit 124 = the run exceeded PLATOON_TIMEOUT=60 s" in out


def test_step9_reports_a_missing_perf_log_instead_of_passing_silently(tmp_path):
    proc = _run_step9(tmp_path, env={"CHECKPOINT_ROOT": str(tmp_path / "elsewhere")})
    out = proc.stdout + proc.stderr
    assert "FAIL trainer completed >= 2 optim steps" in out
    assert "optim_windows_with_datums=0" in out


def test_step9_fails_when_mcp_bridges_stop_connecting(tmp_path):
    """The 15-node regression: trajectories die at step 0 with `MCPError: MCP Connection Failure`."""
    proc = _run_step9(tmp_path, mcp_failures=5)
    out = proc.stdout + proc.stderr
    assert "FAIL MCP bridges connected" in out, out
    facts = (tmp_path / "jobdir" / "platoon-smoke" / "step9-facts.txt").read_text()
    assert "trajectories_finished=10 trajectories_errored=5 mcp_connection_failures=5" in facts
    assert "mcp_failure_rate_pct=50" in facts
    assert proc.returncode == 1


def test_step9_does_not_pass_the_mcp_check_when_no_trajectory_finished(tmp_path):
    """rate 0 out of 0 must not read as a clean run."""
    proc = _run_step9(tmp_path, env={"SMOKE_TRIAL": "no-such-trial"})
    out = proc.stdout + proc.stderr
    assert "FAIL MCP bridges connected" in out, out
