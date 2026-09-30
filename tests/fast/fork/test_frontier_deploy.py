"""Frontier-CS endpoint routing and supervised service deployment without PBS."""

import os
import subprocess

import pytest

from tests.fast.fork.test_deploy_dry_run import DEPLOY, PBS, REPO, SECRET_RE, run_pbs_dry


def test_frontier_services_and_client_share_port_8085_pool(tmp_path):
    ips = [f"10.0.0.{index}" for index in range(1, 11)]
    output, jobdir = run_pbs_dry(
        tmp_path, [f"hnode{index:03}" for index in range(1, 11)], ips,
        SERVICES="1", TRAIN_NODES="5", SERVICE_ENVIRONMENT="frontier_cs",
        OPENREWARD_REWARD_MODE="partial",
        CLIENT_SCRIPT=str(DEPLOY / "clients/platoon_toolathlon.sh"),
    )
    service_lines = [line for line in output.splitlines() if line.startswith("DRY_RUN bg[services-")]
    assert len(service_lines) == 5
    assert all("frontier_cs_service.sh" in line and "FRONTIER_ENV=" in line for line in service_lines)
    assert all("env -i" in line for line in service_lines)
    expected = ",".join(f"http://{ip}:8085" for ip in ips[5:])
    assert f"export OPENREWARD_SESSION_URLS_FRONTIER_CS={expected}" in output
    assert f"export OPENREWARD_SESSION_URLS={expected}" in output
    assert "export SERVICE_ENVIRONMENT=frontier_cs" in output
    assert "OPENREWARD_SESSION_URLS_TOOLATHLON=" not in output
    assert "OPENREWARD_SESSION_URLS_SWE_REBENCH=" not in output
    assert not SECRET_RE.search(output)
    assert (jobdir / "roles.tsv").read_text().count("engine") == 5


@pytest.mark.parametrize("healthy,helper_alive", [(True, True), (False, True), (True, False)])
def test_frontier_health_requires_http_and_live_helper(healthy, helper_alive):
    source = PBS.read_text()
    sweep = source.split("SWEEP_REPORT=\n", 1)[1].split('if [[ $SERVICES == 1 ]] && ! dry; then', 1)[0]
    script = f"""
set -euo pipefail
SERVICE_ENVIRONMENT=frontier_cs; SERVICE_PORT=8085
ENGINE_HOSTS=(host); ENGINE_IPS=(10.0.0.1)
SERVICE_PIDS=({'$$' if helper_alive else '999999999'})
probe() {{ [[ "$*" == http://10.0.0.1:8085/health ]] && {'true' if healthy else 'false'}; }}
{sweep}
sweep_services
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if healthy and helper_alive else 1), result.stderr


def test_mixture_labels_resolve_to_the_live_shared_endpoint_pool():
    source = (DEPLOY / "clients/platoon_env.sh").read_text()
    exports = source.split("export SERVICE_ENVIRONMENT=", 1)[1].split("unset PLATOON_TRIAL_REWARD_MODE", 1)[0]
    script = """
set -euo pipefail
PLATOON_TRIAL_SERVICE=frontier_cs
PLATOON_TRIAL_URLS=http://node-a:8085,http://node-b:8085
PLATOON_TRIAL_REWARD_MODE=partial
export SERVICE_ENVIRONMENT=""" + exports + """
[[ $OPENREWARD_SESSION_URLS_FCS_ALGORITHMIC == $PLATOON_TRIAL_URLS ]]
[[ $OPENREWARD_SESSION_URLS_FCS_2_0 == $PLATOON_TRIAL_URLS ]]
[[ $OPENREWARD_SESSION_URLS_FRONTIER_CS == $PLATOON_TRIAL_URLS ]]
[[ $OPENREWARD_REWARD_MODE == partial ]]
"""
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("environment", ["frontier_cs", "toolathlon", "swe_rebench"])
def test_frontier_native_grading_and_queued_creation_deadlines(environment):
    source = (DEPLOY / "clients/platoon_env.sh").read_text()
    settings = source.split("export OPENHANDS_SUPPRESS_BANNER=1", 1)[1].split("export WANDB_MODE", 1)[0]
    script = "unset OPENREWARD_SESSION_CREATION_TIMEOUT\nexport OPENHANDS_SUPPRESS_BANNER=1" + settings
    script += '\nprintf "%s %s" "$OPENREWARD_MCP_TIMEOUT" "${OPENREWARD_SESSION_CREATION_TIMEOUT:-default}"\n'
    result = subprocess.run(
        ["bash", "-eu", "-c", script], capture_output=True, text=True, check=True, timeout=10,
        env={**os.environ, "SERVICE_ENVIRONMENT": environment},
    )
    assert result.stdout == ("25200 25200" if environment == "frontier_cs" else "1800 default")


def test_frontier_helper_keeps_memory_guard_and_host_launcher(tmp_path):
    platoon = tmp_path / "platoon"
    scripts = platoon / "pbs-scripts"
    scripts.mkdir(parents=True)
    server = scripts / "openreward-frontiercs-singularity-server.sh"
    server.touch()
    result = subprocess.run(
        ["bash", str(DEPLOY / "frontier_cs_service.sh")], capture_output=True, text=True, timeout=30,
        env={
            "PATH": os.environ["PATH"], "HOME": str(tmp_path), "DRY_RUN": "1",
            "PBS_JOBID": "2245678.pbs1", "PBS_LOCALDIR": str(tmp_path),
            "PLATOON_ROOT": str(platoon), "MILES_SRC": str(REPO),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reward=native-score/100" in result.stdout
    assert str(server) in result.stdout
    assert "--fallback-available-bytes 274877906944" in result.stdout
    assert not SECRET_RE.search(result.stdout)
