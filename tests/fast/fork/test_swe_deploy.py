"""Exercise SWE endpoint routing and service supervision without a cluster."""

import os
import subprocess

import pytest

from tests.fast.fork.test_deploy_dry_run import DEPLOY, PBS, REPO, SECRET_RE, run_pbs_dry


def test_swe_services_and_client_receive_the_same_endpoint_pool(tmp_path):
    hosts = [f"hnode{i:03d}" for i in range(1, 9)]
    ips = [f"10.0.0.{i}" for i in range(1, 9)]
    output, jobdir = run_pbs_dry(
        tmp_path, hosts, ips, SERVICES="1", TRAIN_NODES="2",
        SERVICE_ENVIRONMENT="swe_rebench", OPENREWARD_REWARD_MODE="binary",
        CLIENT_SCRIPT=str(DEPLOY / "clients/platoon_toolathlon.sh"),
    )
    service_lines = [line for line in output.splitlines() if line.startswith("DRY_RUN bg[services-")]
    assert len(service_lines) == 6
    assert all("swe_rebench_service.sh" in line and "OPENREWARD_REWARD_MODE=binary" in line for line in service_lines)
    assert all("SWE_ENV=" in line and "env -i" in line for line in service_lines)
    # the head (hnode001, the lowest IP) never trains: it is the last engine node
    expected = ",".join(f"http://{ip}:8084" for ip in ips[3:] + ips[:1])
    assert f"export OPENREWARD_SESSION_URLS_SWE_REBENCH={expected}" in output
    assert f"export OPENREWARD_SESSION_URLS={expected}" in output
    assert "export SERVICE_ENVIRONMENT=swe_rebench" in output
    assert "OPENREWARD_SESSION_URLS_TOOLATHLON=" not in output
    assert not SECRET_RE.search(output)
    assert (jobdir / "roles.tsv").read_text().count("engine") == 6


@pytest.mark.parametrize("healthy,helper_alive", [(True, True), (False, True), (True, False)])
def test_swe_health_requires_its_http_endpoint_and_live_helper(healthy, helper_alive):
    source = PBS.read_text()
    sweep = source.split("SWEEP_REPORT=\n", 1)[1].split('if [[ $SERVICES == 1 ]] && ! dry; then', 1)[0]
    script = f"""
set -euo pipefail
SERVICE_ENVIRONMENT=swe_rebench; SERVICE_PORT=8084
ENGINE_HOSTS=(host); ENGINE_IPS=(10.0.0.1)
SERVICE_PIDS=({'$$' if helper_alive else '999999999'})
probe() {{ [[ "$*" == http://10.0.0.1:8084/health ]] && {'true' if healthy else 'false'}; }}
{sweep}
sweep_services
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if healthy and helper_alive else 1), result.stderr


def test_swe_helper_uses_the_existing_server_and_memory_guard(tmp_path):
    platoon = tmp_path / "platoon"
    scripts = platoon / "pbs-scripts"
    scripts.mkdir(parents=True)
    (scripts / "openreward-swe-singularity-server.sh").touch()
    result = subprocess.run(
        ["bash", str(DEPLOY / "swe_rebench_service.sh")], capture_output=True, text=True, timeout=30,
        env={
            "PATH": os.environ["PATH"], "HOME": str(tmp_path), "DRY_RUN": "1",
            "PBS_JOBID": "2245678.pbs1", "PBS_LOCALDIR": str(tmp_path),
            "PLATOON_ROOT": str(platoon), "MILES_SRC": str(REPO), "OPENREWARD_REWARD_MODE": "binary",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reward_mode=binary" in result.stdout
    assert str(scripts / "openreward-swe-singularity-server.sh") in result.stdout
    assert "--fallback-available-bytes 274877906944" in result.stdout
    assert not SECRET_RE.search(result.stdout)
