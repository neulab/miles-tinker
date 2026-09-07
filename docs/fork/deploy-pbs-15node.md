# N-node PBS deployment with Toolathlon services (deploy/abci)

How `deploy/abci/tinker_gateway.pbs` maps a PBS allocation of N 8xH200 nodes onto the
gateway (4 Megatron trainer nodes + N-4 SGLang TP8 engine nodes), starts one Toolathlon
service per engine node, runs the platoon client on the head node, and tears everything
down. Design contracts C10 (launcher env) and C12 (service env) live here; the runbook with
commands is `deploy/abci/README.md`.

## Files

| File | Role |
|---|---|
| `deploy/abci/tinker_gateway.pbs` | the job: node map, roles, services, Ray, gateway, client, 4-phase teardown; `DRY_RUN=1` prints every pbsdsh/singularity line on a login node |
| `deploy/abci/submit_toolathlon.sh` | `qsub` presets `smoke` (5 nodes, 6 h), `recursive` / `nonrecursive` (15 nodes, 168 h); `--dry-run` prints the exact qsub line after `qstat -Q R9920261300` |
| `deploy/abci/toolathlon_service.sh` | one Toolathlon service + the node memory guard on one engine node (trimmed from the release `openreward-node-services.sh`) |
| `deploy/abci/toolathlon.env` | C12: SIF identity (path, sha256, base/OCI/portability labels), scripts root, singularity scratch, guard 5-tuple; non-secret, never `abci.env` |
| `deploy/abci/openreward-node-memory-guard.py` | verbatim copy of `platoon-v8r36-07bebf9-057108f/pbs-scripts/openreward-node-memory-guard.py` (a test asserts byte equality when the release is present) |
| `deploy/abci/kill-leftovers.sh` | teardown sweep on every node, extended to the Toolathlon process names (`nginx postgres uvicorn node chrome ... /opt/venv/bin/ /app/openreward`) |
| `deploy/abci/clients/platoon_toolathlon.sh` | CLIENT_SCRIPT: platoon Toolathlon RL under `restart_wrapper`, checkpoint pruner, `perf_report.py` |
| `deploy/abci/clients/prune_tinker_checkpoints.py` | prunes `<checkpoint_root>/<model_id>/{weights,sampler_weights}/<name>` from `checkpoints.jsonl` (stdlib; `tests/fast/fork/test_prune_tinker_checkpoints.py`) |
| `deploy/abci/clients/smoke_q36.sh` | CLIENT_SCRIPT: design 10.2 steps 1-8 on the 5-node smoke |
| `deploy/clients/tinker_r3_probe.py` | SDK 0.16.1 probe: `routing_id` on every sequence, `routing_handle` datum, C6 optim metrics |
| `tests/fast/fork/test_deploy_dry_run.py` | `bash -n`, PBS DRY_RUN (2 and 15 fake nodes), service-helper DRY_RUN, guard self-test, submit presets |

## Topology and roles (C10)

- `NODES` = unique hosts of `PBS_NODEFILE` (>= 2); `TRAIN_NODES` defaults to 4, or 1 on two
  nodes; `ROLLOUT_GPUS = 8 * (NODES - TRAIN_NODES)`; both are always passed to the launcher as
  `--actor-num-nodes $TRAIN_NODES --rollout-num-gpus $ROLLOUT_GPUS` **ahead of** `LAUNCHER_ARGS`.
  Requesting exactly `8 * NODES` GPUs is what keeps every TP8 engine on one node.
- The node-local init runs on every node through `pbsdsh` (`node-init.sh`): creates
  `$PBS_LOCALDIR/miles/tmp`, prints `init_ok <host> <IPoIB ip>` plus `init_info` lines with the
  job cgroup's `memory.max`/`memory.current` (walked up from `/proc/self/cgroup`), `PBS_LOCALDIR`,
  `nproc`, `MemTotal` and `df -h $PBS_LOCALDIR` - the facts the smoke records.
- Trainer nodes = the `TRAIN_NODES` lowest IPoIB IPs sorted numerically per octet
  (`sort -t. -k1,1n -k2,2n -k3,3n -k4,4n`), the same order `miles/ray/placement_group.py:sort_key`
  uses for actor slots, so the service nodes coincide with the engine nodes. `$JOBDIR/roles.tsv`
  holds `host  vnode-index  ip  role` in that order; the PBS head node (batch shell, Ray head,
  gateway HTTP, client) can be either a trainer or an engine node.
- `TINKER_PORT = 30000 + jobid % 9000` (outside miles' dynamic 20000+ range and Toolathlon's
  8082 / 28100-28107), bumped while `ss -ltn` shows it bound. Every container gets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Start order: Toolathlon services (so the miles port allocator, which starts at 20000 and skips
  bound ports, never takes 28100-28107) -> Ray workers (one pbsdsh + container per node,
  `ray-worker-<host>.log` / `worker-<host>.log`) -> head container (`ray status` waits 180 x 10 s
  for `8*NODES` GPUs) -> service health quorum (<= 1200 s) -> gateway `/api/v1/healthz`
  (<= 3600 s) -> monitor -> client.
- `GATEWAY_LAUNCHER` is the launcher knob (repo-relative, default the Qwen3-4B one); `LAUNCHER`
  is accepted as an alias.

## Toolathlon services (7.2, C12)

`SERVICES=1` starts, on every engine node, one pbsdsh-held copy of `toolathlon_service.sh` under
`/usr/bin/env -i PATH=... HOME USER LOGNAME PBS_JOBID PBS_LOCALDIR PLATOON_ROOT TOOLATHLON_ENV
MILES_SRC OPENREWARD_REWARD_MODE=partial OPENREWARD_ENVIRONMENT_SET=toolathlon_only`, logging to
`$JOBDIR/services-<host>.log`. The helper sources `toolathlon.env` only, unsets the ten secret
names defensively, loads `singularity-ce/4.4.1` when `module` exists and otherwise uses
`/usr/bin/singularity` by absolute path, verifies the SIF sha256 and its
`org.openreward.toolathlon.{base-sif-sha256,base-oci-digest,pbs-portability}` labels, stages the
SIF to `$PBS_LOCALDIR/toolathlon/images/`, stages PGDATA with `singularity exec --fakeroot`,
builds the 16 GiB sparse ext3 overlay, starts the memory guard with exactly
`--node-limit-bytes 2061584302080 --trigger-percent 85 --fallback-available-bytes 274877906944
--poll-seconds 2 --telemetry-seconds 60` (any other tuple exits 2), and runs the container
command of the release helper (lines 571-596: `--cleanenv --no-home --overlay`, the portable
Playwright/uv env, `OPENREWARD_PORT=8082 OPENREWARD_WORKER_BASE_PORT=28100 OPENREWARD_WORKERS=8
OPENREWARD_PG_PORT=5432 OPENREWARD_PG_MAX_CONNECTIONS=512 OPENREWARD_WORKER_PID_ISOLATION=0`).
The supervisor loop exits 1 when either child dies (TERMing the other) and appends
`ps -eo rss,comm --sort=-rss | head -8` every 300 s; the guard's 60 s `event=telemetry` lines
land in the same log. On TERM: TERM the container, 90 s, KILL; TERM the guard, 10 s; assert
`:8082` is closed.

Health (release quorum): per host >= 1 of 4 `GET http://<ip>:8082/health` probes with an
`X-Session-ID` header and >= 5 of 8 `GET http://<ip>:281{00..07}/health`, `--noproxy '*'`, URLs
built from IPoIB IPs. The monitor sweeps every 30 s; 10 consecutive failures (or a dead head
container) TERMs the batch shell -> cleanup.

Exports to `CLIENT_SCRIPT` (C10): `TINKER_BASE_URL TINKER_API_KEY ENDPOINT_FILE JOBDIR RUN_NAME
HF_HOME no_proxy MILES_SRC PLATOON_ROOT SERVICE_NODES OPENREWARD_SESSION_URLS_TOOLATHLON
OPENREWARD_SESSION_URLS PBS_LOCALDIR WORKERS`; the URL lists are `http://<engine-ip>:8082` in IP
order (empty with `SERVICES=0`). The client runs in its own session (`setsid`) so a monitor-fatal
teardown can TERM its whole process group.

## Secrets

The batch shell unsets `WANDB_API_KEY OPENAI_API_KEY OPENAI_BASE_URL LITELLM_API_KEY
LITELLM_BASE_URL RUBRIC_DEFAULT_LLM HF_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN` first thing and never sources `abci.env`/`secrets.env`; every pbsdsh line runs
under `/usr/bin/env -i` and every container under `--cleanenv`, so no credential can reach a
node helper or a container. Only `clients/platoon_toolathlon.sh` sources `abci.env` (after
checking `secrets.env` is mode 0600), inside the client process. The DRY_RUN tests assert that
none of the ten names (nor a planted value) appears in the launcher or helper output.

## Teardown (4 phases)

1. TERM the service helpers, wait up to 120 s (their own bounded cleanup is 90 s + 10 s).
2. TERM the head + worker containers (singularity's starter; `--containall` takes the whole PID
   namespace down), wait 30 s.
3. KILL whatever is left; `wait` on the explicit pid list only (a bare `wait` blocks forever on the
   `tee` process substitution behind `launcher.log`).
4. `kill-leftovers.sh` locally and via `pbsdsh` on every other node (in parallel, 90 s budget),
   logged to `launcher.log` / `teardown-<host>.log`.

## Client, restart, checkpoints (7.3)

`clients/platoon_toolathlon.sh` runs `platoon.train.tinker.restart_wrapper --max-restarts 20
--restart-delay 700 -- python -m platoon.openreward.train_scripts.tinker.train_tinker --config
<yaml> --tinker_base_url ... --log_path <runs>/<RUN_NAME>/logs --train.num_concurrent_rollout_workflow_workers $WORKERS`
with `TINKER_TELEMETRY=1` (SESSION_END frees the single full-FT slot on a clean exit; the
gateway's 600 s lease frees it otherwise, so 700 s >= lease + 30 s sweep + margin), and resumes
from `<log_path>/<experiment>/<trial>/checkpoints.jsonl` automatically. The client saves on an
AReaL-style two-tier cadence: a permanent `<batch>` checkpoint every `checkpoint.every` steps and
a disposable `recovery-<batch>` one every `checkpoint.recovery_every` steps (plus one per
(re)launch), each a full ~490 GB `kind=both` save. Every 600 s
`prune_tinker_checkpoints.py --keep 2` deletes `recovery-*` directories across all `model_id`s
except the newest two recovery records', the last `state_path` record's, and any directory
modified in the last 30 min; permanent names (`<batch>`, `<batch>_interrupted`, `final`) are
never deleted.

## Validation without a job

```bash
P=/home/aci18912it/miles-tinker/test-venv/bin/python
PYTHONPATH=$PWD $P -m pytest tests/fast/fork -p no:cacheprovider --confcutdir=tests/fast/fork -o addopts="" -q
# by hand: 15 fake nodes with unsorted IPs, every pbsdsh/singularity line, roles.tsv
printf 'hnode%03d\n' 31 48 200 12 55 7 99 100 101 102 103 104 105 106 107 > /tmp/nf
DRY_RUN=1 DRY_RUN_IPS=10.0.1.9,10.0.1.10,10.0.0.200,10.0.1.100,10.0.0.5,10.0.2.1,10.0.0.7,10.0.0.99,10.0.0.100,10.0.0.101,10.0.0.102,10.0.0.103,10.0.0.104,10.0.0.105,10.0.0.106 \
  PBS_NODEFILE=/tmp/nf PBS_JOBID=1.pbs1 PBS_LOCALDIR=/tmp PROJECT_ROOT=/tmp/pr MILES_SRC=$PWD HF_SNAPSHOT=/x \
  MODE=full SERVICES=1 CLIENT_SCRIPT=$PWD/deploy/abci/clients/smoke_q36.sh bash deploy/abci/tinker_gateway.pbs
DRY_RUN=1 MILES_SRC=$PWD bash deploy/abci/toolathlon_service.sh    # resolved SIF + sha 11d158f7..., guard argv, container line
deploy/abci/submit_toolathlon.sh smoke --dry-run
```
