# Tinker gateway on ABCI (PBS + Singularity + Ray)

End-to-end runbook for serving `serve_tinker.py` from this fork on N 8xH200 ABCI nodes
(2 for the Qwen3-4B smoke, 5 / 15 for the Qwen3.6-35B-A3B Toolathlon jobs) and training
against it with tinker-cookbook or platoon from inside the job. Cluster facts used throughout:

| What | Value |
|---|---|
| PBS | project `gai51740`, reservation queue `R9920261300`, `RTYPE=rt_HF` (8 H200, 192 cores, 1920 GiB memory chunk per node, nodes exclusive) |
| Project dir | `PROJECT_ROOT=/groups/gai51740/aci18912it/miles-tinker` (`images/`, `rootfs/`, `extra-site/`, `jobs/`, `checkpoints/`, `name-resolve/`, `runs/`) |
| Image | `images/miles-latest-cu13.sif` = `radixark/miles:latest` (cu13) pulled by digest |
| Rootfs | `rootfs/` bound over `/root`: `Megatron-LM/` (clone matching the image pin), `models/`, `shared_data/`, `.cache/` |
| Fork checkout | `/home/aci18912it/source/miles` bound at `/root/miles` (`MILES_SRC`; the job serves whichever checkout `MILES_SRC` names) |
| Extra site | `extra-site/` = `pip --target` dir with `tinker` 0.27.1 (the gateway's proto codec needs it); prepended to `PYTHONPATH` |
| HF cache | `HF_HOME=/groups/gai51740/aci18912it/rasp/cache/huggingface` (`Qwen/Qwen3-4B-Instruct-2507`, `apurvaga/Qwen3.6-35B-A3B-preserve-thinking`), `HF_HUB_OFFLINE=1` in jobs |
| Toolathlon | derived SIF `/groups/gai51740/aci18912it/platoon-next/toolathlon-audit/ghcr-c66ff0d/abci-sif/openreward-toolathlon-c66ff0d-pbs-7034e1a61a4a.sif` (`ghcr.io/apga/toolathlon-gym:c66ff0d` with the PBS portability patch: the grader fixes and the snowflake tools; sha256 `2fb0ee8b...`, pinned in `deploy/abci/toolathlon.env`), entrypoint + PGDATA stager from `$PLATOON_ROOT/plugins/openreward/scripts` |
| Platoon | `PLATOON_ROOT=/home/aci18912it/source/platoon-tinker` (branch `miles-tinker-toolathlon`), venv `plugins/openreward/.venv` (tinker 0.16.1) |
| Logs | `/home/aci18912it/miles-tinker/logs/<job>.OU` (PBS), `$PROJECT_ROOT/jobs/<run>/` (launcher, head, workers, services, client) |
| Clients | the gateway job's head node via `CLIENT_SCRIPT` or `CLIENT_CMD` (never a login node); cookbook venv `/home/aci18912it/miles-tinker/cookbook-venv` (tinker 0.27.1) |

Commands below are typed on a login node (qsub, tail, git) but every client and probe runs inside the job on the head compute node: the launchers in `deploy/clients/` and the client scripts in `deploy/abci/clients/` refuse to start on a `login*` host (`MILES_ALLOW_LOGIN_NODE=1` only for a seconds-long smoke).

## 1. Pull the image (once per image tag)

The login node cannot pull; `pull_image.pbs` does it on one node with the
Singularity cache on `/local`, pins the amd64 manifest digest, copies the SIF to
`images/` and writes `<name>.sif.sha256` plus a `<name>.receipt.txt` with the
labels and the miles/Megatron/sglang commits baked into the image.

```bash
qsub -v TAG=latest,DIGEST=sha256:37945072c6cfd99cedb4368a2f14bf72a9bafd032bb9ba33e5c5f0add444febe,NAME=miles-latest-cu13 deploy/abci/pull_image.pbs
# find the digest of a newer tag from a machine with docker: docker manifest inspect radixark/miles:latest
```

`latest` is retagged twice a day; always record the digest (the receipt does)
and base the fork on the miles commit the receipt lists when re-pinning.

## 2. Prepare the rootfs and extra-site (once, or after a new image)

`/root` inside the image is `0700 root`, so the job binds `rootfs/` over it and
the editable installs (`/root/miles`, `/root/Megatron-LM`) resolve there.

```bash
P=/groups/gai51740/aci18912it/miles-tinker
mkdir -p $P/rootfs/{models,datasets,shared_data} $P/images $P/jobs $P/checkpoints $P/name-resolve $P/runs
# Megatron-LM at the commit the image was built from (see images/<name>.receipt.txt)
git clone https://github.com/NVIDIA/Megatron-LM $P/rootfs/Megatron-LM && git -C $P/rootfs/Megatron-LM checkout <commit>
# tinker SDK for the gateway (protobuf codec); installed with the container's python ABI in mind (pure python + protobuf)
module load python/3.12/3.12.9
python -m pip install --target $P/extra-site "tinker==0.27.1"
```

Then validate the stack on one node (imports, fast tests inside the container,
NCCL all-reduce, tokenizer offline, router binary):

```bash
qsub deploy/abci/smoke_container.pbs                # all sections
qsub -v SECTIONS="1 2 4" deploy/abci/smoke_container.pbs
tail -50 /home/aci18912it/miles-tinker/logs/miles-tinker-smoke.o<jobid>
```

## 3. Topology: what `tinker_gateway.pbs` does with N nodes

`tinker_gateway.pbs` takes any allocation of N >= 2 nodes (`-l select=N:mpiprocs=8`):

- PBS node 0 (the batch shell) runs the Ray head, the miles driver
  (`<GATEWAY_LAUNCHER> serve`) and the gateway HTTP server in one container; every
  other node joins as a Ray worker in its own container through `pbsdsh`.
- `TRAIN_NODES` (default 4, or 1 on two nodes) nodes train (Megatron) and the other
  `N - TRAIN_NODES` sample (one SGLang engine per node, TP8 for the 35B recipe). The
  launcher always gets `--actor-num-nodes $TRAIN_NODES --rollout-num-gpus $((8*(N-TRAIN_NODES)))`
  ahead of `LAUNCHER_ARGS`. **Trainer nodes are the `TRAIN_NODES` lowest IPoIB IPs**
  (numeric per-octet order, exactly how `miles/ray/placement_group.py` assigns actor slots),
  not PBS nodes 0-3; `$JOBDIR/roles.tsv` lists `host vnode ip role`.
- `SERVICES=1` starts one Toolathlon service (nginx `:8082`, 8 uvicorn workers
  `28100-28107`, postgres on localhost) plus the node memory guard on every engine
  node before Ray comes up, waits for the release health quorum, monitors it, and
  exports the URLs to the client (section 5). Details and contracts:
  `docs/fork/deploy-pbs-15node.md`.
- The job ends when the client (`CLIENT_SCRIPT` or `CLIENT_CMD`) exits (or `KEEP_GATEWAY=1` keeps serving).

## 4. Launch

### Two-node Qwen3-4B gateway (LoRA or full)

```bash
# LoRA gateway (4 adapter slots, rank <= 32); RUN_NAME defaults to gw-<mode>-<jobid>
qsub -v MODE=lora,RUN_NAME=gw-lora deploy/abci/tinker_gateway.pbs
# full fine-tuning gateway (single tenant, base weights trained and pushed)
qsub -v MODE=full,RUN_NAME=gw-full deploy/abci/tinker_gateway.pbs
# knobs: LAUNCHER_ARGS go to the launcher, EXTRA_ARGS straight to serve_tinker.py
qsub -v MODE=lora,LAUNCHER_ARGS="--n-adapters 8 --max-tokens-per-gpu 24576",EXTRA_ARGS="--tinker-lease-timeout-s 7200" deploy/abci/tinker_gateway.pbs
```

Every knob is a `qsub -v` variable: `PRESET` (below), `MODE`, `RUN_NAME`, `TRAIN_NODES`, `SERVICES`,
`GATEWAY_LAUNCHER` (repo-relative launcher; `LAUNCHER` is an alias), `HF_SNAPSHOT`
(its `--hf-checkpoint`), `LAUNCHER_ARGS`, `EXTRA_ARGS`, `CLIENT_SCRIPT` or `CLIENT_CMD` and
`CLIENT_DIR` (section 5), `KEEP_GATEWAY`,
`WORKERS` (client rollout workers, 3), `PLATOON_ROOT`, `TOOLATHLON_ENV`, `TINKER_PORT`
(default `30000 + jobid % 9000`, bumped while bound), `MILES_SRC`, `MILES_SIF`, `WANDB_MODE`.

### Gateway presets (`PRESET=`)

A preset pins one model's gateway: the launcher, its options, the engine flags and the client's
sampling timeout. `PRESET=<name>` reads `deploy/abci/presets/<name>.env` in `MILES_SRC`, the
checkout the job serves, whatever checkout you run `qsub` from. So a launch that serves a checkout
of its own passes `MILES_SRC`; without it, the job reads the default checkout's presets and serves
its code. `PRESET=<path>` reads a file of your own in the same format. (The Toolathlon "presets"
below are `qsub` lines; these are gateways.)

| Preset | Model (`HF_MODEL` @ `HF_REVISION`) | Gateway |
|---|---|---|
| `qwen3.8-27b-128k` | `apurvaga/Qwen3.8-27B-preserve-thinking` @ `97c72561` | the fcs-q38 gateway, as rasp's `launch/abci/submit.sh frontiercs-*` submits it: full fine-tuning, `--model-variant 27b` with TP4 training and TP8 engines, 128K context, FA3 engines, initial lr 1e-6, the cache-aware router (thresholds 16 and 1.5), sampling timeout 1800 s. It advertises the HF id, not the snapshot path, so a trainer config names `model_name: apurvaga/Qwen3.8-27B-preserve-thinking` |

```bash
# fcs-q38's layout: 10 nodes, 5 train (TP4 x DP10) + 5 TP8 engines. Run from the training project: the client is
# its command line and runs in this directory. MILES is the miles-tinker checkout the job serves.
qsub -l select=10:mpiprocs=8 -l walltime=72:00:00 \
  -v RTYPE=rt_HF,PRESET=qwen3.8-27b-128k,TRAIN_NODES=5,RUN_NAME=my-run,MILES_SRC=$MILES,CLIENT_CMD="$PWD/.venv/bin/platoon train $PWD/train.yaml" \
  $MILES/deploy/abci/tinker_gateway.pbs
```

- A preset holds `KEY=value` lines for `GATEWAY_LAUNCHER MODE HF_MODEL HF_REVISION LAUNCHER_ARGS
  EXTRA_ARGS TINKER_SAMPLE_TIMEOUT_SECONDS`; a value with spaces goes in one pair of quotes. The job
  reads the file and never sources it, so nothing in it is expanded, and an unknown key, a second
  value for a key or a line it cannot read fails the job before any node starts.
- A non-empty job variable replaces the preset's value; it does not add to it, so changing one
  launcher option means passing the whole `LAUNCHER_ARGS`. `launcher.log` lists every key with the
  value used and marks the job's own.
- `HF_MODEL` and `HF_REVISION` select `$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>` as
  `HF_SNAPSHOT`, unless the job gives `HF_SNAPSHOT`. A missing snapshot fails the job with the
  `hf download` command that fetches it.
- `TINKER_SAMPLE_TIMEOUT_SECONDS` is the client's deadline for one sample request; the job exports it
  to the client with the section 5 variables.
- The gateway loads the snapshot by path. A client that names the model by id resolves it through
  `refs/main` of the hub cache, so with `HF_HUB_OFFLINE=1` it needs `refs/main` to name the
  revision; `launcher.log` prints a note when it does not (the rasp cache has no `refs/` for
  `apurvaga/Qwen3.8-27B-preserve-thinking`).

Flag by flag, with the `serve_tinker.py` argv it produces: `docs/fork/flags-and-launcher.md`.

### Context parallelism (full fine-tuning, `run_qwen3_6_35b_a3b_gateway.py`)

`--cp N` in `LAUNCHER_ARGS` splits every datum over N trainer ranks (zigzag chunks;
DP becomes GPUs / (TP * N)). `--max-tokens-per-gpu` is then a per-rank budget and
defaults to the shard of one `--context-len + 1` row. The GatedDeltaNet layers run
headwise unless `--linear-cp-mode chunkwise` is given (headwise needs TP*CP to
divide the 16 linear key heads). The two-node Qwen3.5-4B parity gateway
(`--model-variant 4b`, a bridge VL model with GDN layers):

```bash
qsub -v MODE=full,GATEWAY_LAUNCHER=examples/tinker/run_qwen3_6_35b_a3b_gateway.py,HF_SNAPSHOT=$(ls -d /groups/gai51740/aci18912it/rasp/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*),LAUNCHER_ARGS="--model-variant 4b --cp 2 --context-len 8192 --max-tokens-per-datum 8193" deploy/abci/tinker_gateway.pbs
```

Layouts, budgets and limits: `docs/fork/full-finetuning.md` (section "Context parallelism").

### Qwen3.6-35B-A3B Toolathlon presets (5 / 15 nodes)

`deploy/abci/submit_toolathlon.sh` prints `qstat -Q R9920261300`, a reminder that the
reservation's end time is not visible to this account (confirm it covers 168 h before a
job submission), the exact `qsub` line, and submits it:

```bash
deploy/abci/submit_toolathlon.sh smoke --dry-run        # 5 nodes (4 train + 1 engine/service), 06:00:00, clients/smoke_q36.sh
deploy/abci/submit_toolathlon.sh smoke                  # submit
deploy/abci/submit_toolathlon.sh recursive              # 15 nodes (4 + 11), 168:00:00, RUN_NAME=ta15-rec-q36-r3-v1, WORKERS=3
deploy/abci/submit_toolathlon.sh nonrecursive           # 15 nodes, RUN_NAME=ta15-nonrec-q36-r3-v1
# overrides: HF_SNAPSHOT, MILES_SRC (this checkout), PLATOON_ROOT, WORKERS, LAUNCHER_ARGS, EXTRA_ARGS, --run-name, --walltime
```

All presets use `MODE=full SERVICES=1 TRAIN_NODES=4 GATEWAY_LAUNCHER=examples/tinker/run_qwen3_6_35b_a3b_gateway.py`
with the `apurvaga/Qwen3.6-35B-A3B-preserve-thinking` snapshot; the jobs run
`clients/platoon_toolathlon.sh` with the recursive / non-recursive platoon yaml
(`platoon/openreward/configs/tinker/abci_toolathlon_15node_{recursive,nonrecursive}_qwen36_miles_tinker_r3_v1.yaml`).

For deterministic inference, set `LAUNCHER_ARGS=--sglang-deterministic-inference`.
The launcher matches `NCCL_ALGO=allreduce:tree` and eight NCCL channels on the
trainer and engines, including the shared weight-broadcast communicator. The
channel count can be changed with `--deterministic-nccl-nchannels`. Passing only
`--sglang-enable-deterministic-inference` through `EXTRA_ARGS` configures only the
engines and can fail the first weight push with an NCCL bootstrap size mismatch.
Full routing is selected independently with
`EXTRA_ARGS='--tinker-routing-capture-mode full'`.

Startup takes a few minutes on two nodes and longer on 15 (Ray join of 14 workers, bridge-mode
HF load, TP8 engines, `warmup`, the ~50 s Toolathlon services). Watch it:

```bash
qstat -u $USER
J=$PROJECT_ROOT/jobs/<RUN_NAME>
tail -f $J/launcher.log             # node facts (init_info: cgroup memory.max, PBS_LOCALDIR, df), roles.tsv, "services healthy", "gateway healthy at http://hnodeNNN:PORT"
tail -f $J/head.log                 # ray head + driver + gateway (serve_tinker.py) output
tail -f $J/worker-<host>.log        # one per worker node (ray-worker-<host>.log holds pbsdsh-level errors)
tail -f $J/services-<host>.log      # one per Toolathlon service node: staging, guard telemetry (60 s), top-8 RSS (300 s)
```

### Endpoint file

When the Ray cluster is up the job writes
`$PROJECT_ROOT/name-resolve/<RUN_NAME>.endpoint` containing
`http://<head-host>:<port>`; the health loop then polls
`GET /api/v1/healthz`. Clients read the file (the launchers under
`deploy/clients/` take `--endpoint-file`), or set:

```bash
export TINKER_BASE_URL=$(cat $PROJECT_ROOT/name-resolve/gw-lora.endpoint)
export TINKER_API_KEY=tml-local        # SDK 0.16.1 requires a "tml-" prefix; the gateway keys tenants on this value
curl -s --noproxy '*' $TINKER_BASE_URL/api/v1/healthz
```

The file is removed when the job ends. Only the gateway port is used by
clients; the SGLang router and engines bind the compute nodes' IPoIB
addresses and are reached by the driver only.

## 5. Run clients inside the job (head node)

ABCI login nodes are shared and monitored; a Tinker client (rollouts, rendering, per-token logprobs) pins a
core for hours, so it runs on the gateway job's head compute node. Pass a bash script with `CLIENT_SCRIPT`, or a
command line with `CLIENT_CMD`; the launcher starts it (in its own process group) once the gateway and the services
are healthy, with these exports:

`TINKER_BASE_URL TINKER_API_KEY ENDPOINT_FILE JOBDIR RUN_NAME HF_HOME no_proxy MILES_SRC PLATOON_ROOT
SERVICE_NODES OPENREWARD_SESSION_URLS_TOOLATHLON OPENREWARD_SESSION_URLS PBS_LOCALDIR WORKERS`

(the URL lists are `http://<engine-IPoIB-ip>:8082,...` in IP order, empty with `SERVICES=0`; a `PRESET` adds its
`TINKER_SAMPLE_TIMEOUT_SECONDS`), logs it to `$JOBDIR/client.log` (exit code in `client.status`), and ends the job
when the client finishes, with the client's exit status (the nodes are released about ten seconds later;
`KEEP_GATEWAY=1` keeps serving).

`CLIENT_CMD` needs no script file. The job writes it verbatim to `$JOBDIR/client-cmd.sh` and runs that file exactly
as it runs a `CLIENT_SCRIPT`, in `CLIENT_DIR`: by default `PBS_O_WORKDIR`, the directory you ran `qsub` in. (A
`CLIENT_SCRIPT` runs in `MILES_SRC` unless `CLIENT_DIR` is set.) The command line runs as bash reads it, so its
quotes group words and its `$VARIABLES` expand on the head node, the exports above included. Before any node
starts, the job fails if both `CLIENT_SCRIPT` and `CLIENT_CMD` are set, if `CLIENT_DIR` is not a directory, or if
`bash -n` rejects the command line.

```bash
qsub -l select=2:mpiprocs=8 -v RTYPE=rt_HF,PRESET=qwen3.8-27b-128k,MILES_SRC=$MILES,CLIENT_CMD="$PWD/.venv/bin/platoon train $PWD/train.yaml" $MILES/deploy/abci/tinker_gateway.pbs
```

`qsub -v` rewrites a value written after `=`: a comma ends the value unless the value is in quotes
(`-v "CLIENT_CMD='echo a,b'"`), and the quotes are removed, because they are `qsub`'s own syntax. A command line
without quotes, commas or backslashes passes as is. To keep quotes and commas, put the command line in `qsub`'s
environment and name it in `-v` without a value; `qsub` then copies it unchanged:

```bash
CLIENT_CMD='platoon train train.yaml "stats.notes=lr 1e-6, group 8"' \
  qsub -l select=2:mpiprocs=8 -v RTYPE=rt_HF,PRESET=qwen3.8-27b-128k,MILES_SRC=$MILES,CLIENT_CMD $MILES/deploy/abci/tinker_gateway.pbs
```

Backslashes survive neither way (PBS doubles them), so a command line that needs one belongs in a `CLIENT_SCRIPT`.
Ready-made scripts:

```bash
# SDK smoke + sampler/trainer probe + short MFU benchmark (LoRA or full, two nodes)
qsub -v RTYPE=rt_HF,MODE=lora,RUN_NAME=smoke-lora,CLIENT_SCRIPT=$PWD/deploy/abci/clients/smoke.sh deploy/abci/tinker_gateway.pbs
# tinker-cookbook guess_number, async RL, 25 steps (GN_ARGS="learning_rate=1e-5" etc.)
qsub -v RTYPE=rt_HF,MODE=lora,RUN_NAME=gn-lora,CLIENT_SCRIPT=$PWD/deploy/abci/clients/cookbook_guess_number.sh,GN_ARGS="learning_rate=1e-5" deploy/abci/tinker_gateway.pbs
# platoon number-search, 100 steps, W&B online (run `wandb login` once beforehand; the key lives in ~/.netrc)
qsub -v RTYPE=rt_HF,MODE=full,RUN_NAME=ns-full,CLIENT_SCRIPT=$PWD/deploy/abci/clients/platoon_number_search.sh,STEPS=100 -l walltime=06:00:00 deploy/abci/tinker_gateway.pbs
# Qwen3.6-35B-A3B: the 5-node smoke (clients/smoke_q36.sh) and the Toolathlon jobs (clients/platoon_toolathlon.sh) via submit_toolathlon.sh (section 4)
```

`clients/smoke_q36.sh` runs design 10.2 steps 1-8: node facts + engine flags from the logs
(`enable_return_routed_experts`, `mamba_radix_cache_strategy`, `enable_fp32_lm_head`,
`HostCache[routed_experts] allocated`, warmup `check_weights`), the SDK 0.27.1 smoke, the 0.16.1 R3 probe
(`deploy/clients/tinker_r3_probe.py`: `routing_id` on every sequence of three prefix-extending turns, one cispo
datum with a `routing_handle` and one without -> `routing_covered_datums == 1`, `routing_uncovered_datums == 1`),
the 64k memory bench (`tinker_mfu_bench.py --num-datums 32 --seq-len 65536`), sampler-vs-trainer numerics,
the MFU band (0.05-0.35), weight-push timings, and the resume gate (`tinker_sdk_smoke.py --unload-before-resume`).
Checks that need another gateway configuration are relaunches of the same preset: R3 equivalence with
`LAUNCHER_ARGS="--check-weight-update-equal --r3-check"` (the launcher emits `--tinker-r3-check`) and `R3_CHECK=1`,
the strict policy with `LAUNCHER_ARGS="--check-weight-update-equal --routing-policy strict"` (or
`EXTRA_ARGS="--tinker-routing-replay-policy strict"`, appended last so it wins) and `R3_POLICY=strict`, the
fused-head off compare with `LAUNCHER_ARGS="--check-weight-update-equal --no-fused-lm-head"` and `FUSED_HEAD=0`.
`submit_toolathlon.sh` forwards `LAUNCHER_ARGS` / `EXTRA_ARGS` from its environment; the `R3_CHECK` / `R3_POLICY` /
`FUSED_HEAD` knobs are read by the client script from the job environment, so append them to the `qsub -v` list
that `--dry-run` prints and submit that line by hand.

`clients/platoon_toolathlon.sh` sources `/home/aci18912it/rasp/config/abci.env` (secrets.env must be 0600) in the
client process only, exports the C11 client env (`TINKER_TELEMETRY=1`, HF offline, `OPENREWARD_*`), runs the trainer
under `platoon.train.tinker.restart_wrapper --max-restarts 20 --restart-delay 700` (the gateway frees the full-FT slot
on SESSION_END or after the 600 s lease, so a killed trainer re-acquires it and resumes from `checkpoints.jsonl`
without operator action), and prunes checkpoints every 10 min with `clients/prune_tinker_checkpoints.py --keep 2`.
Knobs: `CONFIG`, `WORKERS` (3; ramp 6 -> 12 by restarting the client), `KEEP_CKPTS`, `WANDB_TRIAL`, `EXTRA`.

Your own client is any bash script that reads `$ENDPOINT_FILE` or `$TINKER_BASE_URL`; keep venvs on `/home` or
`/groups` (compute nodes see them, have internet access for W&B/HF, and `module load python/3.12/3.12.9`).
Watch progress from a login node with `tail -f $P/jobs/<RUN_NAME>/client.log`. Two clients against one LoRA
gateway: launch the gateway with `KEEP_GATEWAY=1` and one `CLIENT_SCRIPT`, then run the second client from a
separate 1-node job that reads the endpoint file (or chain both in one script).

A quick interactive check from a login node (`MILES_ALLOW_LOGIN_NODE=1 ... tinker_sdk_smoke.py --steps 1`) is
acceptable because it finishes in about a minute; anything that samples rollouts is not.

## 6. Where things land

| Artifact | Path |
|---|---|
| Gateway checkpoints (`tinker://<model_id>/weights/<name>`, `.../sampler_weights/<name>`, `tinker_meta.json` sidecars) | `$PROJECT_ROOT/checkpoints/<RUN_NAME>/<model_id>/...` (`--checkpoint-root`); pruned by `clients/prune_tinker_checkpoints.py` on the Toolathlon jobs |
| Launcher / head / worker / service / client logs | `$PROJECT_ROOT/jobs/<RUN_NAME>/{launcher,head}.log`, `worker-<host>.log`, `ray-worker-<host>.log`, `services-<host>.log`, `client.log` + `client.status` (+ `client-cmd.sh` with `CLIENT_CMD`), `teardown-<host>.log`, `roles.tsv`, `nodes.tsv`, `container_args.txt`, `worker.sh`, `node-init.sh`, `kill-leftovers.sh` |
| PBS stdout/err | `/home/aci18912it/miles-tinker/logs/<jobname>.o<jobid>` |
| Ray session logs (node-local, gone with the job) | `/local/<jobid>/miles/tmp/ray/session_latest/logs` |
| Toolathlon node-local state (gone with the job) | `/local/<jobid>/toolathlon/{images,state,rootfs-overlay.img,singularity-cache,singularity-tmp}` |
| Gateway MFU / step metrics (`actor_train_mfu`, `num_tokens`, `routing_*` ... per `optim_step`) | `$PROJECT_ROOT/checkpoints/<RUN_NAME>/perf.jsonl` (one line per slot per optim window), the same records as `tinker perf: {...}` lines in `head.log`, and the client's `metrics.jsonl` (`docs/fork/acceptance.md` section C) |
| Client logs | the `log_path` you pass: cookbook `metrics.jsonl`, `checkpoints.jsonl`; platoon `$PROJECT_ROOT/runs/<RUN_NAME>/logs/<experiment>/<trial>/` |

## 7. Memory guard and node telemetry

### SWE-rebench V2 services

Set `SERVICES=1,SERVICE_ENVIRONMENT=swe_rebench` to run one SWE server on each
inference node. The launcher probes port 8084 and forwards the same IPoIB URL
pool as `OPENREWARD_SESSION_URLS_SWE_REBENCH` to Platoon. Toolathlon remains the
default service environment. Set `OPENREWARD_REWARD_MODE=binary` for strict
test-pass rewards.

`swe_rebench_service.sh` supervises the existing Platoon
`pbs-scripts/openreward-swe-singularity-server.sh` and the node memory guard.
The pinned SWE backend extracts a private writable directory for each episode
and retains exact UUID session cleanup. Directory containers avoid the nodes'
ten-loop-device limit and make root-owned task build caches writable by the
service UID. They use more node-local scratch space than sparse overlays.
`swe_rebench.env` contains only dataset, image-cache, source,
and job-local directory settings; `SWE_ENV` can select another credential-free
file. The environment server runs on the host so it can start Singularity task
instances.

The existing `clients/platoon_toolathlon.sh` also runs SWE configs; its filename
is retained for existing submissions. `CONFIG` accepts an absolute path or a
path relative to the OpenReward plugin. The config's environment must be
`nebius/SWE-rebench-V2`, with session URL port 8084 and
`session_urls_env_var: OPENREWARD_SESSION_URLS_SWE_REBENCH`.

Each service node runs `deploy/abci/openreward-node-memory-guard.py` (verbatim release copy) with exactly
`--node-limit-bytes 2061584302080 --trigger-percent 85 --fallback-available-bytes 274877906944 --poll-seconds 2
--telemetry-seconds 60`: it watches the job cgroup that carries the 1920 GiB `memory.max` (falling back to
`MemAvailable`), prints `event=telemetry ... memory_current_bytes=...` every 60 s into `services-<host>.log`, and
exits 4 at 85 % (1632 GiB) - which fails the service helper closed, the launcher's monitor counts 10 sweeps (5 min)
and tears the job down cleanly instead of PBS killing the whole allocation. The helper also appends the node's
top-8 RSS processes every 5 min. Budget per engine node: SGLang TP8 schedulers incl. the pinned R3 host cache
(~50 GB per node at mem fraction 0.75) + Toolathlon (8 uvicorn + postgres + MCP/Chromium per session); keep RSS
below ~1.2 TB (`--sglang-max-total-tokens` bounds the pool otherwise) and ramp `WORKERS` 3 -> 6 -> 12 only while it
stays flat.

## 8. Stop, restart, troubleshoot

- Stop: `qdel <jobid>`, or let the client finish (the job ends by itself unless `KEEP_GATEWAY=1`). The
  `cleanup` trap runs four phases: (1) TERM the service helpers and wait up to 120 s (each TERMs its container,
  waits 90 s, KILLs, stops the guard and asserts `:8082` is closed), (2) TERM the miles containers (each is its
  own PID namespace, so every ray / SGLang process goes down with it) and wait 30 s, (3) KILL what is left and wait
  on those explicit pids only (a bare `wait` blocks forever on the `tee` behind `launcher.log`, bash 5.1), (4)
  `kill-leftovers.sh` locally and via `pbsdsh` on every other node - the Toolathlon container has no PID namespace,
  so nginx/postgres/uvicorn/MCP processes are swept by name (`teardown-<host>.log`). PBS releases the nodes about
  ten seconds after the client exits.
- Clients survive a gateway restart only through checkpoints: cookbook resumes
  from `checkpoints.jsonl` with `behavior_if_log_dir_exists=resume`, platoon
  through its restart wrapper; both call `/weights_info` + `load_state` on the
  last `state_path`, which the gateway resolves from the `tinker_meta.json`
  sidecars, so restart the gateway with the **same** `RUN_NAME`
  (same `--checkpoint-root`) and, for platoon, the same `TINKER_API_KEY` tenant.
- "workers never joined": read `ray-worker-<host>.log` / `worker-<host>.log`; usually the IPoIB address
  detection (`ip -o -4 addr show`, the `10.0.x.y` interface) or `/local` init (`init_ok` lines in `launcher.log`).
- "services not healthy after 1200 s": `services-<host>.log` shows staging (SIF sha, PGDATA, overlay), the
  guard's `event=start`, and the entrypoint's output; the quorum line names the host and the failing probes
  (`nginx=k/4 workers=m/8`).
- "head container exited early": `tail -100 jobs/<run>/head.log`; the usual
  suspects are a missing `extra-site` (`ModuleNotFoundError: tinker`), an HF
  snapshot path, or SGLang engines failing to bind (`--sglang-mem-fraction-static`).
- Port in use / unreachable: `TINKER_PORT` derives from the job id (30000-38999) and is bumped while bound;
  check `curl --noproxy '*' http://<head>:<port>/api/v1/healthz` from the login node (compute hostnames
  resolve via `getent hosts hnodeNNN`).
- Dry run on a login node (no PBS, no containers): `DRY_RUN=1` with a fake `PBS_NODEFILE`/`PBS_JOBID`/
  `PBS_LOCALDIR` prints `roles.tsv` and every pbsdsh/singularity line; `DRY_RUN=1 bash deploy/abci/toolathlon_service.sh`
  prints the resolved SIF + sha, the guard argv and the container command; `tests/fast/fork/test_deploy_dry_run.py`
  automates both (`docs/fork/deploy-pbs-15node.md`).
- Image drift: re-run `smoke_container.pbs`; section [2] lists every import the
  gateway needs and section [3] the pinned trees.
