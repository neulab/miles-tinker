# Tinker gateway on ABCI (PBS + Singularity + Ray)

End-to-end runbook for serving `serve_tinker.py` from this fork on two 8xH200
ABCI nodes and training against it with tinker-cookbook or platoon from the
login node. Cluster facts used throughout:

| What | Value |
|---|---|
| PBS | project `gai51740`, reservation queue `R9920261300`, `RTYPE=rt_HF` (8 H200 per node, nodes exclusive) |
| Project dir | `PROJECT_ROOT=/groups/gai51740/aci18912it/miles-tinker` (`images/`, `rootfs/`, `extra-site/`, `jobs/`, `checkpoints/`, `name-resolve/`) |
| Image | `images/miles-latest-cu13.sif` = `radixark/miles:latest` (cu13) pulled by digest |
| Rootfs | `rootfs/` bound over `/root`: `Megatron-LM/` (clone matching the image pin), `models/`, `shared_data/`, `.cache/` |
| Fork checkout | `/home/aci18912it/source/miles` bound at `/root/miles` (`MILES_SRC`) |
| Extra site | `extra-site/` = `pip --target` dir with `tinker` 0.27.1 (the gateway's proto codec needs it); prepended to `PYTHONPATH` |
| HF cache | `HF_HOME=/groups/gai51740/aci18912it/rasp/cache/huggingface` (has `Qwen/Qwen3-4B-Instruct-2507`), `HF_HUB_OFFLINE=1` in jobs |
| Logs | `/home/aci18912it/miles-tinker/logs/<job>.OU` (PBS), `$PROJECT_ROOT/jobs/<run>/` (launcher, head, worker) |
| Clients | the gateway job's head node via `CLIENT_SCRIPT` (never a login node); cookbook venv `/home/aci18912it/miles-tinker/cookbook-venv` (tinker 0.27.1), platoon venvs under `/home/aci18912it/source/platoon/plugins/*/.venv` (tinker 0.16.1) |

Commands below are typed on a login node (qsub, tail, git) but every client and probe runs inside the job on the head compute node: the launchers in `deploy/clients/` refuse to start on a `login*` host (`MILES_ALLOW_LOGIN_NODE=1` only for a seconds-long smoke).

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
mkdir -p $P/rootfs/{models,datasets,shared_data} $P/images $P/jobs $P/checkpoints $P/name-resolve
# Megatron-LM at the commit the image was built from (see images/<name>.receipt.txt)
git clone https://github.com/NVIDIA/Megatron-LM $P/rootfs/Megatron-LM && git -C $P/rootfs/Megatron-LM checkout <commit>
# tinker SDK for the gateway (protobuf codec); installed with the container's python ABI in mind (pure python + protobuf)
module load python/3.12/3.12.9
python -m pip install --target $P/extra-site "tinker==0.27.1"
# model: the HF cache already has Qwen3-4B-Instruct-2507; otherwise, from a node with egress:
#   HF_HOME=/groups/gai51740/aci18912it/rasp/cache/huggingface hf download Qwen/Qwen3-4B-Instruct-2507
```

Then validate the stack on one node (imports, fast tests inside the container,
NCCL all-reduce, tokenizer offline, router binary):

```bash
qsub deploy/abci/smoke_container.pbs                # all sections
qsub -v SECTIONS="1 2 4" deploy/abci/smoke_container.pbs
tail -50 /home/aci18912it/miles-tinker/logs/miles-tinker-smoke.o<jobid>
```

## 3. Launch the gateway

`tinker_gateway.pbs` takes two nodes: node 0 runs the Ray head, the miles
driver (`examples/tinker/run_qwen3_4b_gateway.py serve`) and the gateway HTTP
server in one container; node 1 joins as a Ray worker through `pbsdsh`. The
launcher trains on node 0 (8 GPUs, Megatron TP1/DP8) and samples on node 1
(8 SGLang engines, TP1) - see `examples/tinker/run_qwen3_4b_gateway.py` for the
flags.

```bash
# LoRA gateway (4 adapter slots, rank <= 32); RUN_NAME defaults to gw-<mode>-<jobid>
qsub -v MODE=lora,RUN_NAME=gw-lora deploy/abci/tinker_gateway.pbs
# full fine-tuning gateway (single tenant, base weights trained and pushed)
qsub -v MODE=full,RUN_NAME=gw-full deploy/abci/tinker_gateway.pbs
# knobs: LAUNCHER_ARGS go to run_qwen3_4b_gateway.py, EXTRA_ARGS straight to serve_tinker.py
qsub -v MODE=lora,LAUNCHER_ARGS="--n-adapters 8 --max-tokens-per-gpu 24576",EXTRA_ARGS="--tinker-lease-timeout-s 7200" deploy/abci/tinker_gateway.pbs
# other overrides: MILES_SRC (checkout to serve), MILES_SIF, TINKER_PORT (default 20000 + jobid % 20000), WANDB_MODE
```

Startup takes a few minutes (Ray join, bridge-mode HF load, 8 SGLang engines,
`warmup`). Watch it:

```bash
qstat -u $USER
J=$PROJECT_ROOT/jobs/gw-lora        # $PROJECT_ROOT/jobs/<RUN_NAME>
tail -f $J/launcher.log             # node map, "gateway healthy at http://hnodeNNN:PORT"
tail -f $J/head.log                 # ray head + driver + gateway (serve_tinker.py) output
tail -f $J/ray-worker.log           # node 1
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

## 4. Run clients inside the job (head node)

ABCI login nodes are shared and monitored; a Tinker client (rollouts, rendering, per-token logprobs) pins a
core for hours, so it runs on the gateway job's head compute node. Pass a bash script with `CLIENT_SCRIPT`; the
launcher starts it once the gateway is healthy with `TINKER_BASE_URL`, `ENDPOINT_FILE`, `JOBDIR`, `RUN_NAME`,
`HF_HOME` and `no_proxy` exported, logs it to `$JOBDIR/client.log` (exit code in `client.status`), and ends the
job when the client finishes (`KEEP_GATEWAY=1` keeps serving). Ready-made scripts:

```bash
# SDK smoke + sampler/trainer probe + short MFU benchmark (LoRA or full)
qsub -v RTYPE=rt_HF,MODE=lora,RUN_NAME=smoke-lora,CLIENT_SCRIPT=$PWD/deploy/abci/clients/smoke.sh deploy/abci/tinker_gateway.pbs
# tinker-cookbook guess_number, async RL, 25 steps (GN_ARGS="learning_rate=1e-5" etc.)
qsub -v RTYPE=rt_HF,MODE=lora,RUN_NAME=gn-lora,CLIENT_SCRIPT=$PWD/deploy/abci/clients/cookbook_guess_number.sh,GN_ARGS="learning_rate=1e-5" deploy/abci/tinker_gateway.pbs
# platoon number-search, 100 steps, W&B online (run `wandb login` once beforehand; the key lives in ~/.netrc)
qsub -v RTYPE=rt_HF,MODE=full,RUN_NAME=ns-full,CLIENT_SCRIPT=$PWD/deploy/abci/clients/platoon_number_search.sh,STEPS=100 -l walltime=06:00:00 deploy/abci/tinker_gateway.pbs
```

Your own client is any bash script that reads `$ENDPOINT_FILE` or `$TINKER_BASE_URL`; keep venvs on `/home` or
`/groups` (compute nodes see them, have internet access for W&B/HF, and `module load python/3.12/3.12.9`).
Watch progress from a login node with `tail -f $P/jobs/<RUN_NAME>/client.log`. Two clients against one LoRA
gateway: launch the gateway with `KEEP_GATEWAY=1` and one `CLIENT_SCRIPT`, then run the second client from a
separate 1-node job that reads the endpoint file (or chain both in one script).

A quick interactive check from a login node (`MILES_ALLOW_LOGIN_NODE=1 ... tinker_sdk_smoke.py --steps 1`) is
acceptable because it finishes in about a minute; anything that samples rollouts is not.

## 5. Where things land

| Artifact | Path |
|---|---|
| Gateway checkpoints (`tinker://<model_id>/weights/<name>`, `.../sampler_weights/<name>`, `tinker_meta.json` sidecars) | `$PROJECT_ROOT/checkpoints/<RUN_NAME>/<model_id>/...` (`--checkpoint-root`) |
| Launcher / head / worker logs | `$PROJECT_ROOT/jobs/<RUN_NAME>/{launcher,head,ray-worker}.log`, `container_args.txt`, `worker.sh`, `nodes.tsv` |
| PBS stdout/err | `/home/aci18912it/miles-tinker/logs/<jobname>.o<jobid>` |
| Ray session logs (node-local, gone with the job) | `/local/<jobid>/miles/tmp/ray/session_latest/logs` |
| Gateway MFU / step metrics (`actor_train_mfu`, `num_tokens` ... per `optim_step`) | `$PROJECT_ROOT/checkpoints/<RUN_NAME>/perf.jsonl` (one line per slot per optim window), the same records as `tinker perf: {...}` lines in `head.log`, and the client's `metrics.jsonl` (`docs/fork/acceptance.md` section C) |
| Client logs | the `log_path` you pass: cookbook `metrics.jsonl`, `checkpoints.jsonl`; platoon `<log_path>/<experiment>/<trial>/` |

## 6. Stop, restart, troubleshoot

- Stop: `qdel <jobid>`. The job's `cleanup` trap TERMs the head container and
  the worker `pbsdsh`, removes the endpoint file and exits; nothing survives on
  the nodes (each node runs a single `singularity exec` with its own PID
  namespace).
- Clients survive a gateway restart only through checkpoints: cookbook resumes
  from `checkpoints.jsonl` with `behavior_if_log_dir_exists=resume`, platoon
  through its restart wrapper; both call `/weights_info` + `load_state` on the
  last `state_path`, which the gateway resolves from the `tinker_meta.json`
  sidecars, so restart the gateway with the **same** `RUN_NAME`
  (same `--checkpoint-root`).
- "worker never joined": read `ray-worker.log`; usually the IPoIB address
  detection (`ip -o -4 addr show`, the `10.0.x.y` interface) or `/local` init.
- "head container exited early": `tail -100 jobs/<run>/head.log`; the usual
  suspects are a missing `extra-site` (`ModuleNotFoundError: tinker`), an HF
  snapshot path, or SGLang engines failing to bind (`--sglang-mem-fraction-static`).
- Port in use / unreachable: `TINKER_PORT` derives from the job id and is always
  >= 20000; check `curl --noproxy '*' http://<head>:<port>/api/v1/healthz` from
  the login node (compute hostnames resolve via `getent hosts hnodeNNN`).
- Image drift: re-run `smoke_container.pbs`; section [2] lists every import the
  gateway needs and section [3] the pinned trees.
