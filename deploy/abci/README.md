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
| Clients | login node, `module load python/3.12/3.12.9`; cookbook venv `/home/aci18912it/miles-tinker/cookbook-venv` (tinker 0.27.1), platoon venv `/home/aci18912it/source/platoon/plugins/textcraft/.venv` (tinker 0.16.1) |

Everything below is run from the login node in the fork checkout unless stated.

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

## 4. Run clients on the login node

All clients speak HTTP only (no Ray, no `/dev/shm`), so the login node is the
right place for interactive runs; keep them light (the rollout/env code of
guess_number and textcraft is small). Long unattended runs can instead be
started on node 0 from a plain host process (see `notes` in
`docs/fork/acceptance.md`).

```bash
E=$PROJECT_ROOT/name-resolve/gw-lora.endpoint
# raw SDK smoke (0.27.1): train/save/sample/resume/stop tokens, PASS/FAIL per check
/home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/tinker_sdk_smoke.py --endpoint-file $E --lora-rank 16
# cookbook math_rl arithmetic (single-turn, async, 20 steps)
deploy/clients/cookbook_math_arithmetic_smoke.sh $E /groups/gai51740/aci18912it/miles-tinker/runs/math-smoke
# cookbook guess_number, multi-turn, AsyncConfig(max_steps_off_policy=1), 25 steps (acceptance run A)
/home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/cookbook_guess_number_async.py \
    endpoint_file=$E log_path=/groups/gai51740/aci18912it/miles-tinker/runs/gn-async
# platoon textcraft synth (SDK 0.16.1, cispo), small overrides (acceptance run B)
deploy/clients/run_platoon_textcraft.sh $E /groups/gai51740/aci18912it/miles-tinker/runs/textcraft
```

Against a `MODE=full` gateway the same commands work unchanged: the gateway
accepts and ignores `lora_config` (`lora_rank` on the client is a no-op) and
`weights_info` still reports `is_lora: true` so both SDKs' resume path works
(`docs/fork/full-finetuning.md`). Run one client at a time on a full gateway.

## 5. Where things land

| Artifact | Path |
|---|---|
| Gateway checkpoints (`tinker://<model_id>/weights/<name>`, `.../sampler_weights/<name>`, `tinker_meta.json` sidecars) | `$PROJECT_ROOT/checkpoints/<RUN_NAME>/<model_id>/...` (`--checkpoint-root`) |
| Launcher / head / worker logs | `$PROJECT_ROOT/jobs/<RUN_NAME>/{launcher,head,ray-worker}.log`, `container_args.txt`, `worker.sh`, `nodes.tsv` |
| PBS stdout/err | `/home/aci18912it/miles-tinker/logs/<jobname>.o<jobid>` |
| Ray session logs (node-local, gone with the job) | `/local/<jobid>/miles/tmp/ray/session_latest/logs` |
| miles perf metrics (`perf/actor_train_mfu` ...) | `head.log` lines `perf <step>: {...}`; wandb offline dir `$PROJECT_ROOT/jobs/<RUN_NAME>/wandb` when `WANDB_MODE=offline` |
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
