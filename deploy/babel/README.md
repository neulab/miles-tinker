# Tinker gateway on Babel

The deployment has two independent parts that only share an HTTP URL:

- **Server** (`server.py`, `server.sbatch`, `container/`): the Miles Tinker gateway on one or
  more GPU nodes, inside the Miles Apptainer image.
- **Client** (`client/`): Platoon number-search training against any gateway URL. It needs no
  GPU, no container and nothing else from this repository; delete the directory, and
  `train.sbatch`, if you do not need it.

## Assumptions

- `apptainer` is on PATH on the GPU nodes. The launcher does not install it.
- The client needs Python 3 and [`uv`](https://docs.astral.sh/uv/) on whichever machine runs it.
- For multi-node servers the work directory (default `.cache/babel` in the checkout under
  Slurm) is on a filesystem every node sees.
- Export `WANDB_API_KEY` before submitting a client, or have an existing W&B login; batch
  jobs cannot answer login prompts.

## One job: serve and train

```bash
sbatch deploy/babel/train.sbatch            # 100 steps; --steps 2 for a smoke test
```

One node with four GPUs fully fine-tunes Qwen3-4B-Instruct-2507 (two learner and two rollout
GPUs, LR 1e-6, W&B online). Arguments go to the client. The server is stopped when the client
exits. Logs: `babel-<jobid>.log`; outputs: `.cache/babel/output/`.

## Serve only

```bash
sbatch deploy/babel/server.sbatch --use-fa2 --mode full --max-tokens-per-gpu 1024
```

The gateway serves until the time limit or `scancel`. Once it is healthy its URL is written to
`.cache/babel/endpoints/<name>.endpoint` (default name `gateway-<jobid>`, set with `--name`) and
removed again when it stops. Logs and checkpoints go to `.cache/babel/output/<name>/`
(`gateway.log`, `prepare.log`, `checkpoints/`). Restart with the same `--name` to let a client
resume from its checkpoints.

Server options (`python deploy/babel/server.py --help`):

- `--use-fa2`: run Transformer Engine attention with FlashAttention 2 on SM8x GPUs (A100,
  A6000, L40S). The image's FlashAttention 4 has no backward pass there, so training needs this
  flag on those GPUs; newer GPUs keep their backend.
- `--port`: defaults to `20000 + jobid % 20000`.
- `--image`: defaults to a pinned `radixark/miles` digest (`DEFAULT_IMAGE` in `server.py`). The
  `latest` tag moves daily and newer SGLang releases can break the gateway, so change the pin
  deliberately. A local SIF path also works and skips the pull.
- `--work-dir`, `--dry-run`.

Every other option configures the gateway (`container/gateway.py`):

- `--mode full` trains the base weights for one client; `--mode lora` (default) serves up to
  `--n-adapters` clients at once.
- `--model-name` and `--megatron-model-type` select the model: an HF id and the matching
  definition under `scripts/models/`, e.g.
  `--model-name Qwen/Qwen3-1.7B --megatron-model-type qwen3-1.7B`.
- `--actor-num-nodes`, `--actor-num-gpus-per-node`, `--rollout-num-gpus`: the default is half
  the GPUs each on one node, and half the nodes each on several.
- `--tp`, `--max-tokens-per-gpu`, `--recompute-args`, the `--sglang-*` options, and
  `--extra-args` for anything else `serve_tinker.py` accepts.

### Multiple nodes

Edit `#SBATCH --nodes` in `server.sbatch`, or override it:

```bash
sbatch --nodes=2 deploy/babel/server.sbatch --use-fa2 --mode lora
```

The first task builds the server venv and runs the Ray head and the gateway; the others join
as Ray workers. Set `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` before submission if NCCL picks
the wrong interface.

## Client only

Against a gateway job, from another Slurm job:

```bash
sbatch deploy/babel/client/client.sbatch --endpoint-file .cache/babel/endpoints/<name>.endpoint
```

It waits up to an hour for the endpoint, so it can be submitted together with the server
(`--dependency=after:<server-jobid>`). Locally, as a smoke test, from a node or a laptop that can
reach the gateway (through `ssh -L 20613:<head-node>:<port> babel` if needed):

```bash
python deploy/babel/client/run.py --endpoint http://127.0.0.1:20613 --steps 2 --wandb-mode disabled
```

`--model-name` and `--renderer` must match the served model. Arguments after `--` go to Platoon.
Any other Tinker client works too: set `TINKER_BASE_URL` to the endpoint and `TINKER_API_KEY`
to any `tml-` value.

## Caveats

- The gateway has no authentication. The API key only separates tenants, so anyone who can
  reach the port can train or sample. Keep it inside the cluster.
- The gateway stops at the Slurm time limit. Clients resume only through their checkpoints,
  against a gateway started with the same `--name`.
- The server uses the tinker SDK 0.27.1 for its codec. Platoon's own pinned SDK is compatible.
