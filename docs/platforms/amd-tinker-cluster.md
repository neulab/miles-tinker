# Multinode Tinker training on MI350

Run Tinker across allocated MI350 nodes with one or more GPUs each. This example
uses three single-GPU nodes, two for training and one for inference; see
[Nodes with several GPUs](#nodes-with-several-gpus) for 8-GPU nodes. The
launcher starts Ray and Tinker in tmux sessions over SSH.

## Setup

Complete the one-time [AMD setup](amd-tinker.md#setup). Keep the repository,
image, venv, models, and checkpoints at the same shared paths on every node.
Each node needs Python 3.9+, Apptainer, tmux, `ip`, and GNU `tee`.

Run the commands below from a shell on the first node in `NODES`; this node is
the Ray head. It needs passwordless SSH access to every listed node, including
itself. You can use a tmux session for this shell.

Set `MILES_AMD_ROOT` to your existing setup directory and replace the node names
with your allocated hosts:

```bash
cd ~/miles_final
export MILES_AMD_ROOT="$WORK/miles_artifacts"
export MILES_AMD_CHECKPOINTS="$MILES_AMD_ROOT/checkpoints"
NODES=(HEAD_HOSTNAME WORKER_HOSTNAME_1 WORKER_HOSTNAME_2)
TRAIN_GPUS=2
INFERENCE_GPUS=1
RAY_CPUS_PER_NODE=24
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=qwen3-4B-Instruct-2507
```

The GPU counts must add up to the number of nodes times `--gpus-per-node`. Ray
assigns which nodes train and which serve inference. Set `RAY_CPUS_PER_NODE` no higher than the CPUs in
each node's allocation. The launcher uses existing allocations; stop it before
releasing the nodes. Use nodes without another Ray/Tinker deployment running.

## Nodes with several GPUs

With `--gpus-per-node 8`, training takes whole nodes first, in node IP order, so
its tensor-parallel groups stay inside a node. Training that spans nodes must
take a multiple of `--gpus-per-node` GPUs; for two 8-GPU nodes, use
`--train-gpus 8 --rollout-gpus 8`. Weights then reach the inference engines over
the network after each step. On mi3508x, add `--roce` so RCCL uses the RoCE NICs.
It sets `NCCL_NET_GDR_LEVEL=0`: with GPU-direct RDMA, the image's RCCL 2.27.7
fails the weight push from one training GPU to eight engine GPUs on another node
("Message truncated").

`PYTORCH_HIP_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF`, `NCCL_DEBUG`, and
`NCCL_DEBUG_SUBSYS` set on the head when you run `start` are applied on every
node. `deploy/amd/serve_openhands.sbatch` wraps this launcher for `NODES=...`.

## Usage / serve

Run this once on the head:

```bash
python3 tools/amd_tinker_cluster.py start \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --nodes "${NODES[@]}" \
  --train-gpus "$TRAIN_GPUS" --rollout-gpus "$INFERENCE_GPUS" \
  --cpus-per-node "$RAY_CPUS_PER_NODE" \
  --interface eth0 --ray-port 16379 --port 11613 -- \
  --model-id "$MODEL_ID" --model-type "$MODEL_TYPE" \
  --context-length 66000 --max-tokens-per-gpu 50000 --mem-fraction 0.9 \
  --keep-checkpoints 3 --modal-relay -- \
  --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --balance-data
```

| Cluster flag | Meaning |
| --- | --- |
| `--nodes` | Allocated hostnames, with the head first. |
| `--train-gpus`, `--rollout-gpus` | Number of training and inference GPUs; together all GPUs of all nodes. |
| `--gpus-per-node` | GPUs on each node, all of them used (default 1). |
| `--cpus-per-node` | CPUs advertised to Ray on each node. |
| `--object-store-gb` | Ray object store on each node (default 2). |
| `--roce` | Add the AMD HPC Fund mi3508x RoCE NCCL settings on every node. |
| `--interface` | Network interface for node addresses and GPU communication. |
| `--ray-port` | Ray head port. |
| `--port` | Tinker HTTP port on the head. |

The first `--` forwards options to `amd_tinker.py serve`; the second forwards
training options. The launcher derives the multinode topology from the GPU
counts. The [AMD guide](amd-tinker.md#usage--serve) explains the remaining flags.
Change `MODEL_ID` and `MODEL_TYPE` for another model; Qwen3.5 requires the
conversion described in the setup guide.

The command waits for the gateway and Modal relay to become healthy, then prints
the client connection settings in this terminal:

```text
Tinker is ready. Client settings:
export TINKER_BASE_URL=https://...
export TINKER_API_KEY=...
```

It also prints a command to attach to the serve tmux session. Detach with
**Ctrl+B, then D** to leave the server running. Server and relay logs are saved
under `$MILES_AMD_CHECKPOINTS/logs/server-*.log`; Ray startup logs are alongside
them as `cluster-ray-*.log`. Keep these logs private because they contain the
API key. The Modal relay has a 24-hour lifetime.

Check readiness and print the connection settings again:

```bash
python3 tools/amd_tinker_cluster.py status --root "$MILES_AMD_ROOT"
```

Startup waits up to 900 seconds. Add `--startup-timeout SECONDS` before the first
`--` to change that limit, or `--dry-run` to preview the derived command without
starting anything. If startup times out, use `status` to recheck; the sessions
remain running.

To stop or change the GPU split, first stop the client and let its checkpoint
finish, then stop the deployment across all nodes:

```bash
python3 tools/amd_tinker_cluster.py stop --root "$MILES_AMD_ROOT"
```

Use this command for full cleanup; Ctrl+C in the serve pane alone does not stop
the whole cluster. Checkpoints, logs, images, and venvs are retained. To launch
again, rerun `start` with the desired settings; one-time setup does not need
repeating. Use the newly printed URL and key for the client.
