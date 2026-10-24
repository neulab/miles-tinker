# Start a PPO server on ABCI

[`multi_node.pbs`](deploy/abci/multi_node.pbs) starts everything in one
PBS job. By default it requests **two nodes for six hours**:

- One 8×H200 node trains the model (the **actor**) and its value estimator
  (the **critic**). Both stay in GPU memory; they take turns training.
- The other 8×H200 node generates responses.
- A Modal relay provides a public URL and API key for the server.

Ray connects the two nodes. The script starts Ray, the server, and the relay
for you. Offloading is disabled by default.

## Before the first run

Complete [ENVIRONMENT.md](ENVIRONMENT.md) once. The script expects your model,
container, and Python environments in `~/miles-env`, your code in `~/miles`,
and an authenticated Modal account. Reuse your existing installation if it
is already set up.

The default model is **Qwen/Qwen3-4B-Instruct-2507**, with a 66,000-token
context limit.

## Change server settings

Edit the variables directly below the `#PBS` lines in
[`multi_node.pbs`](deploy/abci/multi_node.pbs):

```bash
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=qwen3-4B-Instruct-2507
TRAIN_GPUS=8
TRAIN_NODES=1
ROLLOUT_GPUS=8
ROLLOUT_GPUS_PER_ENGINE=1
ENABLE_CRITIC=true
OFFLOAD_TRAIN=false
TP=1
PP=1
CP=1
CONTEXT_LENGTH=66000
MAX_TOKENS_PER_GPU=50000
INFERENCE_MEM_FRACTION=0.95
```

Training and inference GPU counts are totals. The critic shares the training
GPUs. `TP`, `PP`, and `CP` set tensor, pipeline, and context parallelism;
keep `CP=1` for PPO. `ROLLOUT_GPUS_PER_ENGINE` sets inference parallelism.
Use a model already downloaded under `~/miles-env/models/$MODEL_ID` and its
matching definition from `scripts/models/$MODEL_TYPE.py`.

Change the allocated node count in `#PBS -l select=...`:

| Nodes (`select`) | `TRAIN_GPUS` | `TRAIN_NODES` | `ROLLOUT_GPUS` |
|---|---|---|---|
| 1 | 4 | 1 | 4 |
| 2 | 8 | 1 | 8 |
| 3 | 16 | 2 | 8 |
| 4 | 16 | 2 | 16 |

`TRAIN_GPUS` must divide evenly by `TRAIN_NODES`, with at most eight training
GPUs per node. Training plus inference must fit within eight GPUs per allocated
node. PBS reads its directives before Bash runs, so `select` cannot use a
shell variable. The script starts Ray workers on every additional allocated node.

PBS keeps the submitted script; edits apply only to new submissions.

## Start the job

On an ABCI login node, check your jobs, then submit:

```bash
qstat -n -u "$USER"
qsub ~/multi_node.pbs
```

Leave room for the requested nodes within your 15-node limit. `qsub` prints
a job ID such as `1234567.pbs1`. Keep that ID for the commands below.

## Watch startup and get the URL

Replace `JOB_ID` below with your full job ID, including `.pbs1`:

```bash
JOB_ID=<your job ID>
J="$HOME/miles-env/jobs/tinker-$JOB_ID"
tail -f "$J/head.log"
```

Loading the model takes several minutes. Press Ctrl+C to stop watching the
log; this does not stop the job. Once the server is healthy, the script
starts the relay. Display its URL and key with:

```bash
grep '^TINKER_' "$J/relay.log" | tail -2
```

The relay gets one initial attempt plus three retries, with 10 seconds between
failures. This also covers failures after startup; a restart can change the URL.
If they have not appeared yet, check `relay.log`. For other startup problems,
check `launcher.log`, `ray-worker-<hostname>.log`, and `worker-<hostname>.log` in the same directory.

Logs and checkpoints survive after the job ends. Checkpoints are saved in
`~/miles-env/checkpoints/<model_id>/`. The checkpoint root is shared across
jobs; pruning retains three completed checkpoints per model lineage. A client
can resume its saved actor/critic pair on a new job using the same Tinker paths
and its PPO checkpoint manifest. Temporary files go on each node's
local scratch disk and are deleted with the job. As in the older script,
that disk is mounted at `/tmp` inside the container; it is not the small host `/tmp`.

## Stop the job

```bash
qdel JOB_ID
```

This releases the allocated nodes and stops the relay. The job also ends if the server
or a worker exits, or the relay exits successfully or exhausts its retries.
The script attempts to remove the Modal
sandbox during shutdown. The relay uses its existing default timeout (24 hours);
there is no PBS walltime conversion or six-hour relay override.

This uses the same approach as the older [multi-node script](tinker_gateway.pbs),
for Ray startup, with configurable model settings and an automatic relay.
It calls `serve_tinker.py` directly so `MODEL_TYPE` is configurable. Shell
syntax has been checked; this version has not been tested on an allocation.
