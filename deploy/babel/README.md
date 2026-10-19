# Training on Babel

From the repository root on Babel, submit the Slurm template:

```bash
sbatch deploy/babel/train.sbatch
```

Slurm allocates one node with four GPUs, 32 CPUs and 256 GB RAM for up to four
hours, then starts setup and training. Edit the `#SBATCH` resource requests as
needed; no GPU model or account is pinned. Python and `uv` must be available on
the host. The launcher uses an installed Apptainer or installs a local copy on
Debian/Ubuntu without sudo. Export `WANDB_API_KEY` before submission or use an
existing W&B login; batch jobs cannot answer login prompts.

The job installs the server and Platoon client, downloads the model and dataset,
starts the server, and runs **100 number-search training steps**
with **Qwen3-4B-Instruct-2507**, **LR 1e-6**, and **W&B online**. Your existing
setup and downloads are reused on later submissions.

The server runs inside the container; the Platoon client runs on the host, in
its own uv environment under `<work-dir>/client`, so that rollouts can start
containers of their own. The container shares the host network namespace, so
the client reaches the server on `127.0.0.1`.

Visible GPUs are split between learner and rollout (two each on four GPUs).
The learner uses DP, a 1024-token microbatch target and CPU optimizer offload;
the training batch is 32 groups of eight. This separate-server recipe needs
at least two GPUs. The server is stopped when the client exits.

Slurm logs go to `babel-<jobid>.log`; training outputs go to `.cache/babel/output`
in the checkout. `sbatch deploy/babel/train.sbatch --steps 2` submits a smoke
test. Arguments after the template name are passed to `deploy/babel/run.py`;
use `--work-dir PATH` to choose another storage location.

Inside an existing allocation, run `python deploy/babel/run.py` directly.
This uses `/tmp/miles-<uid>` for the runtime, models, data and outputs.

## Terminal-agent RL

`--task litecoder-terminal` trains on
[LiteCoder-Terminal-RL-preview](https://huggingface.co/datasets/hamishivi/agent-task-litecoder-terminal-rl-preview)
instead of number search. Each rollout runs the Vanillux harness (one `<bash>`
command per turn) inside a per-rollout Apptainer instance started from the
task's own image, and the task's verifier scores the episode when the agent
issues `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

Submit it with its own template, which sizes the job for terminal rollouts and
reaps leftover sandboxes when the job ends:

```bash
sbatch deploy/babel/train_terminal.sbatch
```

Stage the task pool first; the whole pool is 602 images and about 446 GB, so the
default is the 16 smallest:

```bash
uv run <science-agent>/experiments/qwen38_27b_tinker_amd/pool/prepare_litecoder.py --output /data/user_data/$USER/litecoder
```

Then train against it:

```bash
python deploy/babel/run.py --task litecoder-terminal \
    --data-dir /data/user_data/$USER/litecoder
```

The plugin lives in `deploy/babel/terminal_rl/` and is copied into the Platoon
checkout at launch. It serves both terminal recipes. The sandbox backend and the Vanillux prompts are vendored
from [tmax](https://github.com/hamishivi/tmax); `backend.py` and
`vanillux_prompts.yaml` record what was changed and why. Rollouts need
Apptainer on the host with working `--fakeroot`, which is why the client runs
outside the Miles container.

`--client-arg` passes an argument straight to the Platoon client, overriding
the plugin's config. Use it to shrink a first run, which otherwise starts 32
sandboxes per step:

```bash
python deploy/babel/run.py --task litecoder-terminal \
    --data-dir /data/user_data/$USER/litecoder \
    --steps 2 --wandb-mode offline \
    --client-arg=--train.batch_size=1 \
    --client-arg=--train.workflow_config.group_size=2
```

To watch a single episode instead of a training run, use the plugin's debug
entry point from the client environment built by a previous launch:

```bash
<work-dir>/client-<task>/bin/python -m platoon.terminal_rl.rollout_once \
    --data-dir /data/user_data/$USER/litecoder --oracle
```

`--oracle` replays the task's reference solution and needs no model; it should
end with reward 1. tmax tasks ship no solution, so use `--noop` there: it submits
immediately and should score 0 with a test failure naming missing work rather
than a broken harness. Point `--base-url` at a served model to watch the real
agent instead.

Episodes are long, so the batch is small: 4 groups of 8, up to 100 bash calls
per episode, 8 concurrent rollouts. Each sandbox is sized by its `task.toml`
(typically 1 CPU and 2 GB), so concurrency is bounded by the job's CPUs and RAM
rather than by GPUs.

### tmax-15k

`--task tmax-15k` trains on [allenai/tmax-15k-open-instruct](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct),
the set the tmax paper trained on. It publishes no Apptainer pool, so
`prepare_tmax.py` builds each task's image locally from Docker Hub; pulls are
limited to roughly 100 per hour per source IP, so budget about five hours for
500 tasks. The build is resumable and records each success in `selected.json`
as it lands, so a partial pool is usable.

```bash
uv run <science-agent>/experiments/qwen38_27b_tinker_amd/pool/prepare_tmax.py --output /data/user_data/$USER/tmax15k --tasks 500
TASK=tmax-15k sbatch deploy/babel/train_terminal.sbatch --steps 50
```

Unlike the other pools, these tasks ship a `setup.sh` that installs their own
dependencies and writes their input files. tmax's own sandbox skips it, assuming
images are prebuilt with setup applied; that does not hold for these images, so
the executor runs it at reset. Tasks ship no reference solution, so `--oracle`
is unavailable and `--noop` is the harness check.
