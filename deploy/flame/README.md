# Training on Flame

From the repository root on Flame, submit the Slurm template:

```bash
sbatch deploy/flame/train.sbatch
```

The template requests one node with eight GPUs, 32 CPUs and 256 GB RAM for up to
four hours. Set `--qos` to your assigned Flame QoS and adjust the resource
requests as needed. Python must be available. The launcher uses an installed
Apptainer or downloads an unprivileged copy into its runtime directory on Flame's
Ubuntu nodes; no administrator installation is needed.
Export `WANDB_API_KEY` before submission or use an existing W&B login; batch
jobs cannot answer login prompts.

The job installs the server and Platoon client, downloads the model and dataset,
starts the server, and runs **100 number-search training steps**
with **Qwen3-4B-Instruct-2507**, **LR 1e-6**, and **W&B online**. Your existing
setup and downloads are reused on later submissions.

Visible GPUs are split between learner and rollout (four each on eight GPUs).
The learner uses DP, a 1024-token microbatch target and CPU optimizer offload;
the training batch is 32 groups of eight. This separate-server recipe needs
at least two GPUs. The server is stopped when the client exits.

Slurm logs go to `flame-<jobid>.log`; training outputs go to `.cache/flame/output`
in the checkout. `sbatch deploy/flame/train.sbatch --steps 2` submits a smoke
test. Arguments after the template name are passed to `deploy/flame/run.py`;
use `--work-dir PATH` to choose another storage location.

For testing inside an existing allocation, run directly on the allocated node:

```bash
python deploy/flame/run.py --steps 2
```

This uses `/tmp/miles-<uid>` for the runtime, models, data and outputs.
