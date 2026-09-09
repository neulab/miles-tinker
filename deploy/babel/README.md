# Training on Babel

From the repository root on Babel, submit the Slurm template:

```bash
sbatch deploy/babel/train.sbatch
```

Slurm allocates one node with four GPUs, 32 CPUs and 256 GB RAM for up to four
hours, then starts setup and training. Edit the `#SBATCH` resource requests as
needed; no GPU model or account is pinned. Python and Apptainer must be available.
Export `WANDB_API_KEY` before submission or use an existing W&B login; batch
jobs cannot answer login prompts.

The job installs the server and Platoon client, downloads the model and dataset,
starts the server, and runs **100 number-search training steps**
with **Qwen3-4B-Instruct-2507**, **LR 1e-6**, and **W&B online**. Your existing
setup and downloads are reused on later submissions.

Visible GPUs are split between learner and rollout (two each on four GPUs).
The learner uses DP, a 1024-token microbatch target and CPU optimizer offload;
the training batch is 32 groups of eight. This separate-server recipe needs
at least two GPUs. The server is stopped when the client exits.

Slurm logs go to `babel-<jobid>.log`; training outputs go to `.cache/babel/output`
in the checkout. `sbatch deploy/babel/train.sbatch --steps 2` submits a smoke
test. Arguments after the template name are passed to `deploy/babel/run.py`;
use `--work-dir PATH` to choose another storage location.
