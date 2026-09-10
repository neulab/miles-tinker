# Training on Flame

Flame uses the [Babel launcher, setup, client and training defaults](../babel/README.md).
From the repository root on Flame, submit the Slurm template:

```bash
sbatch deploy/flame/train.sbatch
```

The template requests one node with eight GPUs, 32 CPUs and 256 GB RAM for up to
four hours. Set `--qos` to your assigned Flame QoS and adjust resources as needed.
Python must be available; the shared launcher handles Apptainer setup.
Export `WANDB_API_KEY` before submission or use an existing W&B login; batch
jobs cannot answer login prompts.

Inside an existing allocation, run directly on the allocated node:

```bash
python deploy/flame/run.py
```

Both commands run 100 steps by default; append `--steps 2` for a smoke test.
Slurm logs go to `flame-<jobid>.log` and runtime files to `.cache/flame` in the
checkout. Direct runs use `/tmp/miles-<uid>`. Use `--work-dir PATH` to choose
another storage location.
