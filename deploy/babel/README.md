# Training on Babel

With Python, Apptainer and GPUs available:

```bash
python deploy/babel/run.py
```

This installs the server and Platoon client, downloads the model and dataset,
starts the server, waits for it, and runs **100 number-search training steps**
with **Qwen3-4B-Instruct-2507**, **LR 1e-6**, and **W&B online**. Your existing
W&B login is used; otherwise the client prompts for it during setup.

Visible GPUs are split between learner and rollout (two each on four GPUs).
The learner uses DP, a 1024-token microbatch target and CPU optimizer offload;
the training batch is 32 groups of eight. This separate-server recipe needs
at least two GPUs. The server is stopped when the client exits.

Outputs are under `/tmp/miles-<uid>/output`; use `--work-dir PATH` for persistent
storage. `--steps 2` runs a short smoke test. See `--help` for optional settings.
An explicit command after `--` still runs another repository recipe instead.
