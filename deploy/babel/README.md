# Training on Babel

Requires Python and Apptainer. Run any repository training recipe with:

```bash
python deploy/babel/run.py -- python scripts/run_qwen3_dense.py --help
```

The runner prepares the container and uv environment, installs this checkout,
and runs setup before every command. The recipe controls the model, data, GPU
count and parallelism. With no command, the runner only performs setup.

Options before `--`:

- `--work-dir PATH`: runtime storage, mounted at `/run/miles`; defaults to node-local `/tmp/miles-<uid>`. Use persistent storage to retain outputs.
- `--image IMAGE`: local SIF or compatible image; defaults to `docker://radixark/miles:latest`.
- `--dry-run`: print the command without executing it.

## Tinker example

Prepare Qwen3-4B-Instruct-2507:

```bash
python deploy/babel/run.py -- python examples/tinker/run_qwen3_4b_gateway.py prepare
```

Start the server with two learner GPUs (DP=2) and two rollout GPUs, as in the
[52m37s number-search run](https://wandb.ai/comp-code-llm/number-search-platoon-tinker/runs/d2ol7twe):

```bash
python deploy/babel/run.py -- python examples/tinker/run_qwen3_4b_gateway.py serve \
  --mode full \
  --num-gpus-per-node 4 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 --tp 1 \
  --max-tokens-per-gpu 1024 \
  --extra-args "--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 64"
```

Connect your Tinker client to `http://<server-host>:10613`.
