---
title: "Qwen3.5-4B scientific GRPO with OpenHands"
description: "Qwen3.5-4B scientific GRPO with OpenHands, Harbor/Sail verifiers, and four-GPU gradient accumulation."
# Generated from examples/science_rl/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
The recipe in `scripts/run_qwen35_4b_science.py` uses the native GRPO loop in
neulab/miles-tinker. OpenHands calls the policy through Miles' Qwen3.5 TITO
session server, which preserves sampled token IDs, policy log probabilities,
and assistant-only loss masks. `SciencePolicyLLM` preserves reasoning and
whitespace in SDK history; `loose_tool_call` permits equivalent JSON argument
serialization while preserving tool identities and message content. Tools execute in hardened Harbor/Sail learner
containers; hidden verifiers execute separately. An infrastructure failure
aborts the sample and filters its entire group before optimization.
Use the runtime patch in [paper2rlenv PR #14](https://github.com/neulab/paper2rlenv/pull/14)
(commit `71c1cc5`):
recoverable action-validation errors can recover normally, and the training
adapter explicitly allows hidden-verifier grading after the action budget
ends. Native agent wall-clock deadlines are accepted only when Harbor supplies
a binary hidden-verifier reward. Exhausting the action budget does not itself bypass the verifier or
determine a reward. Actual API and sandbox failures still abort the group.
Training also truncates oversized tool observations to the byte limit with
an explicit notice, preserving the command exit code and allowing the agent
to continue. The default curation runtime remains strict.
Training bounds learner commands with GNU `timeout`, returning exit code 124
as a tool observation so the policy can recover from a slow computation. The
remote transport gets termination grace; its failures still abort the group.
A missing timeout executable is an infrastructure failure.
If a solver omits an artifact and the hidden verifier returns 0, training
retains that policy failure. Files captured by the SDK must still transfer
successfully; missing transfers remain infrastructure errors.

The batch is **32 prompts × 8 independent responses = 256 trajectories**.
Training uses four GPUs, tensor parallelism 4 and data parallelism 1, with
microbatch size 1 and token-budget packing. Gradients accumulate to the global
batch; the number of packed microbatches depends on trajectory lengths. Four
single-GPU SGLang engines alternate with training on the same GPUs using
Miles' colocation/offloading. Explicit `qwen3` reasoning and `qwen3_coder` tool parsers ensure emitted
XML tool calls reach OpenHands as structured actions. Full activation
recomputation, loss recomputation, and 1,024-token probability chunks limit
memory on L40S GPUs. Tensor parallelism 4 provides room for long trajectories
and the model's 248,320-token vocabulary. The initial recipe runs 100 updates at learning rate 1e-6 and
saves a checkpoint every 10 updates.

Each completed rollout batch is saved under `OUTPUT_DIR/rollouts/{rollout_id}.pt`
before training. These files contain the original policy tokens, log
probabilities, loss masks, and verifier rewards; preserve them to diagnose or
recover from a training failure without rerunning the agent trajectories.

Prepare a pinned snapshot of `neulab/science-rl`, verify every task against its
manifest, and create `science-rl.jsonl` with `prompt` (a user-message list) and
`metadata` (`task_id`, `domain`) columns. Keep the full tasks on the trusted
agent host. Preconvert the base model to Megatron `torch_dist` using the
existing Qwen3.5-4B model definition. Never substitute another task corpus or
mount gold solutions in a learner container.

Install `paper2rlenv` with the Harbor Sail backend and OpenHands SDK in a host
Python environment separate from CUDA training dependencies. Run the server:

```bash
export SCIENCE_TASK_ROOT=/datasets/science-rl/tasks
export SCIENCE_TRIAL_ROOT=/runs/science/trials
export SCIENCE_CREDENTIAL_FILE=/private/credentials.json
export SCIENCE_AGENT_AUTH_FILE=/private/agent-auth
export SCIENCE_AGENT_WORKERS=64
python -m uvicorn examples.science_rl.agent_server:app --host 127.0.0.1 --port 18085
```

The credential JSON contains `SAIL_API_KEY` (and `HF_TOKEN` for data download).
The authorization file contains a random server token. Set both files to mode
600. The agent server reads these on the host; they are never task inputs.
W&B authentication uses a scoped host `.netrc`; set `WANDB_API_KEY` for the
Miles helper to enable logging, and ensure the container home contains that
`.netrc` so Ray workers authenticate without printing the key in argv.

```bash
python scripts/run_qwen35_4b_science.py \
  --model-dir /models --data-dir /datasets/science-rl --output-dir /runs/science \
  --rollout-batch-size 32 --group-size 8 --num-rollout 100 \
  --extra-env-vars '{"SCIENCE_AGENT_SERVER_URL":"http://127.0.0.1:18085","SCIENCE_AGENT_AUTH_FILE":"/private/agent-auth","WANDB_MODE":"online"}'
```

The server `/health` endpoint reports active trials and completed trial
statuses. Preserve individual Harbor traces, verifier evidence, dataset
revision, task digests and the W&B run URL alongside checkpoints. These are
training-set reward metrics; they do not constitute held-out scientific
evaluation.

The run-specific Babel staging, provisioning, monitoring, and cleanup pipeline
is published in [science-agent PR #17](https://github.com/neulab/science-agent/pull/17).
[Miles issue #3](https://github.com/neulab/miles-tinker/issues/3) tracks oversized
refills and the missing science-adapter abort hook. The current integration
drains surplus external trials before starting optimization.
