# Fresh-machine MI300 number-search reproduction

This is the complete recipe for the MI300 run
[zsr75q6r](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/zsr75q6r):
100 optimizer updates, zero skipped updates, first-batch logprob mismatch
0.0010036 and mean mismatch 0.0038031. The MI350 reference mean was 0.0041416.
The training configuration matches the historical MI350 run; the two AITER
variables below correct MI300's BF16 rounding. No model-specific patch is needed.

## 1. Prepare the GPU host

Use an allocated Linux x86-64 host with eight MI300X GPUs, a working AMD GPU
driver, access to `/dev/kfd` and `/dev/dri`, Git, ripgrep, Python 3.9+ and Apptainer.
The validated host used Rocky Linux 9 and Apptainer 1.4.5. The container
provides the GPU Python libraries and ROCm userspace; do not install the
trainer's Torch, SGLang, or Transformer Engine into the host Python.
Allow at least 100 GiB free on local `/tmp` for setup and build caches, plus
roughly 300 GiB on the checkpoint filesystem for this run's saved states.
The Platoon client can run on a separate Linux x86-64 machine without a GPU.

If Apptainer is already installed, skip installation. On Rocky/RHEL with
administrator access:

```bash
sudo dnf install -y epel-release
sudo dnf install -y apptainer git ripgrep python3 tmux
```

On Ubuntu with administrator access:

```bash
sudo apt update
sudo apt install -y software-properties-common git ripgrep python3 python3-venv tmux
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

These package commands follow the [Apptainer installation guide](https://apptainer.org/docs/admin/main/installation.html).
On a managed cluster, use its installed Apptainer/module and GPU allocation.
The host must permit Apptainer's user namespaces; see its installation guide
if the container fails before Python starts.

```bash
apptainer --version
python3 --version
ls -l /dev/kfd /dev/dri
```

## 2. Get this Miles revision

The checkout must include the commit containing this guide: it forwards the
optional AITER variables through Apptainer's `--cleanenv`. Once that commit
is available on the remote branch:

```bash
git clone --branch miles-tinker-amd https://github.com/neulab/miles-tinker.git /tmp/miles-tinker-amd
cd /tmp/miles-tinker-amd
git rev-parse HEAD
```

A local commit is not automatically published. To reproduce an unpublished
commit, create a bundle in the original checkout and transfer it to the new
GPU host instead:

```bash
# Original machine; replace MI300_HOST with the new allocated host.
git -C /home1/adityabs/miles-tinker bundle create /tmp/miles-mi300-repro.bundle miles-tinker-amd
scp /tmp/miles-mi300-repro.bundle MI300_HOST:/tmp/
```

Then use this clone command on the new host in place of the GitHub clone:

```bash
git clone --branch miles-tinker-amd /tmp/miles-mi300-repro.bundle /tmp/miles-tinker-amd
cd /tmp/miles-tinker-amd
```

Use an unused checkout/runtime directory for a fresh trial. Do not copy the
old `/tmp/miles-mi300-runtime` tree or its compiled kernels to the new host.

## 3. Set the MI300 environment and provision

Run on the GPU host. Keep this environment file and source it again in every
new server shell. Change `MILES_CHECKPOINT_ROOT` to a writable filesystem with
the capacity described above; the default below is persistent under your home
and may need changing if your home quota is small.

```bash
mkdir -p /tmp/miles-mi300-runtime
cat > /tmp/miles-mi300-runtime/mi300.env <<'ENV'
export MILES_AMD_IMAGE=docker://rlsys/miles@sha256:545241a0dbbf8543ef9a6c6d14313b45ba2e78b49ce1ff8370f2c7ab354180e3
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
export AITER_JIT_DIR=/tmp/miles-mi300-runtime/aiter-jit-bf16-rne
export MILES_CHECKPOINT_ROOT="$HOME/miles-mi300-checkpoints"
ENV
source /tmp/miles-mi300-runtime/mi300.env
mkdir -p "$MILES_CHECKPOINT_ROOT"
cd /tmp/miles-tinker-amd

python3 tools/amd_tinker.py setup \
  --root /tmp/miles-mi300-runtime \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /app \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554

python3 tools/amd_tinker.py doctor \
  --root /tmp/miles-mi300-runtime \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /app \
  --peak-tflops 1307.4
```

`setup` pulls the pinned MI300 image, creates the server venv, installs this
Miles checkout, prepares compatible Bridge/SGLang dependencies, and downloads
the fixed model snapshot. This requires network access and can take time.
Do not substitute the default MI350 image: some of its compiled libraries
target only `gfx950`. The doctor should report eight GPUs, `gfx942`, and
successful matmul/MFU checks.

`CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0` requests round-to-nearest-even instead
of the pinned build's default truncation. The separate `AITER_JIT_DIR` makes
the first invocation compile the kernels with that setting. AITER does not
invalidate an existing binary when compiler flags change. This fresh cache
also handles images that already contain precompiled kernels; no cache
removal or kernel source edits are required.

## 4. Start the MI300 server

Run in a persistent shell, for example `tmux new -s miles-mi300`. This uses
four training GPUs and four TP1 inference engines, matching the historical
MI350 run rather than the different topology of its later job.

```bash
cd /tmp/miles-tinker-amd
source /tmp/miles-mi300-runtime/mi300.env
python3 tools/amd_tinker.py serve \
  --root /tmp/miles-mi300-runtime \
  --checkpoint-root "$MILES_CHECKPOINT_ROOT" \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /app \
  --train-gpus 4 --rollout-gpus 4 --rollout-gpus-per-engine 1 \
  --tp 1 --pp 1 --cp 1 \
  --context-length 65000 --max-tokens-per-gpu 65000 \
  --mem-fraction 0.9 --peak-tflops 1307.4 --port 10613 \
  -- \
  --seed 42 --bf16 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full --recompute-method uniform \
  --recompute-num-layers 1 --check-weight-update-equal \
  > /tmp/miles-mi300-runtime/logs/server.log 2>&1
```

First startup can compile kernels and capture graphs. Check from another shell:

```bash
tail -f /tmp/miles-mi300-runtime/logs/server.log
```

Once ready:

```bash
curl --fail http://127.0.0.1:10613/health
rg 'module_rmsnorm_quant|check_weights\(compare\)' /tmp/miles-mi300-runtime/logs/server.log
```

The RMSNorm import should name `aiter-jit-bf16-rne/module_rmsnorm_quant.so`.
The initial weight comparison should report no mismatches. SGLang still uses
AITER attention; the trainer uses TE FlashAttention. No additional attention,
KV-cache, graph, or sampling-backend override is part of this recipe.

## 5. Connect the client

For a separate client such as `babel-compute-node`, run this in another
persistent shell on the GPU host. Replace that SSH alias with your client
host if necessary:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:11613:127.0.0.1:10613 babel-compute-node
```

Keep that SSH connection alive. On the client host:

```bash
export TINKER_BASE_URL=http://127.0.0.1:11613
export TINKER_API_KEY=tml-local-mi300-diagnostic
export TINKER_TELEMETRY=0
curl --fail "$TINKER_BASE_URL/health"
```

The key above is a local tenant identifier for this direct gateway, not a
hosted Tinker credential. This recipe uses an SSH tunnel without Modal.
If the client runs on the GPU host itself, omit the tunnel, use port 10613
in `TINKER_BASE_URL`, and change the YAML's `tinker_base_url` accordingly.

## 6. Install the pinned number-search client

Run on the client host. A Python with venv support bootstraps `uv`; uv then
installs the Python 3.12.9 used for the validated client. All client assets
stay in `/tmp`.

```bash
python3 -m venv /tmp/platoon-bootstrap
/tmp/platoon-bootstrap/bin/pip install uv==0.11.8
export PATH=/tmp/platoon-bootstrap/bin:$PATH
export UV_PYTHON_INSTALL_DIR=/tmp/platoon-python
export UV_CACHE_DIR=/tmp/platoon-uv-cache
uv python install 3.12.9

git clone --branch legacy_code_areal https://github.com/adityasoni9998/platoon.git /tmp/platoon-mi300-match
cd /tmp/platoon-mi300-match
git checkout --detach cea67e438c222c2b642dcd1f04e348b354297763
uv venv --python 3.12.9 .venv
cat > client-constraints.txt <<'CONSTRAINTS'
pydantic_core==2.41.5
ai-rubric==0.2.4
pydantic-settings==2.12.0
chz==0.4.0
ipython_pygments_lexers==1.1.1
ipython==9.8.0
torchvision==0.26.0+cu129
pydantic-extra-types==2.11.0
tinker==0.16.1
tokenizers==0.22.2
pydantic==2.12.5
tensordict==0.10.0
torch==2.11.0+cu129
tinker_cookbook==0.1.1.dev198+g0be951bd3
datasets==4.3.0
networkx==3.3
transformers==5.3.0
wandb==0.26.1
numpy==2.4.0
litellm==1.83.4
CONSTRAINTS
uv pip install --python .venv/bin/python \
  --override client-constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  -e '.[tinker]' -e plugins/number-search
```

Run the install from the Platoon checkout so uv also reads its dependency
overrides. The client intentionally retains the validated CUDA Torch wheel;
it makes Tinker requests rather than executing training on that client GPU.
The MI300 server's ROCm environment is separate. These constraints pin the
important client packages; they are not a complete transitive lockfile.

## 7. Write the matched training configuration

On the client host, use this complete YAML. The run name/output paths differ
from the reference; training settings are the same, including
`max_staleness: 3` and `max_training_steps: 100`.

```bash
cd /tmp/platoon-mi300-match
cat > mi350-matched.yaml <<'YAML'
eval:
  every: 100000
  strategy: step
  workflow_config:
    group_size: 1
    rollout_config:
      train: false
      timeout: null
      verbose: true
      max_steps: null
      model_name: null
      output_dir: rollout_results
      return_dict: false
      step_timeout: 300
      model_api_key: null
      model_endpoint: null
      inference_params:
        top_p: 1
        temperature: 1
        max_completion_tokens: 512
      propogate_root_success: false
      skip_subagent_reward_computation: false
    depth_level_weighting: false
    leave_one_out_baseline: false
    filter_zero_variance_groups: true
  num_concurrent_rollout_workflow_workers: 256
stats:
  wandb:
    mode: online
    name: null
    tags: []
    group: null
    notes: null
    entity: null
    api_key: null
    project: number-search-platoon-tinker
    base_url: null
    resume_run_id: null
  trial_name: mi300-fresh-machine-check
  experiment_name: mi300-number-search
train:
  loss_fn: cispo
  lora_rank: 0
  optimizer:
    eps: 1.0e-08
    beta1: 0.9
    beta2: 0.95
    weight_decay: 0.017
    learning_rate: 1.0e-06
    grad_clip_norm: 1
  batch_size: 32
  model_name: Qwen/Qwen3-4B-Instruct-2507
  num_epochs: null
  max_staleness: 3
  renderer_name: qwen3_instruct_exact
  loss_fn_config:
    clip_low_threshold: 0
    clip_high_threshold: 5
  num_minibatches: 1
  workflow_config:
    group_size: 8
    rollout_config:
      train: false
      timeout: 300
      verbose: false
      max_steps: 10
      model_name: null
      output_dir: /tmp/platoon-mi300-match-results/rollouts
      return_dict: false
      step_timeout: 300
      model_api_key: null
      model_endpoint: null
      inference_params:
        top_p: 1
        temperature: 1
        max_completion_tokens: 2048
      propogate_root_success: false
      skip_subagent_reward_computation: false
    depth_level_weighting: false
    leave_one_out_baseline: false
    filter_zero_variance_groups: false
  num_microbatches: 1
  max_training_steps: 100
  context_window_length: 65000
  num_concurrent_rollout_workflow_workers: 32
log_path: /tmp/platoon-mi300-match-results
watchdog:
  enabled: true
  exit_code: 2
  timeout_seconds: 2400
checkpoint:
  every: 25
  strategy: step
  load_checkpoint_path: null
environments:
- package: null
  rollout: null
  workflow: group_rollout
  task_loader: null
  eval_rollout: null
  dataset_kwargs: {}
  dataset_loader: null
  trainer_config: null
  workflow_kwargs: {}
  reward_processor: null
  eval_dataset_kwargs: {}
  eval_dataset_loader: null
  eval_workflow_kwargs: {}
  discover_entry_points: false
tinker_base_url: http://127.0.0.1:11613
YAML
```

## 8. Run and check the result

Keep the client in a persistent shell as well. Log in to your own W&B account
or provide its existing `WANDB_API_KEY`. Adjust `WANDB_ENTITY` if you are
using a different team.

```bash
cd /tmp/platoon-mi300-match
source .venv/bin/activate
export TINKER_BASE_URL=http://127.0.0.1:11613
export TINKER_API_KEY=tml-local-mi300-diagnostic
export TINKER_TELEMETRY=0
export WANDB_ENTITY=9998_team
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
wandb login
python -m platoon.number_search.train_tinker --config mi350-matched.yaml \
  > /tmp/platoon-mi300-match-training.log 2>&1
```

From another client shell:

```bash
tail -f /tmp/platoon-mi300-match-training.log
```

The log prints the W&B run URL. Inspect `train/optim/kl_sample_train_v2`:
the validated first batch was 0.0010036, versus 0.0010306 on MI350 and
0.0604402 on the original MI300 run. These are reference observations, not
bit-exact acceptance thresholds: independently sampled batches vary. Across
100 updates, the corrected MI300 mean was 0.0038031, compared with MI350's
0.0041416. The measure is half the mean squared sampled-token logprob difference.

Server optimizer records are at `$MILES_CHECKPOINT_ROOT/perf.jsonl` and
client-requested checkpoints at
`$MILES_CHECKPOINT_ROOT/<model-id>/weights/<checkpoint-name>/`. The validated
run completed 100 updates without skipped steps. Keep checkpoints and client
logs if you want to resume later; restarting the server alone does not resume
a run. The 65000-token limit is a configured budget, not a validation of an
individual maximum-length sequence.

## 9. Stop only this runtime

After the client finishes, run on the MI300 host:

```bash
cd /tmp/miles-tinker-amd
python3 tools/amd_tinker.py stop --root /tmp/miles-mi300-runtime
```

Stop the dedicated SSH tunnel with Ctrl-C in its shell. This retains the
runtime files and checkpoints. Run these commands only on the MI300 host;
there is no step that starts, stops, or modifies the MI350 job.

The full numerical investigation is in
[MI300 logprob investigation](mi300-logprob-investigation.md). This guide
records the validated stack and checks its commands/configuration; the user
will perform the independent clean-machine reproduction.
