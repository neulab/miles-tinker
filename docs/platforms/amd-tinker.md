# Tinker on the AMD cluster

## Setup

Run on an allocated GPU node with Python 3.9+ and Apptainer. MI300 has been
used with the Qwen3 family; use MI350 for Qwen3.5 and newer models.

```bash
git clone --branch soni/hyperion https://github.com/neulab/miles-tinker.git ~/miles_final
cd ~/miles_final
export MILES_AMD_ROOT="$WORK/miles-tinker"
export MILES_AMD_CHECKPOINTS="$MILES_AMD_ROOT/checkpoints"

python3 tools/amd_tinker.py setup --model-family qwen3 \
  --root "$MILES_AMD_ROOT" \
  --image-dir "$MILES_AMD_ROOT/images" --venv-dir "$MILES_AMD_ROOT/venvs"
```

Use `--model-family qwen3.5` for Qwen3.5. Setup detects MI300/MI350, installs
the matching image and environment, prepares Modal (prompting for login if
needed), and runs the GPU doctor. Run it once on each GPU type you use;
subsequent jobs reuse the files. The root defaults to `$WORK/miles-tinker`,
and image/venv paths default to its `images`/`venvs` directories. Keep these
files on `$WORK`; temporary build files use node-local scratch. If `/tmp`
is too small for setup, prefix the command with `MILES_AMD_TMPDIR=/dev/shm`.
Modal credentials stay in your home directory. To select an existing Modal
profile, export `MODAL_PROFILE` before setup and serve.

**Model conversion:** Qwen3.5 requires conversion before serving. The Qwen3
policy-only example below loads directly through HF bridge; the
[PPO example](#ppo-example) uses converted Qwen3 weights as well.
Choose a matching definition from
[`scripts/models`](../../scripts/models).

```bash
python3 tools/amd_tinker.py convert --root "$MILES_AMD_ROOT" \
  --model-family qwen3.5 --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B
```

## Usage / serve

In each new job, set the same `MILES_AMD_ROOT` and `MILES_AMD_CHECKPOINTS` as
above. Set the model repository and matching Miles architecture definition:

```bash
cd ~/miles_final
export MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
export MODEL_TYPE=qwen3-4B-Instruct-2507

python3 tools/amd_tinker.py serve \
  --model-id "$MODEL_ID" --model-type "$MODEL_TYPE" \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --train-gpus 2 --rollout-gpus 6 \
  --context-length 66000 --max-tokens-per-gpu 50000 --mem-fraction 0.9 \
  --keep-checkpoints 3 --modal-relay -- \
  --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --balance-data
```

This example needs eight GPUs on one node. Adjust the two GPU counts for
your single-node allocation. For separate single-GPU nodes, follow the
[multinode guide](amd-tinker-cluster.md), which starts Ray and Tinker across
the allocated nodes using SSH and tmux.
Model weights download on first use and are reused. For
Qwen3.5, change `MODEL_ID` and `MODEL_TYPE` to the converted model.

| Flag | Meaning |
| --- | --- |
| `--model-id` | Hugging Face model repository, also advertised to Tinker clients. |
| `--model-type` | Matching architecture filename in `scripts/models`, without `.py`. |
| `--root` | Persistent setup, model downloads, and logs. |
| `--checkpoint-root` | Shared training checkpoint directory for all GPU types. |
| `--train-gpus 2` | GPUs assigned to training. |
| `--rollout-gpus 6` | GPUs assigned to inference, one engine per GPU by default. |
| `--context-length 66000` | Maximum inference sequence length, including prompt and response. |
| `--max-tokens-per-gpu 50000` | Training token budget used to form dynamic microbatches. |
| `--mem-fraction 0.9` | SGLang's static GPU memory fraction for model weights and KV cache; default 0.9. |
| `--keep-checkpoints 3` | Retain up to three completed training states per experiment across resumes; default 3. |
| `--modal-relay` | Start an authenticated Modal HTTPS endpoint for remote clients. |
| `--` | Pass the remaining flags to the trainer. |
| `--bf16` | Train in BF16 precision. |
| `--optimizer adam` | Use Adam. |
| `--lr 1e-6` | Initial learning rate; Tinker clients can supply it on optimizer steps. |
| `--lr-decay-style constant` | Keep the configured learning-rate schedule constant. |
| `--adam-beta1 0.9`, `--adam-beta2 0.95` | Adam first- and second-moment decay rates. |
| `--adam-eps 1e-8` | Adam numerical-stability epsilon. |
| `--weight-decay 0.017` | Optimizer weight decay. |
| `--clip-grad 1.0` | Gradient norm clipping threshold. |
| `--recompute-granularity full` | Recompute complete transformer layers to reduce activation memory. |
| `--recompute-method uniform` | Recompute layers in uniform groups. |
| `--recompute-num-layers 1` | One layer per recomputation group. |
| `--balance-data` | Balance training token workload across data-parallel GPUs; also enabled by the launcher. |

Each serve launch automatically copies server and relay output to a unique
`server-*.log` under `$MILES_AMD_CHECKPOINTS/logs/`, as well as the terminal.
The launcher prints the log path; use `tail -f` on that file to follow it from
another terminal. Logs have owner-only permissions because they include relay
credentials. Checkpoint retention does not delete these logs. Worker output
forwarded through the server is included; separate Ray startup sessions still
have their own output. Client optimizer parameters take precedence where the
Tinker API supplies them.

The log prints
`TINKER_BASE_URL` and `TINKER_API_KEY`; set those in your GRPO client's environment
and use the same model ID. Treat the API key and log as private. Ctrl+C on the
serve command shuts down the server and its Modal relay.

Resume with your client's saved `tinker://.../weights/...` path. Retention
follows that resume relationship, prefers the latest resume's copy of a saved
state, and leaves unrelated experiments alone. It runs in the background
after successful saves; temporarily more than three checkpoints may exist.
Older checkpoints without resume metadata join the lineage when loaded;
unlinked historical model IDs are left alone.

Full-training Tinker currently requires separate training and inference GPUs;
colocated GRPO with inference offload is not supported by this serving path.

## PPO example

After the MI350 setup above, convert Qwen3-4B once, then start a PPO server.
Actor and critic share four training GPUs with offloading; four dedicated
GPUs serve inference. Use a fresh run name for each experiment.

```bash
PPO_RUN=tinker-ppo-dapo
python3 tools/amd_tinker.py convert --root "$MILES_AMD_ROOT" \
  --gpu mi350 --model-family qwen3 \
  --model-id Qwen/Qwen3-4B --model-type qwen3-4B

python3 tools/amd_tinker.py serve --root "$MILES_AMD_ROOT" --gpu mi350 \
  --model-id Qwen/Qwen3-4B --model-type qwen3-4B \
  --train-gpus 4 --rollout-gpus 4 --tp 1 --pp 1 --cp 1 \
  --mem-fraction 0.9 --keep-checkpoints 3 \
  --checkpoint-root "$MILES_AMD_ROOT/runs/$PPO_RUN/checkpoints" -- \
  --bf16 --megatron-to-hf-mode raw \
  --load "$MILES_AMD_ROOT/models/Qwen3-4B_torch_dist" \
  --critic-load "$MILES_AMD_ROOT/models/Qwen3-4B_torch_dist" \
  --tinker-enable-critic --offload-train --balance-data \
  --rollout-temperature 0.8 --seed 1234 \
  --sglang-max-running-requests 512 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
```

In a second terminal, follow the [DAPO-Math-17K client instructions](../fork/tinker-ppo-client.md#client--terminal-2),
setting `PPO_ROOT` to the same directory as `MILES_AMD_ROOT` and using the same
`PPO_RUN`. That page includes client environment and dataset preparation,
PPO flag descriptions, and async/resume options. For the API and supported
GPU layouts, see [PPO through Tinker](../fork/tinker-ppo.md).
