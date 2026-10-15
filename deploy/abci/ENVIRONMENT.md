# ABCI single-node NVIDIA environment

Run this one-time setup in a host Bash shell on an allocated 8xH200 ABCI node. Then follow the [serving guide](SINGLE_NODE.md).

Set scratch in each host shell. If `PBS_LOCALDIR` is absent, set `ABCI_SCRATCH`
to your current allocation's directory under `/local`:

```bash
# If needed, replace JOB_ID with your current PBS job ID (including .pbs1).
# export ABCI_SCRATCH=/local/JOB_ID
export ABCI_SCRATCH="${ABCI_SCRATCH:-${PBS_LOCALDIR:-}}"
: "${ABCI_SCRATCH:?Set ABCI_SCRATCH to the local scratch directory for your current allocation}"
```

## Persistent files and pinned environment

```bash
export PROJECT_ROOT="$HOME/miles-env"
export MILES_SRC="$HOME/miles"
export MILES_SIF="$PROJECT_ROOT/images/miles-sglang-a359142-cu13.sif"
```

`$PROJECT_ROOT` stores all env related files and checkpoints.

| Contents | Location |
|---|---|
| NVIDIA image and receipts | `$PROJECT_ROOT/images/` |
| Megatron checkout | `$PROJECT_ROOT/rootfs/Megatron-LM/` |
| Additional Python packages | `$PROJECT_ROOT/extra-site/` |
| Your HF downloads and cache | `$PROJECT_ROOT/models/`, `$PROJECT_ROOT/hf-home/` |
| Persistent kernel caches | `$PROJECT_ROOT/cache/` |
| Checkpoints and logs | `$PROJECT_ROOT/checkpoints/` |
| Host Modal environment | `$PROJECT_ROOT/venvs/modal/` |
| Relay credentials and metadata | `$PROJECT_ROOT/relay/runtime/` |

The image digest matches the [parent ABCI guide](README.md#1-pull-the-image-once-per-image-tag),
with SGLang `0.5.19.dev56` and its matching Megatron commit. Keep these pins
paired; pulling Miles source does not update them.

## One-time server installation

Skip if already installed. Set the paths above first.

```bash
module load singularity-ce/4.4.1
mkdir -p "$PROJECT_ROOT"/{images,rootfs,extra-site,models,hf-home,cache,checkpoints}
PREP_TMP=$(mktemp -d "${ABCI_SCRATCH:?Set ABCI_SCRATCH first}/miles-pull.XXXXXX")
export SINGULARITY_CACHEDIR="$PREP_TMP/cache"
export SINGULARITY_TMPDIR="$PREP_TMP/tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

test -f "$MILES_SIF" || singularity pull --arch amd64 "$MILES_SIF" \
  docker://radixark/miles@sha256:37945072c6cfd99cedb4368a2f14bf72a9bafd032bb9ba33e5c5f0add444febe
sha256sum "$MILES_SIF" > "$MILES_SIF.sha256"
singularity inspect --labels "$MILES_SIF" > "$MILES_SIF.labels.json"

# Fresh checkout only; preserve an existing Megatron tree.
if [ ! -d "$PROJECT_ROOT/rootfs/Megatron-LM" ]; then
  git clone https://github.com/radixark/Megatron-LM "$PROJECT_ROOT/rootfs/Megatron-LM" && \
    git -C "$PROJECT_ROOT/rootfs/Megatron-LM" checkout 8c1e05747eb612b382df2632783df5c83a853646
fi

INSTALL_TMP=$(mktemp -d "${ABCI_SCRATCH:?Set ABCI_SCRATCH first}/miles-pip.XXXXXX")
singularity exec --cleanenv --containall --no-mount cwd \
  --home "$PROJECT_ROOT/rootfs:/root" \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT,$INSTALL_TMP:/tmp" --pwd /tmp \
  --env TMPDIR=/tmp \
  "$MILES_SIF" python -m pip install --no-deps --no-cache-dir --target "$PROJECT_ROOT/extra-site" \
  'tinker==0.27.1' 'pyqwest==0.12.0'
```

The image pull and pip use allocation-local scratch. The pinned image supplies
the extra packages' dependencies. After success, `PREP_TMP` and `INSTALL_TMP`
can be removed. Binding over `/root` exposes the
image's editable installs to your host user.

## One-time Modal installation and authentication

Install the relay's Python environment on the host, outside the GPU container:

```bash
module load python/3.12/3.12.9
python -m venv "$PROJECT_ROOT/venvs/modal"
"$PROJECT_ROOT/venvs/modal/bin/python" -m pip install 'modal==1.2.6'
```

Authenticate once:

```bash
"$PROJECT_ROOT/venvs/modal/bin/python" -m modal token new
```

Open the printed URL in your browser and sign in. Check the modal profile being used:

```bash
"$PROJECT_ROOT/venvs/modal/bin/python" -m modal profile list
```

Credentials live in `~/.modal.toml` by default.

## Save the host-shell environment once

This saves your paths and `miles_python`, a shortcut for containerized Python.

```bash
mkdir -p "$PROJECT_ROOT"
cat > "$PROJECT_ROOT/abci-env.sh" <<'ENV'
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/miles-env}"
export MILES_SRC="${MILES_SRC:-$HOME/miles}"
export MILES_SIF="${MILES_SIF:-$PROJECT_ROOT/images/miles-sglang-a359142-cu13.sif}"
export HF_HOME="$PROJECT_ROOT/hf-home"
export ABCI_SCRATCH="${ABCI_SCRATCH:-${PBS_LOCALDIR:-}}"
: "${ABCI_SCRATCH:?Set ABCI_SCRATCH to the local scratch directory for your current allocation}"

source /etc/profile.d/modules.sh
module load singularity-ce/4.4.1

miles_python() (
  set -e
  umask 077
  export NVIDIA_VISIBLE_DEVICES=all
  local_tmp="$ABCI_SCRATCH/miles-$USER/tmp"
  mkdir -p "$local_tmp" "$PROJECT_ROOT/models" "$PROJECT_ROOT/hf-home" "$PROJECT_ROOT/cache"
  chmod 1777 "$local_tmp"
  exec singularity exec --nv --cleanenv --containall --no-mount cwd \
    --home "$PROJECT_ROOT/rootfs:/root" \
    --bind "$MILES_SRC:/root/miles,$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$local_tmp:/tmp,/dev/infiniband:/dev/infiniband" --pwd /root/miles \
    --env "PROJECT_ROOT=$PROJECT_ROOT,HF_HOME=$PROJECT_ROOT/hf-home" \
    --env "PYTHONPATH=/root/miles:/root/Megatron-LM:$PROJECT_ROOT/extra-site" \
    --env PYTHONNOUSERSITE=1,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1 \
    --env TMPDIR=/tmp,RAY_TMPDIR=/tmp/ray,RAY_ADDRESS=local \
    --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
    --env "XDG_CACHE_HOME=$PROJECT_ROOT/cache,TRITON_CACHE_DIR=$PROJECT_ROOT/cache/triton" \
    --env "TORCHINDUCTOR_CACHE_DIR=$PROJECT_ROOT/cache/inductor,FLASHINFER_WORKSPACE_DIR=$PROJECT_ROOT/cache/flashinfer" \
    --env "TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/cache/torch-ext,CUDA_CACHE_PATH=$PROJECT_ROOT/cache/nv-cache" \
    --env "DG_JIT_CACHE_DIR=$PROJECT_ROOT/cache/deepgemm" \
    "$MILES_SIF" python "$@"
)
ENV
```

Load it in each new host shell:

```bash
source "$HOME/miles-env/abci-env.sh"
```

## Verify the environment

```bash
miles_python -c 'import torch, ray, tinker, megatron.bridge; import miles.tinker.server.proto_codec; assert torch.cuda.device_count() == 8; x = torch.ones(1, device="cuda"); print(torch.cuda.get_device_name(0), x.item())'
miles_python serve_tinker.py --help
```

These check imports, eight visible GPUs, CUDA and argument parsing, not full
training. Continue to the [GRPO/PPO serving guide](SINGLE_NODE.md).
