#!/bin/bash
# Build the Muse-Glimmer-30B overlay of the miles-latest-cu13 image: a root filesystem and an extra-site directory that
# tinker_gateway.pbs takes as the ROOTFS and EXTRA_SITE job variables. The image itself does not change; README.md
# beside this file says why each piece is needed.
#   OUT=<new directory> [variables below] deploy/abci/overlays/muse-glimmer/build.sh
# Run it in a CPU job (it runs pip and Python inside the image, and needs PyPI and GitHub unless the sources are local):
#   OUT            the overlay to make: OUT/rootfs (ROOTFS) and OUT/extra-site (EXTRA_SITE); it must not exist
#   SIF            the image (default: miles-latest-cu13.sif of the miles-tinker project)
#   ROOTFS_SRC     a rootfs of that image to copy all but Megatron-LM from (its caches and mount points)
#   EXTRA_SITE_SRC an extra-site to start from (the tinker SDK); unset: pip installs tinker==$TINKER_VERSION
#   MEGATRON_LM / MEGATRON_BRIDGE  local clones holding the commits below (default: clone them from GitHub)
#   CHECK_SNAPSHOT a Muse-Glimmer-30B snapshot directory for the import check (optional)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$(realpath -m "${OUT:?set OUT to the overlay directory to make}")
[[ ! -e $OUT ]] || { echo "OUT=$OUT exists; the overlay is built into a new directory" >&2; exit 2; }
PROJECT=/groups/gai51740/aci18912it
SIF=${SIF:-$PROJECT/miles-tinker/images/miles-latest-cu13.sif}
ROOTFS_SRC=${ROOTFS_SRC:-$PROJECT/platoon-next/miles-rootfs}
MEGATRON_LM=${MEGATRON_LM:-https://github.com/radixark/Megatron-LM.git}
MEGATRON_BRIDGE=${MEGATRON_BRIDGE:-https://github.com/radixark/Megatron-Bridge.git}
TINKER_VERSION=${TINKER_VERSION:-0.27.1}
SIF_SHA256=$(cut -d' ' -f1 "$SIF.sha256" 2>/dev/null || true)
# The pins: Megatron-LM with the Megatron-Core APIs the Glimmer bridge imports (radixark 8a5dbe5f, in fd15ee20), the
# Megatron-Bridge commit upstream miles pins (NVIDIA's builder-backed Glimmer bridge, #5479 and #5955) with NVIDIA
# Megatron-Bridge PR #6256 on top, and the first transformers release with Glimmer's configuration (#47867).
MEGATRON_LM_COMMIT=fd15ee20a4f03b03680529baf4b8f7eeed64df1d
MEGATRON_BRIDGE_COMMIT=8cd3466d14d2337c8492827b3712482c2b3e4866
TRANSFORMERS_VERSION=5.15.0
PATCH=$HERE/megatron-bridge-pr6256.diff
for path in "$SIF" "$ROOTFS_SRC" "$PATCH" ${EXTRA_SITE_SRC:+"$EXTRA_SITE_SRC"}; do
  [[ -e $path ]] || { echo "missing $path" >&2; exit 2; }
done
source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity-ce/4.4.1 2>/dev/null || true
umask 027
WORK=$(mktemp -d "${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/glimmer-overlay.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/tmp" "$WORK/home" && chmod 1777 "$WORK/tmp"
export SINGULARITY_TMPDIR=$WORK/singularity
mkdir -p "$SINGULARITY_TMPDIR" "$OUT"
step() { echo "== $* ($(date -Is))"; }
in_image() {  # in_image <command...>: the image's python with the overlay mounted as a gateway job mounts it
  singularity exec --cleanenv --containall --no-mount cwd --home "$WORK/home" --bind "$WORK/tmp:/tmp" \
    --bind "$OUT/rootfs:/root" --bind "$WORK:$WORK" --bind "$OUT:$OUT" \
    ${CHECK_SNAPSHOT:+--bind "$CHECK_SNAPSHOT:$CHECK_SNAPSHOT:ro" --env "CHECK_SNAPSHOT=$CHECK_SNAPSHOT"} \
    --env TMPDIR=/tmp --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONNOUSERSITE=1 --env PIP_CACHE_DIR=/tmp/pip-cache \
    --env PIP_DISABLE_PIP_VERSION_CHECK=1 --env "PYTHONPATH=/root/Megatron-LM:$OUT/extra-site" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HOME=/tmp/hf \
    "$SIF" "$@"
}
checkout() {  # checkout <clone or URL> <commit> <dir>: that commit's tree in <dir>, a git checkout
  git clone --quiet --filter=blob:none --no-checkout "$1" "$3"
  git -C "$3" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$3" fetch --quiet origin "$2"
  git -C "$3" -c advice.detachedHead=false checkout --quiet "$2"
}

step "rootfs: $ROOTFS_SRC without its Megatron-LM, and radixark/Megatron-LM $MEGATRON_LM_COMMIT"
mkdir "$OUT/rootfs"
for entry in "$ROOTFS_SRC"/* "$ROOTFS_SRC"/.[!.]*; do
  [[ -e $entry && ${entry##*/} != Megatron-LM ]] && cp -a "$entry" "$OUT/rootfs/"
done
checkout "$MEGATRON_LM" "$MEGATRON_LM_COMMIT" "$OUT/rootfs/Megatron-LM"

step "extra-site: the tinker SDK, transformers $TRANSFORMERS_VERSION, Megatron-Bridge $MEGATRON_BRIDGE_COMMIT + PR #6256"
if [[ -n ${EXTRA_SITE_SRC:-} ]]; then cp -a "$EXTRA_SITE_SRC" "$OUT/extra-site"; else mkdir "$OUT/extra-site"; fi
checkout "$MEGATRON_BRIDGE" "$MEGATRON_BRIDGE_COMMIT" "$WORK/megatron-bridge"
git -C "$WORK/megatron-bridge" apply "$PATCH"
# --no-deps: the image already holds every dependency these need, at versions they accept (README.md)
[[ -n ${EXTRA_SITE_SRC:-} ]] || in_image python -m pip install --quiet --no-deps --target "$OUT/extra-site" "tinker==$TINKER_VERSION"
in_image python -m pip install --quiet --no-deps --target "$OUT/extra-site" "transformers==$TRANSFORMERS_VERSION"
in_image python -m pip install --quiet --no-deps --no-build-isolation --target "$OUT/extra-site" "$WORK/megatron-bridge"
# The image's SGLang (0.5.19.dev56) registers qwen3_asr / qwen3_asr_thinker with AutoConfig without exist_ok, and
# transformers >= 5.13 ships both, so `import sglang.srt.configs` (and miles' argument parsing) would fail. Upstream
# SGLang passes exist_ok since 97c69783. Tolerate exactly these two names: tolerating every name would let SGLang's own
# flat muse_glimmer config replace transformers' and hand megatron-bridge an 11x attention scale.
in_image python - "$OUT/extra-site/transformers/models/auto/configuration_auto.py" <<'PY'
import sys

path = sys.argv[1]
source = open(path).read()
old = """        if key in self._mapping and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
"""
new = """        if key in self._mapping and not exist_ok:
            # glimmer overlay patch: SGLang 0.5.19.dev56 (a3591427) registers these two without exist_ok, and
            # transformers >= 5.13 ships them; upstream SGLang passes exist_ok=True since 97c69783 (2026-09-06).
            if key not in ("qwen3_asr", "qwen3_asr_thinker"):
                raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
"""
assert source.count(old) == 1, "the AutoConfig.register anchor moved: check the transformers version"
open(path, "w").write(source.replace(old, new))
PY
find "$OUT/extra-site" -name __pycache__ -type d -prune -exec rm -rf {} +

step "check: imports and the versions the overlay resolves to"
# on a CPU node, CUDA's stub driver lets torch and SGLang import (a GPU node has the real one)
STUB='[[ -e /dev/nvidia0 ]] || { mkdir -p /tmp/cuda-stub && ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 && export LD_LIBRARY_PATH=/tmp/cuda-stub:${LD_LIBRARY_PATH:-}; }; exec python -'
in_image bash -c "$STUB" <<'PY'
import importlib.metadata as metadata
import os

import transformers
from transformers import AutoConfig

import megatron.bridge
import megatron.core
from megatron.bridge.models.muse_glimmer import MuseGlimmerBridge, MuseGlimmerModelBuilder  # noqa: F401
from megatron.core.models.backends import get_backend  # noqa: F401  (needs Megatron-LM >= 8a5dbe5f)
import sglang.srt.configs  # noqa: F401  (the qwen3_asr registration)

print("transformers", transformers.__version__, transformers.__file__)
print("megatron-bridge", metadata.version("megatron-bridge"), megatron.bridge.__file__)
print("megatron-core", megatron.core.__file__)
snapshot = os.environ.get("CHECK_SNAPSHOT")
if snapshot:
    config = AutoConfig.from_pretrained(snapshot)
    assert type(config).__module__.startswith("transformers."), f"AutoConfig gave {type(config)}, not transformers' class"
    print("AutoConfig", type(config).__name__, "qk_scale_factor", config.text_config.qk_scale_factor)
PY

step "stamp"
{
  echo "image $SIF${SIF_SHA256:+ (sha256 $SIF_SHA256)}"
  echo "rootfs $ROOTFS_SRC without Megatron-LM; Megatron-LM $(git -C "$OUT/rootfs/Megatron-LM" rev-parse HEAD)"
  echo "extra-site ${EXTRA_SITE_SRC:-tinker==$TINKER_VERSION}; transformers==$TRANSFORMERS_VERSION with the qwen3_asr patch;" \
       "megatron-bridge $(git -C "$WORK/megatron-bridge" rev-parse HEAD) + $(basename "$PATCH") ($(sha256sum "$PATCH" | cut -c1-16))"
  echo "built $(date -Is) on $(hostname -s) by $0"
} | tee "$OUT/BUILD.txt"
