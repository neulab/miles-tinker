# Muse-Glimmer-30B overlay for the miles-latest-cu13 image

The image serves Muse-Glimmer-30B through SGLang, but it cannot train it: its transformers, Megatron-Bridge and
Megatron-LM are too old. The three pieces are pure Python, so they go in an overlay instead of a new image: a root
filesystem with a newer Megatron-LM (the job's `ROOTFS`) and an extra-site directory on `PYTHONPATH` with a newer
transformers and Megatron-Bridge (the job's `EXTRA_SITE`). The gateway presets `muse-glimmer-30b-128k` (5 nodes) and
`muse-glimmer-30b-64k` (2 nodes) need both. Use the overlay for Glimmer jobs only: its Megatron-Bridge lacks a Qwen-VL
fix (radixark/Megatron-Bridge PR #9) that context-parallel Qwen3.5/3.6 runs rely on.

| Piece | Image | Overlay | Why |
|---|---|---|---|
| transformers | 5.12.1 | 5.15.0, with a 3-line patch | 5.15.0 is the first release with Glimmer's configuration (#47867). miles' argument parsing and Megatron-Bridge read the checkpoint's config through `AutoConfig`. |
| Megatron-Bridge | 0.5.0 (radixark 582783a0) | radixark 8cd3466d, the commit upstream miles pins, plus NVIDIA PR #6256 | It has NVIDIA's Glimmer bridge (#5479, #5955), which builds through Megatron-Bridge's ModelBuilder. PR #6256 (still open) keeps the final norm's gain uncentered, so the weights round-trip bit-exactly in bf16. Without it, 1,120 of the 6,656 gains change on import and on every export, so the first weight push could never equal the engines' own load. |
| Megatron-LM | radixark 8c1e0574 | radixark fd15ee20 | That Megatron-Bridge imports Megatron-Core APIs added in radixark 8a5dbe5f (`megatron.core.models.backends`, `mla_qk_norm_config`). |
| SGLang, torch, Transformer Engine, CUDA | unchanged | unchanged | The image's SGLang (0.5.19.dev56) already serves Glimmer, with its tool and reasoning parsers, `--language-model-only` and the fp32 head. |

**The transformers patch.** With transformers 5.13 or newer, the image's SGLang fails on import: it registers
`qwen3_asr` and `qwen3_asr_thinker` with `AutoConfig` without `exist_ok`, and transformers now ships both (upstream SGLang
fixed this in 97c69783). The patch tolerates exactly these two names. A broader one would let SGLang's own flat Glimmer
config replace transformers' in the trainer, and that config hands Megatron-Bridge an attention scale 11 times too large.
SGLang's engines keep using their own config either way: they load it from SGLang's registry.

**Dependencies.** `pip --no-deps`: the image already holds what the three need, at accepted versions (transformers
5.15.0: huggingface-hub 1.28, tokenizers 0.22.2, safetensors 0.8; Megatron-Bridge 8cd3466d: `transformers>=5.12.1,<=5.15.0`
and the image's modelopt and pydantic). The newer image `miles-dev-202610102218` runs the same Megatron-Bridge and
Megatron-LM pair on the same torch, Transformer Engine, CUDA and Python.

## Build

`build.sh` makes a new overlay directory from GitHub and PyPI (or local clones), checks that it imports, and writes
`BUILD.txt`. Run it in a CPU job of the reservation, for example with the validation wrapper:

```bash
cat > /groups/gai51740/aci18912it/platoon-next/glimmer/build-overlay.sh <<'EOF'
OUT=/groups/gai51740/aci18912it/platoon-next/glimmer/overlay \
EXTRA_SITE_SRC=/groups/gai51740/aci18912it/platoon-next/miles-extra-site \
CHECK_SNAPSHOT=/groups/gai51740/aci18912it/platoon-next/glimmer/hf-home/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  bash /path/to/miles-tinker/deploy/abci/overlays/muse-glimmer/build.sh
EOF
qsub -q R9920261300 -v RTYPE=rt_HC,CMD_FILE=/groups/gai51740/aci18912it/platoon-next/glimmer/build-overlay.sh,RUN_LABEL=glimmer-overlay \
  -l select=1:ncpus=8 -l walltime=01:00:00 /home/aci18912it/platoon-refactor/validation/pbs/cpu_run.pbs
```

`EXTRA_SITE_SRC` starts the extra-site from an existing one (the tinker SDK 0.27.1); without it, pip installs the SDK.
`ROOTFS_SRC` (default `platoon-next/miles-rootfs`) gives the caches and mount points of the root filesystem;
`MEGATRON_LM` and `MEGATRON_BRIDGE` may name local clones that hold the pinned commits. Each run makes a new directory,
so an overlay a job is using is never rewritten.

## Use

Besides the preset, a Glimmer job takes the overlay and a Hugging Face cache that holds the snapshot as real files.
The container binds only the group directory, so links into another user's cache would dangle inside it. `refs/main`
must name the revision, so that a client loads the tokenizer by the model's id offline:

```bash
G=/groups/gai51740/aci18912it/platoon-next/glimmer
qsub -l select=5:mpiprocs=8 -l walltime=72:00:00 \
  -v RTYPE=rt_HF,PRESET=muse-glimmer-30b-128k,TRAIN_NODES=3,ROOTFS=$G/overlay/rootfs,EXTRA_SITE=$G/overlay/extra-site,HF_HOME=$G/hf-home,MILES_SRC=$MILES,... \
  $MILES/deploy/abci/tinker_gateway.pbs
```

The job binds `ROOTFS` writable as the containers' home, so caches land in it. Jobs may share an overlay; a rebuild
always goes to a new directory.
