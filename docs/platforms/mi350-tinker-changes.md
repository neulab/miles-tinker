# MI350 Tinker: ROCm 7.2 implementation notes

This branch targets MI350 with one pinned image:
`rocm/sgl-dev:miles-rocm720-mi35x-20260914`, digest
`sha256:f37c0e4d76f64185fe61c2a4362ae96f121d299f113319098c21f2e298afe185`.
The default model is `Qwen/Qwen3-4B-Instruct-2507`; use the
[endpoint commands](mi350-tinker-server.md) to provision and serve it.

## What the integration changes

- **Reuse the image's native stack.** Apptainer runs this checkout as an editable
  overlay, retaining the image's PyTorch, ROCm, SGLang, Megatron, Bridge, Ray,
  and ModelOpt versions. Setup installs missing gateway packages without
  resolving or upgrading the image's dependencies. There are no MI300 image
  selectors, source rebuilds, or old-image package-replacement paths.
- **Make container writes possible.** Runtime files live under local `/tmp`,
  AITER gets a writable copy for kernel compilation, and Ray uses a short,
  deployment-specific socket directory. Checkpoints can live on persistent
  storage outside the runtime directory.
- **Use native weight updates and checkpoint writing.** SGLang uses its
  weight-session API and Megatron uses its threaded checkpoint writer. The
  legacy weight protocol and forked-writer monkey patch have been removed.
  ROCm still uses the SGLang device name `cuda`.
- **Retain small compatibility adapters.** The image still runs Python 3.10,
  so the gateway loads Tinker's generated protobuf schema without importing
  its newer-Python client, and catches asynchronous timeouts compatibly.
  Defensive Megatron argument, SGLang CLI/launch-gate, lazy template-import,
  and Bridge process-group adapters remain pending validation on this image;
  they do not install or modify third-party source files.
- **Publish the endpoint through Modal.** `serve --modal-relay` starts an
  authenticated HTTPS relay, prints the client URL/key, and supervises its
  lifecycle with the server. Modal authentication is a one-time host setup.
- **Use one configurable context limit.** `--context-length` reaches both
  inference and the gateway's per-datum validation. The old fixed gateway
  limit is gone; 32,768 is the default. Training batching has its own
  `--max-tokens-per-gpu` budget.

GPU diagnostics, explicit MI350 MFU accounting, checkpoint pruning, configurable
checkpoint storage, SSH forwarding, and server/relay shutdown handling remain
available. Full fine-tuning serves one live training model at a time. Samplers
use the latest pushed weights; they are not immutable evaluation snapshots.

## Qwen3.5: small additional setup

The existing text-only `Qwen/Qwen3.5-4B` path needs a one-time conversion from HF
weights to a Megatron checkpoint. Use a separate runtime/checkpoint directory,
select `--model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B` for setup, conversion,
and serving, and run the launcher's `convert` action after `setup`.
The validated HF revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
pass it with `--revision` during setup.

The launcher automatically selects the existing raw text-only model definition
and converted checkpoint for that model type. No extra FLA, Triton, AITER, or
Megatron source patches are required by the recorded ROCm 7.2 configuration.
Use a 16,384-token context and an 8,192-token training batching budget to start.
Conversion requires GPU 0 to be free.

This covers text-only full fine-tuning. Reasoning-history and tool-call rendering
belong to the client; the endpoint does not repair a client's changed token
prefixes. Multimodal training and context parallelism are outside this setup.

## Validation status

Historical artifacts for the pinned image recorded Qwen3.5 with one training
GPU, four inference GPUs, BF16, and TP/PP/CP all one. They show three optimizer
updates, changed post-training logprobs, a checkpoint reload with zero measured
logprob error, and a separate completed five-update workload. That workload
also recorded one context-limit error. These results do not establish long-run
stability or larger context windows.

Qwen3-4B-Instruct worked on the older image; the commands on this branch still
await validation on the pinned image. The simplified launcher and the combined
new-image/Modal path also await an eight-GPU run. Acceptance should cover
sampling, training, inference weight refresh, checkpoint save/reload, and a
request through the public relay. Legacy recipes and experiment notes are
preserved in git at `847662ec`.
