# Multi-LoRA Tinker Gateway

Serve concurrent LoRA fine-tuning clients on one shared base model through the Tinker protocol.

`serve_tinker.py` turns a miles deployment into a [Tinker](https://tinker-docs.thinkingmachines.ai/)-compatible
training service: each client trains its own LoRA adapter in an isolated slot of a shared
Qwen3-30B-A3B base, and samples from its own saved adapter versions — all through the
official `tinker` SDK, with no miles code on the client side.

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`) and the per-expert MoE projections (`linear_fc1`, `linear_fc2`).

## Run

Start the gateway (idles until clients connect):

```bash
python examples/multi_lora/run_gateway.py prepare   # once per node
python examples/multi_lora/run_gateway.py serve     # Tinker API on :10613
```

Then drive it with the smoke client, which only needs `pip install tinker`.
Each client is its own tenant: it teaches its adapter a private marker phrase,
then samples greedily and checks the completion reproduces the marker.

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/client.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

Any Tinker training loop works the same way — point the SDK at the gateway:

```python
import tinker

service = tinker.ServiceClient(base_url="http://127.0.0.1:10613", api_key="my-tenant")
training = service.create_lora_training_client(base_model="/root/models/Qwen3-30B-A3B", rank=32)
```

## Gateway protocol notes

The gateway serves the unmodified `tinker` SDK (0.16.1 JSON bodies and 0.27.1 protobuf
bodies alike). A few choices are deliberate deviations from, or refinements of, the hosted
Tinker service and are worth knowing before pointing a training loop at it.

**Tenancy and leases.** The `X-API-Key` header (or an `Authorization: Bearer` token) is the
tenant; every model, checkpoint, promise, and sampling session belongs to the tenant that
created it. Leases are per `session_id`: a session whose heartbeat is older than
`lease_timeout_s` loses its models (slot freed, resident adapter versions unloaded), its
sampling sessions, and any in-flight samples. Training state dies with the lease; only
checkpoints on disk survive.

**Sampler versions (LoRA).** Every `save_weights_for_sampler` pushes the adapter to the
engines under `<model_id>@<version>` and returns `tinker://<model_id>/sampler_weights/<name>`,
where `<name>` is the client's name (or the version number for unnamed saves, which also
mint the sampling session the SDK asks for). Only the newest `sampler_versions_to_keep`
versions of each model stay resident: older ones are unloaded from the engines and deleted
from disk once no sample is running on them, and sampling from a retired path fails with
category `user`. Base-model sampling sessions always use the pristine base weights.

**Full fine-tuning (`--tinker-train-mode full`).** The gateway serves one model (slot 0 is
the base model itself); `create_model` accepts and logs away any `lora_config`, and
`get_info` / `GET training_runs/{id}` report `is_lora: false`. Two compatibility choices:

- `POST /weights_info` answers `is_lora: true, lora_rank: 0` for full checkpoints too. Both
  SDK generations assert `is_lora` before resuming; the rank-0 `create_model` they send next is
  accepted by a full-mode gateway. The sidecar's `train_mode` field carries the truth, and a
  full checkpoint cannot be loaded into a LoRA model (or vice versa).
- Sampling is latest-only. A full push syncs the base weights into every engine, so every
  sampler path (any version up to the newest push) and every `base_model` session resolves
  to the engines' current weights. Tinker pins a sampler to its version; here staleness is
  bounded by the client's own pipelining (`max_steps_off_policy`), and the returned per-token
  `logprobs` are always those of the weights that produced the tokens, so importance-sampling
  losses stay valid.

**Resume across restarts.** Every `save_weights` / `save_weights_for_sampler` writes a
`tinker_meta.json` sidecar next to the checkpoint (tenant, base model, mode, rank, alpha,
version). `weights_info` and `load_weights` resolve ownership and LoRA shape from it, so
`create_training_client_from_state[_with_optimizer]` works after a gateway restart and from
another session of the same tenant.

**Polling and failures.** `retrieve_future` holds the request until the promise settles or
`retrieve_poll_timeout_s` elapses, then answers HTTP 408 (the SDK's quiet "poll again"). A
`forward_backward` that fails inside the trainer poisons its gradient window: the next
`optim_step` of that model discards the partial gradients and fails with category `user`
instead of stepping; `load_weights` clears the flag.

`GatewayConfig` knobs (`serve_tinker.py` flags in parentheses):

| Field | Default | Meaning |
|---|---|---|
| `train_mode` (`--tinker-train-mode`) | `lora` | `lora` (multi-slot adapters) or `full` (one model, latest-only sampling) |
| `lease_timeout_s` (`--tinker-lease-timeout-s`) | 600 | heartbeat silence after which a session's models and samplers are reclaimed |
| `retrieve_poll_timeout_s` (`--tinker-retrieve-poll-timeout-s`) | 20 | long-poll cap on `retrieve_future`; keep it under the SDK's 45 s request timeout |
| `sampler_versions_to_keep` (`--tinker-sampler-versions-to-keep`) | 3 | LoRA adapter versions kept resident per model |
| `unit_token_budget` (`--tinker-unit-token-budget`) | 262144 | tokens packed into one trainer call across models |
| `coalesce_wait_s` (`--tinker-coalesce-wait-s`) | 0.3 | how long an under-filled unit waits for more chunks of the same window |
| `coalesce_max_wait_s` | 2.0 | upper bound on that wait, counted from the first ready row |
| `max_concurrent_samples` (`--tinker-max-concurrent-samples`) | 512 | in-flight `/generate` posts per gateway |
