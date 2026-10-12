# PPO through Tinker

Miles supports a policy (actor) and a scalar-value critic through unmodified
`tinker==0.16.1`. Both use standard `TrainingClient` methods. The client owns
rollout scheduling, rewards, advantages, and return targets; the server owns
model execution, optimizers, GPU residency, and checkpoints.

Enable `--tinker-train-mode full --tinker-enable-critic` with the Megatron
backend. The server supports one actor/critic pair for one tenant at a time.
Critic LoRA, disjoint actor/critic GPU pools, and concurrent actor/critic
training are unsupported. Context parallelism must be `1`.

## API contract

### Critic creation

SDK method: `await client.models.create(request: CreateModelRequest, extra_body: dict)`.

| Request field | Type / value |
|---|---|
| `request.session_id` | `str`; existing SDK session |
| `request.model_seq_id` | `int`; SDK-allocated training-client sequence ID |
| `request.base_model` | `str`; server's configured base model |
| `request.lora_config` | `LoraConfig(rank=0)`; full fine-tuning |
| `extra_body["model_role"]` | `"critic"`; omitted role defaults to `"policy"` |

The response contains `request_id`. Polling
`client.futures.retrieve(request: FutureRetrieveRequest)` resolves it to
`CreateModelResponse(model_id: str)`. Construct the standard handle with
`tinker.TrainingClient(holder, model_seq_id, model_id)`.

`ServiceClient.create_lora_training_client()` accepts neither `model_role`
nor `extra_body`; `rank=0` alone creates a policy. Existing policy-only clients
can continue to omit the role. No custom SDK class is required.

### Critic inputs

| TrainingClient method | Result after `.result()` |
|---|---|
| `forward(data: list[Datum], loss_fn="ppo", loss_fn_config=None)` | `ForwardBackwardOutput`; values without gradients |
| `forward_backward(data: list[Datum], loss_fn="ppo", loss_fn_config=None)` | `ForwardBackwardOutput`; values and loss, with gradients accumulated |

Critics accept only `loss_fn="ppo"`. `loss_fn_config` accepts only
`value_clip: float` (default `0.2`, finite and nonnegative).

Each `Datum` contains `model_input: ModelInput` and
`loss_fn_inputs: dict[str, TensorData]`. For `T` model-input tokens:

| Input field | Tensor dtype / shape | Required by | Meaning |
|---|---|---|---|
| `target_tokens` | `int64`, `[T]` | Both methods | Next-token targets |
| `values` | `float32`, `[T]` | `forward_backward` | Fixed critic predictions from before updates |
| `returns` | `float32`, `[T]` | `forward_backward` | Fixed regression targets |
| `weights` | `float32`, `[T]` | `forward_backward` | Nonnegative loss weights; zero masks a position |

All floating-point inputs must be finite. Targets follow next-token alignment:
`target_tokens[i] == model_input[i+1]` for `i < T-1`. Output `values[i]`
evaluates the context through `model_input[i]`, before `target_tokens[i]`.
Critic values are not temperature-scaled.

The critic loss is the weighted sum of the larger of the unclipped and
clipped squared errors, without a factor of `0.5`. Clipping bounds the current
prediction around the supplied old `values` by `value_clip`. The client
normalizes the loss through `weights` and keeps `values` and `returns` fixed
across updates on the same data.

### Critic outputs

| `ForwardBackwardOutput` field | Schema |
|---|---|
| `loss_fn_output_type` | `"ArrayRecord"` |
| `loss_fn_outputs` | One dictionary per datum, in input order |
| `loss_fn_outputs[i]["values"]` | `TensorData(float32, shape=[T])`; current predictions |
| `loss_fn_outputs[i]["loss:sum"]` | `TensorData(float32, shape=[1])`; per-datum loss |
| `metrics["loss:sum"]` | `float`; sum of per-datum losses |
| `metrics["num_tokens:sum"]` | `float`; returned value positions, including masked positions |
| `metrics["num_datums:sum"]` | `float`; number of datums |

Critic outputs contain `values` instead of policy `logprobs`. Both roles'
`forward` calls return placeholder zero losses; only `forward_backward`
computes training loss. Do not use `forward` loss metrics for validation.

### Optimizer and state methods

| TrainingClient method | Behavior / result |
|---|---|
| `optim_step(adam_params: AdamParams)` | Updates the selected model with its own optimizer; returns standard optimizer metrics |
| `save_state(name: str)` | Saves parameters and optimizer state; result has `path: str` |
| `load_state(path: str)` | Restores parameters with a fresh optimizer |
| `load_state_with_optimizer(path: str)` | Restores parameters and optimizer state |

These methods return SDK futures. Existing `_async` variants use the same
arguments and result schemas. Cross-role checkpoint loads are rejected.
Only the policy can publish weights using `save_weights_for_sampler` or
`save_weights_and_get_sampling_client`; critics cannot generate tokens.

### Server metadata

| Response field | Value |
|---|---|
| `get_info.model_role` | `"policy"` or `"critic"` |
| `get_server_capabilities.extensions.critic` | Whether critic creation is enabled |
| `get_server_capabilities.extensions.value_loss` | `"ppo"` |
| `get_server_capabilities.extensions.critic_ppo_reduction` | `"weighted_sum"` |
| `get_server_capabilities.extensions.value_alignment` | `"before_target"` |
| `get_server_capabilities.extensions.offload_pending_gradients` | `false` |

## GPU residency

Actor and critic share the training GPU pool and execute sequentially in
both supported modes:

| Mode | Behavior |
|---|---|
| `--offload-train` | Automatically offload the resident model and restore the requested model when switching; preserve weights and optimizer state. |
| `--no-offload-train` | Keep both resident; both must fit in the shared training GPU memory. |

Inference requires dedicated GPUs. `--colocate` is unsupported. Inference
can run concurrently with training in either mode.

With offloading, complete a model's `forward_backward` calls and `optim_step`
before requesting work on the other model, including `forward`. Multiple
backward calls may accumulate gradients before the step. Accumulated gradients
cannot be offloaded; the server rejects a switch while they are pending.

## Client scheduling and async training

The client collects rollouts, requests critic values, computes advantages and
returns, completes critic updates, completes actor updates, and publishes actor
weights to inference. The server does not select a sync or async PPO schedule.

An async client can sample future batches while training the current batch.
It must bound how old those batches become and retain the log probabilities
from the policy that generated them. Replacing those probabilities with scores
from an updated actor loses the behavior-policy denominator. No historical
policy copies are required merely to retain these scores.

Actor training scores use the server's `--rollout-temperature`; sampling must
use the same temperature. When consuming SGLang log probabilities, use
`SGLANG_RETURN_ORIGINAL_LOGPROB=0` so they include sampling temperature.
Non-critic gateways continue to require temperature `1`.

The full-model sampler serves only the latest published weights. Finish
in-flight generation before publishing a new version to keep each response
under one policy. SDK async methods alone do not enforce rollout staleness.
A one-batch lookahead is not a strict two-optimizer-step limit: with two actor
updates per batch, lag can be two updates at the first minibatch and three at
the second.

## Checkpoints and resume

After optimizer steps complete, save each model with
`save_state(name).result().path`. Critic checkpoints include its backbone,
linear value head, and optimizer state for all those parameters. Restore with
`load_state_with_optimizer(path).result()`; the saved head is not reinitialized.

Resume uses `tinker://<model_id>/weights/<name>`, not the original session ID.
New sessions and model handles can load the saved paths. Keep checkpoint files
available to the server and retain the same training parallel topology.

Actor and critic saves are separate. Record a matching pair only after both
saves succeed, together with client progress such as dataset position.
`--tinker-checkpoints-to-keep K` retains the most recent K completed checkpoints
per model lineage, including across resume. Keep at least two so an incomplete
new pair does not prune the previous pair. RNG state and queued rollouts are
not restored for exact sampling replay.

## Numerical stability

Tinker actor losses `ppo` and `importance_sampling` use native Miles' numerical
guard: bound the log-ratio to `[-20, 20]` and exponentiate in FP32. This prevents
ratio overflow from producing NaN losses or gradients, including for zero
advantages. Within this range, both retain their usual objectives; `ppo`
also applies its configured PPO clipping.

`cispo` clips and detaches its ratio before multiplying, avoiding this
particular failure. These guards do not guarantee safety for all other
non-finite inputs or arithmetic.

For server and client commands, see the [DAPO-Math-17K example](tinker-ppo-client.md).

## Critic-head initialization

For a fresh critic, HF Bridge initializes the linear head with mean-zero Gaussian
weights, standard deviation `1 / (hidden_size + 1)`, and zero bias. For Qwen3-4B,
the standard deviation is approximately `0.0003905`. No checkpoint conversion
is needed. The legacy Qwen3-4B raw Megatron loading path still zeroes the head
internally when starting from converted LM weights.
In either mode, resuming a critic training checkpoint restores the saved head
and optimizer state without resetting them.

Language-model PPO implementations use several initialization choices:

| Method | Reference |
|---|---|
| Zero head weights and bias | [OpenAI `lm-human-preferences`](https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/language/model.py#L368) passes `scale=0` for the value head. |
| Random linear-layer initialization | [TRL v0.11.4](https://github.com/huggingface/trl/blob/v0.11.4/trl/models/modeling_value_head.py) defaults to `nn.Linear` initialization; an optional normal initializer controls the standard deviation. |
| Small random weights | [Open-Reasoner-Zero critic setup](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/actors.py) enables a [normal initializer](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/models.py) with standard deviation `1 / (hidden_size + 1)` when actor and critic start from the same checkpoint. [OpenRLHF v0.8.0](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.0/openrlhf/trainer/ray/ppo_critic.py) uses the same condition and [initialization](https://github.com/OpenRLHF/OpenRLHF/blob/v0.8.0/openrlhf/models/model.py). |
| Trained reward-model initialization | [InstructGPT, §3.5 and Appendix C.4](https://arxiv.org/pdf/2203.02155) initializes the value function from the reward model. |

These are implementation choices, not evidence that one is universally best.
The OpenAI source confirms zero initialization but does not explain its rationale.
