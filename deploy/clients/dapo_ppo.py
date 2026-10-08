"""DAPO-Math-17K PPO using unmodified tinker==0.16.1 and a separate critic.

Requires a running full-model PPO gateway, a local tokenizer, and JSONL data
with prompt/label fields. Uses Miles' existing math grader and batch balancing;
all model operations use ordinary Tinker SDK clients.

Example:
    PYTHONPATH=. python deploy/clients/dapo_ppo.py --base-url http://server:10613 \
        --model-path /models/Qwen3-4B --data /data/dapo-math-17k.jsonl --output ./ppo-run
"""

import argparse
import asyncio
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

import tinker
from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType

logger = logging.getLogger(__name__)


def gae(values, reward, gamma=1.0, lambd=1.0):
    """Terminal reward and zero final bootstrap, including length caps, as in native Miles."""
    values = np.asarray(values, dtype=np.float32)
    advantages = np.empty_like(values)
    carry = 0.0
    for index in reversed(range(len(values))):
        following = values[index + 1] if index + 1 < len(values) else 0.0
        delta = (reward if index == len(values) - 1 else 0.0) + gamma * following - values[index]
        carry = delta + gamma * lambd * carry
        advantages[index] = carry
    return advantages, advantages + values


def whiten(advantages):
    """Whole-rollout, action-token statistics with native Miles' sample variance."""
    flat = np.concatenate(advantages).astype(np.float32)
    mean = flat.mean()
    variance = (np.mean(flat * flat) - mean * mean) * (len(flat) / max(len(flat) - 1, 1))
    return [(advantage - mean) / np.sqrt(max(variance, 0.0) + 1e-8) for advantage in advantages]


def balance_batches(trajectories, batch_size, reference_dp):
    """Preserve the old client's DP2 minibatch membership, independent of server GPU count."""
    partitions = get_seqlen_balanced_partitions(
        [len(t["prompt"]) + len(t["tokens"]) for t in trajectories],
        reference_dp,
        equal_size=True,
    )
    per_rank = batch_size // reference_dp
    order = [
        index
        for start in range(0, len(partitions[0]), per_rank)
        for partition in partitions
        for index in partition[start : start + per_rank]
    ]
    return [trajectories[index] for index in order]


def datum(trajectory, role=None, batch_size=32):
    tokens = trajectory["prompt"] + trajectory["tokens"]
    prefix = len(trajectory["prompt"]) - 1
    length = len(trajectory["tokens"])
    if prefix < 0 or length == 0:
        raise ValueError("PPO needs a nonempty prompt and response")
    inputs = {"target_tokens": tokens[1:]}
    if role == "critic":
        inputs.update(
            values=[0.0] * prefix + trajectory["values"],
            returns=[0.0] * prefix + trajectory["returns"],
            weights=[0.0] * prefix + [1.0 / (batch_size * length)] * length,
        )
    elif role == "policy":
        inputs.update(
            logprobs=[0.0] * prefix + trajectory["logprobs"],
            advantages=[0.0] * prefix + [a / (batch_size * length) for a in trajectory["advantages"]],
        )
    else:
        inputs["weights"] = [0.0] * (len(tokens) - 1)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            key: types.TensorData(
                data=value, dtype="int64" if key == "target_tokens" else "float32", shape=[len(value)]
            )
            for key, value in inputs.items()
        },
    )


async def _model_result(holder, future):
    """Poll SDK model lifecycle futures on the SDK's own event loop."""
    while True:
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            result = await client.futures.retrieve(request=types.FutureRetrieveRequest(request_id=future.request_id))
        if getattr(result, "type", None) == "try_again":
            await asyncio.sleep(0.2)
            continue
        if isinstance(result, types.RequestFailedResponse):
            raise RuntimeError(result.error)
        return result


async def create_critic(service, base_model, seed=1234):
    """Use the SDK's existing create method and return an ordinary TrainingClient."""
    holder = service.holder
    model_seq_id = holder.get_training_client_id()

    async def create():
        request = types.CreateModelRequest(
            session_id=holder.get_session_id(),
            model_seq_id=model_seq_id,
            base_model=base_model,
            lora_config=types.LoraConfig(rank=0, seed=seed, train_mlp=True, train_attn=True, train_unembed=True),
        )
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.create(request=request, extra_body={"model_role": "critic"})
        result = await _model_result(holder, future)
        if not isinstance(result, types.CreateModelResponse):
            raise TypeError(f"Unexpected create response: {type(result)!r}")
        return result.model_id

    model_id = await holder.run_coroutine_threadsafe(create())
    return tinker.TrainingClient(holder, model_seq_id=model_seq_id, model_id=model_id)


async def unload_models(service, models):
    async def unload(model):
        with service.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.unload(request=types.UnloadModelRequest(model_id=model.model_id))
        await _model_result(service.holder, future)

    for model in reversed(models):
        try:
            await asyncio.wait_for(service.holder.run_coroutine_threadsafe(unload(model)), timeout=60)
        except Exception:
            logger.exception("Could not release model %s; server lease cleanup remains available", model.model_id)


async def evaluate(client, trajectories, role, chunk_size=32):
    key = "values" if role == "critic" else "logprobs"
    for offset in range(0, len(trajectories), chunk_size):
        chunk = trajectories[offset : offset + chunk_size]
        future = await client.forward_async([datum(t) for t in chunk], "ppo" if role == "critic" else "cross_entropy")
        result = await future.result_async()
        for trajectory, output in zip(chunk, result.loss_fn_outputs, strict=True):
            vector = list(output[key].data)
            expected_length = len(trajectory["prompt"]) + len(trajectory["tokens"]) - 1
            if len(vector) != expected_length or not np.isfinite(vector).all():
                raise ValueError(f"Invalid {key} output: expected {expected_length} finite entries")
            trajectory[key] = vector[-len(trajectory["tokens"]) :]


async def collect_rollout(sampler, tokenizer, records, rollout, args):
    async def sample(row):
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        result = await sampler.sample_async(
            types.ModelInput.from_ints(prompt),
            args.samples_per_prompt,
            types.SamplingParams(
                max_tokens=args.max_response_tokens,
                temperature=args.temperature,
                top_p=1.0,
                top_k=-1,
                stop=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            ),
        )
        if len(result.sequences) != args.samples_per_prompt:
            raise ValueError("Sampler returned an incomplete prompt group")
        trajectories = []
        for sequence in result.sequences:
            if not sequence.tokens or sequence.logprobs is None or len(sequence.logprobs) != len(sequence.tokens):
                raise ValueError("Sampling must return nonempty tokens with matching log probabilities")
            if not np.isfinite(sequence.logprobs).all():
                raise ValueError("Sampler returned nonfinite log probabilities")
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
            reward = await asyncio.to_thread(get_deepscaler_rule_based_reward, text, row["label"])
            trajectories.append(
                dict(
                    prompt=prompt,
                    tokens=list(sequence.tokens),
                    logprobs=list(sequence.logprobs),
                    reward=float(reward),
                    stop_reason=sequence.stop_reason,
                )
            )
        return trajectories

    rows = [records[(rollout * args.prompts_per_rollout + i) % len(records)] for i in range(args.prompts_per_rollout)]
    groups = await asyncio.gather(*(sample(row) for row in rows))
    return [trajectory for group in groups for trajectory in group]


async def update(client, role, trajectories, args, rollout=0, run=None):
    metrics = []
    learning_rate = args.critic_lr if role == "critic" else args.actor_lr
    adam = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
        weight_decay=0.1,
        grad_clip_norm=1.0,
    )
    config = (
        {"value_clip": args.value_clip}
        if role == "critic"
        else {
            "clip_low_threshold": 1 - args.policy_clip,
            "clip_high_threshold": 1 + args.policy_clip,
        }
    )
    for offset in range(0, len(trajectories), args.batch_size):
        batch = trajectories[offset : offset + args.batch_size]
        future = await client.forward_backward_async([datum(t, role, len(batch)) for t in batch], "ppo", config)
        forward = await future.result_async()
        future = await client.optim_step_async(adam)
        result = await future.result_async()
        if result.metrics.get("skipped"):
            raise RuntimeError(f"{role} optimizer skipped; restart from a complete checkpoint pair")
        loss = float(forward.metrics["loss:sum"])
        metrics.append({**result.metrics, "loss": loss})
        if run:
            run.log(
                {
                    "train/step": rollout * (len(trajectories) // args.batch_size) + offset // args.batch_size,
                    "train/critic-value_loss" if role == "critic" else "train/pg_loss": loss,
                    "train/critic-grad_norm" if role == "critic" else "train/grad_norm": result.metrics["grad_norm"],
                    "train/critic-lr-pg_0" if role == "critic" else "train/lr-pg_0": learning_rate,
                }
            )
    return {f"{role}/{key}": float(np.mean([m[key] for m in metrics])) for key in metrics[0]}


def _training_settings(args):
    names = (
        "base_model",
        "dataset_seed",
        "prompts_per_rollout",
        "samples_per_prompt",
        "max_response_tokens",
        "batch_size",
        "gamma",
        "lambd",
        "actor_lr",
        "critic_lr",
        "critic_warmup",
        "policy_clip",
        "value_clip",
        "temperature",
        "reference_dp",
    )
    return {name: getattr(args, name) for name in names}


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


async def save_pair(policy, critic, args, rollout, trajectory, dataset_hash, wandb_id):
    # Short, fixed-shape probe for checking model outputs after a restart.
    probe = {"prompt": trajectory["prompt"], "tokens": trajectory["tokens"][:32]}
    paths = {}
    for role, model in (("critic", critic), ("policy", policy)):
        await evaluate(model, [probe], role)
        future = await model.save_state_async(f"rollout-{rollout + 1:04d}")
        paths[role] = (await future.result_async()).path
    state = dict(
        next_rollout=rollout + 1,
        paths=paths,
        dataset_sha256=dataset_hash,
        settings=_training_settings(args),
        wandb_id=wandb_id,
        probe=probe,
    )
    # Do not advance the manifest if either model's save fails.
    _write_json(args.output / "latest.json", state)
    logger.info("Saved actor/critic checkpoint pair: %s", args.output / "latest.json")


async def restore_pair(policy, critic, state, verify=False):
    for role, model in (("critic", critic), ("policy", policy)):
        future = await model.load_state_with_optimizer_async(state["paths"][role])
        await future.result_async()
        if verify:
            probe = {key: state["probe"][key] for key in ("prompt", "tokens")}
            await evaluate(model, [probe], role)
            key = "values" if role == "critic" else "logprobs"
            np.testing.assert_allclose(probe[key], state["probe"][key], atol=1e-5, rtol=1e-5)


async def train(args, policy, critic, tokenizer, records, dataset_hash, state=None, run=None):
    start = state["next_rollout"] if state else 0
    if state:
        await restore_pair(policy, critic, state, verify=args.verify_resume)
    sampler = await policy.save_weights_and_get_sampling_client_async()
    prefetched = None
    pending = None
    try:
        for rollout in range(start, args.rollouts):
            started = time.monotonic()
            trajectories = prefetched
            prefetched = None
            lag = int(trajectories is not None)
            if trajectories is None:
                trajectories = await collect_rollout(sampler, tokenizer, records, rollout, args)
            if args.async_rollouts and rollout + 1 < args.rollouts:
                pending = asyncio.create_task(collect_rollout(sampler, tokenizer, records, rollout + 1, args))

            # Match the old synchronous client: score with the unchanged actor before
            # any training. Async rollouts retain scores from their generating policy.
            if not args.async_rollouts and rollout >= args.critic_warmup:
                await evaluate(policy, trajectories, "policy", args.batch_size)
            await evaluate(critic, trajectories, "critic", args.batch_size)
            estimates = [gae(t["values"], t["reward"], args.gamma, args.lambd) for t in trajectories]
            advantages = whiten([a for a, _ in estimates])
            for trajectory, advantage, (_, returns) in zip(trajectories, advantages, estimates, strict=True):
                trajectory.update(advantages=advantage.tolist(), returns=returns.tolist())
            trajectories = balance_batches(trajectories, args.batch_size, args.reference_dp)
            metrics = await update(critic, "critic", trajectories, args, rollout, run)
            if rollout >= args.critic_warmup:
                metrics.update(await update(policy, "policy", trajectories, args, rollout, run))

            # This gateway has latest-only sampler weights. Finish old-policy generation
            # before publishing, so one response cannot cross a policy update.
            if pending is not None:
                prefetched = await pending
                pending = None
            sampler = await policy.save_weights_and_get_sampling_client_async()
            metrics.update(
                {
                    "rollout/step": rollout,
                    "rollout/rewards": float(np.mean([t["reward"] for t in trajectories])),
                    "rollout/response_lengths": float(np.mean([len(t["tokens"]) for t in trajectories])),
                    "rollout/truncated_ratio": float(np.mean([t["stop_reason"] == "length" for t in trajectories])),
                    "rollout/policy_lag_iterations": lag,
                    "rollout/seconds": time.monotonic() - started,
                }
            )
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if run:
                run.log(metrics)
            if (rollout + 1) % args.save_interval == 0 or rollout + 1 == args.rollouts:
                await save_pair(policy, critic, args, rollout, trajectories[0], dataset_hash, run.id if run else None)
                # Resume starts at the next prompt batch with fresh sampling; no queue is checkpointed.
                prefetched = None
                if args.stop_after_first_checkpoint:
                    break
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-path", required=True, help="Tokenizer path or Hugging Face identifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=300)
    parser.add_argument("--prompts-per-rollout", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8, help="Must match server --rollout-temperature")
    parser.add_argument(
        "--reference-dp", type=int, default=2, help="Preserve the old run's optimizer minibatch membership"
    )
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--critic-warmup", type=int, default=1, help="Number of critic-only rollouts")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambd", type=float, default=1.0)
    parser.add_argument("--policy-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--resume", type=Path, help="Client latest.json checkpoint-pair manifest")
    parser.add_argument(
        "--verify-resume", action="store_true", help="Compare saved actor/critic probe outputs on resume"
    )
    parser.add_argument("--stop-after-first-checkpoint", action="store_true")
    parser.add_argument(
        "--async-rollouts", action="store_true", help="Overlap training with one lookahead rollout batch"
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args(argv)
    for name in (
        "rollouts",
        "prompts_per_rollout",
        "samples_per_prompt",
        "batch_size",
        "max_response_tokens",
        "save_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.prompts_per_rollout * args.samples_per_prompt % args.batch_size:
        parser.error("rollout sample count must be divisible by --batch-size")
    if args.reference_dp <= 0 or args.batch_size % args.reference_dp:
        parser.error("--batch-size must be divisible by positive --reference-dp")
    if not 0 < args.temperature < float("inf"):
        parser.error("--temperature must be positive and finite")
    if args.critic_warmup < 0 or not 0 <= args.gamma <= 1 or not 0 <= args.lambd <= 1:
        parser.error("critic-warmup must be nonnegative; gamma/lambd must be in [0, 1]")
    if not 0 <= args.policy_clip < 1 or args.value_clip < 0:
        parser.error("policy-clip must be in [0, 1); value-clip must be nonnegative")
    if args.verify_resume and not args.resume:
        parser.error("--verify-resume requires --resume")
    return args


async def _main(args):
    # Defer optional/heavy integrations so --help and numerical tests need no tokenizer or W&B.
    from transformers import AutoTokenizer

    if tinker.__version__ != "0.16.1":
        raise RuntimeError("Install the pinned client requirements: tinker==0.16.1")
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    records = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not records or any("prompt" not in row or "label" not in row for row in records):
        raise ValueError("Expected nonempty JSONL data with prompt and label fields")
    random.Random(args.dataset_seed).shuffle(records)
    state = json.loads(args.resume.read_text()) if args.resume else None
    if state and (state["dataset_sha256"] != dataset_hash or state["settings"] != _training_settings(args)):
        raise ValueError("Resume dataset or PPO settings differ from the saved run")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    _write_json(args.output / "client_config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    run = None
    service = tinker.ServiceClient(base_url=args.base_url)
    models = []
    try:
        if args.wandb_project:
            import wandb

            previous_id = state.get("wandb_id") if state else None
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.output.name,
                dir=str(args.output),
                config=config,
                id=previous_id,
                resume="must" if previous_id else None,
            )
            run.define_metric("rollout/*", step_metric="rollout/step")
            run.define_metric("train/*", step_metric="train/step")
        policy = await service.create_lora_training_client_async(args.base_model, rank=0, seed=1234)
        models.append(policy)
        critic = await create_critic(service, args.base_model)
        models.append(critic)
        await train(args, policy, critic, tokenizer, records, dataset_hash, state, run)
    finally:
        await unload_models(service, models)
        if run:
            run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(_parse_args()))
