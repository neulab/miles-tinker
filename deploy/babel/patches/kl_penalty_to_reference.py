"""Anchor the policy to its starting weights with a reward-level KL penalty.

Nothing in this setup keeps the policy near the model it started from. Tinker's
loss surface is ``cross_entropy, importance_sampling, ppo, cispo, dro`` with a
``Dict[str, float]`` config -- there is no KL coefficient and no entropy bonus,
the loss runs server-side, and the logits never reach the client. So the usual
guard against a policy shedding capabilities it is not currently rewarded for is
simply absent.

Job 10617579 shows what that costs. Over 24 steps ``tool_json`` fell from 98.8%
to 79.7% -- one rollout in five could no longer emit a well-formed tool call --
while entropy rose 0.284 -> 0.436 and the true solve rate fell 32.1% -> 21.2%
(permutation p = 0.0044). Masking and token alignment were audited and are
correct (``importance_weight_mean`` = 1.0000, std 0.024), and ``clip_frac_total``
was 0.000 at every step, so no trust region ever engaged. The policy was free to
drift and did.

This adds the penalty where it can be added from the client: to the reward,
before the group baseline is subtracted. For each trajectory,

    KL_i = mean over trained tokens of (logprob_sampled - logprob_reference)
    reward_i <- reward_i - beta * KL_i

Reference logprobs come from a SamplingClient pinned to the base model, via
``compute_logprobs``. Subtracting before centring is deliberate: GRPO removes
anything constant across a group, so a per-rollout KL only biases the gradient
where rollouts actually differ in how far they have drifted, which is the
behaviour wanted.

The penalty is k1 (``logp_sampled - logp_ref``), the straightforward estimator.
It is unbiased but can go negative for an individual sample; that is correct
here, since a rollout that stayed closer to the reference than average should be
rewarded relative to one that did not.

Cost: one extra forward pass over the batch per step, roughly 13M tokens at the
observed sizes. Expect step time to grow by about a third. Set
PLATOON_KL_BETA=0 to disable without reverting, or tune it there; the default is
deliberately small because the reward scale here is 0.0-1.0 and a KL term that
dominates the task signal would stall learning instead of anchoring it.

Usage: python kl_penalty_to_reference.py <platoon-checkout>
"""

import sys
from pathlib import Path

MARKER = "_kl_to_reference"
TARGET = "platoon/train/tinker/workflows/group_rollout_workflow.py"

# The workflow cannot reach the trainer's ServiceClient, and the gateway URL
# arrives as a CLI flag rather than the TINKER_BASE_URL the SDK falls back to.
# Exporting it where the trainer builds its own client is the smallest bridge.
RL_TARGET = "platoon/train/tinker/rl.py"
RL_MARKER = 'os.environ.setdefault("TINKER_BASE_URL"'
RL_ANCHOR = "        service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)\n"
RL_REPLACEMENT = (
    "        service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)\n"
    "        # So workflows can build their own client for the KL reference policy.\n"
    '        os.environ.setdefault("TINKER_BASE_URL", self.config.tinker_base_url)\n'
)

HELPER_ANCHOR = "class GroupRolloutWorkflow"

HELPER = '''# Measuring and applying are separate on purpose: run with PLATOON_KL_BETA=0 to
# log kl_to_reference without changing any reward, read the real magnitude off a
# few steps, then choose beta from that instead of guessing. Measuring costs the
# same forward pass either way, so PLATOON_KL_MEASURE=0 turns the whole thing off.
_KL_BETA = float(os.environ.get("PLATOON_KL_BETA", "0.0"))
_KL_MEASURE = os.environ.get("PLATOON_KL_MEASURE", "1") != "0"
_kl_reference_client = None
_kl_reference_error = None
_kl_reference_lock = asyncio.Lock()


async def _reference_client(model_info):
    """A SamplingClient pinned to the weights training started from.

    Built once and reused. Tinker has no server-side KL, so the reference policy
    has to be queried explicitly; create_sampling_client(base_model=...) returns
    the untrained model, which is where this run began. The ServiceClient reads
    TINKER_BASE_URL, which the trainer exports when it builds its own.

    A failure is remembered, not retried. Every attempt builds a ServiceClient,
    whose session heartbeats every 10s and is never closed; retrying per rollout
    leaked 4002 of them in job 155424, and their heartbeats saturated the Tinker
    SDK's thread -- which every sample_async also runs on -- until sampling took
    40s+ a call while SGLang sat idle.
    """
    global _kl_reference_client, _kl_reference_error
    async with _kl_reference_lock:
        if _kl_reference_error is not None:
            raise _kl_reference_error
        if _kl_reference_client is None:
            # ModelInfo.model_name carries LiteLLM's provider prefix
            # ("platoon-tinker/Qwen/..."); the gateway knows the bare name.
            base = getattr(model_info.llm, "model_name", None) or model_info.model_name.removeprefix("platoon-tinker/")
            try:
                service = tinker.ServiceClient()
                _kl_reference_client = await service.create_sampling_client_async(base_model=base)
            except Exception as error:
                _kl_reference_error = error
                raise
            logger.info("KL reference pinned to %s (beta=%.4g, %s)", base, _KL_BETA,
                        "penalty ACTIVE" if _KL_BETA else "measuring only, reward unchanged")
    return _kl_reference_client


async def _kl_to_reference(datums, model_info) -> float:
    """Mean (logp_sampled - logp_reference) over the tokens that train.

    Returns 0.0 when the reference cannot be reached, so a transient failure
    costs the penalty for one rollout rather than the rollout itself.
    """
    if not _KL_MEASURE or not datums:
        return 0.0
    try:
        client = await _reference_client(model_info)
    except Exception as error:
        logger.warning("KL reference unavailable (%s); skipping the penalty", error)
        return 0.0

    total, count = 0.0, 0
    for datum in datums:
        mask = datum.loss_fn_inputs["mask"].to_torch()
        sampled = datum.loss_fn_inputs["logprobs"].to_torch()
        try:
            reference = await client.compute_logprobs_async(datum.model_input)
        except Exception as error:
            logger.warning("compute_logprobs failed (%s); skipping the penalty", error)
            return 0.0
        # compute_logprobs returns one entry per prompt token, None where it
        # could not be computed; the datum's tensors are right-shifted by one,
        # so line them up from the end and ignore anything unmatched.
        offset = len(reference) - len(sampled)
        for i in range(len(sampled)):
            if mask[i] <= 0:
                continue
            j = i + offset
            if j < 0 or j >= len(reference) or reference[j] is None:
                continue
            total += float(sampled[i]) - float(reference[j])
            count += 1
    return total / count if count else 0.0


class GroupRolloutWorkflow'''

ANCHOR = """        # Compute group-centered advantages
        mean_task_reward = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
"""

REPLACEMENT = '''        # Penalise drift from the starting weights before the baseline is taken.
        # Subtracting after centring would be a no-op for anything the group
        # shares, which is most of the drift; subtracting before lets the
        # baseline see it.
        if _KL_MEASURE:
            kls = [await _kl_to_reference(result.datums, self.model_info) for result in valid_results]
            if any(kls):
                mean_kl = sum(kls) / len(kls)
                self.tracker.scalar(kl_to_reference=mean_kl)
                if _KL_BETA != 0.0:
                    for i, (result, kl) in enumerate(zip(valid_results, kls)):
                        task_rewards[i] -= _KL_BETA * kl
                        result.task_reward -= _KL_BETA * kl
                    self.tracker.scalar(kl_penalty_applied=_KL_BETA * mean_kl)
                logger.info(
                    "task %s mean KL to reference %.5f (penalty %.5f)",
                    data["task_id"], mean_kl, _KL_BETA * mean_kl,
                )

        # Compute group-centered advantages
        mean_task_reward = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / TARGET
    if not target.is_file():
        raise SystemExit(f"No group_rollout_workflow at {target}")
    source = target.read_text()
    if MARKER in source:
        print(f"KL penalty already present in {target}")
        return
    for name, anchor in (("helper", HELPER_ANCHOR), ("advantage", ANCHOR)):
        if source.count(anchor) != 1:
            raise SystemExit(
                f"Could not find a unique {name} anchor in {target}; "
                "Platoon changed upstream and this patch needs review."
            )
    if "\nimport os\n" not in source:
        raise SystemExit(f"{target} does not import os; this patch needs review.")
    if "\nimport asyncio\n" not in source:
        raise SystemExit(f"{target} does not import asyncio; this patch needs review.")
    source = source.replace(HELPER_ANCHOR, HELPER, 1)
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    target.write_text(source)

    rl = Path(sys.argv[1]) / RL_TARGET
    rl_source = rl.read_text()
    if RL_MARKER not in rl_source:
        if rl_source.count(RL_ANCHOR) != 1:
            raise SystemExit(f"Could not find a unique ServiceClient anchor in {rl}")
        rl.write_text(rl_source.replace(RL_ANCHOR, RL_REPLACEMENT, 1))
        print(f"exported TINKER_BASE_URL in {rl}")
    print(f"added the KL penalty to the reference policy in {target}")


if __name__ == "__main__":
    main()
