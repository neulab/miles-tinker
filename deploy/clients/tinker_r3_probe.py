"""R3 (rollout routing replay) probe against a live gateway with the platoon-side Tinker SDK (0.16.1).

Exercises the C1/C2/C6 contracts end to end:
  1. ``SampleResponse``/``SampledSequence`` are re-bound with ``extra="allow"`` (as platoon's proxy does)
     so the JSON ``routing_id`` the gateway adds to every sequence survives pydantic parsing.
  2. Three prefix-extending chat turns are sampled (turn n's prompt = turn n-1's prompt + its sampled
     tokens + a new user turn), ``num_samples=2`` on the first turn: every sequence must carry a
     positive ``routing_id`` < 2**53, unique across all sequences.
  3. Two cispo datums are trained: datum A = the third turn's prompt + generated tokens with
     ``loss_fn_inputs["routing_handle"]`` = int64 ``[1]`` holding that call's ``routing_id``; datum B =
     a synthetic prompt/target that was never sampled and carries no handle. Under the gateway's
     ``natural`` policy both train and ``optim_step`` metrics must report
     ``routing_covered_datums == 1`` and ``routing_uncovered_datums == 1``. Under ``strict`` datum A
     is stepped alone first (covered 1 / uncovered 0), then datum B's ``forward_backward`` must fail
     with a user error AND poison its gradient window: the following ``optim_step`` fails typed with
     "gradients discarded" (miles/tinker/core/service.py ``_discard_dirty``).
     The optim metrics must also carry ``routing_prefix_hits >= 2`` (turns 2-3 reused a cached parent).
Prints one PASS/FAIL line per check, the optim metrics, and exits non-zero on any violation.

Args:
  --endpoint-file: file holding ``http://host:port`` (or --base-url / $TINKER_BASE_URL).
  --base-model: model name the gateway advertises.
  --tokenizer: HF snapshot path for the tokenizer (default: the training client's tokenizer).
  --policy: natural|strict, the gateway's --tinker-routing-replay-policy (default natural).
  --max-tokens: generated tokens per turn (default 48).
  --keep-model: leave the model loaded at exit (default: unload so the single full-FT slot is freed).

Example:
  /home/aci18912it/source/platoon-tinker/plugins/openreward/.venv/bin/python deploy/clients/tinker_r3_probe.py \\
    --endpoint-file $ENDPOINT_FILE --base-model apurvaga/Qwen3.6-35B-A3B-preserve-thinking
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostguard  # noqa: E402,F401  (refuses login nodes; MILES_ALLOW_LOGIN_NODE=1 overrides)

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pydantic import ConfigDict  # noqa: E402

os.environ.setdefault("TINKER_API_KEY", "tml-local")
os.environ.setdefault("TINKER_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", "/groups/gai51740/aci18912it/rasp/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import tinker  # noqa: E402  (env defaults above must precede the import)
from tinker import types  # noqa: E402
import tinker.types.sample_response as sample_response_module  # noqa: E402
import tinker.types.sampled_sequence as sampled_sequence_module  # noqa: E402

ROUTING_ID_MAX = 2**53
ROUTING_METRIC_KEYS = (
    "routing_covered_datums",
    "routing_uncovered_datums",
    "routing_uncovered_rows",
    "routing_capture_rejected",
    "routing_materialize_s",
    "routing_cache_entries",
    "routing_cache_bytes",
    "routing_cache_pinned",
    "routing_cache_evictions",
    "routing_prefix_hits",
)
TURNS = [
    "List three prime numbers below twenty and explain briefly why each is prime.",
    "Now add the next prime to your list.",
    "Which of the four is the largest? Answer with one number.",
]
SYNTHETIC_PROMPT = "The quick brown fox"
SYNTHETIC_TARGET = " jumps over the lazy dog and keeps running through the field."


def patch_sample_response_extra_allow() -> None:
    """Keep unknown keys (``routing_id``) on the 0.16.1 sample models, as platoon's proxy does (design 8.1)."""
    assert not os.environ.get("TINKER_SUBPROCESS_SAMPLING"), "TINKER_SUBPROCESS_SAMPLING would bypass the patched models"

    class _SampledSequence(types.SampledSequence):
        model_config = ConfigDict(frozen=True, extra="allow")

    class _SampleResponse(types.SampleResponse):
        model_config = ConfigDict(frozen=True, extra="allow")
        sequences: list[_SampledSequence]  # type: ignore[assignment]

    for module in (types, sample_response_module):
        module.SampleResponse = _SampleResponse  # type: ignore[attr-defined]
    for module in (types, sampled_sequence_module):
        module.SampledSequence = _SampledSequence  # type: ignore[attr-defined]


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        self.failed += 0 if ok else 1
        return ok


def resolve_base_url(args: argparse.Namespace) -> str:
    if args.endpoint_file:
        return Path(args.endpoint_file).read_text().strip()
    url = args.base_url or os.environ.get("TINKER_BASE_URL")
    if not url:
        raise SystemExit("pass --endpoint-file or --base-url (or set TINKER_BASE_URL)")
    return url


def unload_model(base_url: str, model_id: str) -> str:
    """POST /api/v1/unload_model with this client's API key (a gateway route without an SDK method).

    A 400 naming an unknown model means the gateway already released it (the lease sweep
    races this call when a failed unit kills the trainer), which is the outcome we want.
    """
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/v1/unload_model",
        json={"model_id": model_id},
        headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
        timeout=120,
    )
    if response.status_code == 400 and "unknown model" in response.text:
        return "already released by the gateway"
    response.raise_for_status()
    return "unloaded"


def render_turn(tokenizer, history: list[dict], user_text: str) -> list[int]:
    """Token ids of the chat prompt for ``history + [user]`` with the generation prompt appended."""
    messages = history + [{"role": "user", "content": user_text}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return [int(t) for t in tokenizer.encode(rendered, add_special_tokens=False)]


def routing_id_of(sequence) -> int | None:
    value = getattr(sequence, "routing_id", None)
    if value is None:
        extra = getattr(sequence, "model_extra", None) or {}
        value = extra.get("routing_id")
    return value


def sample_turns(checks: Checks, sampling_client, tokenizer, stop_ids: list[int], max_tokens: int) -> list[dict]:
    """Three prefix-extending turns; returns one record per turn with prompt, tokens, logprobs and routing_id."""
    records = []
    seen_ids: set[int] = set()
    prompt = render_turn(tokenizer, [], TURNS[0])
    for turn, user_text in enumerate(TURNS):
        if turn > 0:
            previous = records[-1]["prompt"] + records[-1]["tokens"]
            prompt = extend_prompt(tokenizer, records[-1]["prompt"], records[-1]["tokens"], user_text, stop_ids)
            checks.check(f"turn {turn + 1} prompt extends turn {turn}", prompt[: len(previous)] == previous, f"len {len(previous)} -> {len(prompt)}")
        num_samples = 2 if turn == 0 else 1
        params = types.SamplingParams(max_tokens=max_tokens, temperature=1.0, stop=stop_ids)
        t0 = time.time()
        response = sampling_client.sample(types.ModelInput.from_ints(prompt), num_samples=num_samples, sampling_params=params).result()
        checks.check(f"turn {turn + 1} sample", len(response.sequences) == num_samples, f"{len(response.sequences)} sequences in {time.time() - t0:.1f}s")
        for i, seq in enumerate(response.sequences):
            rid = routing_id_of(seq)
            ok = isinstance(rid, int) and not isinstance(rid, bool) and 0 < rid < ROUTING_ID_MAX and rid not in seen_ids
            checks.check(f"turn {turn + 1} sample {i} routing_id", ok, f"routing_id={rid!r} tokens={len(seq.tokens)} stop={seq.stop_reason}")
            if isinstance(rid, int):
                seen_ids.add(rid)
        seq = response.sequences[0]
        records.append({"prompt": prompt, "tokens": list(seq.tokens), "logprobs": list(seq.logprobs or []), "routing_id": routing_id_of(seq)})
    return records


def extend_prompt(tokenizer, prompt: list[int], generated: list[int], user_text: str, stop_ids: list[int]) -> list[int]:
    """prompt + generated (the assistant turn, stop token included) + the next user turn + the generation prompt.

    The suffix is what the chat template appends for one more user turn after a finished assistant turn, taken as
    the difference of two renders so thinking prefixes and the generation prompt follow the template exactly.
    """
    base = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    stub = tokenizer.apply_chat_template(base, tokenize=False)
    full = tokenizer.apply_chat_template(base + [{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=False)
    suffix = full[len(stub) :] if full.startswith(stub) else full[full.rfind("<|im_start|>user") :]
    closed = generated if generated and generated[-1] in stop_ids else generated + stop_ids[:1]
    # the template writes "<|im_end|>\n" after an assistant turn; the sampled turn ends at the stop token itself
    between = stub[stub.rfind("<|im_end|>") + len("<|im_end|>") :]
    return prompt + closed + [int(t) for t in tokenizer.encode(between + suffix, add_special_tokens=False)]


def make_datum(tokens: list[int], prompt_len: int, logprobs: list[float] | None, routing_id: int | None) -> types.Datum:
    """cispo datum over ``tokens``: prompt positions weighted 0, generated positions 1, advantage 1."""
    n_target = len(tokens) - 1
    weights = np.zeros(n_target, dtype=np.float32)
    weights[prompt_len - 1 :] = 1.0
    if logprobs is not None and len(logprobs) == len(tokens) - prompt_len:
        sampled_logprobs = np.concatenate([np.zeros(prompt_len - 1, dtype=np.float32), np.asarray(logprobs, dtype=np.float32)])
    else:
        sampled_logprobs = np.zeros(n_target, dtype=np.float32)
    inputs = {
        "target_tokens": np.asarray(tokens[1:], dtype=np.int64),
        "weights": weights,
        "advantages": weights.copy(),
        "logprobs": sampled_logprobs,
    }
    if routing_id is not None:
        inputs["routing_handle"] = types.TensorData.from_numpy(np.asarray([routing_id], dtype=np.int64))
    return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs)


def train_datums(checks: Checks, client, records: list[dict], tokenizer, policy: str) -> dict:
    last = records[-1]
    tokens_a = last["prompt"] + last["tokens"]
    datum_a = make_datum(tokens_a, len(last["prompt"]), last["logprobs"], last["routing_id"])
    handle = datum_a.loss_fn_inputs["routing_handle"]
    checks.check("datum A routing_handle int64 [1]", handle.dtype == "int64" and list(handle.shape or []) == [1] and handle.data == [last["routing_id"]], f"{handle}")
    prompt_b = tokenizer.encode(SYNTHETIC_PROMPT)
    tokens_b = prompt_b + tokenizer.encode(SYNTHETIC_TARGET)
    datum_b = make_datum(tokens_b, len(prompt_b), None, None)
    checks.check("datum B has no routing_handle", "routing_handle" not in datum_b.loss_fn_inputs)

    fb_a = client.forward_backward([datum_a], loss_fn="cispo").result()
    checks.check("datum A forward_backward (handle)", len(fb_a.loss_fn_outputs) == 1, f"metrics {sorted(fb_a.metrics or {})}")
    b_error = None
    if policy == "strict":
        # a strict failure poisons the whole gradient window (service._discard_dirty), so datum A must be
        # stepped on its own before datum B is submitted
        metrics = _optim_step_metrics(client)
        _check_window_metrics(checks, metrics, policy, expected_uncovered=0)
        try:
            client.forward_backward([datum_b], loss_fn="cispo").result()
            checks.check("datum B forward_backward fails typed under strict", False, "it trained")
        except Exception as exc:  # noqa: BLE001  (the SDK surfaces the gateway's user error as its own exception type)
            b_error = exc
            checks.check("datum B forward_backward fails typed under strict", "routing" in str(exc).lower(), f"error={type(exc).__name__}: {str(exc)[:200]}")
        try:
            client.optim_step(types.AdamParams(learning_rate=1e-6)).result()
            checks.check("optim_step after the strict failure fails typed (window poisoned)", False, "it stepped")
        except Exception as exc:  # noqa: BLE001
            checks.check("optim_step after the strict failure fails typed (window poisoned)", "gradients discarded" in str(exc), f"error={type(exc).__name__}: {str(exc)[:200]}")
        return {"metrics": metrics, "b_error": b_error}
    try:
        fb_b = client.forward_backward([datum_b], loss_fn="cispo").result()
        checks.check("datum B forward_backward (no handle)", len(fb_b.loss_fn_outputs) == 1, f"policy={policy} metrics {sorted(fb_b.metrics or {})}")
    except Exception as exc:  # noqa: BLE001
        b_error = exc
        checks.check("datum B forward_backward (no handle)", False, f"policy={policy} error={type(exc).__name__}: {str(exc)[:200]}")
    metrics = _optim_step_metrics(client)
    _check_window_metrics(checks, metrics, policy, expected_uncovered=1)
    return {"metrics": metrics, "b_error": b_error}


def _optim_step_metrics(client) -> dict:
    optim = client.optim_step(types.AdamParams(learning_rate=1e-6)).result()
    metrics = dict(optim.metrics or {})
    print("optim metrics:", {k: metrics[k] for k in sorted(metrics)}, flush=True)
    return metrics


def _check_window_metrics(checks: Checks, metrics: dict, policy: str, expected_uncovered: int) -> None:
    missing = [k for k in ROUTING_METRIC_KEYS if k not in metrics]
    checks.check("optim metrics carry the C6 routing keys", not missing, f"missing {missing}" if missing else "all present")
    covered, uncovered = metrics.get("routing_covered_datums"), metrics.get("routing_uncovered_datums")
    checks.check("routing_covered_datums == 1", covered == 1, f"routing_covered_datums={covered}")
    checks.check(f"routing_uncovered_datums == {expected_uncovered} ({policy})", uncovered == expected_uncovered, f"routing_uncovered_datums={uncovered}")
    checks.check("routing_capture_rejected == 0", metrics.get("routing_capture_rejected", 0) == 0, f"{metrics.get('routing_capture_rejected')}")
    # turns 2 and 3 extend the previous turn's sequence: each must have reused a cached parent (P > 0);
    # the counter is process-wide, so earlier clients only add to it
    hits = metrics.get("routing_prefix_hits")
    checks.check("routing_prefix_hits >= 2 (turns 2-3 reused a cached parent)", isinstance(hits, (int, float)) and hits >= 2, f"routing_prefix_hits={hits}")
    checks.check("grad_norm finite", np.isfinite(metrics.get("grad_norm", float("nan"))), f"grad_norm={metrics.get('grad_norm')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint-file")
    parser.add_argument("--base-url")
    parser.add_argument("--base-model", default="apurvaga/Qwen3.6-35B-A3B-preserve-thinking")
    parser.add_argument("--tokenizer", help="HF snapshot path for the tokenizer (default: the training client's)")
    parser.add_argument("--policy", choices=["natural", "strict"], default="natural")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--keep-model", action="store_true")
    args = parser.parse_args()
    patch_sample_response_extra_allow()
    checks = Checks()
    base_url = resolve_base_url(args)
    print(f"r3 probe against {base_url} model={args.base_model} policy={args.policy} sdk={tinker.__version__}", flush=True)
    service = tinker.ServiceClient(base_url=base_url)
    client = service.create_lora_training_client(base_model=args.base_model, rank=8)
    info = client.get_info()
    model_id = info.model_id
    checks.check("create training client", bool(model_id), f"model_id={model_id}")
    try:
        if args.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        else:
            tokenizer = client.get_tokenizer()
        stop_ids = [t for t in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id) if t is not None]
        sampling_client = client.save_weights_and_get_sampling_client("r3-probe")
        checks.check("save_weights_and_get_sampling_client", sampling_client is not None)
        records = sample_turns(checks, sampling_client, tokenizer, stop_ids, args.max_tokens)
        if all(isinstance(r["routing_id"], int) for r in records):
            train_datums(checks, client, records, tokenizer, args.policy)
        else:
            checks.check("training skipped: no routing_id on a turn", False)
    finally:
        # the single full-FT slot is this client's to give back: a leaked model holds it
        # for the whole lease and every later client gets "serves one model at a time"
        if not args.keep_model:
            for attempt in range(3):
                try:
                    print(f"{model_id}: {unload_model(base_url, model_id)}", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"WARN unload_model attempt {attempt + 1} failed: {exc}", flush=True)
                    time.sleep(2)
    print(f"{'PASS' if checks.failed == 0 else 'FAIL'} r3 probe: {checks.failed} failed check(s)", flush=True)
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
