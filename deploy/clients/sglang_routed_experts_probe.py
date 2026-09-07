"""Ask the gateway's SGLang engines for routed experts (the R3 replay payload) on one request.

Posts ``/generate`` to the miles router with ``return_routed_experts`` set and
reports whether ``meta_info.routed_experts`` came back and how many int32
entries it decodes to. Engines only fill it when launched with
``enable_return_routed_experts`` (``--use-rollout-routing-replay``, or a
``--sglang-config`` YAML override); an engine without it answers normally and
the field is absent, which this probe reports as exit status 2.

The router address is read from the job's ``head.log`` (``Router ready at
host='..' port=..``) unless ``--router-url`` is given. The request is the
equivalent of::

  curl -s --noproxy '*' -X POST http://<router>/generate -H 'content-type: application/json' \\
    -d '{"input_ids": [...], "sampling_params": {"max_new_tokens": 16, "temperature": 1.0}, "return_routed_experts": true}'

Args:
  --head-log: the gateway job's head.log to parse the router address from.
  --router-url: ``http://host:port`` of the miles router (overrides --head-log).
  --num-prompt-tokens / --max-new-tokens: request shape (default 50 / 16).
  --num-layers / --topk: expected layout (default 40 / 8): (ntok - 1) * layers * topk entries.
  --retries: attempts on 503 (engines not yet router-visible) or connection errors, 10 s apart.

Example:
  /home/aci18912it/miles-tinker/cookbook-venv/bin/python deploy/clients/sglang_routed_experts_probe.py \\
    --head-log $P/jobs/smoke-35b/head.log
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hostguard  # noqa: E402,F401  (refuses login nodes; MILES_ALLOW_LOGIN_NODE=1 overrides)

import argparse  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402

RETRY_PAUSE_S = 10.0
_ROUTER_READY = re.compile(r"Router ready at host='(?P<host>[^']+)' port=(?P<port>\d+)")


def router_url_from_head_log(path: Path) -> str:
    for line in path.read_text(errors="replace").splitlines():
        match = _ROUTER_READY.search(line)
        if match:
            return f"http://{match['host']}:{match['port']}"
    raise SystemExit(f"no 'Router ready at ...' line in {path}")


def build_payload(num_prompt_tokens: int, max_new_tokens: int) -> dict:
    # plain ids well inside the vocabulary; the content does not matter for the probe
    return {
        "input_ids": [1000 + i for i in range(num_prompt_tokens)],
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 1.0},
        "return_routed_experts": True,
    }


def post_with_retries(session, url: str, payload: dict, timeout: float, retries: int):
    """The router answers 503 no_available_workers until the first weight push marks the engines serving
    (the gateway's /healthz goes green a few seconds earlier), so retry that and connection errors."""
    response = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            print(f"attempt {attempt}/{retries}: {type(e).__name__}: {e}", flush=True)
        else:
            if response.status_code != 503:
                return response
            print(f"attempt {attempt}/{retries}: HTTP 503 {response.text[:200]}", flush=True)
        time.sleep(RETRY_PAUSE_S)
    return response


def decode_routed_experts(encoded: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoded.encode("ascii")), dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--head-log")
    parser.add_argument("--router-url")
    parser.add_argument("--num-prompt-tokens", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=18, help="attempts on 503 / connection errors, 10 s apart")
    args = parser.parse_args()
    if not args.router_url and not args.head_log:
        raise SystemExit("pass --router-url or --head-log")
    router_url = args.router_url or router_url_from_head_log(Path(args.head_log))

    payload = build_payload(args.num_prompt_tokens, args.max_new_tokens)
    print(f"routed-experts probe: POST {router_url}/generate {json.dumps(payload)}", flush=True)
    session = requests.Session()
    session.trust_env = False  # the router sits on the IPoIB network; never go through a proxy
    response = post_with_retries(session, f"{router_url}/generate", payload, args.timeout, args.retries)
    if response is None or response.status_code != 200:
        detail = "no response" if response is None else f"HTTP {response.status_code}: {response.text[:500]}"
        print(f"FAIL /generate -> {detail}")
        return 1
    output = response.json()
    meta = output.get("meta_info", {})
    completion_tokens = meta.get("completion_tokens")
    print(f"HTTP 200 finish_reason={meta.get('finish_reason')} completion_tokens={completion_tokens} meta_info keys={sorted(meta)}")

    encoded = meta.get("routed_experts")
    if not encoded:
        print("ABSENT: meta_info.routed_experts missing (engine launched without enable_return_routed_experts?)")
        return 2
    routed = decode_routed_experts(encoded)
    ntok = args.num_prompt_tokens + int(completion_tokens or 0)
    expected = (ntok - 1) * args.num_layers * args.topk
    layout = f"(ntok-1={ntok - 1}) x layers({args.num_layers}) x topk({args.topk}) = {expected}"
    ok = routed.size == expected
    print(
        f"{'PASS' if ok else 'FAIL'}: routed_experts present, {routed.size} int32 entries "
        f"({len(encoded)} base64 chars), expected {layout}; expert id range [{routed.min()}, {routed.max()}]"
    )
    if ok:
        per_layer = routed.reshape(ntok - 1, args.num_layers, args.topk)
        print(f"  first token, first 2 layers: {per_layer[0, :2].tolist()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
