"""Reproduce the weight-sync keep-alive race on CPU, with miles' own engine client.

The trainer posts every weight-sync bucket to every SGLang engine through
SGLangApiClient (httpx via GeneralHttpClientProvider). SGLang serves HTTP with uvicorn
(uvloop, h11) and closes a keep-alive connection that stays idle for
SGLANG_TIMEOUT_KEEP_ALIVE seconds (5 by default); httpx drops an idle pooled connection
after its keepalive_expiry (5 s by default). A request written on a connection the server
is closing at that moment is answered with a TCP reset: httpx.ReadError.

The servers here are SGLang-like: FastAPI on uvicorn.run(..., loop="uvloop",
timeout_keep_alive=<keep-alive>), exactly the call in sglang.srt.entrypoints.http_server.
The client is whatever `miles` PYTHONPATH resolves to, so the same script measures the
code before and after a fix.

Modes:
  scan       one client process per engine (pinned to its own core). After each response
             the next request goes out GAP - offset seconds later, the offset drawn
             uniformly from [OFFSET_MIN, OFFSET_MAX] ms, so the send time sweeps across
             the moment the server's idle timer fires.
  broadcast  one client process fans every bucket out to all engines through
             miles.utils.async_utils, as protocols/broadcast.py does, then waits GAP - offset
             before the next bucket; --burners N busy loops share the client's core, like the
             CPU-saturated head node of the incident.

Output: one JSON line per request in --out, then a summary on stdout.

Example, on a CPU node with the miles checkout under test on PYTHONPATH:
  python tools/fork/keepalive_race_repro.py --mode broadcast --engines 8 --rounds 60 --burners 4 --out race.jsonl

miles is imported inside the client functions only: the servers are spawned processes that re-import this module, and
they stand in for SGLang, so they must not load miles.
"""

import argparse
import asyncio
import collections
import json
import logging
import multiprocessing as mp
import os
import random
import socket
import sys
import time

ENDPOINTS = (
    "/update_weights_from_distributed",
    "/pause_generation",
    "/continue_generation",
    "/update_weight_version",
    "/begin_weight_update",
    "/end_weight_update",
)


def _pin(cpu: int | None) -> None:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})


def run_server(port: int, keepalive: int, cpu: int | None) -> None:
    _pin(cpu)
    import uvicorn
    from fastapi import FastAPI, Request

    app = FastAPI()

    async def handle(request: Request):
        await request.body()
        return {
            "success": True,
            "message": "Success",
            "client_port": request.client.port,
            "server_keepalive": keepalive,
        }

    for path in ENDPOINTS:
        app.add_api_route(path, handle, methods=["POST"])
    # sglang.srt.entrypoints.http_server: uvicorn.run(app, ..., timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
    # loop="uvloop"); h11 (no httptools in the image)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", timeout_keep_alive=keepalive, loop="uvloop")


def run_burner(cpu: int | None) -> None:
    _pin(cpu)
    x = 0
    while True:
        x += 1


def _free_port_range(count: int) -> int:
    """A base port whose next `count` ports nobody listens on: concurrent runs on one node must not share servers."""
    rng = random.SystemRandom()
    for _ in range(200):
        base = rng.randrange(20000, 60000 - count)
        try:
            for port in range(base, base + count):
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
    raise RuntimeError(f"no {count} free consecutive ports")


def _wait_port(port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server on {port} not up")


def _client_info() -> dict:
    import httpcore
    import httpx

    from miles.utils import http_utils

    limits = http_utils.GeneralHttpClientProvider._LIMITS
    return {
        "miles": os.path.dirname(os.path.dirname(os.path.abspath(http_utils.__file__))),
        "keepalive_expiry": limits.keepalive_expiry,
        "httpx": httpx.__version__,
        "httpcore": httpcore.__version__,
    }


def _payload_kwargs() -> dict:
    return dict(names=["w"], dtypes=["bfloat16"], shapes=[[1]], group_name="miles-pp_0")


def _make_call(api, call: str):
    """The client call each request makes: the weight-sync bucket post (never retried) or an idempotent control call."""
    if call == "update_weights_from_distributed":
        return lambda: api.update_weights_from_distributed(**_payload_kwargs())
    if call == "pause_generation":
        return lambda: api.pause_generation(mode="retract")
    raise ValueError(call)


def _body_of(result) -> dict:
    # update_weights_from_distributed answers the JSON body; pause_generation the httpx.Response
    return result if isinstance(result, dict) else result.json()


class _RetryCounter(logging.Handler):
    """Counts the client's own "retrying once" warnings (miles' idempotent-call retry after a connection reset)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "retrying once" in record.getMessage():
            self.count += 1


def _install_retry_counter() -> _RetryCounter:
    counter = _RetryCounter()
    logging.getLogger("miles.backends.sglang_utils.sglang_api_client").addHandler(counter)
    return counter


def _outcome(error: BaseException | None) -> str:
    return "ok" if error is None else type(error).__name__


def run_scan_client(engine: int, url: str, args, cpu: int | None, out_path: str) -> None:
    _pin(cpu)
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient

    rng = random.Random(args.seed * 1000 + engine)
    retries = _install_retry_counter()

    async def main() -> None:
        call = _make_call(SGLangApiClient(server_url=url), args.call)
        last_port = _body_of(await call())["client_port"]
        t_done = time.monotonic()
        with open(out_path, "a") as out:
            for round_index in range(args.rounds):
                offset_ms = rng.uniform(args.offset_min_ms, args.offset_max_ms)
                target = t_done + args.gap - offset_ms / 1000.0
                await asyncio.sleep(max(0.0, target - time.monotonic() - 0.003))
                while time.monotonic() < target:
                    pass
                t_send = time.monotonic()
                error, body, retries_before = None, None, retries.count
                try:
                    body = _body_of(await call())
                except Exception as exc:  # noqa: BLE001  every failure is a data point
                    error = exc
                port = None if body is None else body["client_port"]
                t_end = time.monotonic()
                record = {
                    "mode": "scan",
                    "call": args.call,
                    "retries": retries.count - retries_before,
                    "engine": engine,
                    "round": round_index,
                    "offset_ms": round(offset_ms, 4),
                    "idle_s": round(t_send - t_done, 6),
                    "outcome": _outcome(error),
                    "reused": None if port is None else port == last_port,
                    "server_keepalive": None if body is None else body["server_keepalive"],
                    "latency_ms": round((t_end - t_send) * 1000, 3),
                    "error": None if error is None else repr(error)[:200],
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                last_port = port
                t_done = t_end

    asyncio.run(main())


def run_broadcast_client(urls: list[str], args, cpu: int | None, out_path: str) -> None:
    _pin(cpu)
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
    from miles.utils import async_utils

    rng = random.Random(args.seed)
    retries = _install_retry_counter()
    calls = [_make_call(SGLangApiClient(server_url=url), args.call) for url in urls]
    last_ports: list[int | None] = [None] * len(calls)

    def send_bucket() -> list:
        return [async_utils.submit(call()) for call in calls]

    for future, index in zip(send_bucket(), range(len(calls)), strict=True):
        last_ports[index] = _body_of(future.result())["client_port"]
    t_done = time.monotonic()
    with open(out_path, "a") as out:
        for round_index in range(args.rounds):
            offset_ms = rng.uniform(args.offset_min_ms, args.offset_max_ms)
            target = t_done + args.gap - offset_ms / 1000.0
            time.sleep(max(0.0, target - time.monotonic() - 0.003))
            while time.monotonic() < target:
                pass
            t_send = time.monotonic()
            retries_before = retries.count
            futures = send_bucket()
            first_error = None
            try:
                async_utils.wait_futures(futures)  # the trainer's own fan-in: raises the first error
            except Exception as exc:  # noqa: BLE001
                first_error = exc
            t_end = time.monotonic()
            for index, future in enumerate(futures):
                error = future.exception()
                body = None if error is not None else _body_of(future.result())
                port = None if body is None else body["client_port"]
                record = {
                    "mode": "broadcast",
                    "call": args.call,
                    "retries": (retries.count - retries_before) if index == 0 else 0,  # per bucket, on its first row
                    "engine": index,
                    "round": round_index,
                    "offset_ms": round(offset_ms, 4),
                    "idle_s": round(t_send - t_done, 6),
                    "outcome": _outcome(error),
                    "reused": None if port is None else port == last_ports[index],
                    "server_keepalive": None if body is None else body["server_keepalive"],
                    "latency_ms": round((t_end - t_send) * 1000, 3),
                    "error": None if error is None else repr(error)[:200],
                    "bucket_failed": first_error is not None,
                }
                out.write(json.dumps(record) + "\n")
                last_ports[index] = port
            out.flush()
            t_done = t_end


def summarize(out_path: str, info: dict, args) -> dict:
    records = [json.loads(line) for line in open(out_path)]
    outcomes = collections.Counter(r["outcome"] for r in records)
    reused = collections.Counter(("reused" if r["reused"] else "new") for r in records if r["outcome"] == "ok")
    errors = [r for r in records if r["outcome"] != "ok"]
    failed_buckets = len({r["round"] for r in records if r.get("bucket_failed")})
    summary = {
        "client": info,
        "mode": args.mode,
        "server_keepalive_s": args.server_keepalive,
        "engines": args.engines,
        "rounds": args.rounds,
        "gap_s": args.gap,
        "offset_ms": [args.offset_min_ms, args.offset_max_ms],
        "burners": args.burners,
        "requests": len(records),
        "outcomes": dict(outcomes),
        "ok_connections": dict(reused),
        "server_keepalive_seen": sorted({r["server_keepalive"] for r in records if r["server_keepalive"] is not None}),
        "call": args.call,
        "client_retries": sum(r.get("retries", 0) for r in records),
        "error_idle_s": [
            min((r["idle_s"] for r in errors), default=None),
            max((r["idle_s"] for r in errors), default=None),
        ],
        "error_offsets_ms": sorted(r["offset_ms"] for r in errors)[:50],
        "error_examples": [r["error"] for r in errors[:3]],
    }
    if args.mode == "broadcast":
        summary["failed_buckets"] = failed_buckets
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("scan", "broadcast"), default="scan")
    parser.add_argument("--engines", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--gap", type=float, default=5.0, help="seconds from a response to the next request")
    parser.add_argument("--offset-min-ms", type=float, default=0.0, help="the next request goes out GAP - offset")
    parser.add_argument("--offset-max-ms", type=float, default=1.5)
    parser.add_argument(
        "--server-keepalive",
        type=int,
        default=int(os.environ.get("SGLANG_TIMEOUT_KEEP_ALIVE", "5")),
        help="uvicorn timeout_keep_alive of the engines (default: $SGLANG_TIMEOUT_KEEP_ALIVE, else SGLang's 5)",
    )
    parser.add_argument("--burners", type=int, default=0, help="busy loops on the broadcast client's core")
    parser.add_argument(
        "--call",
        choices=("update_weights_from_distributed", "pause_generation"),
        default="update_weights_from_distributed",
        help="the SGLangApiClient call each request makes",
    )
    parser.add_argument("--base-port", type=int, default=0, help="first server port (default: a free range)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cpus = sorted(os.sched_getaffinity(0))
    spawn = mp.get_context("spawn")
    info = _client_info()
    print(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "cpus": len(cpus), **info}), flush=True)

    servers = []
    base_port = args.base_port or _free_port_range(args.engines)
    ports = [base_port + i for i in range(args.engines)]
    for i, port in enumerate(ports):
        proc = spawn.Process(target=run_server, args=(port, args.server_keepalive, cpus[i % len(cpus)]), daemon=True)
        proc.start()
        servers.append(proc)
    for port in ports:
        _wait_port(port)
    time.sleep(1.0)
    dead = [port for port, proc in zip(ports, servers, strict=True) if not proc.is_alive()]
    if dead:  # a server that could not bind leaves its port to someone else's server
        for proc in servers:
            proc.terminate()
        raise RuntimeError(f"servers on ports {dead} exited at startup")
    urls = [f"http://127.0.0.1:{port}" for port in ports]
    open(args.out, "w").close()

    helpers = []
    try:
        if args.mode == "scan":
            clients = [
                spawn.Process(
                    target=run_scan_client,
                    args=(i, url, args, cpus[(args.engines + i) % len(cpus)], args.out),
                )
                for i, url in enumerate(urls)
            ]
            for proc in clients:
                proc.start()
            for proc in clients:
                proc.join()
        else:
            client_cpu = cpus[args.engines % len(cpus)]
            for _ in range(args.burners):
                proc = spawn.Process(target=run_burner, args=(client_cpu,), daemon=True)
                proc.start()
                helpers.append(proc)
            client = spawn.Process(target=run_broadcast_client, args=(urls, args, client_cpu, args.out))
            client.start()
            client.join()
    finally:
        for proc in servers + helpers:
            proc.terminate()
        for proc in servers + helpers:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()

    print(json.dumps(summarize(args.out, info, args)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
