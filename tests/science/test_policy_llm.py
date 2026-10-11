"""No GPU/model call: SDK wire history must preserve the sampled assistant turn."""

from examples.science_rl.policy_llm import SciencePolicyLLM
from openhands.sdk import Message, TextContent
from openhands.sdk.llm.message import MessageToolCall
from pydantic import SecretStr
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from paper2rlenv.runtime import PolicyTokenBudgetReached, ProxyLLM
from paper2rlenv.runtime import ContainerExecuteTool, _process, _sdk_run
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import threading


@pytest.mark.parametrize(
    "finish_reason,total_tokens,expected",
    [
        ("tool_calls", 100, None),
        ("length", 4096, "model_call_length"),
        ("tool_calls", 32768, "episode_context_limit"),
        ("stop", 40000, "episode_context_limit"),
    ],
)
def test_policy_budget_blocks_over_budget_response_before_sdk_execution(finish_reason, total_tokens, expected):
    llm = SciencePolicyLLM(model="openai/test", api_key=SecretStr("local-test"), episode_token_limit=32768)
    response = SimpleNamespace(
        raw_response=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason)],
            usage=SimpleNamespace(total_tokens=total_tokens),
        )
    )
    with patch.object(ProxyLLM, "generate", return_value=response):
        if expected is None:
            assert llm.generate([]) is response
        else:
            with pytest.raises(PolicyTokenBudgetReached, match=expected):
                llm.generate([])


def test_missing_policy_usage_remains_an_error():
    llm = SciencePolicyLLM(model="openai/test", api_key=SecretStr("local-test"), episode_token_limit=32768)
    response = SimpleNamespace(raw_response=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")], usage=None))
    with patch.object(ProxyLLM, "generate", return_value=response):
        with pytest.raises(ValueError, match="requires policy token usage"):
            llm.generate([])


@pytest.mark.parametrize("finish_reason,total_tokens", [("length", 4097), ("tool_calls", 32768)])
def test_real_sdk_policy_budget_retains_only_executed_artifact(tmp_path, monkeypatch, finish_reason, total_tokens):
    artifact = tmp_path / "answer.txt"
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            number = len(requests)
            value = 42 if number == 1 else 0
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": str(number),
                        "type": "function",
                        "function": {
                            "name": ContainerExecuteTool.name,
                            "arguments": json.dumps({"command": f"from pathlib import Path; Path({str(artifact)!r}).write_text('{value}')"}),
                        },
                    }
                ],
            }
            total = 2 if number == 1 else total_tokens
            payload = json.dumps(
                {
                    "id": f"call-{number}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if number == 1 else finish_reason}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": total - 1, "total_tokens": total},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    class Sandbox:
        name = "science-policy-budget-test"
        failure = None

        def exec(self, code):
            return _process([sys.executable, "-c", code])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LLM_API_KEY", "local-test")
    monkeypatch.setattr("paper2rlenv.runtime.ProxyLLM", partial(SciencePolicyLLM, episode_token_limit=32768))
    events = tmp_path / "events.jsonl"
    try:
        _sdk_run("Compute", "openai/test", f"http://127.0.0.1:{server.server_port}/v1", events, 30, sandbox=Sandbox(), allow_token_budget=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(requests) == 2
    assert artifact.read_text() == "42"
    assert '"execution_status": "token_budget"' in events.read_text()


def test_qwen_reasoning_and_tool_history_survive_sdk_serialization():
    llm = SciencePolicyLLM(
        model="openai/Qwen3.5-4B",
        api_key=SecretStr("local-test"),
        base_url="http://127.0.0.1:1/v1",
        capability_overrides={"supports_vision": True, "supports_prompt_cache_key": False},
    )
    reasoning = "Derive the regression, then execute it.\n"
    call = MessageToolCall(id="call_test", name="container_execute", arguments='{"command":"ls /work"}', origin="completion")
    messages = [
        Message(role="user", content=[TextContent(text="Read the data.")]),
        Message(role="assistant", content=[TextContent(text="\n\n")], reasoning_content=reasoning, tool_calls=[call]),
        Message(role="tool", content=[TextContent(text="table.json\n")], tool_call_id="call_test", name="container_execute"),
    ]
    wire = llm.format_messages_for_llm(messages)
    assert wire[0]["content"] == "Read the data."
    assert wire[1]["content"] == "\n\n"
    assert wire[1]["reasoning_content"] == reasoning
    assert wire[1]["tool_calls"][0]["id"] == "call_test"
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == call.arguments
    assert wire[2]["content"] == "table.json\n"
