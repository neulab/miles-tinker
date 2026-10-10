"""No GPU/model call: SDK wire history must preserve the sampled assistant turn."""

from examples.science_rl.policy_llm import SciencePolicyLLM
from openhands.sdk import Message, TextContent
from openhands.sdk.llm.message import MessageToolCall
from pydantic import SecretStr


def test_qwen_reasoning_and_tool_history_survive_sdk_serialization():
    llm = SciencePolicyLLM(
        model="openai/Qwen3.5-4B",
        api_key=SecretStr("local-test"),
        base_url="http://127.0.0.1:1/v1",
        capability_overrides={"supports_vision": True, "supports_prompt_cache_key": False},
    )
    reasoning = "Derive the regression, then execute it.\n"
    call = MessageToolCall(
        id="call_test", name="container_execute", arguments='{"command":"ls /work"}', origin="completion"
    )
    messages = [
        Message(role="user", content=[TextContent(text="Read the data.")]),
        Message(role="assistant", content=[TextContent(text="\n\n")], reasoning_content=reasoning, tool_calls=[call]),
        Message(
            role="tool", content=[TextContent(text="table.json\n")], tool_call_id="call_test", name="container_execute"
        ),
    ]
    wire = llm.format_messages_for_llm(messages)
    assert wire[0]["content"] == "Read the data."
    assert wire[1]["content"] == "\n\n"
    assert wire[1]["reasoning_content"] == reasoning
    assert wire[1]["tool_calls"][0]["id"] == "call_test"
    assert wire[1]["tool_calls"][0]["function"]["arguments"] == call.arguments
    assert wire[2]["content"] == "table.json\n"
