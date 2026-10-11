"""Replay Qwen messages without losing reasoning or changing text representation."""

from dataclasses import replace

from openhands.sdk import TextContent
from paper2rlenv.runtime import PolicyTokenBudgetReached, ProxyLLM
from pydantic import Field


class SciencePolicyLLM(ProxyLLM):
    # SGLang returns text strings; SDK list serialization changes replayed history.
    force_string_serializer: bool = True
    episode_token_limit: int | None = Field(default=None, gt=0, exclude=True)

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        if self.episode_token_limit is not None:
            response = result.raw_response
            if response.choices[0].finish_reason == "length":
                raise PolicyTokenBudgetReached("model_call_length")
            usage = response.usage
            if usage is None or usage.total_tokens is None:
                raise ValueError("Token-budget enforcement requires policy token usage")
            if usage.total_tokens >= self.episode_token_limit:
                raise PolicyTokenBudgetReached("episode_context_limit")
        return result

    def _model_features(self):
        # The SDK's model registry does not currently enable Qwen reasoning replay.
        return replace(super()._model_features(), send_reasoning_content=True)

    def _to_chat_dicts(self, messages):
        wire = super()._to_chat_dicts(messages)
        for message, serialized in zip(messages, wire, strict=True):
            if message.role == "assistant":
                texts = [item.text for item in message.content if isinstance(item, TextContent)]
                if texts:
                    # SDK removes whitespace-only assistant content on tool turns.
                    # It is part of the sampled prefix and must survive exact replay.
                    serialized["content"] = "\n".join(texts)
        return wire
