"""Replay Qwen messages without losing reasoning or changing text representation."""

from dataclasses import replace

from openhands.sdk import TextContent
from paper2rlenv.runtime import ProxyLLM


class SciencePolicyLLM(ProxyLLM):
    # SGLang returns text strings; SDK list serialization changes replayed history.
    force_string_serializer: bool = True

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
