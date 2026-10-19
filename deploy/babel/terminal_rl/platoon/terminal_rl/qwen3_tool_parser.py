"""Parse Qwen3.5 tool calls the way the serving stack does.

tinker-cookbook's ``Qwen3_5Renderer._parse_qwen3_5_tool_call_xml`` reads each
parameter with ``<parameter=name>\\s*(.*?)\\s*</parameter>`` and then
``strip("\\n")``, so it removes *all* leading and trailing whitespace from a value.
The chat template wraps a value in exactly one newline on each side. What the
proxy hands openhands is therefore not what the model wrote, and openhands
executes it: in job 155391 a ``str_replace`` whose ``new_str`` was
``"    double accumulator = base;"`` arrived as ``"double accumulator = base;"``,
and ``file_text`` lost its trailing newline -- 34 tool-call mismatches in the
first 30 rollouts. The policy was then scored on edits it did not make, and
history showed it the altered call. Values were also typed by trying
``json.loads`` on everything, so a string parameter that looked like a number
became one, and ``true`` came back into history as ``True``.

This replaces that method with SGLang's ``Qwen3CoderDetector`` semantics
(sglang 0.5.20, ``srt/function_call/qwen3_coder_detector.py``), which is what
an evaluation served by SGLang parses with:

* a value is the raw text between ``<parameter=name>`` and ``</parameter>`` (or
  the next ``<parameter=`` / ``</function>``), minus exactly one leading and one
  trailing ``\\n``;
* it is typed by the tool's JSON schema: strings verbatim, ``null`` as None,
  integers and numbers parsed, ``true``/``false`` as booleans, objects and arrays
  through ``json.loads`` then ``ast.literal_eval``; a parameter the schema does
  not declare stays a string.

The schema comes from the request: the proxy calls ``set_tools`` when it builds
the prompt, and parsing that call's response reads it back through a context
variable, so concurrent rollouts do not see each other's tools.

``TERMINAL_RL_SGLANG_TOOL_PARSER=0`` keeps the cookbook parser.
"""

from __future__ import annotations

import ast
import contextvars
import json
import logging
import os
import re
import warnings
from typing import Any

logger = logging.getLogger("platoon.terminal_rl.qwen3_tool_parser")

_tools: contextvars.ContextVar[list | None] = contextvars.ContextVar("terminal_rl_tool_specs", default=None)

_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)


def enabled() -> bool:
    return os.environ.get("TERMINAL_RL_SGLANG_TOOL_PARSER", "1") != "0"


def set_tools(tools: Any) -> None:
    """Record the request's tool specs for parsing this call's response."""
    _tools.set(list(tools) if tools else None)


# ----------------------------------------------------------------------------
# Schema helpers, as sglang/srt/function_call/utils.py
# ----------------------------------------------------------------------------
def _schema_properties(schema: Any) -> dict:
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return properties
    merged: dict = {}
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                for key, value in _schema_properties(branch).items():
                    merged.setdefault(key, value)
    return merged


def _infer_type(schema: Any) -> str | None:
    if not isinstance(schema, dict):
        return None
    if "type" in schema:
        value = schema["type"]
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            non_null = [t for t in value if t != "null"]
            return non_null[0] if non_null else "string"
    if "anyOf" in schema or "oneOf" in schema:
        branches = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(branches, list):
            types = [t for t in (_infer_type(b) for b in branches) if t]
            if types:
                if len(set(types)) == 1:
                    return types[0]
                if len(set(types)) == 2 and "null" in types:
                    return [t for t in types if t != "null"][0]
                if "string" in types:
                    return "string"
                return types[0]
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"
        kinds = set()
        for value in schema["enum"]:
            if value is None:
                kinds.add("null")
            elif isinstance(value, bool):
                kinds.add("boolean")
            elif isinstance(value, int):
                kinds.add("integer")
            elif isinstance(value, float):
                kinds.add("number")
            elif isinstance(value, str):
                kinds.add("string")
            elif isinstance(value, list):
                kinds.add("array")
            elif isinstance(value, dict):
                kinds.add("object")
        return kinds.pop() if len(kinds) == 1 else "string"
    if "allOf" in schema and isinstance(schema["allOf"], list):
        for branch in schema["allOf"]:
            inferred = _infer_type(branch)
            if inferred and inferred != "string":
                return inferred
        return "string"
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return None


def _arguments_config(name: str, tools: list | None) -> dict:
    for tool in tools or []:
        tool = tool.model_dump() if hasattr(tool, "model_dump") else tool
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        if tool.get("type", "function") == "function" and isinstance(function, dict) and function.get("name") == name:
            params = function.get("parameters")
            if isinstance(params, dict):
                properties = _schema_properties(params)
                return properties if properties or "properties" in params else params
            return {}
    return {}


def _literal(value: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.literal_eval(value)


def _convert(value: str, name: str, config: dict) -> Any:
    """As Qwen3CoderDetector._convert_param_value."""
    if value.lower() == "null":
        return None
    if name not in config:
        return value
    kind = str(_infer_type(config[name]) or "string").strip().lower()
    if kind in ("string", "str", "text", "varchar", "char", "enum"):
        return value
    if kind.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(value)
        except Exception:
            return value
    if kind.startswith(("num", "float")):
        try:
            integral = not ("." in value or "e" in value.lower())
            number = float(value)
            return int(number) if integral and number.is_integer() else number
        except Exception:
            return value
    if kind in ("boolean", "bool", "binary"):
        return value.lower() == "true"
    if kind in ("object", "array", "arr") or kind.startswith(("dict", "list")):
        try:
            return json.loads(value)
        except Exception:
            pass
    try:
        return _literal(value)
    except Exception:
        return value


def parse_arguments(raw_text: str, tools: list | None) -> tuple[str, dict] | None:
    """One ``<tool_call>`` block -> (function name, typed arguments), or None."""
    body = raw_text
    match = re.search(r"<tool_call>(.*?)</tool_call>", raw_text, re.DOTALL)
    if match:
        body = match.group(1)
    function = next((a or b for a, b in _FUNCTION_RE.findall(body) if ">" in (a or b)), None)
    if function is None:
        return None
    name, params = function[: function.index(">")], function[function.index(">") + 1 :]
    if not name:
        return None
    config = _arguments_config(name, tools)
    arguments: dict[str, Any] = {}
    for parameter in _PARAMETER_RE.findall(params):
        if ">" not in parameter:
            continue
        key, value = parameter[: parameter.index(">")], parameter[parameter.index(">") + 1 :]
        # Exactly one newline each side: the template's own "<parameter=k>\n" and "\n</parameter>".
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        arguments[key] = _convert(value, key, config)
    return name, arguments


def parse_tool_call(raw_text: str, tools: list | None = None):
    """One ``<tool_call>...</tool_call>`` block -> tinker-cookbook ToolCall."""
    from tinker_cookbook.renderers.base import ToolCall, UnparsedToolCall

    parsed = parse_arguments(raw_text, tools if tools is not None else _tools.get())
    if parsed is None:
        return UnparsedToolCall(raw_text=raw_text, error="No <function=name> block")
    name, arguments = parsed
    return ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))


def install(renderer: Any) -> None:
    """Use this parser for the renderer's Qwen3.5 XML tool calls."""
    if not enabled() or not hasattr(renderer, "_parse_qwen3_5_tool_call_xml"):
        return
    renderer._parse_qwen3_5_tool_call_xml = lambda raw_text: parse_tool_call(raw_text)
    logger.info("[render] Qwen3.5 tool calls are parsed with SGLang's Qwen3Coder semantics (TERMINAL_RL_SGLANG_TOOL_PARSER=0 to disable)")
