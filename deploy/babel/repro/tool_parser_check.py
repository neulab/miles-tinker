"""Compare the proxy's tool-call parser with SGLang's on what the model generated.

Reads the GENERATED token ids that TERMINAL_RL_DUMP_PROMPTS writes into each
rollout's prompts_*.txt, decodes them, and parses every response that calls a
tool with both platoon.terminal_rl.qwen3_tool_parser and SGLang's
Qwen3CoderDetector, comparing function names and typed arguments, against the
tool specs in TOOL_SPECS. Run it in the Miles image, which has SGLang:

    TOOL_SPECS=tools.json apptainer exec --bind /home,/tmp <image.sif> python3 deploy/babel/repro/tool_parser_check.py \\
        <run>/litecoder-terminal-platoon-tinker/<run>/rollouts <work>/models/Qwen3.5-4B
"""

import glob
import importlib.util
import json
import os
import re
import sys

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from transformers import AutoTokenizer

rollouts, model_dir = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "qwen3_tool_parser", os.path.join(here, "../terminal_rl/platoon/terminal_rl/qwen3_tool_parser.py")
)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
tok = AutoTokenizer.from_pretrained(model_dir)

responses = []
for path in glob.glob(os.path.join(rollouts, "*/*/events/prompts_*.txt")):
    text = open(path, errors="replace").read()
    for ids in re.findall(r"--- GENERATED token ids ---\n\[([0-9, ]+)\]", text):
        decoded = tok.decode([int(x) for x in ids.split(",")])
        if "<tool_call>" in decoded:
            responses.append((os.path.basename(path), decoded))

# The prompt dumps keep only the head of the system turn, so the tool specs come
# from a file: TOOL_SPECS=<json list of OpenAI tool specs>.
spec_file = os.environ.get("TOOL_SPECS")
tools = json.load(open(spec_file)) if spec_file else None
if not tools:
    raise SystemExit("set TOOL_SPECS to a JSON list of the OpenAI tool specs (e.g. from a saved prompt's <tools> block)")

same = differ = calls = 0
for name, decoded in responses:
    text = decoded.split("</think>", 1)[-1].split("<|im_end|>")[0]
    ref = [(c.name, json.loads(c.parameters)) for c in Qwen3CoderDetector().detect_and_parse(text, [Tool(**t) for t in tools]).calls]
    ours = [p for p in (parser.parse_arguments(b, tools) for b in re.findall(r"<tool_call>.*?</tool_call>", text, re.S)) if p]
    calls += len(ref)
    if json.dumps([list(x) for x in ours], sort_keys=True) == json.dumps([list(x) for x in ref], sort_keys=True):
        same += 1
    else:
        differ += 1
        print(f"DIFFERS {name}\n  sglang: {json.dumps(ref)[:300]}\n  ours  : {json.dumps(ours)[:300]}")
print(f"{same}/{same + differ} responses ({calls} tool calls): qwen3_tool_parser == SGLang Qwen3CoderDetector")
