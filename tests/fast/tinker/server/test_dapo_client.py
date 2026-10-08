"""Run the actual SDK 0.16.1 PPO client against a CPU gateway with recording trainers."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from tests.fast.tinker.harness import make_config
from tests.fast.tinker.server.test_sdk_flow import _Gateway
from tests.fast.tinker.test_residency import ResidentBackend

from miles.tinker.core.service import TinkerService
from miles.tinker.full_backend import FullModelBackend

CLIENT_RUNNER = r"""
import asyncio
import importlib.util
import sys
from pathlib import Path
import transformers

root, url, output, asynchronous = sys.argv[1:]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("dapo_ppo", Path(root) / "deploy/clients/dapo_ppo.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

class Tokenizer:
    eos_token_id = 2
    def apply_chat_template(self, *args, **kwargs): return "question"
    def encode(self, *args, **kwargs): return [10, 20]
    def decode(self, *args, **kwargs): return r"</think>\boxed{4}"

transformers.AutoTokenizer.from_pretrained = lambda *args, **kwargs: Tokenizer()
output = Path(output)
common = ["--base-url", url, "--model-path", "test-tokenizer", "--data", str(output / "data.jsonl"),
          "--output", str(output), "--prompts-per-rollout", "1", "--samples-per-prompt", "2",
          "--batch-size", "2", "--reference-dp", "1", "--save-interval", "2"]
if asynchronous == "True": common += ["--async-rollouts"]
asyncio.run(client._main(client._parse_args(common + ["--rollouts", "2"])))
asyncio.run(client._main(client._parse_args(common + ["--rollouts", "3", "--resume",
                                                     str(output / "latest.json"), "--verify-resume"])))
print("PPO SDK train and resume passed")
"""


class CriticBackend(ResidentBackend):
    async def forward_only(self, *args):
        outputs = await super().forward_only(*args)
        for output in outputs:
            output["values"] = output.pop("logprobs")
        return outputs

    async def forward_backward(self, *args):
        outputs = await super().forward_backward(*args)
        for output in outputs:
            output["values"] = output.pop("logprobs")
        return outputs


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_ppo_client_train_checkpoint_and_resume(tmp_path, asynchronous):
    sdk_python = os.environ.get("TINKER016_PYTHON")
    if sdk_python is None:
        pytest.skip("set TINKER016_PYTHON to the standalone PPO client environment")
    resident, events = set(), []
    policy = ResidentBackend(0, resident, events)
    critic = CriticBackend(1, resident, events)
    backend = FullModelBackend(policy, critic, offload=True)
    config = make_config(
        base_model="Qwen/Qwen3-4B",
        train_mode="full",
        enable_critic=True,
        n_slots=2,
        checkpoint_root=str(tmp_path / "checkpoints"),
        coalesce_wait_s=0,
        retrieve_poll_timeout_s=0.1,
    )
    (tmp_path / "data.jsonl").write_text(json.dumps({"prompt": "What is 2 + 2?", "label": "4"}) + "\n")
    root = Path(__file__).resolve().parents[4]
    gateway = _Gateway(lambda: TinkerService(backend, config)).start()
    try:
        result = subprocess.run(
            [sdk_python, "-c", CLIENT_RUNNER, str(root), gateway.url, str(tmp_path), str(asynchronous)],
            cwd=tmp_path,
            env={**os.environ, "TINKER_API_KEY": "tml-dapo-test"},
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PPO SDK train and resume passed" in result.stdout
        assert len(critic.named("optim_step")) == 3
        assert len(policy.named("optim_step")) == 2, "the first rollout is critic-only warmup"
        for trainer in (policy, critic):
            steps = trainer.named("optim_step")
            adam = steps[0]["adam_params_by_slot"][0]
            assert (adam["beta1"], adam["beta2"], adam["eps"], adam["weight_decay"]) == (0.9, 0.98, 1e-8, 0.1)
            assert any(call["ckpt_path"] and call["load_optimizer"] for call in trainer.named("load_slot"))
        assert all(call["payload"]["sampling_params"]["temperature"] == 0.8 for call in policy.named("sample"))
        assert not resident and not gateway.service.models, "SDK unload must release both model handles"
        manifest = json.loads((tmp_path / "latest.json").read_text())
        assert manifest["next_rollout"] == 3 and set(manifest["paths"]) == {"policy", "critic"}
        for call in critic.named("forward_backward"):
            assert call["loss_fn"] == "ppo"
            assert "values" in call["slot_rows"][0][1] and "old_values" not in call["slot_rows"][0][1]
    finally:
        gateway.stop()
