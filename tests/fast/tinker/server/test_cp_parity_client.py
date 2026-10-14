"""deploy/clients/tinker_cp_parity.py ``run`` drives a real SDK session against the fake-backend
gateway (forward, forward_backward for each loss, optim_step, unload) and its dump compares clean
against itself."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.tinker.server.test_sdk_flow import BASE_MODEL, _Gateway

pytest.importorskip("tinker")

PARITY = Path(__file__).resolve().parents[4] / "deploy" / "clients" / "tinker_cp_parity.py"


@pytest.fixture(scope="module")
def parity():
    spec = importlib.util.spec_from_file_location("tinker_cp_parity_e2e", PARITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_dumps_every_probe_and_compares_clean(parity, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MILES_ALLOW_LOGIN_NODE", "1")  # hostguard: a seconds-long run against a local fake
    monkeypatch.setenv("TINKER_API_KEY", "tml-cp-parity-test")
    gateway = _Gateway().start()
    try:
        out = tmp_path / "parity.json"
        args = Namespace(
            base_url=gateway.url,
            endpoint_file=None,
            base_model=BASE_MODEL,
            lora_rank=8,
            out=str(out),
            cp=1,
            label="fake",
            max_cp=2,
            max_len=64,
            steps=3,
            learning_rate=1e-6,
            seed=0,
            tokens="motif",  # no tokenizer download in the fast suite; --tokens text is the GPU default
            keep_model=False,
        )
        assert parity.run(args) == 0
        assert [name for name, _ in gateway.service.backend.calls].count("optim_step") == 3
        assert gateway.service.backend.named("unload_slot"), "the single-tenant slot is freed at exit"
    finally:
        gateway.stop()

    dump = json.loads(out.read_text())
    lengths = dump["lengths"]
    assert lengths == [1, 2, 3, 4, 31] and dump["single_length"] == 4  # the length closest to 11
    for key in ("forward_before", "forward_before_permuted", "forward_after"):
        assert [len(v) for v in dump[key]["logprobs"]] == lengths
    assert [step["loss_fn"] for step in dump["steps"]] == ["cross_entropy", "importance_sampling", "cispo"]
    assert all(step["grad_norm"] == 0.5 for step in dump["steps"])
    assert "forward_backward_single" in dump["steps"][0]

    compare = Namespace(
        ref=str(out), cand=str(out), cp=2, json_out=str(tmp_path / "report.json"), noise=str(out), noise_factor=2.0
    )
    for key, value in parity.DEFAULT_THRESHOLDS.items():
        setattr(compare, key, value)
    assert parity.compare(compare) == 0
    assert json.loads((tmp_path / "report.json").read_text())["failures"] == []
    assert "PASS" in capsys.readouterr().out
