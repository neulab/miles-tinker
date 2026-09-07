"""deploy/clients/tinker_sdk_smoke.py frees the models it created at exit: the SDK never sends
SESSION_END on a clean exit, so a leaked model would hold the single full-FT slot for the lease."""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SDK_SMOKE = REPO_ROOT / "deploy" / "clients" / "tinker_sdk_smoke.py"


@pytest.fixture(scope="module")
def sdk_smoke():
    os.environ.setdefault("MILES_ALLOW_LOGIN_NODE", "1")  # hostguard refuses login nodes at import
    sys.path.insert(0, str(SDK_SMOKE.parent))
    try:
        spec = importlib.util.spec_from_file_location("tinker_sdk_smoke", SDK_SMOKE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SDK_SMOKE.parent))


def test_unload_at_exit_frees_every_resident_model_and_warns_on_failure(sdk_smoke, capsys):
    calls = []

    def unload(base_url, model_id):
        calls.append((base_url, model_id))
        if model_id == "model-b":
            raise RuntimeError("gateway gone")

    checks = sdk_smoke.Checks()
    loaded = ["model-a", "model-b", "model-c"]
    sdk_smoke.unload_at_exit(checks, "http://gw", loaded, keep=False, unload=unload)
    assert calls == [("http://gw", "model-a"), ("http://gw", "model-b"), ("http://gw", "model-c")]
    assert loaded == ["model-b"], "a failed unload stays listed; the lease sweep frees it later"
    assert checks.failed == 0, "an unload failure is a WARN, the checks' verdict is unchanged"
    out = capsys.readouterr().out
    assert out.count("PASS unload_model at exit") == 2 and "WARN unload_model at exit: model-b" in out


def test_keep_model_leaves_the_models_loaded(sdk_smoke, capsys):
    calls = []
    loaded = ["model-a"]
    sdk_smoke.unload_at_exit(sdk_smoke.Checks(), "http://gw", loaded, keep=True, unload=lambda *a: calls.append(a))
    assert calls == [] and loaded == ["model-a"]
    assert "keeping" in capsys.readouterr().out


def test_main_runs_the_checks_under_a_finally_that_unloads(sdk_smoke):
    """The unload must run whatever the outcome (a FAIL exit, an SDK exception, a KeyboardInterrupt)."""
    tree = ast.parse(SDK_SMOKE.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    trys = [node for node in ast.walk(main) if isinstance(node, ast.Try)]
    assert trys and trys[0].finalbody, "main() has no try/finally"
    finally_src = ast.unparse(trys[0].finalbody)
    assert "unload_at_exit" in finally_src and "keep=args.keep_model" in finally_src
    body_src = ast.unparse(trys[0].body)
    assert "_run_checks(" in body_src
    # both create calls register their model with the exit unload, and the pre-resume unload deregisters
    run_checks = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_checks")
    src = ast.unparse(run_checks)
    assert "loaded.append(resumed_id)" in src and "loaded.remove(info.model_id)" in src
    assert sdk_smoke.main.__doc__ is None or "unload" in sdk_smoke.__doc__
