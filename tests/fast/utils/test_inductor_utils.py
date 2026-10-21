import functools
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch._inductor.config as inductor_config

from miles.utils import inductor_utils
from miles.utils.inductor_utils import QUIESCE_ENV, disable_compile_pool_quiesce


@pytest.fixture
def restore_quiesce_config(monkeypatch):
    """Undo the process-wide switch after each test."""
    monkeypatch.setattr(inductor_config, "quiesce_async_compile_pool", inductor_config.quiesce_async_compile_pool)
    monkeypatch.delenv(QUIESCE_ENV, raising=False)
    yield


class _FakeTimer:
    def __init__(self):
        self.quit_calls = 0

    def quit(self):
        self.quit_calls += 1


class _FakePool:
    def __init__(self):
        self.timer = _FakeTimer()


def _fake_process_pool(pool):
    @functools.lru_cache(1)
    def process_pool():
        return pool

    return process_pool


class TestDisableCompilePoolQuiesce:
    def test_turns_off_inductor_quiesce(self, restore_quiesce_config):
        """The Inductor option is off and the env var is set for child processes."""
        inductor_config.quiesce_async_compile_pool = True
        disable_compile_pool_quiesce()
        assert inductor_config.quiesce_async_compile_pool is False
        assert os.environ[QUIESCE_ENV] == "0"

    def test_running_pool_loses_its_idle_timer(self, restore_quiesce_config, monkeypatch):
        """A pool created before the call keeps running without its quiesce timer."""
        from torch._inductor.async_compile import AsyncCompile

        pool = _FakePool()
        timer = pool.timer
        process_pool = _fake_process_pool(pool)
        process_pool()  # created before the call
        monkeypatch.setattr(AsyncCompile, "process_pool", staticmethod(process_pool))
        disable_compile_pool_quiesce()
        assert timer.quit_calls == 1
        assert pool.timer is None

    def test_no_pool_is_created(self, restore_quiesce_config, monkeypatch):
        """Without a running pool nothing is started (the first compile creates it, without quiesce)."""
        from torch._inductor.async_compile import AsyncCompile

        pool = _FakePool()
        process_pool = _fake_process_pool(pool)
        monkeypatch.setattr(AsyncCompile, "process_pool", staticmethod(process_pool))
        disable_compile_pool_quiesce()
        assert process_pool.cache_info().currsize == 0
        assert pool.timer.quit_calls == 0

    def test_missing_option_is_a_noop(self, restore_quiesce_config, monkeypatch):
        """A torch without the quiesce option has nothing to turn off."""
        monkeypatch.delattr(inductor_config, "quiesce_async_compile_pool", raising=False)
        disable_compile_pool_quiesce()
        assert not hasattr(inductor_config, "quiesce_async_compile_pool")


def test_trainer_actor_disables_quiesce_before_any_compile():
    """TrainRayActor.__init__ calls the switch first thing, before anything can torch.compile.

    Read from the source: importing miles.ray.train_actor pulls in Ray and SGLang.
    """
    source = (Path(inductor_utils.__file__).parents[1] / "ray" / "train_actor.py").read_text()
    init = source[source.index("class TrainRayActor") :]
    init = init[init.index("    def __init__") : init.index("    def propose_master_addr_and_port")]
    assert "from miles.utils.inductor_utils import disable_compile_pool_quiesce" in source
    assert "disable_compile_pool_quiesce()" in init
    assert init.index("disable_compile_pool_quiesce()") < init.index("object_store.init_instance")


def test_real_pool_never_quiesces(restore_quiesce_config):
    """End to end on torch's SubprocPool: with the switch on, an idle pool keeps its workers.

    With quiesce on, the sidecar shuts its fork pool down after ``quiesce_async_compile_time`` idle seconds and
    the next job forks a new one, the step that can deadlock every new worker. With the switch, the pool is
    built without the idle timer, stays up across the idle period, and keeps answering jobs.
    """
    subproc_pool = pytest.importorskip("torch._inductor.compile_worker.subproc_pool")
    disable_compile_pool_quiesce()
    pool = subproc_pool.SubprocPool(2, quiesce=inductor_config.quiesce_async_compile_pool)
    try:
        assert pool.timer is None
        assert pool.submit(functools.partial(pow, 2, 10)).result(timeout=120) == 1024
        time.sleep(1.5)  # longer than a shortened quiesce time; nothing may quiesce here
        assert pool.quiesce_waitcounter is None
        assert pool.submit(functools.partial(pow, 3, 4)).result(timeout=120) == 81
    finally:
        pool.shutdown()


def test_does_not_import_async_compile():
    """The switch is cheap at actor start: it never imports Inductor's async_compile (nor creates a pool)."""
    code = (
        "import sys; from miles.utils.inductor_utils import disable_compile_pool_quiesce; "
        "disable_compile_pool_quiesce(); print('torch._inductor.async_compile' in sys.modules)"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
