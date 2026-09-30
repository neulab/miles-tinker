"""Exercise launcher signals in isolated processes without GPUs or Modal."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


def _wait_for_file(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None, f"Supervisor exited before creating {path}"
        assert time.monotonic() < deadline, f"Timed out waiting for {path}"
        time.sleep(0.02)


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_signals_clean_up_server_and_relay_even_when_repeated(tmp_path, stop_signal):
    python = tmp_path / "venvs/modal/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    relay = tmp_path / "repo/tools/amd_tinker_modal.py"
    relay.parent.mkdir(parents=True)
    server = tmp_path / "server.py"
    child_code = textwrap.dedent(
        """\
        import argparse
        import os
        import signal
        import time
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument('--root', type=Path)
        args, _ = parser.parse_known_args()
        role = Path(__file__).stem

        def terminate(signum, frame):
            (args.root / (role + '.stopped')).touch()
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, terminate)
        (args.root / (role + '.started')).write_text(str(os.getpid()))
        while True:
            time.sleep(1)
        """
    )
    relay.write_text(child_code)
    server.write_text(child_code)
    wrapper = tmp_path / "supervisor.py"
    wrapper.write_text(
        textwrap.dedent(
            """\
            import signal
            import sys
            import time
            from pathlib import Path

            sys.path.insert(0, sys.argv[1])
            from tools import amd_tinker

            root = Path(sys.argv[2])
            children = []
            original_popen = amd_tinker.subprocess.Popen

            def popen(*args, **kwargs):
                child = original_popen(*args, **kwargs)
                children.append(child)
                return child

            def stop(config):
                # Only stop our dummy server; never inspect real GPU processes.
                (root / 'cleanup.started').touch()
                deadline = time.monotonic() + 5
                while not (root / 'second-signal.sent').exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                children[1].terminate()
                children[1].wait(timeout=5)

            amd_tinker.subprocess.Popen = popen
            amd_tinker._stop = stop
            config = amd_tinker.BackendConfig(root=root, repo=root / 'repo', modal_relay=True)
            previous = signal.getsignal(signal.SIGTERM)
            try:
                amd_tinker._serve_with_modal(
                    config, [sys.executable, str(root / 'server.py'), '--root', str(root)]
                )
            finally:
                if signal.getsignal(signal.SIGTERM) == previous:
                    (root / 'handler.restored').touch()
            """
        )
    )
    repo = Path(__file__).resolve().parents[2]
    process = subprocess.Popen([sys.executable, str(wrapper), str(repo), str(tmp_path)], start_new_session=True)
    try:
        for role in ("server", "amd_tinker_modal"):
            _wait_for_file(tmp_path / f"{role}.started", process)
        process.send_signal(stop_signal)
        _wait_for_file(tmp_path / "cleanup.started", process)
        process.send_signal(stop_signal)
        (tmp_path / "second-signal.sent").touch()
        assert process.wait(timeout=10) in (128 + stop_signal, -stop_signal)
        for role in ("server", "amd_tinker_modal"):
            assert (tmp_path / f"{role}.stopped").exists()
        assert (tmp_path / "handler.restored").exists()
    finally:
        # Clean up this isolated process group even if the regression reappears.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
