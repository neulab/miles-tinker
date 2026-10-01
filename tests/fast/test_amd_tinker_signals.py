"""Exercise launcher signals in isolated processes without GPUs or Modal."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import tools.amd_tinker as amd_tinker


@pytest.mark.parametrize("modal_relay", [False, True])
def test_logs_preserve_server_and_relay_output_across_launches(tmp_path, monkeypatch, capfd, modal_relay):
    config = amd_tinker.BackendConfig(
        root=tmp_path / "root",
        checkpoint_root=tmp_path / "checkpoints",
        repo=tmp_path / "repo",
        modal_relay=modal_relay,
    )
    config.root.mkdir()
    ready = config.root / "relay.ready"
    if modal_relay:
        python = config.venv_dir / "modal/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        relay = config.repo / "tools/amd_tinker_modal.py"
        relay.parent.mkdir(parents=True)
        relay.write_text(textwrap.dedent("""\
            import signal
            import sys
            import time
            from pathlib import Path

            def stop(*args):
                print('relay cleanup', flush=True)
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, stop)
            print('relay stdout', flush=True)
            print('relay stderr', file=sys.stderr, flush=True)
            Path(sys.argv[sys.argv.index('--root') + 1], 'relay.ready').touch()
            while True:
                time.sleep(1)
            """))
    # Never inspect or stop real server processes in this test.
    monkeypatch.setattr(amd_tinker, "_stop", lambda config: None)
    saved_logs = {}
    for index in range(2):
        ready.unlink(missing_ok=True)
        code = textwrap.dedent("""\
            import sys
            import time
            from pathlib import Path
            if sys.argv[1] == 'True':
                deadline = time.monotonic() + 5
                while not Path(sys.argv[2]).exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
            print('server stdout ' + sys.argv[3], flush=True)
            print('server stderr ' + sys.argv[3], file=sys.stderr, flush=True)
            """)
        amd_tinker._serve_with_modal(config, [sys.executable, "-c", code, str(modal_relay), str(ready), str(index)])
        logs = list((config.checkpoint_root / "logs").glob("server-*.log"))
        assert len(logs) == index + 1
        for path, contents in saved_logs.items():
            assert path.read_text() == contents
        path = next(path for path in logs if path not in saved_logs)
        contents = path.read_text()
        console = capfd.readouterr().out
        expected = [f"server stdout {index}", f"server stderr {index}"]
        if modal_relay:
            expected += ["relay stdout", "relay stderr", "relay cleanup"]
        for message in expected:
            assert message in contents
            assert message in console
        assert path.stat().st_mode & 0o077 == 0
        saved_logs[path] = contents


def test_failed_server_keeps_error_output_and_exit_status(tmp_path, monkeypatch, capfd):
    config = amd_tinker.BackendConfig(root=tmp_path)
    monkeypatch.setattr(amd_tinker, "_stop", lambda config: None)
    command = [sys.executable, "-c", "import sys; print('startup failed', file=sys.stderr); sys.exit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        amd_tinker._serve_with_modal(config, command)
    assert error.value.returncode == 7
    (path,) = (config.checkpoint_root / "logs").glob("server-*.log")
    assert "startup failed" in path.read_text()
    assert "startup failed" in capfd.readouterr().out


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
            import os
            import signal
            import sys
            import time
            from pathlib import Path

            sys.path.insert(0, sys.argv[1])
            from tools import amd_tinker

            root = Path(sys.argv[2])
            def stop(config):
                # Only stop our dummy server; never inspect real GPU processes.
                (root / 'cleanup.started').touch()
                deadline = time.monotonic() + 5
                while not (root / 'second-signal.sent').exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                os.kill(int((root / 'server.started').read_text()), signal.SIGTERM)

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
