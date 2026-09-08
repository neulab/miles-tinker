"""CPU checks for the Babel launcher: python deploy/babel/test_run.py."""

import io
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run


class TrainingLauncherTests(unittest.TestCase):
    def test_default_command_and_explicit_command_route_correctly(self):
        for arguments, expected in (
            (["--dry-run"], "--in-container --steps=100"),
            (["--dry-run", "--", "python", "recipe.py"], "bash python recipe.py"),
        ):
            with (
                mock.patch.object(run.sys, "argv", ["run.py", *arguments]),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                run.main()
                self.assertIn(expected, output.getvalue())

    def test_defaults_match_training(self):
        args = run._parser().parse_args([])
        gateway, client = run._training_commands(args, 4, Path("/output/run"), 12345)
        for expected in ("--actor-num-gpus-per-node=2", "--rollout-num-gpus=2", "--tp=1"):
            self.assertIn(expected, gateway)
        for expected in (
            "--train.max_training_steps=100",
            "--train.optimizer.learning_rate=1e-06",
            "--train.batch_size=32",
            "--train.workflow_config.group_size=8",
            "--stats.wandb.mode=online",
        ):
            self.assertIn(expected, client)
        self.assertIn("--tinker-port=12345", gateway)
        self.assertIn("--tinker_base_url=http://127.0.0.1:12345", client)

    def test_gpu_split_uses_available_count_and_optional_override(self):
        for count in range(2, 17):
            args = run._parser().parse_args([])
            gateway, _ = run._training_commands(args, count, Path("/output/run"), 12345)
            self.assertIn(f"--actor-num-gpus-per-node={count // 2}", gateway)
            self.assertIn(f"--rollout-num-gpus={count - count // 2}", gateway)
        args = run._parser().parse_args(["--rollout-gpus=3"])
        gateway, _ = run._training_commands(args, 8, Path("/output/run"), 12345)
        self.assertIn("--actor-num-gpus-per-node=5", gateway)
        with self.assertRaises(ValueError):
            run._training_commands(args, 2, Path("/output/run"), 12345)

    def test_client_setup_clones_only_when_missing_and_isolates_dependencies(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/server", "VIRTUAL_ENV": "/server"}):
            for exists in (False, True):
                with (
                    mock.patch.object(Path, "exists", return_value=exists),
                    mock.patch.object(subprocess, "run") as call,
                ):
                    _, env = run._prepare_client()
                    self.assertNotIn("PYTHONPATH", env)
                    self.assertNotIn("VIRTUAL_ENV", env)
                    self.assertEqual(call.call_count, 1 if exists else 2)
                    self.assertEqual(call.call_args.args[0][:2], ["uv", "sync"])

    def test_startup_failure_prevents_client_and_stops_server(self):
        self._check_cleanup(RuntimeError("startup failed"), None)

    def test_client_failure_stops_server(self):
        self._check_cleanup(None, subprocess.CalledProcessError(1, "client"))

    def _check_cleanup(self, startup_error, client_error):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (
                mock.patch.object(subprocess, "Popen") as start,
                mock.patch.object(run, "_wait_for_gateway", side_effect=startup_error),
                mock.patch.object(subprocess, "run", side_effect=client_error) as client,
                mock.patch.object(run, "_stop_gateway") as stop,
            ):
                with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    run._run_training(["server"], ["client"], path, {}, path, 12345)
                stop.assert_called_once_with(start.return_value)
                self.assertEqual(client.call_count, 0 if startup_error else 1)
                if client.call_count:
                    self.assertEqual(client.call_args.kwargs["env"]["TINKER_BASE_URL"], "http://127.0.0.1:12345")

    def test_dead_server_does_not_report_ready(self):
        server = mock.Mock()
        server.poll.return_value = 1
        server.returncode = 1
        with mock.patch.object(run.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "Server exited"):
                run._wait_for_gateway(server, "http://127.0.0.1:12345")
            request.assert_not_called()

    def test_batch_entrypoint_cleans_only_ipython_and_preserves_training_errors(self):
        class Shell:
            def atexit_operations(self):
                pass

        for error in (None, RuntimeError("training failed")):
            train = mock.AsyncMock(side_effect=error)
            shells = [Shell(), Shell()]
            modules = {
                "IPython.core.interactiveshell": SimpleNamespace(InteractiveShell=Shell),
                "platoon.number_search.train_tinker": SimpleNamespace(main=train),
            }
            with (
                mock.patch.dict(run.sys.modules, modules),
                mock.patch.object(run.sys, "argv", ["client.py", "--train.max_training_steps=2"]),
                mock.patch("atexit.unregister") as unregister,
                mock.patch("gc.get_objects", return_value=[*shells, object()]),
            ):
                path = Path(__file__).with_name("client.py")
                if error:
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        runpy.run_path(str(path), run_name="__main__")
                else:
                    runpy.run_path(str(path), run_name="__main__")
                train.assert_awaited_once_with(["--train.max_training_steps=2"])
                self.assertEqual(unregister.call_args_list, [mock.call(shell.atexit_operations) for shell in shells])


if __name__ == "__main__":
    unittest.main()
