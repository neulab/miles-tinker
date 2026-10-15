"""Expose a local Tinker server through an ephemeral, authenticated Modal relay.

An outbound SSH reverse forward carries traffic from the sandbox back to the
GPU host. Modal terminates HTTPS; nginx checks the client's X-API-Key before
forwarding. Keys and relay metadata stay under the deployment's runtime folder.
This helper is launched by ``amd_tinker.py serve --modal-relay``.
Use ``--ssh-over-tls`` on hosts that require the outbound tunnel to use TLS port 443.
"""

import argparse
import json
import os
import secrets
import shlex
import shutil
import signal
import socket
import ssl
import subprocess
import time
from pathlib import Path


def _sandbox_main():
    public_key = os.environ.pop("RELAY_SSH_PUBLIC_KEY")
    api_key = os.environ.pop("RELAY_API_KEY")
    port = int(os.environ["RELAY_LOCAL_PORT"])
    ssh_dir = Path("/root/.ssh")
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    authorized = ssh_dir / "authorized_keys"
    authorized.write_text(public_key + "\n")
    authorized.chmod(0o600)
    Path("/run/sshd").mkdir(exist_ok=True)
    subprocess.run(["ssh-keygen", "-A"], check=True)
    Path("/tmp/relay_sshd_config").write_text(
        "Port 2222\nListenAddress 0.0.0.0\nPermitRootLogin prohibit-password\n"
        "PasswordAuthentication no\nKbdInteractiveAuthentication no\nUsePAM no\n"
        "AllowTcpForwarding remote\nGatewayPorts no\n"
        "ChannelTimeout forwarded-tcpip=6m\n"
        f"PermitListen 127.0.0.1:{port}\n"
    )
    Path("/tmp/relay_nginx.conf").write_text(
        "events { worker_connections 4096; }\nhttp {\naccess_log off;\nserver {\nlisten 8080;\n"
        "client_max_body_size 64m;\nlocation / {\n"
        f'if ($http_x_api_key != "{api_key}") {{ return 401; }}\n'
        f"proxy_pass http://127.0.0.1:{port};\n"
        "proxy_http_version 1.1;\nproxy_set_header Connection '';\n"
        "proxy_read_timeout 3600s;\nproxy_send_timeout 3600s;\n"
        "proxy_request_buffering off;\nproxy_buffering off;\n}\n}\n}\n"
    )
    sshd = subprocess.Popen(["/usr/sbin/sshd", "-D", "-e", "-f", "/tmp/relay_sshd_config"])
    try:
        subprocess.run(["nginx", "-c", "/tmp/relay_nginx.conf", "-g", "daemon off;"], check=True)
    finally:
        sshd.terminate()
        sshd.wait()


def _private_file(path, contents):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(contents)


def _relay_keys(runtime):
    key_path = runtime / "public-api-key"
    if not key_path.exists():
        _private_file(key_path, "tml-" + secrets.token_urlsafe(32) + "\n")
    ssh_key = runtime / "modal-ssh-key"
    if not ssh_key.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(ssh_key)], check=True)
    return key_path.read_text().strip(), ssh_key


def _wait_for_ssh(host, port, *, tls=False):
    context = ssl.create_default_context() if tls else None
    deadline = time.monotonic() + 60
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3) as connection:
                if context is not None:
                    with context.wrap_socket(connection, server_hostname=host) as secured:
                        banner = secured.recv(128)
                else:
                    banner = connection.recv(128)
                if banner.startswith(b"SSH-"):
                    return
        except OSError as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"Could not reach Modal SSH listener at {host}:{port} within 60 seconds") from last_error


def _ssh_command(runtime, ssh_key, port, host, ssh_port, *, tls=False):
    command = [
        "ssh",
        "-N",
        "-T",
        "-i",
        str(ssh_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={runtime / 'modal-known-hosts'}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ConnectTimeout=30",
    ]
    if tls:
        proxy = shlex.join(
            [
                "openssl",
                "s_client",
                "-quiet",
                "-no_ign_eof",
                # SSH ciphertext can start with s_client command letters such as R or Q.
                "-nocommands",
                "-connect",
                f"{host}:{ssh_port}",
                "-servername",
                host,
                "-verify_return_error",
                "-verify_hostname",
                host,
            ]
        )
        command.extend(["-o", f"ProxyCommand={proxy}"])
    return command + ["-R", f"127.0.0.1:{port}:127.0.0.1:{port}", "-p", str(ssh_port), f"root@{host}"]


def _run_relay(root, port, timeout, *, ssh_over_tls=False):
    # Modal is an optional host-side dependency, absent from the GPU environment.
    import modal

    if ssh_over_tls and shutil.which("openssl") is None:
        raise RuntimeError("--ssh-over-tls requires openssl on the host")
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    api_key, ssh_key = _relay_keys(runtime)
    app = modal.App.lookup("miles-tinker-relay", create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("openssh-server", "nginx")
        .add_local_file(__file__, "/opt/relay.py", copy=True)
    )
    sandbox = None
    ssh = None
    try:
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "python",
                "/opt/relay.py",
                "--sandbox",
                app=app,
                image=image,
                cpu=0.125,
                memory=128,
                timeout=timeout,
                encrypted_ports=[8080, 2222] if ssh_over_tls else [8080],
                unencrypted_ports=[] if ssh_over_tls else [2222],
                env={
                    "RELAY_SSH_PUBLIC_KEY": ssh_key.with_suffix(".pub").read_text().strip(),
                    "RELAY_LOCAL_PORT": str(port),
                },
                secrets=[modal.Secret.from_dict({"RELAY_API_KEY": api_key})],
            )
        tunnels = sandbox.tunnels()
        ssh_tunnel = tunnels[2222]
        host = ssh_tunnel.host if ssh_over_tls else ssh_tunnel.unencrypted_host
        ssh_port = ssh_tunnel.port if ssh_over_tls else ssh_tunnel.unencrypted_port
        _wait_for_ssh(host, ssh_port, tls=ssh_over_tls)
        ssh = subprocess.Popen(_ssh_command(runtime, ssh_key, port, host, ssh_port, tls=ssh_over_tls))
        metadata = {
            "url": tunnels[8080].url,
            "sandbox_id": sandbox.object_id,
            "ssh_pid": ssh.pid,
            "api_key_file": str(runtime / "public-api-key"),
        }
        _private_file(runtime / "modal-relay.json", json.dumps(metadata, indent=2) + "\n")
        print(f"TINKER_BASE_URL={metadata['url']}", flush=True)
        print(f"TINKER_API_KEY={api_key}", flush=True)
        print(f"Client TINKER_API_KEY file: {runtime / 'public-api-key'}", flush=True)
        code = ssh.wait()
        if code:
            raise RuntimeError(f"Modal SSH relay exited with status {code}")
    finally:
        # Ctrl+C reaches both launcher and relay; ignore its follow-up terminate
        # while removing the remote sandbox.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if ssh is not None and ssh.poll() is None:
            ssh.terminate()
            ssh.wait()
        if sandbox is not None:
            sandbox.terminate()
        (runtime / "modal-relay.json").unlink(missing_ok=True)


def _terminate(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--port", type=int, default=10613)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--ssh-over-tls", action="store_true", help="Wrap SSH in verified TLS on Modal port 443")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _terminate)
    if args.sandbox:
        _sandbox_main()
    else:
        if args.root is None:
            parser.error("--root is required")
        _run_relay(args.root, args.port, args.timeout, ssh_over_tls=args.ssh_over_tls)


if __name__ == "__main__":
    main()
