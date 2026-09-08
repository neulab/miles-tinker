"""Create a uv environment in the Miles runtime and install this checkout."""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    venv = args.venv.resolve()
    os.environ.setdefault("UV_CACHE_DIR", str(venv.parent / ".cache/uv"))
    subprocess.run(
        ["uv", "venv", "--allow-existing", "--system-site-packages", "--python", sys.executable, str(venv)],
        check=True,
    )
    install = ["uv", "pip", "install", "--python", str(venv / "bin/python")]
    # The runtime supplies the repository's GPU dependencies. Install our source over it.
    subprocess.run([*install, "--no-deps", "--no-build-isolation", "-e", str(repo)], check=True)
    # Same SDK as deploy/abci; retain the repository's dependency constraints.
    subprocess.run([*install, "-c", str(repo / "requirements.txt"), "tinker==0.27.1"], check=True)

    # This runtime's FA4 has no SM8x backward; TE incorrectly selects it over FA2.
    # Correct TE's capability check in the venv, leaving the image unchanged.
    source = Path(importlib.util.find_spec("transformer_engine").origin).parent
    relative = Path("pytorch/attention/dot_product_attention/utils.py")
    if not (source / relative).exists():
        return
    text = (source / relative).read_text()
    old = "# FA4 supports SM80, SM90, SM100, SM120\n    if device_compute_capability < (8, 0):"
    if old in text:
        target = venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/transformer_engine"
        text = text.replace(old, "# FA4 backward requires SM90 or newer.\n    if device_compute_capability < (9, 0):")
        text = text.replace(
            "Disabling FlashAttention 4 for compute capability < sm80",
            "Disabling FlashAttention 4 for compute capability < sm90",
        )
        if not (target / relative).exists() or (target / relative).read_text() != text:
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            (target / relative).write_text(text)
        print("Transformer Engine: FlashAttention 2 on SM8x; newer GPUs retain their backend.", flush=True)


if __name__ == "__main__":
    main()
