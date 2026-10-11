"""Create the server's uv environment on top of the Miles image and install this checkout.

Runs inside the container with the image's Python. The venv sees the image's
packages (Torch, Megatron, SGLang, Transformer Engine) and adds this checkout
plus the tinker SDK. ``--use-fa2`` shadows Transformer Engine with a copy that
uses FlashAttention 2 on SM8x GPUs; without it any earlier shadow copy is removed.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TE_ATTENTION_UTILS = Path("pytorch/attention/dot_product_attention/utils.py")
FA4_CHECK = "# FA4 supports SM80, SM90, SM100, SM120\n    if device_compute_capability < (8, 0):"
FA4_CHECK_SM90 = "# FA4 backward requires SM90 or newer.\n    if device_compute_capability < (9, 0):"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--use-fa2", action="store_true", help="Patch Transformer Engine to use FA2 on SM8x GPUs.")
    args = parser.parse_args()
    venv = args.venv.resolve()
    site_packages = venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    _create_venv(venv=venv, site_packages=site_packages)
    _install(venv)
    shadow = site_packages / "transformer_engine"
    if args.use_fa2:
        _shadow_transformer_engine(shadow)
    elif shadow.exists():
        shutil.rmtree(shadow)
        print("Transformer Engine: removed the FA2 shadow copy; using the image's build.", flush=True)


def _create_venv(*, venv: Path, site_packages: Path) -> None:
    os.environ.setdefault("UV_CACHE_DIR", str(venv.parent / ".cache/uv"))
    subprocess.run(
        ["uv", "venv", "--allow-existing", "--system-site-packages", "--python", sys.executable, str(venv)],
        check=True,
    )
    # Preserve image dependencies when its Python already belongs to a venv.
    (site_packages / "runtime.pth").write_text(
        "".join(f"import site; site.addsitedir({path!r})\n" for path in site.getsitepackages() if Path(path).is_dir())
    )


def _install(venv: Path) -> None:
    install = ["uv", "pip", "install", "--python", str(venv / "bin/python")]
    # The runtime supplies the repository's GPU dependencies. Install our source over it.
    subprocess.run([*install, "--no-deps", "--no-build-isolation", "-e", str(REPO)], check=True)
    # The gateway's proto codec. uv does not count the image's packages as installed, so pin any
    # dependency the image already has to the image's version; the venv must not shadow them.
    image_constraints = venv / "image-constraints.txt"
    image_constraints.write_text(_image_constraints())
    constraints = ["-c", str(REPO / "requirements.txt"), "-c", str(image_constraints)]
    subprocess.run([*install, *constraints, "tinker==0.26.2"], check=True)


def _image_constraints() -> str:
    """``name==version`` for every package of this (image) Python, the first on sys.path winning."""
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions.setdefault(re.sub(r"[-_.]+", "-", name).lower(), distribution.version)
    return "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))


def _shadow_transformer_engine(target: Path) -> None:
    # This runtime's FA4 has no SM8x backward; TE incorrectly selects it over FA2.
    # Correct TE's capability check in a venv copy, leaving the image unchanged.
    source = Path(importlib.util.find_spec("transformer_engine").origin).parent
    if not (source / TE_ATTENTION_UTILS).exists():
        print("Transformer Engine: no attention backend check found; --use-fa2 has no effect.", flush=True)
        return
    text = (source / TE_ATTENTION_UTILS).read_text()
    if FA4_CHECK not in text:
        print("Transformer Engine: this build does not select FA4 on SM8x; --use-fa2 has no effect.", flush=True)
        return
    text = text.replace(FA4_CHECK, FA4_CHECK_SM90).replace(
        "Disabling FlashAttention 4 for compute capability < sm80",
        "Disabling FlashAttention 4 for compute capability < sm90",
    )
    if not (target / TE_ATTENTION_UTILS).exists() or (target / TE_ATTENTION_UTILS).read_text() != text:
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        (target / TE_ATTENTION_UTILS).write_text(text)
    print("Transformer Engine: FlashAttention 2 on SM8x; newer GPUs retain their backend.", flush=True)


if __name__ == "__main__":
    main()
