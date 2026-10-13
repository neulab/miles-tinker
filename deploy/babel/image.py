"""Cache the pinned Miles image as a local SIF for ``server.py --image``.

Runs on the host. ``apptainer build`` downloads, unpacks and converts the layers in a
temporary directory under ``--scratch``; the finished SIF is then moved to ``--output``
and the directory, layer cache included, is deleted. ``image.sbatch`` runs it on a
compute node.

Requires ``apptainer`` on PATH.

Usage: python deploy/babel/image.py --output <sif> [--scratch DIR] [--image URI] [--mksquashfs-mem SIZE]
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# radixark/miles:dev-202610121029 (CUDA 13, SGLang v0.5.21), whose miles.image-inputs label matches
# `python docker/image_inputs.py` for this checkout. The latest tag moves daily; a newer SGLang can
# break the gateway, so update the digest deliberately.
IMAGE = "docker://radixark/miles@sha256:dde749d0e71cc5ed149982f3cffd00cda0f8b9ef942486378aae780550a8b1cd"


def main() -> None:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        sys.exit(f"{output} already exists; pass --force to replace it.")
    args.scratch.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="miles-image-", dir=args.scratch)).resolve()
    try:
        (scratch / "tmp").mkdir()
        (scratch / "cache").mkdir()
        env = {**os.environ, "APPTAINER_TMPDIR": str(scratch / "tmp"), "APPTAINER_CACHEDIR": str(scratch / "cache")}
        sif = scratch / "miles.sif"
        # mksquashfs sizes its caches from the node's RAM and threads from its CPUs, not the job's
        # limits, and gets OOM-killed in a small job; `build`, unlike `pull`, passes it arguments.
        mksquashfs = f"-mem {args.mksquashfs_mem} -processors {len(os.sched_getaffinity(0))}"
        # mksquashfs runs under proot, which fails if it cannot resolve the caller's working directory.
        subprocess.run(
            ["apptainer", "build", "--mksquashfs-args", mksquashfs, str(sif), args.image],
            cwd=scratch,
            env=env,
            check=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # Copy next to the target, then rename, so a reader never sees a partial SIF.
        partial = output.with_name(f"{output.name}.partial")
        shutil.move(str(sif), str(partial))
        os.replace(partial, output)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"Cached {args.image} at {output}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Where to write the SIF.")
    parser.add_argument("--image", default=IMAGE, help="Image URI to pull; defaults to the pinned digest.")
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path("/scratch") / getpass.getuser(),
        help="Local disk for the download, unpacking and conversion; emptied afterwards.",
    )
    parser.add_argument(
        "--mksquashfs-mem",
        default="8G",
        help="mksquashfs cache size (K, M or G); keep it well under the job's memory limit.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing --output.")
    return parser


if __name__ == "__main__":
    main()
