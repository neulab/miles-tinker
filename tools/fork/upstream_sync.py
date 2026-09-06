"""Sync the miles-tinker fork with upstream radixark/miles.

The fork tracks the multi-LoRA v2 PR stack (#2839-#2849, branch
``yueming/multi-lora-example``) until it lands on ``upstream/main``; after that
it tracks ``upstream/main`` directly. This tool picks the right base, merges
(or rebases) it into the fork branch, explains conflicts in terms of FORK.md,
reruns the fast Tinker tests and records the result in ``upstream_state.json``.

Stdlib only; run it from the checkout whose current branch is the fork branch.

Args:
  --repo: git checkout to operate on (default: the repo containing this file).
  --branch: fork branch that must be checked out (default: from the state file).
  --remote: upstream remote name (default: from the state file).
  --rebase: ``git rebase`` onto the base instead of ``git merge --no-ff``.
  --check: report only (fetch, choose the base, preview conflicts); exit 0 when
    the fork already contains the base, 1 when a sync is needed, 2 when the
    preview shows conflicts.
  --dry-run: never run a mutating git command (no fetch, merge, state write).
  --push-mirror: after the fetch, push upstream ``main`` to the fork's origin
    ``main`` (an untouched mirror) and the unmerged stack head to
    ``upstream/multi-lora-v2``, so GitHub shows the fork delta.
  --no-tests: skip the fast test run after a clean merge.
  --python: interpreter for the tests (default: $MILES_FAST_PYTHON or this one).
  --state: path of the state json (default: tools/fork/upstream_state.json).

Example:
  MILES_FAST_PYTHON=/path/to/venv/bin/python python tools/fork/upstream_sync.py
  python tools/fork/upstream_sync.py --check
  python tools/fork/upstream_sync.py --dry-run --check
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "upstream_state.json"
FAST_TEST_DIRS = ("tests/fast/tinker", "tests/fast/fork")
EXIT_OK, EXIT_SYNC_NEEDED, EXIT_CONFLICT, EXIT_PREFLIGHT, EXIT_TESTS_FAILED = 0, 1, 2, 3, 4

_PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")
_CONFLICT_MARKER = re.compile(r"^<{7} ", re.MULTILINE)
_BACKTICKED = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class SyncPlan:
    base_ref: str
    base_commit: str
    stack_merged: bool
    up_to_date: bool
    conflicts: tuple[str, ...]


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}):\n{result.stderr.strip()}")
    return result.stdout.strip()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant], capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git merge-base --is-ancestor {ancestor} {descendant}: {result.stderr.strip()}")
    return result.returncode == 0


def load_state(path: Path) -> dict:
    return json.loads(path.read_text())


def choose_base(remote: str, main_branch: str, stack_branch: str, stack_merged: bool) -> str:
    """The stack branch while the PR stack is open, ``main`` once it has merged."""
    return f"{remote}/{main_branch}" if stack_merged else f"{remote}/{stack_branch}"


def merge_message(base_ref: str, base_commit: str, stack_merged: bool) -> str:
    tracking = "upstream main (multi-lora v2 stack merged)" if stack_merged else "the multi-lora v2 PR stack"
    return f"Merge {base_ref} ({base_commit[:12]}) into the fork\n\nSync with {tracking}; see FORK.md for the fork deltas."


def preview_conflicts(repo: Path, head: str, base_ref: str) -> tuple[str, ...]:
    """Files ``git merge-tree`` reports as conflicting, without touching the working tree."""
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-tree", "--write-tree", "--name-only", head, base_ref],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return ()
    if result.returncode != 1:
        raise RuntimeError(f"git merge-tree failed: {result.stderr.strip()}")
    # first line is the tree oid, then the conflicted paths, then a blank line and informational messages
    paths = []
    for line in result.stdout.splitlines()[1:]:
        if not line:
            break
        paths.append(line)
    return tuple(paths)


def fork_rationale(fork_md: str, paths: tuple[str, ...]) -> dict[str, str]:
    r"""FORK.md table rows (``| `path` | rationale | ... |``) keyed by path, for the paths given.

    A row's first cell may list several backticked paths and carry a qualifier
    (``\`a.py\`, \`b.py\``, ``\`runtime.py\` (sampling functions)``); every path of
    every matching row contributes, so a file with two rows gets both rationales.
    """
    rows = []
    for line in fork_md.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0].startswith("`"):
            rows.append((tuple(_BACKTICKED.findall(cells[0])), cells[1]))
    rationale = {}
    for path in paths:
        found = [
            why
            for listed_paths, why in rows
            if any(path == listed or (listed.endswith("/") and path.startswith(listed)) for listed in listed_paths)
        ]
        if found:
            rationale[path] = " | ".join(found)
    return rationale


def count_conflict_hunks(text: str) -> int:
    return len(_CONFLICT_MARKER.findall(text))


def conflict_report(
    repo: Path, files: tuple[str, ...], merge_base: str, base_ref: str, fork_md: str, rebase: bool = False
) -> str:
    rationale = fork_rationale(fork_md, files)
    action, resume = ("rebasing onto", "git rebase --continue") if rebase else ("merging", "git merge --continue")
    lines = [f"CONFLICTS ({len(files)} files) {action} {base_ref}; resolve, `git add`, then `{resume}`:"]
    for path in files:
        full = repo / path
        hunks = count_conflict_hunks(full.read_text(errors="replace")) if full.is_file() else 0
        upstream = git(repo, "diff", "--shortstat", f"{merge_base}..{base_ref}", "--", path, check=False) or "n/a"
        fork = git(repo, "diff", "--shortstat", f"{merge_base}..HEAD", "--", path, check=False) or "n/a"
        lines.append(f"- {path}: {hunks} conflict hunk(s)")
        lines.append(f"    upstream side: {upstream.strip()}")
        lines.append(f"    fork side:     {fork.strip()}")
        why = rationale.get(path)
        lines.append(f"    FORK.md: {why}" if why else "    FORK.md: not listed -> prefer upstream, keep any fork test passing")
    lines.append("Policy: keep every upstream fix; re-apply the fork intent for files FORK.md lists; prefer upstream otherwise.")
    return "\n".join(lines)


def parse_pytest_summary(output: str) -> dict[str, int]:
    """The ``N passed, M failed`` counts of the last pytest summary line."""
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            counts = {kind.rstrip("s"): int(n) for n, kind in _PYTEST_SUMMARY.findall(line)}
            if counts:
                return counts
    return {}


def test_command(python: str, test_dir: str) -> list[str]:
    return [python, "-m", "pytest", test_dir, "-p", "no:cacheprovider", f"--confcutdir={test_dir}", "-o", "addopts="]


def run_fast_tests(repo: Path, python: str, test_dirs: tuple[str, ...]) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
    summaries, ok = [], True
    for test_dir in test_dirs:
        if not (repo / test_dir).is_dir():
            continue
        result = subprocess.run(test_command(python, test_dir), cwd=repo, env=env, capture_output=True, text=True)
        counts = parse_pytest_summary(result.stdout + result.stderr)
        ok = ok and result.returncode == 0
        summaries.append(f"{test_dir}: exit {result.returncode} {counts or result.stderr.strip()[-200:]}")
    return ok, "\n".join(summaries)


def preflight(repo: Path, branch: str) -> list[str]:
    problems = []
    if git(repo, "branch", "--show-current") != branch:
        problems.append(f"current branch is not {branch} (check it out first)")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        problems.append("working tree has uncommitted changes")
    return problems


def make_plan(repo: Path, state: dict, remote: str, head: str) -> SyncPlan:
    """``merged_into_main`` in the state file is sticky: once true (recorded by a sync, or set by hand
    after a squash merge whose head never becomes an ancestor of main) the stack branch is not consulted."""
    stack = state["stack"]
    stack_merged = bool(stack.get("merged_into_main")) or is_ancestor(repo, stack["head"], f"{remote}/main")
    base_ref = choose_base(remote, "main", stack["branch"], stack_merged)
    base_commit = git(repo, "rev-parse", base_ref)
    up_to_date = is_ancestor(repo, base_commit, head)
    conflicts = () if up_to_date else preview_conflicts(repo, head, base_ref)
    return SyncPlan(base_ref, base_commit, stack_merged, up_to_date, conflicts)


def describe_plan(plan: SyncPlan, branch: str, rebase: bool) -> str:
    action = "rebase onto" if rebase else "merge --no-ff"
    lines = [
        f"stack merged into main: {plan.stack_merged}",
        f"sync base: {plan.base_ref} @ {plan.base_commit[:12]}",
        "fork already contains the base: nothing to do" if plan.up_to_date else f"would {action} {plan.base_ref} into {branch}",
    ]
    if plan.conflicts:
        lines.append(f"expected conflicts ({len(plan.conflicts)}): " + ", ".join(plan.conflicts))
    return "\n".join(lines)


def record_state(path: Path, state: dict, plan: SyncPlan, fork_commit: str, tests: str) -> None:
    updated = dict(state)
    updated["stack"] = {**state["stack"], "merged_into_main": plan.stack_merged}
    updated["last_sync"] = {
        "base_ref": plan.base_ref,
        "base_commit": plan.base_commit,
        "fork_commit": fork_commit,
        "date": dt.date.today().isoformat(),
        "tests": tests,
    }
    path.write_text(json.dumps(updated, indent=2) + "\n")


def mirror_refspecs(remote: str, main_branch: str, stack_branch: str, stack_merged: bool) -> tuple[str, ...]:
    """Ref specs that publish the upstream branches the fork is measured against:
    ``main`` as an untouched mirror of upstream main (GitHub's compare view then
    shows the fork delta) and, while the PR stack is unmerged, its head under
    ``upstream/multi-lora-v2``."""
    specs = [f"refs/remotes/{remote}/{main_branch}:refs/heads/{main_branch}"]
    if not stack_merged:
        specs.append(f"refs/remotes/{remote}/{stack_branch}:refs/heads/upstream/multi-lora-v2")
    return tuple(specs)


def push_mirror(repo: Path, origin: str, refspecs: tuple[str, ...]) -> None:
    for spec in refspecs:
        git(repo, "push", origin, spec)
        print(f"mirrored {spec} -> {origin}")


def integrate(repo: Path, plan: SyncPlan, rebase: bool) -> int:
    if rebase:
        return subprocess.run(["git", "-C", str(repo), "rebase", plan.base_ref]).returncode
    message = merge_message(plan.base_ref, plan.base_commit, plan.stack_merged)
    return subprocess.run(["git", "-C", str(repo), "merge", "--no-ff", "-m", message, plan.base_ref]).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--remote", default=None)
    parser.add_argument("--rebase", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument(
        "--push-mirror",
        action="store_true",
        help="also push upstream main (and the unmerged stack head) to the fork's origin as mirror branches",
    )
    parser.add_argument("--origin", default="origin", help="remote that holds the fork (for --push-mirror)")
    parser.add_argument("--python", default=os.environ.get("MILES_FAST_PYTHON", sys.executable))
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    state = load_state(args.state)
    branch = args.branch or state["fork_branch"]
    remote = args.remote or state["remote"]

    if not args.dry_run:
        git(repo, "fetch", "--quiet", remote)
    head = git(repo, "rev-parse", "HEAD")
    plan = make_plan(repo, state, remote, head)
    print(describe_plan(plan, branch, args.rebase))
    if args.push_mirror and not args.dry_run:
        push_mirror(
            repo,
            args.origin,
            mirror_refspecs(remote, "main", state["stack"]["branch"], plan.stack_merged),
        )
    if plan.up_to_date:
        return EXIT_OK
    if args.check or args.dry_run:
        return EXIT_CONFLICT if plan.conflicts else EXIT_SYNC_NEEDED

    problems = preflight(repo, branch)
    if problems:
        print("preflight failed: " + "; ".join(problems))
        return EXIT_PREFLIGHT
    merge_base = git(repo, "merge-base", head, plan.base_ref)
    if integrate(repo, plan, args.rebase) != 0:
        files = tuple(git(repo, "diff", "--name-only", "--diff-filter=U").splitlines())
        fork_md = (repo / "FORK.md").read_text() if (repo / "FORK.md").is_file() else ""
        print(conflict_report(repo, files, merge_base, plan.base_ref, fork_md, rebase=args.rebase))
        return EXIT_CONFLICT

    tests = "skipped"
    tests_ok = True
    if not args.no_tests:
        tests_ok, tests = run_fast_tests(repo, args.python, FAST_TEST_DIRS)
        print(tests)
    record_state(args.state, state, plan, git(repo, "rev-parse", "HEAD"), tests)
    print(f"state recorded in {args.state}; commit it with the merge follow-ups")
    return EXIT_OK if tests_ok else EXIT_TESTS_FAILED


if __name__ == "__main__":
    sys.exit(main())
