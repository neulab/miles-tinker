import json
import subprocess
from pathlib import Path

import pytest

from tools.fork import upstream_sync as sync

FORK_MD = """
| File | Rationale | Upstream-worthy? |
|---|---|---|
| `serve_tinker.py` | train-mode flag | yes |
| `deploy/abci/` | ABCI launcher | no |
| `miles/tinker/runtime.py` (sampling functions) | seed offset | yes |
| `miles/tinker/runtime.py` (MilesBackend) | dp padding | yes |
| `miles/utils/timer.py`, `miles/utils/flops_utils.py` | per-unit MFU | yes |
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, path: str, text: str, message: str) -> str:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    _git(repo, "add", path)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fork checkout with fake ``upstream/main`` and ``upstream/<stack>`` refs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    _commit(repo, "common.txt", "base\n", "base")
    _git(repo, "update-ref", "refs/remotes/upstream/main", "HEAD")
    _git(repo, "checkout", "-q", "-b", "yueming/multi-lora-example")
    stack_head = _commit(repo, "stack.txt", "stack\n", "stack")
    _git(repo, "update-ref", "refs/remotes/upstream/yueming/multi-lora-example", "HEAD")
    _git(repo, "checkout", "-q", "-b", "miles-tinker")
    _commit(repo, "FORK.md", FORK_MD, "fork docs")
    # local branches double as the upstream so `git fetch upstream` is consistent with the refs above
    _git(repo, "remote", "add", "upstream", str(repo))
    state = {
        "remote": "upstream",
        "fork_branch": "miles-tinker",
        "stack": {"prs": "#1", "branch": "yueming/multi-lora-example", "head": stack_head, "merged_into_main": False},
        "last_sync": {},
    }
    (repo / "state.json").write_text(json.dumps(state))
    return repo


def _state_path(repo: Path) -> Path:
    return repo / "state.json"


def _advance_upstream(repo: Path, branch: str, path: str, text: str, message: str) -> str:
    """Commit on the local branch that stands in for ``upstream/<branch>`` and refresh the tracking ref."""
    _git(repo, "checkout", "-q", branch)
    tip = _commit(repo, path, text, message)
    _git(repo, "update-ref", f"refs/remotes/upstream/{branch}", tip)
    _git(repo, "checkout", "-q", "miles-tinker")
    return tip


def test_choose_base_follows_the_stack_until_it_merges():
    assert sync.choose_base("upstream", "main", "stack", stack_merged=False) == "upstream/stack"
    assert sync.choose_base("upstream", "main", "stack", stack_merged=True) == "upstream/main"


def test_merge_message_names_the_base_and_commit():
    message = sync.merge_message("upstream/main", "0123456789abcdef", stack_merged=True)
    assert message.startswith("Merge upstream/main (0123456789ab) into the fork")
    assert "FORK.md" in message


def test_fork_rationale_matches_files_and_directory_prefixes():
    found = sync.fork_rationale(FORK_MD, ("serve_tinker.py", "deploy/abci/x.pbs", "miles/other.py"))
    assert found == {"serve_tinker.py": "train-mode flag", "deploy/abci/x.pbs": "ABCI launcher"}


def test_fork_rationale_reads_qualified_and_multi_path_rows():
    found = sync.fork_rationale(FORK_MD, ("miles/tinker/runtime.py", "miles/utils/flops_utils.py"))
    assert found == {
        "miles/tinker/runtime.py": "seed offset | dp padding",
        "miles/utils/flops_utils.py": "per-unit MFU",
    }


def test_fork_rationale_covers_every_row_of_the_real_fork_md():
    fork_md = (Path(__file__).resolve().parents[3] / "FORK.md").read_text()
    listed = [row for row in fork_md.splitlines() if row.startswith("| `")]
    assert listed, "FORK.md tables not found"
    sample = (
        "miles/tinker/runtime.py",
        "miles/utils/timer.py",
        "tests/fast/tinker/test_runtime.py",
        "deploy/abci/x.pbs",
    )
    assert set(sync.fork_rationale(fork_md, sample)) == set(sample)


def test_parse_pytest_summary_reads_the_last_summary_line():
    output = "collected 3 items\n...\n=== 2 passed, 1 failed, 1 skipped in 0.1s ===\n"
    assert sync.parse_pytest_summary(output) == {"passed": 2, "failed": 1, "skipped": 1}
    assert sync.parse_pytest_summary("no tests ran") == {}


def test_count_conflict_hunks_counts_markers():
    text = "a\n<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> base\n<<<<<<< HEAD\n"
    assert sync.count_conflict_hunks(text) == 2


def test_test_command_uses_the_confcutdir_trick():
    command = sync.test_command("/venv/bin/python", "tests/fast/tinker")
    assert command[:3] == ["/venv/bin/python", "-m", "pytest"]
    assert "--confcutdir=tests/fast/tinker" in command and "addopts=" in command


def test_plan_tracks_the_stack_branch_while_unmerged(repo: Path):
    state = sync.load_state(_state_path(repo))
    plan = sync.make_plan(repo, state, "upstream", _git(repo, "rev-parse", "HEAD"))
    assert plan.base_ref == "upstream/yueming/multi-lora-example"
    assert not plan.stack_merged and plan.up_to_date and plan.conflicts == ()


def test_plan_switches_to_main_once_the_stack_merged(repo: Path):
    _git(repo, "branch", "-f", "main", "yueming/multi-lora-example")
    _advance_upstream(repo, "main", "common.txt", "upstream fix\n", "upstream fix")

    state = sync.load_state(_state_path(repo))
    plan = sync.make_plan(repo, state, "upstream", _git(repo, "rev-parse", "HEAD"))
    assert plan.stack_merged and plan.base_ref == "upstream/main" and not plan.up_to_date
    assert plan.conflicts == ()


def test_plan_honours_a_recorded_squash_merge(repo: Path):
    """After a squash merge the stack head is never an ancestor of main; the state flag is the override."""
    _advance_upstream(repo, "main", "common.txt", "squashed stack\n", "squash merge of the stack")
    state = sync.load_state(_state_path(repo))
    assert not sync.make_plan(repo, state, "upstream", _git(repo, "rev-parse", "HEAD")).stack_merged

    state["stack"]["merged_into_main"] = True
    plan = sync.make_plan(repo, state, "upstream", _git(repo, "rev-parse", "HEAD"))
    assert plan.stack_merged and plan.base_ref == "upstream/main" and not plan.up_to_date


def test_conflict_report_names_the_rebase_continue_command(tmp_path: Path):
    report = sync.conflict_report(tmp_path, (), "base", "upstream/main", "", rebase=True)
    assert "rebasing onto upstream/main" in report and "git rebase --continue" in report


def test_preview_conflicts_names_the_conflicting_file(repo: Path):
    _advance_upstream(repo, "yueming/multi-lora-example", "common.txt", "upstream side\n", "upstream edit")
    _commit(repo, "common.txt", "fork side\n", "fork edit")

    head = _git(repo, "rev-parse", "HEAD")
    assert sync.preview_conflicts(repo, head, "upstream/yueming/multi-lora-example") == ("common.txt",)


def test_dry_run_check_does_not_touch_the_repo(repo: Path, capsys):
    _advance_upstream(repo, "yueming/multi-lora-example", "stack.txt", "stack v2\n", "stack update")
    before_head, before_state = _git(repo, "rev-parse", "HEAD"), _state_path(repo).read_text()

    code = sync.main(["--repo", str(repo), "--state", str(_state_path(repo)), "--dry-run", "--check"])

    assert code == sync.EXIT_SYNC_NEEDED
    assert "would merge --no-ff upstream/yueming/multi-lora-example into miles-tinker" in capsys.readouterr().out
    assert _git(repo, "rev-parse", "HEAD") == before_head and _state_path(repo).read_text() == before_state


def test_merge_records_the_new_state(repo: Path):
    tip = _advance_upstream(repo, "yueming/multi-lora-example", "stack.txt", "stack v2\n", "stack update")

    code = sync.main(["--repo", str(repo), "--state", str(_state_path(repo)), "--no-tests"])

    assert code == sync.EXIT_OK
    assert _git(repo, "log", "-1", "--format=%s").startswith("Merge upstream/yueming/multi-lora-example")
    recorded = json.loads(_state_path(repo).read_text())["last_sync"]
    assert recorded["base_commit"] == tip and recorded["tests"] == "skipped"
    assert (repo / "stack.txt").read_text() == "stack v2\n"


def test_conflict_stops_with_a_report(repo: Path, capsys):
    _advance_upstream(repo, "yueming/multi-lora-example", "common.txt", "upstream side\n", "upstream edit")
    _commit(repo, "common.txt", "fork side\n", "fork edit")

    code = sync.main(["--repo", str(repo), "--state", str(_state_path(repo)), "--no-tests"])

    out = capsys.readouterr().out
    assert code == sync.EXIT_CONFLICT
    assert "CONFLICTS (1 files)" in out and "common.txt: 1 conflict hunk(s)" in out
    assert "not listed -> prefer upstream" in out


def test_mirror_refspecs_publish_main_and_the_unmerged_stack_head():
    assert sync.mirror_refspecs("upstream", "main", "yueming/multi-lora-example", stack_merged=False) == (
        "refs/remotes/upstream/main:refs/heads/main",
        "refs/remotes/upstream/yueming/multi-lora-example:refs/heads/upstream/multi-lora-v2",
    )
    assert sync.mirror_refspecs("upstream", "main", "yueming/multi-lora-example", stack_merged=True) == (
        "refs/remotes/upstream/main:refs/heads/main",
    )
