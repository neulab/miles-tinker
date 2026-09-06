---
name: upstream-sync
description: "Sync the miles-tinker fork with upstream radixark/miles. Use when the user says 'sync upstream', 'pull upstream miles', 'merge upstream', 'rebase on upstream' or asks whether the fork is behind. Runs tools/fork/upstream_sync.py, resolves conflicts with the FORK.md policy, reruns the fast tests and summarizes."
---

# Upstream Sync

The fork (`FORK.md`) is based on the multi-LoRA v2 PR stack (#2839-#2849,
`upstream/yueming/multi-lora-example`) until that stack lands on
`upstream/main`; then it tracks `upstream/main`. `tools/fork/upstream_sync.py`
decides which base applies from the recorded stack head in
`tools/fork/upstream_state.json` (`git merge-base --is-ancestor`, or the
sticky `merged_into_main: true` flag, which you set by hand after a squash
merge), merges it with `--no-ff`, and records the result. Never sync by hand; the tool keeps the
state file honest.

## Commands

Run everything from a checkout whose current branch is `miles-tinker` (the
main checkout `/home/aci18912it/source/miles`, not an implementer worktree)
with a clean working tree.

```bash
cd /home/aci18912it/source/miles && git switch miles-tinker && git status --short
export MILES_FAST_PYTHON=/tmp/claude-18912/-home-aci18912it/f39085d7-e0f4-47a7-9042-fab61946e8de/scratchpad/miles-fast-venv/bin/python  # any venv with pytest, pytest-asyncio, tinker, fastapi, httpx, numpy, zstandard
python tools/fork/upstream_sync.py --check           # fetch + plan + conflict preview; exit 0 = up to date, 1 = sync needed, 2 = conflicts expected
python tools/fork/upstream_sync.py                   # merge --no-ff, run the fast tests, record the state
python tools/fork/upstream_sync.py --rebase          # only when the user asks for a linear history
python tools/fork/upstream_sync.py --dry-run --check # plan only, no fetch, no writes
```

Exit codes: 0 ok, 1 sync needed (check mode), 2 conflicts, 3 preflight
(wrong branch / dirty tree), 4 merge done but tests failed.

## Conflict-resolution policy

When the tool exits 2 it leaves the merge in progress and prints, per file,
the number of conflict hunks, the upstream and fork shortstats since the merge
base, and the FORK.md rationale for the file. Resolve with:

1. **Files not listed in FORK.md: take upstream.** `git checkout --theirs <file>`
   (merge) or `--ours` under `--rebase`, then re-read the file and make sure no
   fork test in `tests/fast/tinker` or `tests/fast/fork` depends on the old text.
2. **Files listed in FORK.md: re-apply the fork intent on top of upstream.**
   Start from the upstream side, then port the fork hunk described in the
   FORK.md rationale (read `git log -p <merge-base>..HEAD -- <file>` for the
   exact fork change). Keep the fork's tests as the spec for the intent.
3. **Never drop an upstream fix.** If upstream and the fork fixed the same
   thing differently, keep upstream's version and delete the fork's, then note
   it in FORK.md (the delta is gone, say so in the table).
4. **Interfaces in DESIGN.md are the tie-breaker** for `miles/tinker/**`:
   `GatewayConfig` fields, `ExecutorBackend` methods, checkpoint sidecar keys
   and the metrics on the wire must keep their fork names until upstream ships
   an equivalent.
5. Finish with `git add <files>` and `git merge --continue` (or
   `git rebase --continue`), then rerun the tool once more so it runs the tests
   and writes `upstream_state.json`.

## After a clean merge

- Read the printed test summary; every listed dir must be `exit 0`.
- If tests fail, fix them on the fork branch in a follow-up commit (do not
  amend the merge commit) and rerun `python tools/fork/upstream_sync.py --check`
  to confirm exit 0.
- Update `FORK.md` when a fork delta became redundant (upstream shipped it) or
  when a resolution changed a listed file, and bump the "Base" line.
- Commit `tools/fork/upstream_state.json` and any `FORK.md` edit with a message
  like `Sync with upstream/<base> (<commit>)`.

## Final checklist

- [ ] `git status --short` is empty and `git branch --show-current` is `miles-tinker`
- [ ] `python tools/fork/upstream_sync.py --check` exits 0
- [ ] fast tests pass: `tests/fast/tinker` and `tests/fast/fork` (report the counts)
- [ ] `tools/fork/upstream_state.json` committed with the new `base_commit` and date
- [ ] `FORK.md` table still describes every fork delta (removed ones deleted, new ones added)
- [ ] summary to the user: base used, whether the stack has merged into main, files that conflicted and how each was resolved, test counts


## Keeping the GitHub mirror branches current

`neulab/miles-tinker` publishes `main` as an untouched mirror of upstream `main`
and `upstream/multi-lora-v2` as the PR-stack head the fork is built on (see
"Repository layout" in FORK.md). After a sync, refresh them from the fork clone:

```bash
python tools/fork/upstream_sync.py --check --push-mirror     # pushes refs/remotes/upstream/main -> origin main (+ the stack head while unmerged)
```

Never commit on `main`; GitHub's `main...miles-tinker` compare view is the fork delta.
