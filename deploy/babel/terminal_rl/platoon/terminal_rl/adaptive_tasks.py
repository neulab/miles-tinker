"""Adaptive task selection: draw training tasks by how learnable they currently are.

Ported from science-agent's ``science_rl/adaptive_selection.py`` (branch
training/qwen38-adaptive), single source. With PLATOON_TASK_SELECTION=adaptive
the training task loader asks this module for each batch instead of walking a
fixed shuffled list:

* every task keeps a history of scored groups; its pass rate is Beta-smoothed,
  ``p = (passes + 1) / (attempts + 2)``, and its weight is the learnability
  ``4 p (1 - p)`` (at most 1, at p = 0.5) with a floor. An unseen task has
  p = 0.5, so the whole pool is tried before anything is revisited much;
* a task whose last group all passed cools down for COOLDOWN_PASS dispatched
  groups, one that all failed for COOLDOWN_FAIL, one lost to infrastructure for
  COOLDOWN_INFRA; a cooling task keeps only the floor weight;
* EXPLORE of the draws are uniform over the available tasks, so mastered and
  hopeless tasks are still rechecked now and then;
* a task already in flight is not drawn again until its group is observed (or
  STALE dispatches have passed, in case its episode died without a result).

"Pass" is the verifier's task reward >= 0.99, whatever the training reward
scheme does with failures (penalized scores them 0 or -1; both count 0 here).
Only training groups with >= 2 scored rollouts update the history; fewer is an
infrastructure loss. Groups are observed before any filtering (step-limit drop,
all-equal), so filtered groups still inform the sampler.

The state (histories, dispatch count, RNG) is written next to every checkpoint
as task-selection/<name>.json and read back on resume, so a resumed run picks
up the same sampler. Active sampling still works on top: it keeps asking the
loader for groups until it has a batch with a gradient, and this module decides
which tasks those are -- favouring tasks likely to give a gradient.

A prior (PLATOON_ADAPTIVE_PRIOR, a TSV with task / solved / rollouts columns, as
medium_pool_rates.tsv) seeds histories, each task's counts scaled down to at
most PRIOR_WEIGHT attempts so online evidence soon dominates.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger("platoon.terminal_rl.adaptive_tasks")


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PLATOON_ADAPTIVE_{name}", default)


def enabled() -> bool:
    return os.environ.get("PLATOON_TASK_SELECTION", "") == "adaptive"


def learnability(row: dict, floor: float) -> float:
    attempts = row.get("attempts", 0)
    value = row.get("reward_sum", row.get("successes", 0))
    p = (value + 1) / (attempts + 2)
    return max(floor, 4 * p * (1 - p))


class AdaptiveTaskSampler:
    def __init__(self, task_ids: list[str], seed: int = 42, state_dir: str | None = None):
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate task ids")
        self.task_ids = sorted(task_ids)
        self.rng = random.Random(seed)
        self.state_dir = Path(state_dir) if state_dir else None
        self.explore = float(_env("EXPLORE", "0.05"))
        self.floor = float(_env("FLOOR", "0.01"))
        self.cooldown_pass = int(_env("COOLDOWN_PASS", "512"))
        self.cooldown_fail = int(_env("COOLDOWN_FAIL", "64"))
        self.cooldown_infra = int(_env("COOLDOWN_INFRA", "256"))
        self.stale = int(_env("STALE", "400"))
        self.history: dict[str, dict] = {}
        self.in_flight: dict[str, int] = {}  # task -> dispatch number
        self.dispatch_count = 0
        self.counts = {"contrast": 0, "all_pass": 0, "all_fail": 0, "infrastructure": 0}

    # --- selection ---------------------------------------------------------
    def p(self, task: str) -> float:
        row = self.history.get(task, {})
        return (row.get("reward_sum", 0.0) + 1) / (row.get("attempts", 0) + 2)

    def weight(self, task: str) -> float:
        row = self.history.get(task, {})
        if self.dispatch_count < row.get("cooldown_until", 0):
            return self.floor
        return learnability(row, self.floor)

    def _release_stale(self) -> None:
        for task, at in list(self.in_flight.items()):
            if self.dispatch_count - at > self.stale:
                del self.in_flight[task]

    def choose(self) -> str:
        self._release_stale()
        available = [t for t in self.task_ids if t not in self.in_flight]
        if not available:
            raise ValueError("every task is in flight")
        if self.rng.random() < self.explore:
            return self.rng.choice(available)
        return self.rng.choices(available, weights=[self.weight(t) for t in available], k=1)[0]

    def draw(self, n: int) -> list[dict]:
        batch = []
        for _ in range(n):
            task = self.choose()
            self.dispatch_count += 1
            self.in_flight[task] = self.dispatch_count
            batch.append({"task_id": task})
        return batch

    # --- outcomes ----------------------------------------------------------
    def observe(self, task: str, rewards: list[float]) -> str:
        """Record one group's outcome; returns its kind."""
        self.in_flight.pop(task, None)
        row = self.history.setdefault(task, {"attempts": 0, "successes": 0, "reward_sum": 0.0, "groups": 0})
        if len(rewards) < 2:
            row["infrastructure"] = row.get("infrastructure", 0) + 1
            row["cooldown_until"] = self.dispatch_count + self.cooldown_infra
            self.counts["infrastructure"] += 1
            return "infrastructure"
        values = [min(1.0, max(0.0, float(r))) for r in rewards]
        passes = sum(v >= 0.99 for v in values)
        row["attempts"] += len(values)
        row["successes"] += passes
        row["reward_sum"] += sum(values)
        row["groups"] += 1
        if passes == len(values):
            kind, cool = "all_pass", self.cooldown_pass
        elif max(values) - min(values) <= 1e-8:
            kind, cool = "all_fail", self.cooldown_fail
        else:
            kind, cool = "contrast", 0
        if cool:
            row["cooldown_until"] = self.dispatch_count + cool
        self.counts[kind] += 1
        return kind

    def summary(self) -> dict[str, float]:
        seen = [t for t in self.task_ids if self.history.get(t, {}).get("attempts")]
        cooling = sum(1 for t in self.task_ids if self.dispatch_count < self.history.get(t, {}).get("cooldown_until", 0))
        out = {
            "adaptive_dispatched": float(self.dispatch_count),
            "adaptive_seen_tasks": float(len(seen)),
            "adaptive_seen_frac": len(seen) / len(self.task_ids),
            "adaptive_cooling_tasks": float(cooling),
        }
        total = sum(self.counts.values())
        for kind, n in self.counts.items():
            out[f"adaptive_frac_{kind}"] = n / total if total else 0.0
        return out

    # --- prior ---------------------------------------------------------------
    def load_prior(self, path: str, weight: float) -> int:
        n = 0
        for line in Path(path).read_text().splitlines():
            cols = line.split("\t")
            # Any id in the pool: the old `startswith("task_")` matched only tmax ids,
            # so priors for prefixed pools (sci-, tg-, fusion-) were silently dropped.
            # A header row is skipped because it names no task in the pool.
            if len(cols) < 3 or cols[0] not in self.task_ids:
                continue
            solved, rollouts = float(cols[1]), float(cols[2])
            if rollouts <= 0:
                continue
            scale = min(1.0, weight / rollouts)
            row = self.history.setdefault(cols[0], {"attempts": 0, "successes": 0, "reward_sum": 0.0, "groups": 0})
            row["attempts"] += rollouts * scale
            row["successes"] += solved * scale
            row["reward_sum"] += solved * scale
            n += 1
        return n

    # --- persistence ---------------------------------------------------------
    def state(self) -> dict[str, Any]:
        version, internal, gauss = self.rng.getstate()
        return {
            "task_ids": self.task_ids,
            "history": self.history,
            "dispatch_count": self.dispatch_count,
            "counts": self.counts,
            "rng": [version, list(internal), gauss],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state["task_ids"] != self.task_ids:
            raise ValueError("adaptive task selection: resumed task membership differs")
        self.history = state["history"]
        self.dispatch_count = state["dispatch_count"]
        self.counts = state["counts"]
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        self.in_flight = {}  # those episodes died with the old job

    def save(self, name: str) -> Path | None:
        if self.state_dir is None:
            return None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / f"{name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state()))
        tmp.replace(path)
        return path


_SAMPLER: AdaptiveTaskSampler | None = None


def sampler() -> AdaptiveTaskSampler | None:
    return _SAMPLER


def _install_checkpoint_hook() -> None:
    """Write the sampler state next to every checkpoint the trainer saves."""
    from tinker_cookbook import checkpoint_utils

    if getattr(checkpoint_utils.save_checkpoint_async, "_adaptive_hook", False):
        return
    original = checkpoint_utils.save_checkpoint_async

    async def save_with_selection(*args, **kwargs):
        result = await original(*args, **kwargs)
        if _SAMPLER is not None and kwargs.get("name"):
            path = _SAMPLER.save(str(kwargs["name"]))
            logger.info(f"[adaptive] task-selection state saved with checkpoint {kwargs['name']}: {path}")
        return result

    save_with_selection._adaptive_hook = True
    checkpoint_utils.save_checkpoint_async = save_with_selection


def draw(trainer: Any, dataloader: Any, n: int) -> list[dict]:
    """The loader's next batch of n tasks; builds (or resumes) the sampler on first use."""
    global _SAMPLER
    if _SAMPLER is None:
        from tinker_cookbook import checkpoint_utils

        task_ids = list(dataloader.dataset["task_id"])
        state_dir = os.path.join(trainer.run_log_path, "task-selection")
        _SAMPLER = AdaptiveTaskSampler(task_ids, seed=int(_env("SEED", "42")), state_dir=state_dir)
        resume = checkpoint_utils.get_last_checkpoint(trainer.run_log_path)
        state_path = Path(state_dir) / f"{resume.name}.json" if resume is not None else None
        if state_path is not None and state_path.is_file():
            _SAMPLER.load_state(json.loads(state_path.read_text()))
            logger.info(f"[adaptive] resumed task selection from {state_path} (dispatch {_SAMPLER.dispatch_count})")
        else:
            prior = _env("PRIOR", "")
            if prior:
                n_prior = _SAMPLER.load_prior(prior, float(_env("PRIOR_WEIGHT", "4")))
                logger.info(f"[adaptive] prior from {prior}: {n_prior} tasks")
            if resume is not None and resume.batch:
                logger.warning(f"[adaptive] resuming at batch {resume.batch} without saved selection state; starting fresh")
        _install_checkpoint_hook()
        logger.info(
            f"[adaptive] task selection over {len(task_ids)} tasks: explore {_SAMPLER.explore}, "
            f"cooldowns pass/fail/infra {_SAMPLER.cooldown_pass}/{_SAMPLER.cooldown_fail}/{_SAMPLER.cooldown_infra}"
        )
    batch = _SAMPLER.draw(n)
    ps = [_SAMPLER.p(d["task_id"]) for d in batch]
    s = _SAMPLER.summary()
    logger.info(
        f"[adaptive] dispatch {_SAMPLER.dispatch_count}: batch mean p {sum(ps) / len(ps):.2f}, "
        f"seen {int(s['adaptive_seen_tasks'])}/{len(_SAMPLER.task_ids)}, cooling {int(s['adaptive_cooling_tasks'])}"
    )
    if _SAMPLER.state_dir is not None:
        _SAMPLER.state_dir.mkdir(parents=True, exist_ok=True)
        with open(_SAMPLER.state_dir / "dispatched.jsonl", "a") as out:
            out.write(json.dumps({"dispatch": _SAMPLER.dispatch_count, "tasks": [d["task_id"] for d in batch],
                                  "p": [round(x, 3) for x in ps]}) + "\n")
    return batch


def observe(task: str, rewards: list[float], tracker: Any = None) -> None:
    """Report one training group's root task rewards (before any filtering)."""
    if _SAMPLER is None:
        return
    kind = _SAMPLER.observe(task, rewards)
    if tracker is not None:
        try:
            tracker.scalar(**_SAMPLER.summary(), adaptive_group_p=_SAMPLER.p(task))
        except Exception as error:  # metrics never break training
            logger.debug(f"[adaptive] metric logging failed: {error}")
    logger.debug(f"[adaptive] {task}: {kind}")
