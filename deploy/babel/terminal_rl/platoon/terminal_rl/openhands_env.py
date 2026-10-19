"""Terminal tasks driven by openhands-sdk, so training matches evaluation.

The Vanillux recipes give the model a single ``<bash>`` text action. This gives
it the evaluation's real tool-calling interface instead, which is what makes an
RL checkpoint comparable to a terminal-bench number.

The conversation runs in an openhands-agent-server hosted *inside* the task's
sandbox, and Platoon's OpenHandsEnv drives it through a RemoteWorkspace. That
placement is forced: openhands' TerminalTool spawns a local shell rather than
routing through the workspace object, so an agent hosted in the client process
would run the task's commands on the compute node.

Platoon's OpenHandsEnv already owns the conversation and the trajectory steps;
this adds the two things it has no opinion about -- which sandbox the tools act
on, and the reward. Both come from ApptainerBashExecutor, so the SIF pool,
setup.sh handling and verifier are shared with the Vanillux recipes unchanged.

Importing this module requires openhands-sdk.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import replace

from openhands.sdk.conversation import get_agent_final_response
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.workspace import RemoteWorkspace
from platoon.envs.base import Task
from platoon.episode.context import budget_tracker, current_trajectory, current_trajectory_collection, error_message
from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished

from platoon.terminal_rl.executor import ApptainerBashExecutor

logger = logging.getLogger("platoon.terminal_rl.openhands_env")

# Steps the episode loop keeps in reserve after the agent has been told to stop.
# OpenHandsEnv gives the conversation `task.max_steps` iterations and the
# episode loop is budgeted with the same number, so an agent that uses its whole
# budget ends the episode on the very step that ends the conversation: the loop
# halts before it ever observes a terminal status, evaluate() never fires, and
# the task goes unscored. Holding a few steps back lets the ending be seen.
CONVERSATION_HEADROOM = int(os.getenv("TERMINAL_RL_CONVERSATION_HEADROOM", "6"))

# The agent is otherwise never told how many turns it has: openhands announces no
# iteration limit, and the task text says nothing. Running out without solving
# scores FAILURE_PENALTY while stopping with a wrong answer scores 0, so an agent
# blind to the limit can only learn "stop early" from it, not "pace the work".
# The note goes at the end of the task message; the number is the conversation's
# iteration limit (max_steps - CONVERSATION_HEADROOM), one per model response.
ANNOUNCE_BUDGET = os.getenv("TERMINAL_RL_ANNOUNCE_BUDGET", "1") != "0"
BUDGET_NOTE = (
    "You have a budget of {turns} turns for this task. Each response you send uses one turn, "
    "and the conversation ends when the budget runs out. Plan your work to finish, and call the "
    "finish tool, before it runs out."
)

# A response that ends with a raw "<tool_call>" block is the model emitting a
# tool call as prose instead of a real one: openhands sees text with no tool
# call, classifies it CONTENT, and ends the conversation (response_dispatch
# ._handle_content_response). The episode is then scored on an unfinished task,
# so the failure is a formatting one wearing a task failure's clothes.
#
# Shaping it is worth more than it first appears. GRPO drops any group whose
# rewards are all equal, and 70% of groups in job 10590936 scored uniformly
# zero. Of those, 75% contained a mix of clean and malformed endings -- so this
# term alone takes usable groups from 29/96 to 79/96, without which most of the
# batch produces no gradient at all.
#
# The weight is small on purpose. Advantages here are centred but not
# normalised (GroupRolloutWorkflow: reward - mean_reward), so a 0.1 term moves
# the gradient a fifth as much as solving the task does, and a constant offset
# cancels in the centring. It breaks ties; it should not outrank the task.
TOOL_JSON_REWARD_WEIGHT = float(os.getenv("TERMINAL_RL_TOOL_JSON_REWARD_WEIGHT", "0.1"))
TOOL_CALL_MARKERS = ("<tool_call>", "</tool_call>")

# Bounds on the work close() does. run_episode calls env.close() inside its own
# finally, so whatever happens here is still inside the rollout task and is not
# covered by the rollout's own timeout -- an unbounded close let one rollout run
# 1942s against a 700s limit, and since the collection phase ends with its
# slowest member that one rollout set the whole step's collection at 1858s.
#
# Both bounds are generous against the healthy case: a verifier's pytest runs in
# about 0.03s and closing a live conversation is an HTTP call. They only bite
# when the sandbox or the agent-server is already gone, where waiting cannot
# produce a reward anyway.
CLOSE_SCORE_TIMEOUT = float(os.getenv("TERMINAL_RL_CLOSE_SCORE_TIMEOUT", "180"))
CLOSE_TEARDOWN_TIMEOUT = float(os.getenv("TERMINAL_RL_CLOSE_TEARDOWN_TIMEOUT", "60"))

# Openhands ends a conversation the moment the model returns prose with no tool
# call (response_dispatch._handle_content_response), so an agent that narrates
# instead of acting has its episode closed mid-task and scored on an untouched
# filesystem. Calling the finish tool is the only ending that means "I am done"
# rather than "I stopped talking", and the policy does it in 9 of 4281 rollouts
# -- 0.21%.
#
# That rarity is the catch. Advantages are centred, not normalised, so a term
# that is zero for every member of a group cancels; at 0.21% roughly 98% of
# groups see no gradient from this at all. It pays only in the rare group where
# one rollout finishes properly, and relies on that bootstrapping the behaviour.
# The tool-json term above is what keeps the other groups alive meanwhile.
FINISH_REWARD_WEIGHT = float(os.getenv("TERMINAL_RL_FINISH_REWARD_WEIGHT", "0.1"))
# "gated" scores a malformed final response 0 and otherwise pays
# GATED_FLOOR for a well-formed attempt, rising to 1.0 when the task is solved.
# "linear" restores the additive weights above.
# "penalized" scores a malformed final response, or a run into the step limit
# that did not solve the task, FAILURE_PENALTY, and otherwise the task's own
# reward (the verifier's 0/1, or the pass rate under
# TERMINAL_RL_REWARD_MODE=passrate). A solved run that ran out, or whose final
# response is malformed, keeps its 1.0.
REWARD_SCHEME = os.getenv("TERMINAL_RL_REWARD_SCHEME", "gated")
GATED_FLOOR = float(os.getenv("TERMINAL_RL_GATED_FLOOR", "0.2"))
FAILURE_PENALTY = float(os.getenv("TERMINAL_RL_FAILURE_PENALTY", "-1.0"))
# What an unsolved run that hit the step limit scores; defaults to FAILURE_PENALTY.
# The step-limit -1 lands on the longest trajectories, concentrated on the hardest
# tasks (155841: top 10 of 32 tasks held 59% of step-limit rollouts), and pushing
# those long sequences down spreads probability mass onto everything else (sign
# test: never-trained text lost 0.05 nats/token). 0 scores it like a wrong finish.
# "drop" leaves those rollouts out of the training group altogether (no gradient;
# patches/drop_step_limit_rollouts.py); they score 0 for logging.
_STEP_LIMIT_RAW = os.getenv("TERMINAL_RL_STEP_LIMIT_PENALTY", str(FAILURE_PENALTY))
STEP_LIMIT_PENALTY = 0.0 if _STEP_LIMIT_RAW == "drop" else float(_STEP_LIMIT_RAW)
# Output-length penalty (0 = off): the trajectory's longest completion, n tokens
# with reasoning, costs LENGTH_PENALTY_ALPHA * ln(n / N) when n > N, subtracted
# from whatever the trajectory scores and floored at FAILURE_PENALTY, so a solved
# run scores 1 - alpha * ln(n / N). Replaces a hard cap on tool calls per turn:
# without one, openhands runs every call in a response for one step, and the
# policy packed calls (156019) and fell into repetition loops (156562) up to the
# 16k token cap. Subtracted rather than multiplied so long unsolved runs are
# penalised too -- the loops sat mostly in failures -- and so a negative score is
# never shrunk toward zero by length.
LENGTH_PENALTY_N = int(os.getenv("TERMINAL_RL_LENGTH_PENALTY_N", "0"))
LENGTH_PENALTY_ALPHA = float(os.getenv("TERMINAL_RL_LENGTH_PENALTY_ALPHA", "0.3"))
# "trajectory" (default): subtract the penalty from the trajectory reward.
# "targeted": leave the reward alone; the train loop puts the penalty on each long
# response's overflow tokens (patches/targeted_length_credit.py).
LENGTH_PENALTY_CREDIT = os.getenv("TERMINAL_RL_LENGTH_PENALTY_CREDIT", "trajectory")


def length_penalty(response_tokens: list[int]) -> tuple[float, int]:
    """(penalty, longest completion in tokens) for one trajectory."""
    longest = max(response_tokens, default=0)
    if LENGTH_PENALTY_N <= 0 or longest <= LENGTH_PENALTY_N:
        return 0.0, longest
    return LENGTH_PENALTY_ALPHA * math.log(longest / LENGTH_PENALTY_N), longest


# "trajectory" (default): a malformed unsolved ending scores the whole trajectory
# FAILURE_PENALTY. "targeted": the trajectory keeps its task score and the train
# loop puts a fixed penalty on the tokens where the final response broke
# (malformed_break.py, patches/targeted_malformed_credit.py), instead of spreading
# -1 over every action token of the rollout. "off": no malformed penalty; a malformed
# ending scores its task reward like any other (targeted's per-token penalty reached
# |advantage| ~7,700 on a 3-token span and dominated clipped steps).
MALFORMED_CREDIT = os.getenv("TERMINAL_RL_MALFORMED_CREDIT", "trajectory")
# OpenHands' code for a conversation stopped by its own iteration limit.
MAX_ITERATIONS_CODE = "MaxIterationsReached"
FINISH_TOOL_NAME = "finish"

# The shaping terms are carved out of the task reward rather than added on top,
# so a perfect rollout still scores exactly 1.0 and the reward keeps the scale
# the metrics, the checkpoints and any comparison against earlier runs assume.
# Solving still dominates: the worst solved rollout scores TASK_REWARD_WEIGHT
# (0.8), the best unsolved one scores the shaping total (0.2), so no amount of
# well-formed stopping can outrank finishing the task.
TASK_REWARD_WEIGHT = float(
    os.getenv(
        "TERMINAL_RL_TASK_REWARD_WEIGHT",
        str(max(0.0, 1.0 - TOOL_JSON_REWARD_WEIGHT - FINISH_REWARD_WEIGHT)),
    )
)


class OpenHandsTerminalEnv(OpenHandsEnv):
    """An openhands conversation in one task's sandbox, scored by its verifier."""

    def __init__(
        self,
        task: Task,
        agent,
        executor: ApptainerBashExecutor,
        server_url: str,
        working_dir: str = "/home/user",
        response_tokens: list[int] | None = None,
    ):
        self._executor = executor
        # Filled by the proxy bridge, one entry per completion (length penalty).
        self._response_tokens = response_tokens if response_tokens is not None else []
        self._scored = False
        # The sandbox and its agent-server must both be up before the workspace
        # can point at them, so the rollout starts them first; see
        # openhands_rollout.run_rollout.
        super().__init__(
            task=task,
            agent=agent,
            workspace=RemoteWorkspace(host=server_url, working_dir=working_dir),
        )

    # Statuses that mean the agent has stopped acting, whatever the reason.
    # An agent that exhausted its iteration budget still earns whatever its
    # work is worth, so MaxIterationsReached is scored like any other ending.
    _TERMINAL = (
        ConversationExecutionStatus.FINISHED,
        ConversationExecutionStatus.STUCK,
        ConversationExecutionStatus.ERROR,
    )

    #: Set once close() starts; the rollout watchdog reads it.
    closing = False

    def _agent_stopped(self) -> bool:
        """Whether the conversation has ended, by Platoon's test or the SDK's.

        Platoon's is_finished also demands that the episode loop has caught up
        with the conversation's last event. Against a remote conversation the
        final action often has no observation after it, so that never becomes
        true and the episode instead runs out its step budget -- leaving the
        verifier unrun and every reward at zero. The agent having stopped is
        the condition the verifier actually cares about.
        """
        if is_finished(self._state):
            return True
        status = getattr(self._state.conversation_state, "execution_status", None)
        return status in self._TERMINAL

    async def reset(self):
        """Start the conversation with fewer iterations than the episode has steps.

        The reduced budget goes on a *copy* of the task. A GRPO group hands the
        same Task object to all of its rollouts, so subtracting the headroom
        from it in place compounds once per concurrent rollout -- eight members
        took 30 down to 1, and every agent stopped after a single action.

        OpenHandsEnv.reset also attaches the task it was given to the
        trajectory, and the episode's StepBudgetTracker reads
        ``trajectory.task.max_steps`` at every check rather than once up front.
        Left there, the copy put the episode loop on the conversation's budget
        too (both 94 of 100): the loop halted after 93 actions, sometimes while
        the conversation was still running its last one, and close() scored the
        task under it. The original goes back on the trajectory, so the
        conversation's limit is the only one the agent meets.

        With ANNOUNCE_BUDGET the copy's goal also tells the agent that limit.
        """
        original = self._task
        budget = original.max_steps
        if budget:
            turns = max(1, budget - CONVERSATION_HEADROOM)
            goal = original.goal
            if ANNOUNCE_BUDGET and goal:
                goal = f"{goal}\n\n{BUDGET_NOTE.format(turns=turns)}"
            self._task = replace(original, max_steps=turns, goal=goal)
        try:
            return await super().reset()
        finally:
            self._task = original
            trajectory = current_trajectory.get(None)
            collection = current_trajectory_collection.get(None)
            if budget and trajectory is not None and collection is not None:
                collection.set_trajectory_task(trajectory.id, original)

    def _tool_json_reward(self) -> tuple[float, dict]:
        """1.0 unless the agent's final response carries a raw tool-call block.

        An empty or missing final response scores 1.0: there is no malformed
        output to punish, and the episode ended some other way (finish tool,
        step timeout, a dead server).
        """
        try:
            final = get_agent_final_response(self._conversation.state.events) or ""
        except Exception as error:  # a lost conversation must not fail scoring
            logger.warning("[%s] could not read final response: %s", self._task.id, error)
            return 1.0, {"tool_json_error_reward": 1.0, "tool_json_unavailable": True}
        if not final.strip():
            return 1.0, {"tool_json_error_reward": 1.0}
        reward = 0.0 if any(m in final for m in TOOL_CALL_MARKERS) else 1.0
        return reward, {"tool_json_error_reward": reward}

    def _finish_tool_reward(self) -> tuple[float, dict]:
        """1.0 when the agent ended by calling the finish tool, else 0.0.

        Finish is terminal in openhands -- ``_ActionBatch.finalize`` marks the
        conversation FINISHED only when the batch contains it -- so calling it
        at all means calling it last.
        """
        try:
            events = self._conversation.state.events
        except Exception as error:  # a lost conversation must not fail scoring
            logger.warning("[%s] could not read events: %s", self._task.id, error)
            return 0.0, {"finish_tool_reward": 0.0, "finish_tool_unavailable": True}
        called = any(getattr(event, "tool_name", None) == FINISH_TOOL_NAME for event in events)
        reward = 1.0 if called else 0.0
        return reward, {"finish_tool_reward": reward}

    def _step_limit_reward(self) -> tuple[float, dict]:
        """1.0 when the agent ran out of steps rather than stopping on its own.

        Two limits can end a run: the conversation's iteration limit
        (max_steps - CONVERSATION_HEADROOM), which openhands reports as a
        MaxIterationsReached error event and which then scores like any other
        ending, and the episode loop's step budget, which halts the episode and
        leaves scoring to close(). In the batch-32 FLAME run the first ended 776
        of 2958 rollouts and the second 280, all scored as ordinary attempts.
        """
        hit = False
        try:
            hit = any(getattr(e, "code", None) == MAX_ITERATIONS_CODE for e in self._conversation.state.events)
        except Exception as error:  # a lost conversation must not fail scoring
            logger.warning("[%s] could not read events for the step limit: %s", self._task.id, error)
        if not hit:
            hit = "Exhausted budget" in (error_message.get(None) or "")
        if not hit:
            try:
                hit = budget_tracker.get().remaining_budget() <= 0
            except Exception:  # no episode context, e.g. a direct evaluate() call
                pass
        reward = 1.0 if hit else 0.0
        return reward, {"step_limit": reward}

    def _shaped_reward(self, task_reward: float) -> tuple[float, dict]:
        """Task reward gated on well-formed output, and the components behind it.

        Kept in one place because evaluate() and close() both score, and a
        rollout rescued by the close-time safety net must be shaped identically
        to one scored in the loop.

        Two schemes, selected by TERMINAL_RL_REWARD_SCHEME:

        "gated" (default)
            0.0                      a tool-call block leaked into the final
                                     response -- nothing else is scored
            0.2 + 0.8 * task_reward  otherwise, so an attempt that stays
                                     well-formed earns 0.2 and a correct one 1.0

        "linear"
            the additive form this replaced, 0.8*task + 0.1*json + 0.1*finish.

        "penalized"
            FAILURE_PENALTY (-1.0)   a tool-call block leaked into the final
                                     response, or the agent ran out of steps
                                     without solving the task
            task_reward              otherwise -- the verifier's own reward,
                                     including a solved run that ran out

        The additive form priced a format error at 0.1 out of 1.0, and across
        420 groups in job 10617579 the flag was constant within 57% of them --
        so centring removed it and it contributed no gradient at all. That is
        the shape of a term that cannot defend itself: the final response leaked
        raw markup in 9.9% of rollouts and the rate climbed from 1.2% to 20%
        over 24 steps with nothing opposing it. Gating makes the penalty the
        whole reward, and the flag varies within 43% of groups, which is where
        it can act.

        Measured cost of dropping the finish term: 73 of 420 groups (17%) lose
        their only within-group variation and stop producing a gradient, and
        nothing then rewards calling finish -- a rollout that never calls it
        runs to max_steps. Within-group reward spread is otherwise unchanged
        (0.1834 against 0.1842), so the signal is as strong, just differently
        placed. finish is still measured and reported so the rate stays visible.
        """
        tool_reward, tool_info = self._tool_json_reward()
        finish_reward, finish_info = self._finish_tool_reward()
        step_limit, step_info = self._step_limit_reward()
        if REWARD_SCHEME == "penalized":
            # A run that solved the task keeps its reward even if it then used up the
            # step limit (6-8% of step-limit rollouts in jobs 155591/155597 had
            # passed the verifier and kept testing): -1 there punished correct work
            # as hard as total failure.
            ran_out = step_limit > 0.0 and task_reward < 1.0
            # Likewise a solved run whose last message is malformed: the verifier
            # passed, and the evaluation scores it solved.
            malformed = tool_reward <= 0.0 and task_reward < 1.0
            if MALFORMED_CREDIT in ("targeted", "off"):
                # targeted: the malformed ending is penalised on its own tokens by the
                # train loop (patches/targeted_malformed_credit.py); off: not penalised
                # at all. Either way the trajectory keeps its task score.
                malformed = False
            total = FAILURE_PENALTY if malformed else STEP_LIMIT_PENALTY if ran_out else task_reward
        elif REWARD_SCHEME == "gated":
            total = 0.0 if tool_reward <= 0.0 else GATED_FLOOR + (1.0 - GATED_FLOOR) * task_reward
        else:
            total = (
                TASK_REWARD_WEIGHT * task_reward
                + TOOL_JSON_REWARD_WEIGHT * tool_reward
                + FINISH_REWARD_WEIGHT * finish_reward
            )
        penalty, longest = length_penalty(self._response_tokens)
        if penalty > 0.0 and LENGTH_PENALTY_CREDIT != "targeted":
            total = max(total - penalty, FAILURE_PENALTY)
            logger.info(
                "[%s] length penalty %.3f: longest completion %d tokens > %d",
                self._task.id, penalty, longest, LENGTH_PENALTY_N,
            )
        length_info = {"length_penalty": penalty, "max_response_tokens": float(longest)}
        return total, {**tool_info, **finish_info, **step_info, **length_info}

    async def evaluate(self) -> tuple[float, dict]:
        """Score once, when the agent has stopped.

        OpenHandsEnv calls this every step. The verifier is expensive and
        destructive -- it uploads /tests into the sandbox -- so it runs only
        after the agent has stopped, and only once.
        """
        if self._scored or not self._agent_stopped():
            return 0.0, {"scored": self._scored}
        self._scored = True
        task_reward, observation = await self._executor.score()
        reward, shaping = self._shaped_reward(task_reward)
        logger.info(
            "[%s] verifier scored %.2f (task=%.2f tool_json=%.1f finish=%.1f step_limit=%.1f)",
            self._task.id, reward, task_reward,
            shaping["tool_json_error_reward"], shaping["finish_tool_reward"], shaping["step_limit"],
        )
        return reward, {"scored": True, "verifier": observation[-2000:], "task_reward": task_reward, **shaping}

    async def close(self) -> None:
        """Score on the way out if the episode ended without doing so.

        OpenHandsEnv.step calls evaluate() and then, separately, tests whether
        the conversation has finished. Both consult live remote state, so an
        agent that stops between the two calls ends the episode with
        obs.finished set but the verifier never run -- the task is silently
        worth zero. Scoring here as well makes that unreachable; _scored keeps
        the verifier to one run either way.
        """
        # run_episode closes the env inside its own finally, and closing deletes
        # the conversation, which stops the in-sandbox server. The rollout task
        # is still running at that point, so a watchdog polling the process
        # would see it gone and cancel an episode that had already finished --
        # which is what put a spurious CancelledError on 70% of rollouts.
        self.closing = True
        if not self._scored:
            try:
                task_reward, observation = await asyncio.wait_for(
                    self._executor.score(), timeout=CLOSE_SCORE_TIMEOUT
                )
                self._scored = True
                reward, shaping = self._shaped_reward(task_reward)
                logger.info(
                    "[%s] verifier scored %.2f at close (task=%.2f tool_json=%.1f finish=%.1f step_limit=%.1f)",
                    self._task.id, reward, task_reward,
                    shaping["tool_json_error_reward"], shaping["finish_tool_reward"], shaping["step_limit"],
                )
                trajectory = current_trajectory.get(None)
                if trajectory is not None:
                    trajectory.reward = reward
                    trajectory.misc.setdefault("verifier", observation[-2000:])
                    trajectory.misc.setdefault("task_reward", task_reward)
                    trajectory.misc.update(shaping)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] scoring at close exceeded %.0fs; leaving the task unscored",
                    self._task.id, CLOSE_SCORE_TIMEOUT,
                )
            except Exception as error:  # a lost sandbox must not fail the rollout
                logger.warning("[%s] could not score at close: %s", self._task.id, error)
        # Closing the conversation talks to the in-sandbox server over HTTP. If
        # that server is gone the call can retry for minutes, and the sandbox is
        # torn down either way, so bound it rather than let it hold the phase.
        for name, closer in (("conversation", super().close), ("sandbox", self._executor.close)):
            try:
                await asyncio.wait_for(closer(), timeout=CLOSE_TEARDOWN_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] closing the %s exceeded %.0fs; abandoning it",
                    self._task.id, name, CLOSE_TEARDOWN_TIMEOUT,
                )
            except Exception as error:
                logger.warning("[%s] closing the %s failed: %s", self._task.id, name, error)
