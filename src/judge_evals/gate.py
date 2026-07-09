"""Runtime judge gate: wrap an agent function, judge its output, act on failure.

``JudgeGate`` judges an agent/LLM output *at runtime* against a rubric. When a required
criterion fails, a :class:`GatePolicy` decides what to do — retry with judge feedback
injected, escalate, or reject — and the whole thing is recorded in a :class:`GateTrace`.

## What is scaffolded vs. hand-written

Per the project plan, **the retry-with-feedback state machine is intentionally NOT
implemented here** — it is left for the human to write. Concretely:

- Left as ``# TODO(user)`` stubs (raise ``NotImplementedError``): the transition function
  :func:`decide_action` and the orchestration loop :meth:`JudgeGate.gate`. Their
  signatures, docstrings, and (skipped) test specs are provided so the implementation
  slots straight in.
- Fully implemented and tested here: the data types (:class:`GatePolicy`,
  :class:`GateAttempt`, :class:`GateTrace`), the :class:`AgentFn` protocol, and the pure
  helpers the state machine will call — :func:`format_feedback`, :func:`backoff_delay`,
  and the :func:`rescue_rate` metric.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from judge_evals.types import JudgeVerdict

__all__ = [
    "GateAction",
    "GatePolicy",
    "GateAttempt",
    "GateTrace",
    "AgentFn",
    "format_feedback",
    "backoff_delay",
    "rescue_rate",
    "decide_action",
    "JudgeGate",
]


class GateAction(StrEnum):
    """What the gate does after judging an attempt.

    - ``ACCEPT``: the output passed (no required criterion failed) — return it.
    - ``RETRY``: a required criterion failed and retries remain — re-run the agent with
      feedback injected.
    - ``ESCALATE``: terminal — stop retrying and hand off (e.g. to a stronger model or a
      human). The caller decides what escalation means.
    - ``REJECT``: terminal — give up and return the failing output/verdict.
    """

    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"
    REJECT = "reject"


class GatePolicy(BaseModel):
    """Policy governing the gate's behaviour on a required-criterion failure.

    ``max_retries`` is the number of *re-attempts* after the first attempt, so the total
    number of agent calls is at most ``max_retries + 1``. When retries are exhausted the
    gate takes ``on_exhausted`` (``REJECT`` or ``ESCALATE``).
    """

    max_retries: int = Field(default=2, ge=0)
    on_exhausted: GateAction = GateAction.REJECT
    inject_feedback: bool = True
    base_backoff_s: float = Field(default=0.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)


class GateAttempt(BaseModel):
    """One attempt within a gate run: the agent's output and the judge's verdict on it."""

    attempt: int  # 1-indexed
    feedback_in: str | None = None  # feedback injected to produce this output (None on 1st)
    output: str
    verdict: JudgeVerdict
    action: GateAction  # action decided after judging this output


class GateTrace(BaseModel):
    """Full record of a gate run: every attempt plus the terminal action."""

    sample_id: str | None = None
    rubric_name: str
    attempts: list[GateAttempt] = Field(default_factory=list)
    final_action: GateAction

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def n_retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def final_verdict(self) -> JudgeVerdict | None:
        return self.attempts[-1].verdict if self.attempts else None

    @property
    def final_output(self) -> str | None:
        return self.attempts[-1].output if self.attempts else None

    @property
    def passed(self) -> bool:
        """True if the final attempt passed (no required criterion failed)."""
        fv = self.final_verdict
        return fv is not None and fv.passed

    @property
    def rescued(self) -> bool:
        """True if the first attempt failed but a later attempt passed.

        This is the event the rescue-rate metric counts.
        """
        return self.passed and len(self.attempts) > 1 and not self.attempts[0].verdict.passed


class AgentFn(Protocol):
    """The agent/LLM under the gate.

    Called with the task ``input`` and, on a retry, the ``feedback`` string built from the
    previous failing verdict (``None`` on the first attempt). Returns the agent's output.
    """

    async def __call__(self, input: str, feedback: str | None = None) -> str: ...


# --- Pure helpers (implemented + tested) -----------------------------------


def format_feedback(verdict: JudgeVerdict) -> str:
    """Build a feedback string from a failing verdict, to inject into a retry.

    Lists the criteria the output failed — required criteria first — with the judge's
    rationale for each. Returns ``""`` if nothing failed (nothing to correct).
    """
    failed = [s for s in verdict.scores if not s.passed]
    if not failed:
        return ""
    # Required failures first: they are what actually gate the verdict.
    failed.sort(key=lambda s: (not s.required, s.criterion_name))

    lines = ["Your previous response did not meet the following criteria. Please revise:"]
    for s in failed:
        tag = " (required)" if s.required else ""
        rationale = s.rationale.strip() or "(no rationale given)"
        lines.append(f"- {s.criterion_name}{tag} — scored {s.score}: {rationale}")
    return "\n".join(lines)


def backoff_delay(retry_index: int, policy: GatePolicy) -> float:
    """Seconds to wait before a retry.

    ``retry_index`` is 0 for the first retry, 1 for the second, etc. Grows geometrically:
    ``base_backoff_s * backoff_factor ** retry_index``.
    """
    if retry_index < 0:
        raise ValueError(f"retry_index must be >= 0, got {retry_index}")
    return policy.base_backoff_s * (policy.backoff_factor**retry_index)


def rescue_rate(traces: list[GateTrace]) -> float:
    """Fraction of initially-failing tasks that eventually passed after a retry.

    Denominator is the number of traces whose first attempt failed; numerator is those
    among them that ended up passing (``trace.rescued``). Returns 0.0 when no task failed
    initially (nothing to rescue).
    """
    initially_failed = [t for t in traces if t.attempts and not t.attempts[0].verdict.passed]
    if not initially_failed:
        return 0.0
    rescued = sum(1 for t in initially_failed if t.rescued)
    return rescued / len(initially_failed)


# --- Hand-written state machine (interface only; user implements) ----------


def decide_action(verdict: JudgeVerdict, attempt: int, policy: GatePolicy) -> GateAction:
    """Decide the gate's next action after judging ``attempt``.

    Intended contract (define it precisely when you implement it):
    - If ``verdict.passed`` → :attr:`GateAction.ACCEPT`.
    - Else if this was the last allowed attempt (``attempt >= policy.max_retries + 1``)
      → ``policy.on_exhausted`` (``REJECT`` or ``ESCALATE``).
    - Else → :attr:`GateAction.RETRY`.

    ``attempt`` is 1-indexed (the first attempt is 1).
    """
    # TODO(user): implement the transition function described above. This is the decision
    # half of the retry state machine and is intentionally left for you to write.
    raise NotImplementedError("decide_action is hand-written — see TODO(user) in gate.py")


class JudgeGate:
    """Runtime gate that judges an agent's output and applies a policy on failure.

    Construct once with the judge model (and optional cache/policy), then call
    :meth:`gate` per task.
    """

    def __init__(
        self,
        judge_model: str,
        cache: object | None = None,
        policy: GatePolicy | None = None,
    ) -> None:
        self.judge_model = judge_model
        # cache is a judge_evals.cache.VerdictCache | None (typed loosely to avoid a cycle).
        self.cache = cache
        self.policy = policy or GatePolicy()

    async def gate(self, fn: AgentFn, rubric: object, sample: object) -> GateTrace:
        """Run ``fn`` under the gate against ``rubric`` for ``sample``; return a trace.

        Intended loop (the retry-with-feedback state machine — write this yourself):

        1. attempt = 1, feedback = None.
        2. output = await fn(sample.input, feedback).
        3. verdict = await judge_sample(sample-with-output, rubric, self.judge_model,
           self.cache).
        4. action = decide_action(verdict, attempt, self.policy).
        5. Record a GateAttempt(attempt, feedback_in=feedback, output, verdict, action).
        6. If action is RETRY: feedback = format_feedback(verdict) (when
           policy.inject_feedback), await asyncio.sleep(backoff_delay(attempt-1, policy)),
           attempt += 1, go to 2.
        7. Otherwise stop; return GateTrace(..., final_action=action).

        The pieces this loop calls — :func:`decide_action`, :func:`format_feedback`,
        :func:`backoff_delay`, and the :class:`GateAttempt`/:class:`GateTrace` types — are
        all defined in this module (``decide_action`` is likewise a TODO for you).
        """
        # TODO(user): implement the retry-with-feedback state machine described above.
        # Left intentionally unimplemented per the project plan; the scaffolding, types,
        # helpers, and (skipped) test spec in tests/test_gate.py are ready for you.
        raise NotImplementedError("JudgeGate.gate is hand-written — see TODO(user) in gate.py")
