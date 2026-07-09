"""Tests for the runtime judge gate (M3).

Two groups:

- **Implemented scaffolding** — the data types, ``format_feedback``, ``backoff_delay``,
  and ``rescue_rate`` are fully tested here.
- **State-machine spec** — the tests for ``decide_action`` and ``JudgeGate.gate`` are the
  executable specification for the hand-written retry loop. They are ``skip``-marked so
  the suite stays green until the user implements those stubs; un-skip them (flip
  ``STATE_MACHINE_DONE`` to ``True``) as you fill the loop in.
"""

from __future__ import annotations

import pytest

from judge_evals.gate import (
    AgentFn,
    GateAction,
    GateAttempt,
    GatePolicy,
    GateTrace,
    JudgeGate,
    backoff_delay,
    decide_action,
    format_feedback,
    rescue_rate,
)
from judge_evals.types import CriterionScore, JudgeVerdict

# Flip to True once JudgeGate.gate + decide_action are implemented, to run the spec below.
STATE_MACHINE_DONE = False
_SKIP_REASON = "state machine is hand-written by the user — TODO(user) in gate.py"
_skip_sm = pytest.mark.skip(reason=_SKIP_REASON)


# --- fixtures / builders ---------------------------------------------------


def _verdict(*, required_passed: bool, extra_optional_fail: bool = False) -> JudgeVerdict:
    """Build a verdict with one required criterion (pass/fail) and an optional one."""
    scores = [
        CriterionScore(
            criterion_name="accuracy",
            score=5 if required_passed else 1,
            rationale="looks right" if required_passed else "factually wrong",
            passed=required_passed,
            required=True,
        ),
        CriterionScore(
            criterion_name="tone",
            score=2 if extra_optional_fail else 5,
            rationale="too curt" if extra_optional_fail else "polite",
            passed=not extra_optional_fail,
            required=False,
        ),
    ]
    return JudgeVerdict(
        rubric_name="quality",
        rubric_version="v1",
        judge_model="test-model",
        scores=scores,
    )


def _attempt(n: int, *, required_passed: bool, action: GateAction) -> GateAttempt:
    return GateAttempt(
        attempt=n,
        feedback_in=None if n == 1 else "feedback",
        output=f"output-{n}",
        verdict=_verdict(required_passed=required_passed),
        action=action,
    )


# --- GatePolicy ------------------------------------------------------------


def test_policy_defaults():
    p = GatePolicy()
    assert p.max_retries == 2
    assert p.on_exhausted is GateAction.REJECT
    assert p.inject_feedback is True


def test_policy_rejects_negative_retries():
    with pytest.raises(ValueError):
        GatePolicy(max_retries=-1)


# --- GateTrace derived properties ------------------------------------------


def test_trace_pass_first_try_not_rescued():
    trace = GateTrace(
        rubric_name="quality",
        attempts=[_attempt(1, required_passed=True, action=GateAction.ACCEPT)],
        final_action=GateAction.ACCEPT,
    )
    assert trace.passed is True
    assert trace.rescued is False
    assert trace.n_attempts == 1
    assert trace.n_retries == 0
    assert trace.final_output == "output-1"


def test_trace_fail_then_pass_is_rescued():
    trace = GateTrace(
        rubric_name="quality",
        attempts=[
            _attempt(1, required_passed=False, action=GateAction.RETRY),
            _attempt(2, required_passed=True, action=GateAction.ACCEPT),
        ],
        final_action=GateAction.ACCEPT,
    )
    assert trace.passed is True
    assert trace.rescued is True
    assert trace.n_retries == 1


def test_trace_fail_all_not_rescued():
    trace = GateTrace(
        rubric_name="quality",
        attempts=[
            _attempt(1, required_passed=False, action=GateAction.RETRY),
            _attempt(2, required_passed=False, action=GateAction.REJECT),
        ],
        final_action=GateAction.REJECT,
    )
    assert trace.passed is False
    assert trace.rescued is False


def test_trace_empty_attempts():
    trace = GateTrace(rubric_name="quality", attempts=[], final_action=GateAction.REJECT)
    assert trace.passed is False
    assert trace.rescued is False
    assert trace.final_verdict is None
    assert trace.final_output is None


# --- format_feedback -------------------------------------------------------


def test_format_feedback_empty_when_all_passed():
    assert format_feedback(_verdict(required_passed=True)) == ""


def test_format_feedback_lists_failed_criteria():
    fb = format_feedback(_verdict(required_passed=False, extra_optional_fail=True))
    # Required failure listed and tagged, with its rationale
    assert "accuracy (required)" in fb
    assert "factually wrong" in fb
    # Optional failure also listed (untagged)
    assert "tone" in fb
    assert "too curt" in fb


def test_format_feedback_required_listed_before_optional():
    fb = format_feedback(_verdict(required_passed=False, extra_optional_fail=True))
    assert fb.index("accuracy") < fb.index("tone")


# --- backoff_delay ---------------------------------------------------------


def test_backoff_delay_geometric():
    p = GatePolicy(base_backoff_s=0.5, backoff_factor=2.0)
    assert backoff_delay(0, p) == 0.5
    assert backoff_delay(1, p) == 1.0
    assert backoff_delay(2, p) == 2.0


def test_backoff_delay_zero_base():
    assert backoff_delay(3, GatePolicy()) == 0.0


def test_backoff_delay_negative_index_raises():
    with pytest.raises(ValueError):
        backoff_delay(-1, GatePolicy())


# --- rescue_rate -----------------------------------------------------------


def _trace(first_passed: bool, final_passed: bool) -> GateTrace:
    attempts = [_attempt(1, required_passed=first_passed, action=GateAction.ACCEPT)]
    if not first_passed:
        attempts.append(
            _attempt(
                2,
                required_passed=final_passed,
                action=GateAction.ACCEPT if final_passed else GateAction.REJECT,
            )
        )
    return GateTrace(
        rubric_name="quality",
        attempts=attempts,
        final_action=attempts[-1].action,
    )


def test_rescue_rate_basic():
    traces = [
        _trace(first_passed=True, final_passed=True),  # never failed → not in denominator
        _trace(first_passed=False, final_passed=True),  # rescued
        _trace(first_passed=False, final_passed=False),  # failed, not rescued
    ]
    # 2 initially failed, 1 rescued → 0.5
    assert rescue_rate(traces) == 0.5


def test_rescue_rate_no_initial_failures_is_zero():
    traces = [_trace(first_passed=True, final_passed=True)]
    assert rescue_rate(traces) == 0.0


def test_rescue_rate_empty():
    assert rescue_rate([]) == 0.0


# --- AgentFn protocol is structural ----------------------------------------


def test_agentfn_protocol_accepts_matching_callable():
    async def agent(input: str, feedback: str | None = None) -> str:
        return input.upper()

    fn: AgentFn = agent  # should type-check structurally; smoke-assert it's callable
    assert callable(fn)


# --- STATE MACHINE SPEC (skipped until implemented) ------------------------


@_skip_sm
def test_decide_action_accept_when_passed():
    assert decide_action(_verdict(required_passed=True), 1, GatePolicy()) is GateAction.ACCEPT


@_skip_sm
def test_decide_action_retry_when_failed_and_retries_remain():
    assert (
        decide_action(_verdict(required_passed=False), 1, GatePolicy(max_retries=2))
        is GateAction.RETRY
    )


@_skip_sm
def test_decide_action_exhausted_takes_on_exhausted():
    p = GatePolicy(max_retries=1, on_exhausted=GateAction.ESCALATE)
    # attempt 2 == max_retries + 1 → exhausted
    assert decide_action(_verdict(required_passed=False), 2, p) is GateAction.ESCALATE


@_skip_sm
@pytest.mark.asyncio
async def test_gate_accepts_on_first_pass():
    calls = []

    async def agent(input: str, feedback: str | None = None) -> str:
        calls.append(feedback)
        return "good output"

    # NOTE: when implementing, this needs a judge that passes on the first output.
    # Patch judge_evals.runner.litellm.acompletion (see tests/test_runner.py) or inject a
    # stub judge. Expected: trace.passed, trace.n_attempts == 1, calls == [None].
    gate = JudgeGate(judge_model="test-model")
    trace = await gate.gate(agent, rubric=..., sample=...)  # fill rubric/sample
    assert trace.passed
    assert trace.n_attempts == 1


@_skip_sm
@pytest.mark.asyncio
async def test_gate_rescues_after_feedback():
    # Agent fails first, then (given feedback) succeeds. Expected: trace.rescued is True,
    # n_attempts == 2, and the second call received non-None feedback.
    ...


@_skip_sm
@pytest.mark.asyncio
async def test_gate_exhausts_retries_and_rejects():
    # Agent always fails. Expected: n_attempts == policy.max_retries + 1,
    # final_action == policy.on_exhausted, trace.passed is False.
    ...
