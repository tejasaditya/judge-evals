"""Tests for bias controls (pairwise + self-consistency), all litellm calls mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from judge_evals.bias import pairwise_check, self_consistency
from judge_evals.types import Criterion, Rubric, Sample


@pytest.fixture()
def rubric() -> Rubric:
    return Rubric(
        name="quality",
        criteria=[
            Criterion(name="accuracy", description="Is it accurate?", required=True),
            Criterion(name="tone", description="Is the tone good?"),
        ],
    )


def _mock_response(scores: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"scores": scores})))]
    )


def _score(name: str, score: int, passed: bool) -> dict:
    return {
        "criterion_name": name,
        "score": score,
        "rationale": "test",
        "passed": passed,
    }


# --- pairwise_check -------------------------------------------------------


@pytest.mark.asyncio
async def test_pairwise_detects_flip(rubric):
    """When A/B and B/A give different passed values for a criterion, it's a flip."""
    sample_ab = Sample(input="Compare A vs B", output="A is better")
    sample_ba = Sample(input="Compare B vs A", output="B is better")

    resp_ab = _mock_response(
        [
            _score("accuracy", 5, True),
            _score("tone", 4, True),
        ]
    )
    resp_ba = _mock_response(
        [
            _score("accuracy", 2, False),  # flipped!
            _score("tone", 4, True),
        ]
    )

    mock = AsyncMock(side_effect=[resp_ab, resp_ba])
    with patch("judge_evals.runner.litellm.acompletion", mock):
        result = await pairwise_check(sample_ab, sample_ba, rubric, "test-model")

    assert "accuracy" in result.flipped_criteria
    assert "tone" not in result.flipped_criteria


@pytest.mark.asyncio
async def test_pairwise_no_flip(rubric):
    """When both orderings agree, no flips are detected."""
    sample_ab = Sample(input="Compare A vs B", output="A is better")
    sample_ba = Sample(input="Compare B vs A", output="A is better")

    resp = _mock_response(
        [
            _score("accuracy", 5, True),
            _score("tone", 4, True),
        ]
    )

    mock = AsyncMock(return_value=resp)
    with patch("judge_evals.runner.litellm.acompletion", mock):
        result = await pairwise_check(sample_ab, sample_ba, rubric, "test-model")

    assert result.flipped_criteria == []


# --- self_consistency ------------------------------------------------------


@pytest.mark.asyncio
async def test_self_consistency_majority_vote(rubric):
    """2 pass + 1 fail → majority_passed = True."""
    resp_pass = _mock_response(
        [
            _score("accuracy", 5, True),
            _score("tone", 4, True),
        ]
    )
    resp_fail = _mock_response(
        [
            _score("accuracy", 1, False),
            _score("tone", 2, False),
        ]
    )

    mock = AsyncMock(side_effect=[resp_pass, resp_pass, resp_fail])
    sample = Sample(input="q", output="a")

    with patch("judge_evals.runner.litellm.acompletion", mock):
        result = await self_consistency(sample, rubric, "test-model", n=3)

    assert result.majority_passed is True
    assert len(result.verdicts) == 3
    assert "accuracy" in result.per_criterion_variance
    # Variance should be > 0 because scores differ (5, 5, 1)
    assert result.per_criterion_variance["accuracy"] > 0


@pytest.mark.asyncio
async def test_self_consistency_all_agree(rubric):
    """All runs agree → variance should be 0."""
    resp = _mock_response(
        [
            _score("accuracy", 5, True),
            _score("tone", 4, True),
        ]
    )

    mock = AsyncMock(return_value=resp)
    sample = Sample(input="q", output="a")

    with patch("judge_evals.runner.litellm.acompletion", mock):
        result = await self_consistency(sample, rubric, "test-model", n=3)

    assert result.majority_passed is True
    assert result.per_criterion_variance["accuracy"] == 0.0
    assert result.per_criterion_variance["tone"] == 0.0
