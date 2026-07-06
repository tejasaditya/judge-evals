"""Tests for the async batch judge runner (all litellm calls mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from judge_evals.cache import VerdictCache
from judge_evals.prompt import ParseError
from judge_evals.runner import judge_batch, judge_sample
from judge_evals.types import Criterion, Rubric, Sample


@pytest.fixture()
def sample() -> Sample:
    return Sample(id="s1", input="What is 2+2?", output="4")


@pytest.fixture()
def rubric() -> Rubric:
    return Rubric(
        name="math",
        criteria=[
            Criterion(name="correctness", description="Is it correct?", required=True),
        ],
    )


def _mock_response(scores_json: str) -> SimpleNamespace:
    """Fake litellm response with the given JSON content."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=scores_json))])


def _valid_json() -> str:
    return json.dumps(
        {
            "scores": [
                {
                    "criterion_name": "correctness",
                    "score": 5,
                    "rationale": "Correct",
                    "passed": True,
                }
            ]
        }
    )


# --- judge_sample ---------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_sample_returns_verdict(sample, rubric):
    mock = AsyncMock(return_value=_mock_response(_valid_json()))
    with patch("judge_evals.runner.litellm.acompletion", mock):
        verdict = await judge_sample(sample, rubric, "test-model")

    assert verdict.rubric_name == "math"
    assert verdict.passed is True
    assert len(verdict.scores) == 1
    mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_sample_retries_on_parse_failure(sample, rubric):
    """First call returns bad JSON, second returns valid — should succeed."""
    bad_resp = _mock_response("not json")
    good_resp = _mock_response(_valid_json())

    mock = AsyncMock(side_effect=[bad_resp, good_resp])
    with patch("judge_evals.runner.litellm.acompletion", mock):
        verdict = await judge_sample(sample, rubric, "test-model", max_retries=3)

    assert verdict.passed is True
    assert mock.await_count == 2


@pytest.mark.asyncio
async def test_judge_sample_exhausts_retries(sample, rubric):
    """All attempts return bad JSON — should raise ParseError."""
    bad_resp = _mock_response("bad")
    mock = AsyncMock(return_value=bad_resp)

    with patch("judge_evals.runner.litellm.acompletion", mock):
        with pytest.raises(ParseError):
            await judge_sample(sample, rubric, "test-model", max_retries=2)

    assert mock.await_count == 2


@pytest.mark.asyncio
async def test_judge_sample_uses_cache(sample, rubric, tmp_path):
    """Second call should hit cache, not litellm."""
    mock = AsyncMock(return_value=_mock_response(_valid_json()))

    async with VerdictCache(tmp_path / "cache.sqlite") as cache:
        with patch("judge_evals.runner.litellm.acompletion", mock):
            v1 = await judge_sample(sample, rubric, "test-model", cache)
            v2 = await judge_sample(sample, rubric, "test-model", cache)

    assert v1.scores[0].score == v2.scores[0].score
    mock.assert_awaited_once()  # only one litellm call


# --- judge_batch ----------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_batch_processes_multiple_samples(rubric):
    samples = [
        Sample(id="a", input="1+1", output="2"),
        Sample(id="b", input="2+3", output="5"),
        Sample(id="c", input="3+4", output="7"),
    ]
    mock = AsyncMock(return_value=_mock_response(_valid_json()))

    with patch("judge_evals.runner.litellm.acompletion", mock):
        verdicts = await judge_batch(samples, rubric, "test-model", concurrency=2)

    assert len(verdicts) == 3
    assert all(v.passed for v in verdicts)
    assert mock.await_count == 3
