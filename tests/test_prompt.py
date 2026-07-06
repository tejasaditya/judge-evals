"""Tests for the prompt builder and response parser."""

from __future__ import annotations

import json

import pytest

from judge_evals.prompt import ParseError, build_judge_prompt, parse_judge_response
from judge_evals.types import Criterion, Rubric, Sample


@pytest.fixture()
def sample() -> Sample:
    return Sample(input="What is 2+2?", output="4", context="Math quiz")


@pytest.fixture()
def rubric() -> Rubric:
    return Rubric(
        name="math",
        criteria=[
            Criterion(name="correctness", description="Is the answer correct?", required=True),
            Criterion(name="clarity", description="Is the answer clear?"),
        ],
    )


# --- build_judge_prompt ---------------------------------------------------


def test_prompt_returns_two_messages(sample: Sample, rubric: Rubric):
    msgs = build_judge_prompt(sample, rubric)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_prompt_system_contains_all_criteria(sample: Sample, rubric: Rubric):
    msgs = build_judge_prompt(sample, rubric)
    system = msgs[0]["content"]
    assert "correctness" in system
    assert "clarity" in system
    assert "REQUIRED" in system  # correctness is required


def test_prompt_user_contains_sample_content(sample: Sample, rubric: Rubric):
    msgs = build_judge_prompt(sample, rubric)
    user = msgs[1]["content"]
    assert "What is 2+2?" in user
    assert "4" in user
    assert "Math quiz" in user


def test_prompt_omits_none_fields(rubric: Rubric):
    s = Sample(input="q", output="a")  # no context, no reference
    msgs = build_judge_prompt(s, rubric)
    user = msgs[1]["content"]
    assert "Context" not in user
    assert "Reference" not in user


# --- parse_judge_response -------------------------------------------------


def _valid_response(rubric: Rubric) -> str:
    """A valid JSON response matching the rubric."""
    return json.dumps(
        {
            "scores": [
                {
                    "criterion_name": "correctness",
                    "score": 5,
                    "rationale": "Correct answer",
                    "passed": True,
                },
                {
                    "criterion_name": "clarity",
                    "score": 4,
                    "rationale": "Clear enough",
                    "passed": True,
                },
            ]
        }
    )


def test_parse_valid_response(rubric: Rubric):
    scores = parse_judge_response(_valid_response(rubric), rubric)
    assert len(scores) == 2
    assert scores[0].criterion_name == "correctness"
    assert scores[0].required is True  # copied from rubric
    assert scores[1].required is False


def test_parse_strips_markdown_fences(rubric: Rubric):
    raw = f"```json\n{_valid_response(rubric)}\n```"
    scores = parse_judge_response(raw, rubric)
    assert len(scores) == 2


def test_parse_rejects_invalid_json(rubric: Rubric):
    with pytest.raises(ParseError, match="Invalid JSON"):
        parse_judge_response("not json at all", rubric)


def test_parse_rejects_missing_scores_key(rubric: Rubric):
    with pytest.raises(ParseError, match="scores"):
        parse_judge_response('{"results": []}', rubric)


def test_parse_rejects_missing_criterion(rubric: Rubric):
    partial = json.dumps(
        {
            "scores": [
                {
                    "criterion_name": "correctness",
                    "score": 5,
                    "rationale": "ok",
                    "passed": True,
                },
                # missing "clarity"
            ]
        }
    )
    with pytest.raises(ParseError, match="Missing criteria"):
        parse_judge_response(partial, rubric)


def test_parse_rejects_out_of_bounds_score(rubric: Rubric):
    bad = json.dumps(
        {
            "scores": [
                {
                    "criterion_name": "correctness",
                    "score": 99,  # out of 1-5 scale
                    "rationale": "ok",
                    "passed": True,
                },
                {
                    "criterion_name": "clarity",
                    "score": 4,
                    "rationale": "ok",
                    "passed": True,
                },
            ]
        }
    )
    with pytest.raises(ParseError, match="outside scale"):
        parse_judge_response(bad, rubric)


def test_parse_rejects_unknown_criterion(rubric: Rubric):
    bad = json.dumps(
        {
            "scores": [
                {
                    "criterion_name": "correctness",
                    "score": 5,
                    "rationale": "ok",
                    "passed": True,
                },
                {
                    "criterion_name": "clarity",
                    "score": 4,
                    "rationale": "ok",
                    "passed": True,
                },
                {
                    "criterion_name": "bogus",
                    "score": 3,
                    "rationale": "ok",
                    "passed": True,
                },
            ]
        }
    )
    with pytest.raises(ParseError, match="Unknown criterion"):
        parse_judge_response(bad, rubric)
