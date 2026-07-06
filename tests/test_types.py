"""Unit tests for the core domain types."""

import pytest
from pydantic import ValidationError

from judge_evals import (
    Criterion,
    CriterionScore,
    JudgeVerdict,
    Rubric,
    Sample,
    Scale,
)

# --- Scale -----------------------------------------------------------------


def test_scale_defaults_are_one_to_five():
    s = Scale()
    assert (s.min, s.max) == (1, 5)


def test_scale_contains():
    s = Scale(min=1, max=5)
    assert s.contains(1)
    assert s.contains(5)
    assert not s.contains(0)
    assert not s.contains(6)


@pytest.mark.parametrize("bad", [(5, 5), (5, 1), (3, 2)])
def test_scale_rejects_min_ge_max(bad):
    lo, hi = bad
    with pytest.raises(ValidationError):
        Scale(min=lo, max=hi)


# --- Criterion -------------------------------------------------------------


def test_criterion_defaults():
    c = Criterion(name="helpfulness", description="Is it helpful?")
    assert c.required is False
    assert c.scale.min == 1 and c.scale.max == 5


# --- Rubric ----------------------------------------------------------------


def _rubric() -> Rubric:
    return Rubric(
        name="quality",
        criteria=[
            Criterion(name="accuracy", description="Correct?", required=True),
            Criterion(name="tone", description="Polite?"),
        ],
    )


def test_rubric_default_version():
    assert _rubric().version == "v1"


def test_rubric_rejects_empty_criteria():
    with pytest.raises(ValidationError):
        Rubric(name="empty", criteria=[])


def test_rubric_rejects_duplicate_names():
    with pytest.raises(ValidationError):
        Rubric(
            name="dupe",
            criteria=[
                Criterion(name="x", description="a"),
                Criterion(name="x", description="b"),
            ],
        )


def test_rubric_criterion_lookup():
    r = _rubric()
    assert r.criterion("accuracy").required is True


def test_rubric_criterion_lookup_missing_raises_keyerror():
    with pytest.raises(KeyError):
        _rubric().criterion("nope")


# --- JudgeVerdict.passed ---------------------------------------------------


def _verdict(scores: list[CriterionScore]) -> JudgeVerdict:
    return JudgeVerdict(
        rubric_name="quality",
        rubric_version="v1",
        judge_model="test-model",
        scores=scores,
    )


def test_verdict_passes_when_required_passes():
    v = _verdict(
        [
            CriterionScore(
                criterion_name="accuracy", score=5, rationale="ok", passed=True, required=True
            ),
            CriterionScore(
                criterion_name="tone", score=2, rationale="meh", passed=False, required=False
            ),
        ]
    )
    assert v.passed is True  # optional failure does not gate


def test_verdict_fails_when_required_fails():
    v = _verdict(
        [
            CriterionScore(
                criterion_name="accuracy", score=1, rationale="wrong", passed=False, required=True
            ),
            CriterionScore(
                criterion_name="tone", score=5, rationale="great", passed=True, required=False
            ),
        ]
    )
    assert v.passed is False


def test_verdict_with_no_required_criteria_always_passes():
    v = _verdict(
        [
            CriterionScore(
                criterion_name="tone", score=1, rationale="bad", passed=False, required=False
            ),
        ]
    )
    assert v.passed is True


# --- Sample ----------------------------------------------------------------


def test_sample_optional_fields_default_to_none():
    s = Sample(input="q", output="a")
    assert s.id is None
    assert s.context is None
    assert s.reference is None


def test_sample_content_hash_is_stable_for_same_content():
    a = Sample(input="q", output="a", context="c", reference="r")
    b = Sample(input="q", output="a", context="c", reference="r")
    assert a.content_hash == b.content_hash


def test_sample_content_hash_ignores_id():
    a = Sample(input="q", output="a", id="sample-1")
    b = Sample(input="q", output="a", id="sample-2")
    assert a.content_hash == b.content_hash


def test_sample_content_hash_changes_with_content():
    a = Sample(input="q", output="a")
    b = Sample(input="q", output="different")
    assert a.content_hash != b.content_hash
