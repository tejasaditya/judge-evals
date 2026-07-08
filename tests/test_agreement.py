"""Tests for agreement metrics."""

import pytest

from judge_evals.agreement import (
    build_confusion_matrix,
    cohens_kappa,
    compute_agreement,
    exact_agreement,
)
from judge_evals.labels import HumanLabel
from judge_evals.types import Criterion, CriterionScore, JudgeVerdict, Rubric, Scale


def test_exact_agreement():
    assert exact_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert exact_agreement([1, 2, 3], [1, 4, 5]) == 1 / 3
    assert exact_agreement([1, 2], [3, 4]) == 0.0
    assert exact_agreement([], []) == 1.0


def test_exact_agreement_length_mismatch():
    with pytest.raises(ValueError):
        exact_agreement([1], [1, 2])


def test_cohens_kappa():
    # If perfect agreement, kappa = 1.0
    assert cohens_kappa([1, 2, 3], [1, 2, 3]) == 1.0

    # If all same score, kappa undefined by standard formula, we return 1.0
    assert cohens_kappa([5, 5], [5, 5]) == 1.0

    # Test a known case
    # Rater 1: 1, 1, 2, 2
    # Rater 2: 1, 2, 2, 2
    # Agree on 3/4. Expected kappa around 0.5 (varies by formula, just check it's > 0)
    kappa = cohens_kappa([1, 1, 2, 2], [1, 2, 2, 2])
    assert 0.0 < kappa < 1.0


def test_build_confusion_matrix():
    j = [1, 1, 2, 3]
    h = [1, 2, 2, 3]
    cm, labels = build_confusion_matrix(j, h, [1, 2, 3])
    # human = row, judge = col
    # h=1, j=1: 1
    # h=2, j=1: 1
    # h=2, j=2: 1
    # h=3, j=3: 1
    assert labels == [1, 2, 3]
    assert cm == [
        [1, 0, 0],  # h=1
        [1, 1, 0],  # h=2
        [0, 0, 1],  # h=3
    ]


def test_compute_agreement():
    rubric = Rubric(
        name="test",
        criteria=[
            Criterion(name="accuracy", description="", scale=Scale(min=1, max=5)),
            Criterion(name="tone", description="", scale=Scale(min=1, max=3)),
        ],
    )

    verdicts = [
        JudgeVerdict(
            rubric_name="test",
            rubric_version="v1",
            judge_model="m1",
            scores=[
                CriterionScore(
                    criterion_name="accuracy",
                    score=5,
                    rationale="",
                    passed=True,
                    required=False,
                )
            ],
        ),
        JudgeVerdict(
            rubric_name="test",
            rubric_version="v1",
            judge_model="m1",
            scores=[
                CriterionScore(
                    criterion_name="accuracy",
                    score=1,
                    rationale="",
                    passed=False,
                    required=False,
                )
            ],
        ),
    ]

    labels = [
        HumanLabel(sample_id="1", scores={"accuracy": 5, "tone": 3}),
        HumanLabel(sample_id="2", scores={"accuracy": 2, "tone": 1}),
    ]

    reports = compute_agreement(verdicts, labels, rubric)

    # tone should be missing or 0 n_samples, since verdicts have no tone
    # accuracy should have 2 samples
    assert len(reports) == 2

    acc_rep = next(r for r in reports if r.criterion_name == "accuracy")
    assert acc_rep.n_samples == 2
    assert acc_rep.exact_agreement == 0.5  # 1 out of 2 matched exactly (5==5, 1!=2)
    assert acc_rep.labels == [1, 2, 3, 4, 5]

    tone_rep = next(r for r in reports if r.criterion_name == "tone")
    assert tone_rep.n_samples == 0
