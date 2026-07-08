"""Agreement metrics: judge verdicts vs. human labels.

Provides exact agreement, Cohen's kappa, and confusion matrices on a
per-criterion basis. The main entry point is :func:`compute_agreement`.
"""

from __future__ import annotations

from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from judge_evals.labels import HumanLabel
from judge_evals.types import JudgeVerdict, Rubric

__all__ = [
    "AgreementReport",
    "exact_agreement",
    "cohens_kappa",
    "build_confusion_matrix",
    "compute_agreement",
]


class AgreementReport(BaseModel):
    """Agreement metrics for one criterion between judge and human labels."""

    criterion_name: str
    n_samples: int
    exact_agreement: float
    kappa: float
    confusion_matrix: list[list[int]]
    labels: list[int]  # sorted unique score values used as row/col labels


def exact_agreement(judge_scores: list[int], human_scores: list[int]) -> float:
    """Fraction of exact score matches.

    Returns 1.0 if both lists are empty (vacuously agree).
    """
    if not judge_scores and not human_scores:
        return 1.0
    if len(judge_scores) != len(human_scores):
        raise ValueError(f"Length mismatch: {len(judge_scores)} judge vs {len(human_scores)} human")
    matches = sum(1 for j, h in zip(judge_scores, human_scores, strict=True) if j == h)
    return matches / len(judge_scores)


def cohens_kappa(judge_scores: list[int], human_scores: list[int]) -> float:
    """Cohen's kappa between judge and human scores.

    Returns 1.0 when both sides collapse to a single identical label: kappa is
    undefined there (0/0), and identical labels are perfect agreement.
    """
    if len(judge_scores) != len(human_scores):
        raise ValueError(f"Length mismatch: {len(judge_scores)} judge vs {len(human_scores)} human")
    if len(set(judge_scores) | set(human_scores)) <= 1:
        # All identical — kappa is undefined; return perfect agreement
        return 1.0
    return float(cohen_kappa_score(human_scores, judge_scores))


def build_confusion_matrix(
    judge_scores: list[int],
    human_scores: list[int],
    score_labels: list[int] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Build a confusion matrix (rows=human, cols=judge).

    Returns ``(matrix_as_nested_lists, sorted_labels)``.
    """
    if score_labels is None:
        score_labels = sorted(set(judge_scores) | set(human_scores))
    cm = sk_confusion_matrix(human_scores, judge_scores, labels=score_labels)
    return cm.tolist(), score_labels


def compute_agreement(
    verdicts: list[JudgeVerdict],
    labels: list[HumanLabel],
    rubric: Rubric,
) -> list[AgreementReport]:
    """Compute per-criterion agreement between judge verdicts and human labels.

    ``verdicts`` and ``labels`` must be aligned (same order, same samples).
    Returns one :class:`AgreementReport` per criterion in the rubric.
    """
    if len(verdicts) != len(labels):
        raise ValueError(f"Mismatched counts: {len(verdicts)} verdicts vs {len(labels)} labels")

    reports: list[AgreementReport] = []

    for criterion in rubric.criteria:
        cname = criterion.name
        judge_scores: list[int] = []
        human_scores: list[int] = []

        for verdict, label in zip(verdicts, labels, strict=True):
            # Get judge score for this criterion
            judge_score_obj = next((s for s in verdict.scores if s.criterion_name == cname), None)
            human_score = label.scores.get(cname)

            if judge_score_obj is None or human_score is None:
                continue  # skip if either side is missing this criterion

            judge_scores.append(judge_score_obj.score)
            human_scores.append(human_score)

        if not judge_scores:
            # No data for this criterion
            reports.append(
                AgreementReport(
                    criterion_name=cname,
                    n_samples=0,
                    exact_agreement=0.0,
                    kappa=0.0,
                    confusion_matrix=[],
                    labels=[],
                )
            )
            continue

        # All possible score values for this criterion
        score_range = list(range(criterion.scale.min, criterion.scale.max + 1))
        cm, cm_labels = build_confusion_matrix(judge_scores, human_scores, score_range)

        reports.append(
            AgreementReport(
                criterion_name=cname,
                n_samples=len(judge_scores),
                exact_agreement=exact_agreement(judge_scores, human_scores),
                kappa=cohens_kappa(judge_scores, human_scores),
                confusion_matrix=cm,
                labels=cm_labels,
            )
        )

    return reports
