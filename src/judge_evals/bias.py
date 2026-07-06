"""Bias controls for judge evaluations.

Two mechanisms:

1. **Pairwise mode** — runs the judge on A/B and B/A orderings of a sample pair
   and flags criteria where the score flips (position bias detection).
2. **Self-consistency** — runs the judge ``n`` times independently on the same
   sample, reports majority verdict and per-criterion variance.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel

from judge_evals.cache import VerdictCache
from judge_evals.runner import judge_sample
from judge_evals.types import JudgeVerdict, Rubric, Sample

__all__ = [
    "PairwiseResult",
    "ConsistencyResult",
    "pairwise_check",
    "self_consistency",
]


class PairwiseResult(BaseModel):
    """Result of running the judge on A/B and B/A orderings."""

    verdict_ab: JudgeVerdict
    verdict_ba: JudgeVerdict
    flipped_criteria: list[str]
    """Criteria where ``passed`` differed between orderings."""


class ConsistencyResult(BaseModel):
    """Result of running the judge ``n`` times independently."""

    verdicts: list[JudgeVerdict]
    majority_passed: bool
    per_criterion_variance: dict[str, float]
    """Variance of scores for each criterion across the ``n`` runs."""


async def pairwise_check(
    sample_a: Sample,
    sample_b: Sample,
    rubric: Rubric,
    model: str,
    cache: VerdictCache | None = None,
    *,
    max_retries: int = 3,
) -> PairwiseResult:
    """Run the judge on two orderings and detect position bias.

    ``sample_a`` and ``sample_b`` should represent the same underlying comparison
    but with the output/reference swapped (A-then-B vs B-then-A). The caller
    constructs these; this function just runs both and compares.

    A criterion is "flipped" if ``passed`` differs between the two orderings.
    """
    verdict_ab = await judge_sample(sample_a, rubric, model, cache, max_retries=max_retries)
    verdict_ba = await judge_sample(sample_b, rubric, model, cache, max_retries=max_retries)

    scores_ab = {s.criterion_name: s for s in verdict_ab.scores}
    scores_ba = {s.criterion_name: s for s in verdict_ba.scores}

    flipped = [
        name
        for name in scores_ab
        if name in scores_ba and scores_ab[name].passed != scores_ba[name].passed
    ]

    return PairwiseResult(
        verdict_ab=verdict_ab,
        verdict_ba=verdict_ba,
        flipped_criteria=flipped,
    )


async def self_consistency(
    sample: Sample,
    rubric: Rubric,
    model: str,
    *,
    n: int = 3,
    max_retries: int = 3,
    temperature: float = 0.7,
) -> ConsistencyResult:
    """Run the judge ``n`` times and report majority verdict + variance.

    Uses ``temperature > 0`` (default 0.7) to get diverse responses. Cache is
    intentionally bypassed (set to None) so each call is independent.
    """
    verdicts: list[JudgeVerdict] = []
    for _ in range(n):
        v = await judge_sample(
            sample,
            rubric,
            model,
            cache=None,  # No cache — each run must be independent
            max_retries=max_retries,
            temperature=temperature,
        )
        verdicts.append(v)

    # Majority vote on overall pass/fail
    pass_count = sum(1 for v in verdicts if v.passed)
    majority_passed = pass_count > n // 2

    # Per-criterion score variance
    criterion_scores: dict[str, list[int]] = {}
    for v in verdicts:
        for s in v.scores:
            criterion_scores.setdefault(s.criterion_name, []).append(s.score)

    per_criterion_variance: dict[str, float] = {}
    for name, scores in criterion_scores.items():
        if len(scores) >= 2:
            per_criterion_variance[name] = statistics.variance(scores)
        else:
            per_criterion_variance[name] = 0.0

    return ConsistencyResult(
        verdicts=verdicts,
        majority_passed=majority_passed,
        per_criterion_variance=per_criterion_variance,
    )
