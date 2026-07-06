"""Core domain types for judge-evals.

These pydantic models are the vocabulary the whole library speaks in:

- A :class:`Sample` is one thing to be judged (input/output, plus optional context
  and a reference answer).
- A :class:`Rubric` is an ordered list of :class:`Criterion`, each scored on a
  :class:`Scale`. A criterion may be ``required``, meaning a failure on it fails the
  whole verdict.
- A :class:`JudgeVerdict` is the judge's result for one sample against one rubric: a
  :class:`CriterionScore` per criterion, from which overall pass/fail is derived.

The models are deliberately self-contained and JSON-round-trippable so a verdict can be
cached (M1) and replayed without needing the original ``Rubric`` object in memory.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "Scale",
    "Criterion",
    "Rubric",
    "CriterionScore",
    "JudgeVerdict",
    "Sample",
]


class Scale(BaseModel):
    """An integer scoring scale, inclusive on both ends (e.g. 1..5)."""

    min: int = 1
    max: int = 5

    @model_validator(mode="after")
    def _check_bounds(self) -> Scale:
        if self.min >= self.max:
            raise ValueError(f"Scale.min ({self.min}) must be < Scale.max ({self.max})")
        return self

    def contains(self, score: int) -> bool:
        """True if ``score`` falls within ``[min, max]``."""
        return self.min <= score <= self.max


class Criterion(BaseModel):
    """A single thing the judge scores, on its own scale.

    ``required`` marks a gating criterion: if the judge fails it, the overall
    :class:`JudgeVerdict` fails regardless of the other criteria.
    """

    name: str
    description: str
    scale: Scale = Field(default_factory=Scale)
    required: bool = False


class Rubric(BaseModel):
    """An ordered set of criteria, versioned so it can key a cache.

    ``version`` participates in the M1 cache key
    ``(sample, rubric, judge-model, prompt-version)``; bump it whenever the rubric's
    semantics change so stale verdicts are not reused.
    """

    name: str
    criteria: list[Criterion]
    version: str = "v1"

    @model_validator(mode="after")
    def _check_criteria(self) -> Rubric:
        if not self.criteria:
            raise ValueError("Rubric.criteria must be non-empty")
        names = [c.name for c in self.criteria]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Rubric criterion names must be unique; duplicates: {dupes}")
        return self

    def criterion(self, name: str) -> Criterion:
        """Look up a criterion by name, raising ``KeyError`` if absent."""
        for c in self.criteria:
            if c.name == name:
                return c
        raise KeyError(name)


class CriterionScore(BaseModel):
    """The judge's score for one criterion.

    ``required`` is copied from the source :class:`Criterion` at scoring time so the
    enclosing :class:`JudgeVerdict` can compute pass/fail on its own, without the
    original :class:`Rubric` in hand.
    """

    criterion_name: str
    score: int
    rationale: str
    passed: bool
    required: bool = False


class JudgeVerdict(BaseModel):
    """A judge's full result for one sample against one rubric."""

    rubric_name: str
    rubric_version: str
    judge_model: str
    scores: list[CriterionScore]

    @property
    def passed(self) -> bool:
        """Overall pass: no *required* criterion failed.

        Optional criteria never gate the verdict. A verdict with no required criteria
        therefore always passes.
        """
        return all(s.passed for s in self.scores if s.required)


class Sample(BaseModel):
    """One item to be judged."""

    input: str
    output: str
    id: str | None = None
    context: str | None = None
    reference: str | None = None

    @property
    def content_hash(self) -> str:
        """Stable sha256 over the sample's content.

        Independent of ``id`` (a label, not content) so two samples with identical
        content hash the same. This is a building block of the M1 cache key.
        """
        payload = self.model_dump(exclude={"id"})
        canonical = "\x1f".join(f"{k}={payload[k]!r}" for k in sorted(payload))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
