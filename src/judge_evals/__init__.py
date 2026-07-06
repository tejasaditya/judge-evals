"""judge-evals: an LLM-as-judge evaluation harness."""

from judge_evals.types import (
    Criterion,
    CriterionScore,
    JudgeVerdict,
    Rubric,
    Sample,
    Scale,
)

__all__ = [
    "Scale",
    "Criterion",
    "Rubric",
    "CriterionScore",
    "JudgeVerdict",
    "Sample",
    "main",
]


def main() -> None:
    """Placeholder entry point; replaced by the typer CLI in M1."""
    print("judge-evals — LLM-as-judge harness. CLI arrives in M1.")
