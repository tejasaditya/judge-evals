"""judge-evals: an LLM-as-judge evaluation harness."""

from judge_evals.agreement import AgreementReport, compute_agreement
from judge_evals.cache import VerdictCache
from judge_evals.cli import app
from judge_evals.labels import HumanLabel, load_labels
from judge_evals.prompt import PROMPT_VERSION, build_judge_prompt, parse_judge_response
from judge_evals.runner import judge_batch, judge_sample
from judge_evals.types import (
    Criterion,
    CriterionScore,
    JudgeVerdict,
    Rubric,
    Sample,
    Scale,
)

__all__ = [
    # Types
    "Scale",
    "Criterion",
    "Rubric",
    "CriterionScore",
    "JudgeVerdict",
    "Sample",
    "HumanLabel",
    "AgreementReport",
    # Prompt
    "PROMPT_VERSION",
    "build_judge_prompt",
    "parse_judge_response",
    # Cache
    "VerdictCache",
    # Runner
    "judge_sample",
    "judge_batch",
    # Agreement
    "compute_agreement",
    "load_labels",
    # CLI
    "app",
    "main",
]


def main() -> None:
    """Entry point for the ``judge-evals`` console script."""
    app()
