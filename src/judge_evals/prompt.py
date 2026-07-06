"""Prompt builder: rubric + sample → judge LLM messages.

Builds the system and user messages for the judge model, specifying a JSON output
schema. Also parses the judge's raw JSON response back into CriterionScore objects.
"""

from __future__ import annotations

import json

from judge_evals.types import Criterion, CriterionScore, Rubric, Sample

__all__ = ["PROMPT_VERSION", "build_judge_prompt", "parse_judge_response"]

PROMPT_VERSION = "v1"


def _criterion_block(c: Criterion) -> str:
    """Human-readable description of one criterion for the system prompt."""
    req = " [REQUIRED — failing this fails the overall verdict]" if c.required else ""
    return (
        f"- **{c.name}**{req}\n"
        f"  Description: {c.description}\n"
        f"  Score range: {c.scale.min}–{c.scale.max} (higher is better)"
    )


def _json_schema(rubric: Rubric) -> str:
    """The JSON schema the judge must produce."""
    example_score = {
        "criterion_name": "<criterion name>",
        "score": "<int within the criterion's scale>",
        "rationale": "<1-2 sentence justification>",
        "passed": "<true if score meets acceptable threshold, false otherwise>",
    }
    schema = {"scores": [example_score]}
    return json.dumps(schema, indent=2)


def build_judge_prompt(sample: Sample, rubric: Rubric) -> list[dict[str, str]]:
    """Build system + user messages for the judge LLM.

    Returns a list of ``{"role": ..., "content": ...}`` dicts ready for litellm.
    """
    criteria_text = "\n".join(_criterion_block(c) for c in rubric.criteria)
    names_csv = ", ".join(c.name for c in rubric.criteria)

    system = (
        "You are an expert evaluator. You will be given a sample (input and output, "
        "possibly with context and a reference answer) and a rubric with scoring criteria.\n\n"
        "Score the output on EVERY criterion listed below. For each criterion, decide whether "
        "the output passes (meets an acceptable quality bar — generally the upper half of the "
        "scale) or fails.\n\n"
        f"## Criteria\n\n{criteria_text}\n\n"
        "## Output format\n\n"
        "Respond with ONLY valid JSON matching this schema (no markdown fences, no extra text):\n\n"
        f"```\n{_json_schema(rubric)}\n```\n\n"
        f"You MUST include a score for every criterion: {names_csv}.\n"
        "Do NOT add any text outside the JSON object."
    )

    user_parts = [f"**Input:**\n{sample.input}\n", f"**Output:**\n{sample.output}\n"]
    if sample.context is not None:
        user_parts.append(f"**Context:**\n{sample.context}\n")
    if sample.reference is not None:
        user_parts.append(f"**Reference answer:**\n{sample.reference}\n")
    user = "\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class ParseError(Exception):
    """Raised when the judge's response cannot be parsed into valid scores."""


def parse_judge_response(raw: str, rubric: Rubric) -> list[CriterionScore]:
    """Parse the judge model's JSON response into CriterionScore objects.

    Validates:
    - The response is valid JSON with a ``scores`` list.
    - Every criterion in the rubric has exactly one score entry.
    - Each score is within the criterion's scale bounds.

    Raises :class:`ParseError` on any validation failure.
    """
    # Strip markdown fences if the model wrapped its response
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON from judge: {e}") from e

    if not isinstance(data, dict) or "scores" not in data:
        raise ParseError("Judge response must be a JSON object with a 'scores' key")

    raw_scores: list[dict] = data["scores"]
    if not isinstance(raw_scores, list):
        raise ParseError("'scores' must be a list")

    # Build lookup from rubric for validation
    criteria_by_name: dict[str, Criterion] = {c.name: c for c in rubric.criteria}

    # Check all criteria are present
    returned_names = {s.get("criterion_name") for s in raw_scores}
    missing = set(criteria_by_name.keys()) - returned_names
    if missing:
        raise ParseError(f"Missing criteria in judge response: {sorted(missing)}")

    result: list[CriterionScore] = []
    for entry in raw_scores:
        name = entry.get("criterion_name")
        if name not in criteria_by_name:
            raise ParseError(f"Unknown criterion in judge response: {name!r}")

        criterion = criteria_by_name[name]
        score = entry.get("score")
        if not isinstance(score, int):
            raise ParseError(f"Score for {name!r} must be an integer, got {score!r}")
        if not criterion.scale.contains(score):
            raise ParseError(
                f"Score {score} for {name!r} is outside scale "
                f"[{criterion.scale.min}, {criterion.scale.max}]"
            )

        result.append(
            CriterionScore(
                criterion_name=name,
                score=score,
                rationale=str(entry.get("rationale", "")),
                passed=bool(entry.get("passed", False)),
                required=criterion.required,
            )
        )

    return result
