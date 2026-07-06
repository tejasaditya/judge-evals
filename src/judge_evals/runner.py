"""Async batch judge runner.

Runs samples through a judge model via litellm, with caching, retries on parse
failure, and concurrency control.
"""

from __future__ import annotations

import asyncio
import logging

import litellm

from judge_evals.cache import VerdictCache
from judge_evals.prompt import PROMPT_VERSION, ParseError, build_judge_prompt, parse_judge_response
from judge_evals.types import JudgeVerdict, Rubric, Sample

__all__ = ["judge_sample", "judge_batch"]

logger = logging.getLogger(__name__)


async def judge_sample(
    sample: Sample,
    rubric: Rubric,
    model: str,
    cache: VerdictCache | None = None,
    *,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> JudgeVerdict:
    """Judge a single sample against a rubric using the given model.

    1. Check cache → return cached verdict if present.
    2. Build prompt → call litellm.acompletion → parse response.
    3. On ``ParseError``, retry up to ``max_retries`` times.
    4. Store result in cache before returning.

    Raises ``ParseError`` if all retries are exhausted.
    """
    # Cache lookup
    if cache is not None:
        cached = await cache.get(
            sample.content_hash, rubric.name, rubric.version, model, PROMPT_VERSION
        )
        if cached is not None:
            logger.debug("Cache hit for sample %s", sample.id or sample.content_hash[:12])
            return cached

    messages = build_judge_prompt(sample, rubric)
    last_error: ParseError | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            raw = response.choices[0].message.content
            scores = parse_judge_response(raw, rubric)

            verdict = JudgeVerdict(
                rubric_name=rubric.name,
                rubric_version=rubric.version,
                judge_model=model,
                scores=scores,
            )

            # Cache store
            if cache is not None:
                await cache.put(
                    sample.content_hash, rubric.name, rubric.version, model, PROMPT_VERSION, verdict
                )

            return verdict

        except ParseError as e:
            last_error = e
            logger.warning(
                "Parse error on attempt %d/%d for sample %s: %s",
                attempt,
                max_retries,
                sample.id or sample.content_hash[:12],
                e,
            )
            if attempt < max_retries:
                await asyncio.sleep(0.1 * (2 ** (attempt - 1)))  # brief backoff

    raise last_error  # type: ignore[misc]


async def judge_batch(
    samples: list[Sample],
    rubric: Rubric,
    model: str,
    cache: VerdictCache | None = None,
    *,
    concurrency: int = 5,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> list[JudgeVerdict]:
    """Judge a batch of samples with bounded concurrency.

    Returns verdicts in the same order as the input samples.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _run(s: Sample) -> JudgeVerdict:
        async with sem:
            return await judge_sample(
                s, rubric, model, cache, max_retries=max_retries, temperature=temperature
            )

    tasks = [asyncio.create_task(_run(s)) for s in samples]
    return list(await asyncio.gather(*tasks))
