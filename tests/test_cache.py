"""Tests for the async SQLite verdict cache."""

from __future__ import annotations

import pytest

from judge_evals.cache import VerdictCache
from judge_evals.types import CriterionScore, JudgeVerdict


def _verdict(rubric_name: str = "test", score: int = 5) -> JudgeVerdict:
    return JudgeVerdict(
        rubric_name=rubric_name,
        rubric_version="v1",
        judge_model="test-model",
        scores=[
            CriterionScore(
                criterion_name="accuracy",
                score=score,
                rationale="good",
                passed=True,
                required=True,
            )
        ],
    )


@pytest.mark.asyncio
async def test_put_then_get(tmp_path):
    db_path = tmp_path / "test.sqlite"
    verdict = _verdict()

    async with VerdictCache(db_path) as cache:
        await cache.put("hash1", "rubric", "v1", "model", "pv1", verdict)
        result = await cache.get("hash1", "rubric", "v1", "model", "pv1")

    assert result is not None
    assert result.rubric_name == "test"
    assert result.scores[0].score == 5


@pytest.mark.asyncio
async def test_cache_miss_returns_none(tmp_path):
    db_path = tmp_path / "test.sqlite"

    async with VerdictCache(db_path) as cache:
        result = await cache.get("nonexistent", "rubric", "v1", "model", "pv1")

    assert result is None


@pytest.mark.asyncio
async def test_different_keys_no_collision(tmp_path):
    db_path = tmp_path / "test.sqlite"
    v1 = _verdict(score=5)
    v2 = _verdict(score=1)

    async with VerdictCache(db_path) as cache:
        await cache.put("hash_a", "rubric", "v1", "model", "pv1", v1)
        await cache.put("hash_b", "rubric", "v1", "model", "pv1", v2)

        r1 = await cache.get("hash_a", "rubric", "v1", "model", "pv1")
        r2 = await cache.get("hash_b", "rubric", "v1", "model", "pv1")

    assert r1 is not None and r1.scores[0].score == 5
    assert r2 is not None and r2.scores[0].score == 1


@pytest.mark.asyncio
async def test_rubric_version_change_invalidates(tmp_path):
    """Changing rubric version = different cache key = cache miss."""
    db_path = tmp_path / "test.sqlite"
    verdict = _verdict()

    async with VerdictCache(db_path) as cache:
        await cache.put("hash1", "rubric", "v1", "model", "pv1", verdict)
        result = await cache.get("hash1", "rubric", "v2", "model", "pv1")

    assert result is None  # different version = miss


@pytest.mark.asyncio
async def test_clear(tmp_path):
    db_path = tmp_path / "test.sqlite"
    verdict = _verdict()

    async with VerdictCache(db_path) as cache:
        await cache.put("hash1", "rubric", "v1", "model", "pv1", verdict)
        await cache.clear()
        result = await cache.get("hash1", "rubric", "v1", "model", "pv1")

    assert result is None
