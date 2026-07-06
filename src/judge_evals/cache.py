"""Async SQLite cache for judge verdicts.

Keyed on ``(sample.content_hash, rubric.name, rubric.version, judge_model,
prompt_version)`` — so bumping a rubric version or prompt version automatically
invalidates stale entries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import aiosqlite

from judge_evals.types import JudgeVerdict

__all__ = ["VerdictCache"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS verdicts (
    cache_key TEXT PRIMARY KEY,
    verdict_json TEXT NOT NULL
)
"""


def _make_key(
    content_hash: str,
    rubric_name: str,
    rubric_version: str,
    judge_model: str,
    prompt_version: str,
) -> str:
    """Deterministic cache key from the five components."""
    raw = "\x1f".join([content_hash, rubric_name, rubric_version, judge_model, prompt_version])
    return hashlib.sha256(raw.encode()).hexdigest()


class VerdictCache:
    """Async SQLite-backed verdict cache.

    Usage::

        async with VerdictCache() as cache:
            verdict = await cache.get(key_args)
            if verdict is None:
                verdict = await run_judge(...)
                await cache.put(key_args, verdict)
    """

    def __init__(self, path: str | Path = ".judge_evals_cache.sqlite") -> None:
        self._path = str(path)
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> VerdictCache:
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute(_CREATE_TABLE)
        await self._db.commit()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def get(
        self,
        content_hash: str,
        rubric_name: str,
        rubric_version: str,
        judge_model: str,
        prompt_version: str,
    ) -> JudgeVerdict | None:
        """Return a cached verdict, or ``None`` on cache miss."""
        assert self._db is not None, "VerdictCache not opened (use `async with`)"
        key = _make_key(content_hash, rubric_name, rubric_version, judge_model, prompt_version)
        async with self._db.execute(
            "SELECT verdict_json FROM verdicts WHERE cache_key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return JudgeVerdict.model_validate_json(row[0])

    async def put(
        self,
        content_hash: str,
        rubric_name: str,
        rubric_version: str,
        judge_model: str,
        prompt_version: str,
        verdict: JudgeVerdict,
    ) -> None:
        """Store a verdict in the cache (upsert)."""
        assert self._db is not None, "VerdictCache not opened (use `async with`)"
        key = _make_key(content_hash, rubric_name, rubric_version, judge_model, prompt_version)
        verdict_json = verdict.model_dump_json()
        await self._db.execute(
            "INSERT OR REPLACE INTO verdicts (cache_key, verdict_json) VALUES (?, ?)",
            (key, verdict_json),
        )
        await self._db.commit()

    async def clear(self) -> None:
        """Delete all cached verdicts."""
        assert self._db is not None, "VerdictCache not opened (use `async with`)"
        await self._db.execute("DELETE FROM verdicts")
        await self._db.commit()
