"""M3 demo: a small agent wrapped in a JudgeGate over 20 synthetic tasks.

Reports the **retry-rescue rate** — the fraction of tasks that failed the judge on the
first attempt but passed after a judge-triggered retry with feedback injected.

The synthetic agent is deliberately flaky: on a subset of tasks (chosen by a fixed seed)
its first answer is poor, but when handed the judge's feedback it produces the good
answer. The judge itself is a real model called via litellm, so this needs an API key in
`.env` for whichever `--judge` backend you use.

Prerequisite: this exercises ``JudgeGate.gate``, which is the hand-written state machine
left as a ``TODO(user)`` in ``src/judge_evals/gate.py``. Until you implement it, the run
stops early with a clear message. Once implemented, run:

    make demo                 # uses JUDGE_MODEL or defaults to gpt-4o-mini
    JUDGE_MODEL=ollama/llama3 uv run python scripts/gate_demo.py
"""

from __future__ import annotations

import asyncio
import os
import random
import subprocess
from dataclasses import dataclass

from dotenv import load_dotenv

from judge_evals.cache import VerdictCache
from judge_evals.gate import GatePolicy, GateTrace, JudgeGate, rescue_rate
from judge_evals.types import Criterion, Rubric, Sample, Scale

SEED = 7
N_TASKS = 20


@dataclass
class Task:
    """A synthetic task with a question, the good answer, and a deliberately poor one."""

    id: str
    question: str
    good_answer: str
    poor_answer: str
    starts_poor: bool  # if True the agent's first attempt is the poor answer


def build_tasks(seed: int = SEED, n: int = N_TASKS) -> list[Task]:
    """Build ``n`` deterministic synthetic QA tasks; ~half start with a poor answer."""
    rng = random.Random(seed)
    base = [
        ("What is the capital of France?", "Paris.", "A city in Europe."),
        ("What is 12 x 12?", "144.", "Around 140."),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen.", "A British author."),
        ("What is the boiling point of water at sea level in Celsius?", "100°C.", "Pretty hot."),
        ("What is the chemical symbol for gold?", "Au.", "Gd, maybe."),
    ]
    tasks: list[Task] = []
    for i in range(n):
        q, good, poor = base[i % len(base)]
        tasks.append(
            Task(
                id=f"task-{i + 1:02d}",
                question=f"{q} (variant {i // len(base) + 1})",
                good_answer=good,
                poor_answer=poor,
                starts_poor=rng.random() < 0.5,
            )
        )
    return tasks


def make_agent(tasks: list[Task]):
    """A flaky agent: returns the poor answer first (for tasks that start poor), then the
    good answer once feedback is supplied."""
    by_question = {t.question: t for t in tasks}

    async def agent(input: str, feedback: str | None = None) -> str:
        task = by_question[input]
        if feedback is None and task.starts_poor:
            return task.poor_answer
        return task.good_answer

    return agent


def build_rubric() -> Rubric:
    return Rubric(
        name="qa-correctness",
        version="v1",
        criteria=[
            Criterion(
                name="correctness",
                description="Is the answer factually correct and complete for the question?",
                scale=Scale(min=1, max=5),
                required=True,
            )
        ],
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


async def run_demo(judge_model: str) -> list[GateTrace]:
    tasks = build_tasks()
    agent = make_agent(tasks)
    rubric = build_rubric()
    policy = GatePolicy(max_retries=2)

    traces: list[GateTrace] = []
    async with VerdictCache(".gate_demo_cache.sqlite") as cache:
        gate = JudgeGate(judge_model=judge_model, cache=cache, policy=policy)
        for task in tasks:
            sample = Sample(id=task.id, input=task.question, output="")
            # JudgeGate.gate calls the agent, judges its output, and retries with feedback.
            trace = await gate.gate(agent, rubric, sample)
            traces.append(trace)
    return traces


def main() -> None:
    load_dotenv()
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    print(f"Gate demo — seed={SEED}, tasks={N_TASKS}, judge={judge_model}, git={_git_sha()}")

    try:
        traces = asyncio.run(run_demo(judge_model))
    except NotImplementedError as e:
        print("\nJudgeGate.gate is not implemented yet (it is the hand-written state")
        print(f"machine — TODO(user) in src/judge_evals/gate.py):\n  {e}")
        print("Implement it, then re-run `make demo`.")
        raise SystemExit(2) from e

    n = len(traces)
    initially_failed = sum(1 for t in traces if t.attempts and not t.attempts[0].verdict.passed)
    rescued = sum(1 for t in traces if t.rescued)
    passed = sum(1 for t in traces if t.passed)

    print("\n=== Gate demo results ===")
    print(f"Tasks:                 {n}")
    print(f"Passed overall:        {passed}/{n}")
    print(f"Failed on 1st attempt: {initially_failed}")
    print(f"Rescued by retry:      {rescued}")
    print(f"Retry-rescue rate:     {rescue_rate(traces):.1%}")


if __name__ == "__main__":
    main()
