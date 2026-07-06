"""Typer CLI for judge-evals.

Entry points:
    judge-evals run --dataset x.jsonl --rubric r.yaml --judge model-name
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from judge_evals.cache import VerdictCache
from judge_evals.runner import judge_batch
from judge_evals.types import JudgeVerdict, Rubric, Sample

app = typer.Typer(
    name="judge-evals",
    help="LLM-as-judge evaluation harness: rubric scoring offline and as a runtime retry gate.",
    add_completion=False,
)
console = Console()


def _load_samples(path: Path) -> list[Sample]:
    """Load samples from a JSONL file (one JSON object per line)."""
    samples: list[Sample] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(Sample.model_validate_json(line))
            except Exception as e:
                console.print(f"[red]Error parsing line {i} of {path}:[/red] {e}")
                raise typer.Exit(code=1) from e
    return samples


def _load_rubric(path: Path) -> Rubric:
    """Load a rubric from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return Rubric.model_validate(data)
    except Exception as e:
        console.print(f"[red]Error parsing rubric {path}:[/red] {e}")
        raise typer.Exit(code=1) from e


def _print_results(verdicts: list[JudgeVerdict], samples: list[Sample]) -> None:
    """Print a rich table of results."""
    table = Table(title="Judge Results", show_lines=True)
    table.add_column("Sample", style="cyan", no_wrap=True)

    # Collect all criterion names from the first verdict
    if not verdicts:
        console.print("[yellow]No results to display.[/yellow]")
        return

    criterion_names = [s.criterion_name for s in verdicts[0].scores]
    for name in criterion_names:
        table.add_column(name, justify="center")
    table.add_column("Overall", justify="center", style="bold")

    for i, (verdict, sample) in enumerate(zip(verdicts, samples, strict=True)):
        label = sample.id or f"#{i + 1}"
        row = [label]
        scores_by_name = {s.criterion_name: s for s in verdict.scores}
        for name in criterion_names:
            s = scores_by_name.get(name)
            if s is None:
                row.append("—")
            else:
                icon = "✅" if s.passed else "❌"
                row.append(f"{icon} {s.score} ")
        row.append("✅ PASS" if verdict.passed else "❌ FAIL")
        table.add_row(*row)

    console.print(table)

    passed = sum(1 for v in verdicts if v.passed)
    total = len(verdicts)
    console.print(f"\n[bold]{passed}/{total} samples passed.[/bold]")


@app.command()
def run(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to JSONL dataset file."),
    rubric: Path = typer.Option(..., "--rubric", "-r", help="Path to YAML rubric file."),
    judge: str = typer.Option(..., "--judge", "-j", help="Judge model name (any litellm model)."),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent judge calls."),
    cache_path: Path = typer.Option(
        ".judge_evals_cache.sqlite", "--cache-path", help="Path to sqlite cache file."
    ),
    max_retries: int = typer.Option(3, "--max-retries", help="Max retries on parse failure."),
) -> None:
    """Run the judge on a dataset against a rubric."""
    samples = _load_samples(dataset)
    rubric_obj = _load_rubric(rubric)

    console.print(
        f"[bold]Judging {len(samples)} samples with {judge} "
        f"against rubric '{rubric_obj.name}' (v{rubric_obj.version})[/bold]\n"
    )

    async def _run() -> list[JudgeVerdict]:
        async with VerdictCache(cache_path) as cache:
            return await judge_batch(
                samples,
                rubric_obj,
                judge,
                cache,
                concurrency=concurrency,
                max_retries=max_retries,
            )

    verdicts = asyncio.run(_run())
    _print_results(verdicts, samples)

    all_passed = all(v.passed for v in verdicts)
    raise typer.Exit(code=0 if all_passed else 1)
