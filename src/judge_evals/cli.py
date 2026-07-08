"""Typer CLI for judge-evals.

Entry points:
    judge-evals run   --dataset x.jsonl --rubric r.yaml --judge model-name
    judge-evals agree --dataset x.jsonl --labels y.jsonl --rubric r.yaml --judge model-name
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from judge_evals.agreement import AgreementReport, compute_agreement
from judge_evals.cache import VerdictCache
from judge_evals.labels import load_samples_and_labels
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
    """Print a rich table of judge results."""
    table = Table(title="Judge Results", show_lines=True)
    table.add_column("Sample", style="cyan", no_wrap=True)

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


def _print_agreement(reports: list[AgreementReport]) -> None:
    """Print a rich table of agreement metrics."""
    table = Table(title="Agreement: Judge vs. Human Labels", show_lines=True)
    table.add_column("Criterion", style="cyan", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("Exact Agreement", justify="center")
    table.add_column("Cohen's κ", justify="center")

    for r in reports:
        kappa_str = f"{r.kappa:.3f}" if r.n_samples > 0 else "—"
        agree_str = f"{r.exact_agreement:.1%}" if r.n_samples > 0 else "—"
        table.add_row(r.criterion_name, str(r.n_samples), agree_str, kappa_str)

    console.print(table)


def _write_agreement_report(reports: list[AgreementReport], model: str, output_path: Path) -> None:
    """Write agreement report as a markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Agreement Report: {model}",
        "",
        "| Criterion | N | Exact Agreement | Cohen's κ |",
        "|-----------|---|-----------------|-----------|",
    ]
    for r in reports:
        kappa_str = f"{r.kappa:.3f}" if r.n_samples > 0 else "—"
        agree_str = f"{r.exact_agreement:.1%}" if r.n_samples > 0 else "—"
        lines.append(f"| {r.criterion_name} | {r.n_samples} | {agree_str} | {kappa_str} |")

    # Add confusion matrices
    for r in reports:
        if r.n_samples > 0 and r.confusion_matrix:
            lines.append("")
            lines.append(f"### Confusion Matrix: {r.criterion_name}")
            lines.append("")
            lines.append(f"Rows = human, Columns = judge. Labels: {r.labels}")
            lines.append("")
            header = "| |" + "|".join(f" {lb} " for lb in r.labels) + "|"
            sep = "|---|" + "|".join("---" for _ in r.labels) + "|"
            lines.append(header)
            lines.append(sep)
            for row_label, row in zip(r.labels, r.confusion_matrix, strict=True):
                row_str = f"| **{row_label}** |" + "|".join(f" {v} " for v in row) + "|"
                lines.append(row_str)

    lines.append("")
    output_path.write_text("\n".join(lines))
    console.print(f"\n[green]Report written to {output_path}[/green]")


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


@app.command()
def agree(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to JSONL dataset file."),
    labels_path: Path = typer.Option(
        ..., "--labels", "-l", help="Path to JSONL human labels file."
    ),
    rubric: Path = typer.Option(..., "--rubric", "-r", help="Path to YAML rubric file."),
    judge: str = typer.Option(..., "--judge", "-j", help="Judge model name (any litellm model)."),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent judge calls."),
    cache_path: Path = typer.Option(
        ".judge_evals_cache.sqlite", "--cache-path", help="Path to sqlite cache file."
    ),
    max_retries: int = typer.Option(3, "--max-retries", help="Max retries on parse failure."),
    report_path: Path = typer.Option(
        "reports/agreement.md", "--report", help="Path to write the markdown report."
    ),
) -> None:
    """Compute agreement between judge verdicts and human labels."""
    rubric_obj = _load_rubric(rubric)

    # Load and pair samples with labels
    samples, human_labels = load_samples_and_labels(dataset, labels_path)
    if not samples:
        console.print("[red]No matched sample–label pairs found.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Evaluating agreement on {len(samples)} samples with {judge} "
        f"against rubric '{rubric_obj.name}' (v{rubric_obj.version})[/bold]\n"
    )

    # Run judge on the matched samples
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

    # Compute agreement
    reports = compute_agreement(verdicts, human_labels, rubric_obj)
    _print_agreement(reports)
    _write_agreement_report(reports, judge, report_path)
