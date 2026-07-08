"""Run the public benchmark and generate the agreement report.

This script executes the judge model against the example dataset and human labels,
then computes agreement metrics and writes reports/agreement.md.

Note: This calls a real model (requires API keys in .env) and is not run in CI.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from typer.testing import CliRunner

from judge_evals.cli import app


def main() -> None:
    # Load secrets (HF_TOKEN, OPENAI_API_KEY, etc.)
    load_dotenv()

    # Default to a small, fast model if not specified
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    print(f"Running benchmark with judge model: {judge_model}")

    dataset = Path("examples/samples.jsonl")
    labels = Path("examples/labels.jsonl")
    rubric = Path("examples/rubric.yaml")
    report = Path("reports/agreement.md")

    if not dataset.exists() or not labels.exists() or not rubric.exists():
        print("Error: Example files not found. Are you running from the repo root?")
        exit(1)

    # Use the CLI to run the `agree` subcommand
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "agree",
            "--dataset",
            str(dataset),
            "--labels",
            str(labels),
            "--rubric",
            str(rubric),
            "--judge",
            judge_model,
            "--report",
            str(report),
        ],
    )

    print(result.output)

    if result.exit_code != 0:
        print(f"Benchmark failed with exit code {result.exit_code}")
        exit(result.exit_code)
    else:
        print(f"Benchmark completed successfully. See {report} for results.")


if __name__ == "__main__":
    main()
