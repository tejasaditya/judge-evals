"""Tests for the typer CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml
from typer.testing import CliRunner

from judge_evals.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--rubric" in result.output
    assert "--judge" in result.output


def test_run_with_mock(tmp_path):
    """End-to-end CLI run with mocked litellm."""
    # Create dataset JSONL
    dataset_path = tmp_path / "samples.jsonl"
    dataset_path.write_text(
        json.dumps({"input": "What is 2+2?", "output": "4"})
        + "\n"
        + json.dumps({"input": "What is 3+3?", "output": "6"})
        + "\n"
    )

    # Create rubric YAML
    rubric_path = tmp_path / "rubric.yaml"
    rubric_data = {
        "name": "math",
        "version": "v1",
        "criteria": [
            {
                "name": "correctness",
                "description": "Is it correct?",
                "required": True,
                "scale": {"min": 1, "max": 5},
            }
        ],
    }
    rubric_path.write_text(yaml.dump(rubric_data))

    # Mock litellm response
    mock_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "scores": [
                                {
                                    "criterion_name": "correctness",
                                    "score": 5,
                                    "rationale": "Correct",
                                    "passed": True,
                                }
                            ]
                        }
                    )
                )
            )
        ]
    )
    mock = AsyncMock(return_value=mock_resp)

    cache_path = tmp_path / "cache.sqlite"

    with patch("judge_evals.runner.litellm.acompletion", mock):
        result = runner.invoke(
            app,
            [
                "--dataset",
                str(dataset_path),
                "--rubric",
                str(rubric_path),
                "--judge",
                "test-model",
                "--cache-path",
                str(cache_path),
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "2/2" in result.output or "passed" in result.output.lower()
