"""Tests for labels.py."""

from pathlib import Path

import pytest

from judge_evals.labels import load_labels, load_samples_and_labels


def test_load_labels(tmp_path: Path):
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text(
        '{"sample_id": "1", "scores": {"accuracy": 5}}\n'
        '{"sample_id": "2", "scores": {"accuracy": 1}, "passed": {"accuracy": false}}\n'
    )
    labels = load_labels(labels_file)
    assert len(labels) == 2
    assert labels[0].sample_id == "1"
    assert labels[0].scores["accuracy"] == 5
    assert labels[0].passed is None

    assert labels[1].sample_id == "2"
    assert labels[1].passed is not None
    assert labels[1].passed["accuracy"] is False


def test_load_labels_malformed(tmp_path: Path):
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text("not json\n")
    with pytest.raises(ValueError, match="Error parsing label"):
        load_labels(labels_file)


def test_load_samples_and_labels(tmp_path: Path):
    samples_file = tmp_path / "samples.jsonl"
    labels_file = tmp_path / "labels.jsonl"

    samples_file.write_text(
        '{"id": "1", "input": "q1", "output": "a1"}\n'
        '{"id": "2", "input": "q2", "output": "a2"}\n'
        '{"id": "3", "input": "q3", "output": "a3"}\n'  # no label for 3
        '{"input": "q4", "output": "a4"}\n'  # no id, skipped
    )
    labels_file.write_text(
        '{"sample_id": "1", "scores": {"accuracy": 5}}\n'
        '{"sample_id": "2", "scores": {"accuracy": 1}}\n'
        '{"sample_id": "4", "scores": {"accuracy": 3}}\n'  # no sample for 4
    )

    samples, labels = load_samples_and_labels(samples_file, labels_file)
    assert len(samples) == 2
    assert len(labels) == 2
    assert samples[0].id == "1"
    assert labels[0].sample_id == "1"
    assert samples[1].id == "2"
    assert labels[1].sample_id == "2"
