"""Human label ingestion for agreement evaluation.

Loads human-annotated labels from JSONL files and pairs them with samples
for agreement metric computation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from judge_evals.types import Sample

__all__ = ["HumanLabel", "load_labels", "load_samples_and_labels"]

logger = logging.getLogger(__name__)


class HumanLabel(BaseModel):
    """One human annotation for a sample.

    ``scores`` maps criterion name → integer score (matching the rubric's scale);
    these are what :func:`~judge_evals.agreement.compute_agreement` compares against.
    ``passed`` optionally maps criterion name → bool. It is carried through ingestion
    for downstream use but is not yet consumed by agreement computation, which
    operates on ``scores`` only.
    """

    sample_id: str
    scores: dict[str, int]
    passed: dict[str, bool] | None = None


def load_labels(path: Path | str) -> list[HumanLabel]:
    """Load human labels from a JSONL file (one JSON object per line).

    Raises ``ValueError`` on malformed lines.
    """
    path = Path(path)
    labels: list[HumanLabel] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                labels.append(HumanLabel.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Error parsing label at line {i} of {path}: {e}") from e
    return labels


def load_samples_and_labels(
    dataset_path: Path | str,
    labels_path: Path | str,
) -> tuple[list[Sample], list[HumanLabel]]:
    """Load samples and labels, returning only pairs with matching IDs.

    Samples without a matching label (or vice versa) are logged as warnings
    and excluded from the returned lists. Both lists are returned in the same
    order, aligned by ``sample.id == label.sample_id``.
    """
    # Load all samples
    samples_by_id: dict[str, Sample] = {}
    with open(dataset_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            s = Sample.model_validate_json(line)
            if s.id is None:
                logger.warning("Sample at line %d has no id — skipping for agreement", i)
                continue
            samples_by_id[s.id] = s

    all_labels = load_labels(labels_path)
    labels_by_id = {lb.sample_id: lb for lb in all_labels}

    # Find intersection
    common_ids = sorted(set(samples_by_id.keys()) & set(labels_by_id.keys()))

    missing_labels = set(samples_by_id.keys()) - set(labels_by_id.keys())
    missing_samples = set(labels_by_id.keys()) - set(samples_by_id.keys())

    if missing_labels:
        logger.warning("Samples without labels (excluded): %s", sorted(missing_labels))
    if missing_samples:
        logger.warning("Labels without samples (excluded): %s", sorted(missing_samples))

    paired_samples = [samples_by_id[sid] for sid in common_ids]
    paired_labels = [labels_by_id[sid] for sid in common_ids]

    return paired_samples, paired_labels
