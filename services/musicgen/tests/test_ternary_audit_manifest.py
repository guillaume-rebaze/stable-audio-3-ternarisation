from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from audit_ternary_quality import load_split_manifest  # noqa: E402


def test_load_split_manifest_accepts_independent_corpus_selection_manifest(
    tmp_path: Path,
) -> None:
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    first = validation_dir / "first.npy"
    second = validation_dir / "second.npy"
    samples = [
        {"path": str(first), "parent_id": "parent-1"},
        {"path": str(second), "parent_id": "parent-2"},
    ]
    manifest_path = tmp_path / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "onus.ternary-quality/v3/authorized-independent-corpus",
                "status": "prepared",
                "validation": {
                    "sources": [
                        {"staged_latent": str(first), "parent_id": "parent-1"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    selected, report = load_split_manifest(manifest_path, samples, "validation")

    assert selected == [samples[0]]
    assert report["verified"] is True
    assert report["sample_count"] == 1
    assert report["parent_count"] == 1

