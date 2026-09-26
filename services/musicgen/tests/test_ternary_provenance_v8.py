from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from ternary_provenance_v8 import build_contract, validate_contract  # noqa: E402


def _make_dataset(root: Path) -> Path:
    selection = {
        "schema": "test",
        "staged_train_dir": "dataset/train",
        "validation": {"dir": "dataset/validation", "sources": []},
        "test": {"dir": "dataset/test", "sources": []},
    }
    for role, parent in (("train", "train-parent"), ("validation", "validation-parent"), ("test", "test-parent")):
        directory = root / "dataset" / role
        directory.mkdir(parents=True)
        latent = directory / "0000.npy"
        latent.write_bytes(f"latent-{role}".encode())
        (directory / "0000.json").write_text(
            json.dumps({"prompt": f"prompt-{role}", "src_relpath": f"{parent}.wav"}),
            encoding="utf-8",
        )
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return selection_path


def test_v8_contract_build_and_validate(tmp_path: Path) -> None:
    selection_path = _make_dataset(tmp_path)
    contract_path = tmp_path / "contract.json"
    build_contract(selection_path, contract_path, tmp_path)

    report = validate_contract(contract_path, tmp_path)

    assert report["valid"] is True
    assert report["sample_count"] == 3
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}


def test_v8_contract_rejects_mutated_latent(tmp_path: Path) -> None:
    selection_path = _make_dataset(tmp_path)
    contract_path = tmp_path / "contract.json"
    build_contract(selection_path, contract_path, tmp_path)
    (tmp_path / "dataset" / "validation" / "0000.npy").write_bytes(b"changed")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_contract(contract_path, tmp_path)
