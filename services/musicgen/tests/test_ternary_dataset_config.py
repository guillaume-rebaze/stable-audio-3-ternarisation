from __future__ import annotations

import json
from pathlib import Path

import pytest

sys_path = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(sys_path))

from prepare_ternary_dataset import TOP_KEYS, DATASET_KEYS, require_exact_keys  # noqa: E402


def test_v3_config_has_exact_top_level_contract() -> None:
    config = json.loads(Path("configs/ternary_quality_v3.json").read_text())
    assert set(config) == TOP_KEYS
    require_exact_keys(config["dataset"], DATASET_KEYS, "dataset")
    assert config["quantizer"]["mode"] == "symmetric"


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="keys mismatch"):
        require_exact_keys({"train_dir": "x", "unknown": True}, DATASET_KEYS, "dataset")
