from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from ternary_run_config import CONFIG_KEYS, DEFAULT_CONFIG, resolve_config  # noqa: E402


def _payload(values: dict) -> dict:
    return {"schema": "onus.ternary-quality/v6-trainer-config", "values": values}


def test_v6_config_is_strict_and_resolves_values(tmp_path: Path) -> None:
    values = {
        "dataset_dir": "data",
        "teacher_weights": "teacher.npz",
        "output_dir": "out",
        "group_size": 64,
        "quantizer_mode": "symmetric",
        "crop_len": 128,
        "start_block": 0,
        "end_block": None,
        "steps_per_block": 10,
        "max_blocks": 2,
        "max_samples": 4,
        "polish_steps": 0,
        "polish_learning_rate": 1e-6,
        "polish_grad_clip": 0.1,
        "seconds": 12.0,
        "seed": 42,
        "resume_checkpoint": None,
        "checkpoint_every_blocks": 1,
        "checkpoint_every_steps": 5,
        "resume_step_checkpoint": None,
        "learning_rate": 1e-5,
        "learning_rate_end": 1e-6,
        "optimizer_eps": 1e-8,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "gradient_accumulation": 4,
        "sigma_grid": [0.95, 0.5, 0.1],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload(values)))
    result = resolve_config(path)
    assert set(result["values"]) == CONFIG_KEYS
    assert result["values"]["gradient_accumulation"] == 4
    assert result["values"]["sigma_grid"] == [0.95, 0.5, 0.1]


def test_v6_config_rejects_unknown_values(tmp_path: Path) -> None:
    values = {key: None for key in CONFIG_KEYS}
    values["group_size"] = 64
    values["quantizer_mode"] = "symmetric"
    values["unknown"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload(values)))
    with pytest.raises(ValueError, match="unknown"):
        resolve_config(path)


def test_v6_config_rejects_invalid_sigma_grid(tmp_path: Path) -> None:
    values = dict(DEFAULT_CONFIG)
    values["sigma_grid"] = [0.0, 0.5]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload(values)))
    with pytest.raises(ValueError, match="strictly between"):
        resolve_config(path)
