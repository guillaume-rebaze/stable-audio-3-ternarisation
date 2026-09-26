"""Strict, reproducible runtime configuration for the ternary trainer.

The old trainer wrote a JSON file but ignored several of its intended values.
This module makes the consumed configuration explicit and rejects typos before
any model or dataset is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_KEYS = {
    "dataset_dir",
    "teacher_weights",
    "output_dir",
    "group_size",
    "quantizer_mode",
    "crop_len",
    "start_block",
    "end_block",
    "steps_per_block",
    "max_blocks",
    "max_samples",
    "polish_steps",
    "polish_learning_rate",
    "polish_grad_clip",
    "seconds",
    "seed",
    "resume_checkpoint",
    "checkpoint_every_blocks",
    "checkpoint_every_steps",
    "resume_step_checkpoint",
    "learning_rate",
    "learning_rate_end",
    "optimizer_eps",
    "weight_decay",
    "gradient_clip",
    "gradient_accumulation",
    "sigma_grid",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_dir": "output/sample-expertise-pilot/universal-dataset/latents-12s",
    "teacher_weights": str(
        Path.home()
        / ".cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz"
    ),
    "output_dir": "output/sample-expertise-pilot/ternary-quality-recovery/group64-core",
    "group_size": 64,
    "quantizer_mode": "affine_centered",
    "crop_len": 128,
    "start_block": 0,
    "end_block": None,
    "steps_per_block": 200,
    "max_blocks": 24,
    "max_samples": 0,
    "polish_steps": 0,
    "polish_learning_rate": 1e-6,
    "polish_grad_clip": 0.10,
    "seconds": 12.0,
    "seed": 42,
    "resume_checkpoint": None,
    "checkpoint_every_blocks": 1,
    "checkpoint_every_steps": 50,
    "resume_step_checkpoint": None,
    "learning_rate": 5e-5,
    "learning_rate_end": 5e-6,
    "optimizer_eps": 1e-8,
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "gradient_accumulation": 1,
    "sigma_grid": [0.95, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10],
}


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _normalise_path(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string or null")
    return str(Path(value).expanduser())


def _validate(values: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(values)
    for key in ("dataset_dir", "teacher_weights", "output_dir", "resume_checkpoint", "resume_step_checkpoint"):
        resolved[key] = _normalise_path(resolved[key], key)
    if resolved["group_size"] not in (32, 64, 128):
        raise ValueError("group_size must be one of 32, 64, 128")
    if resolved["quantizer_mode"] not in (
        "symmetric",
        "learned_symmetric",
        "learned_affine",
        "affine_centered",
    ):
        raise ValueError(
            "quantizer_mode must be symmetric, learned_symmetric, learned_affine, or affine_centered"
        )
    if int(resolved["start_block"]) < 0:
        raise ValueError("start_block must be zero or positive")
    resolved["start_block"] = int(resolved["start_block"])
    for key in (
        "crop_len",
        "steps_per_block",
        "max_blocks",
        "checkpoint_every_blocks",
    ):
        if int(resolved[key]) <= 0:
            raise ValueError(f"{key} must be positive")
        resolved[key] = int(resolved[key])
    if resolved["end_block"] is not None:
        resolved["end_block"] = int(resolved["end_block"])
        if resolved["end_block"] < resolved["start_block"]:
            raise ValueError("end_block must be >= start_block")
    for key in ("max_samples", "polish_steps", "checkpoint_every_steps", "gradient_accumulation"):
        if int(resolved[key]) < 0:
            raise ValueError(f"{key} must be zero or positive")
        resolved[key] = int(resolved[key])
    if resolved["gradient_accumulation"] < 1:
        raise ValueError("gradient_accumulation must be at least one")
    for key in (
        "polish_learning_rate",
        "polish_grad_clip",
        "seconds",
        "learning_rate",
        "learning_rate_end",
        "optimizer_eps",
        "weight_decay",
        "gradient_clip",
    ):
        resolved[key] = float(resolved[key])
        if key not in {"weight_decay"} and resolved[key] <= 0:
            raise ValueError(f"{key} must be positive")
        if key == "weight_decay" and resolved[key] < 0:
            raise ValueError("weight_decay must be zero or positive")
    grid = resolved["sigma_grid"]
    if not isinstance(grid, list) or not grid:
        raise ValueError("sigma_grid must be a non-empty list")
    resolved["sigma_grid"] = [float(value) for value in grid]
    if any(not 0.0 < value < 1.0 for value in resolved["sigma_grid"]):
        raise ValueError("sigma_grid values must be strictly between zero and one")
    resolved["seed"] = int(resolved["seed"])
    return resolved


def resolve_config(config_path: Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a strict config and apply only explicitly supplied CLI overrides."""
    if config_path is None:
        values = dict(DEFAULT_CONFIG)
        source = None
    else:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "onus.ternary-quality/v6-trainer-config":
            raise ValueError(
                "unsupported trainer config schema; expected "
                "onus.ternary-quality/v6-trainer-config"
            )
        values = dict(payload.get("values", {}))
        _exact_keys(values, CONFIG_KEYS, "trainer config values")
        source = str(config_path)
    for key, value in (overrides or {}).items():
        if value is not None:
            if key not in CONFIG_KEYS:
                raise ValueError(f"unknown trainer override: {key}")
            values[key] = value
    resolved = _validate(values)
    return {
        "schema": "onus.ternary-quality/v6-resolved-config",
        "source_config": source,
        "overrides": {key: value for key, value in (overrides or {}).items() if value is not None},
        "values": resolved,
    }
