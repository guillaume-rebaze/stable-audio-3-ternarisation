"""Shared inference-time numeric contract for ternary SA3 experiments."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


TARGET_CONTRACT_SCHEMA = "onus.ternary-quality/v7-target-contract"
STATE_CACHE_CONTRACT_SCHEMA = "onus.ternary-quality/v7-state-contract"


def timestep_tensor(sigma, batch_size: int = 1) -> mx.array:
    """Build the FP32 timestep consumed by the production DiT forward."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    value = mx.array(sigma, dtype=mx.float32)
    if value.size != 1:
        raise ValueError("sigma must be scalar")
    value = value.reshape((1,))
    result = mx.broadcast_to(value, (batch_size,))
    if result.dtype != mx.float32:
        raise TypeError(f"timestep dtype drift: expected float32, got {result.dtype}")
    return result


def pingpong_transition(
    x: mx.array,
    velocity: mx.array,
    sigma,
    next_sigma,
    noise: mx.array | None,
    step_index: int,
    total_steps: int,
) -> mx.array:
    """Apply one production ARC ping-pong transition without detaching gradients."""
    sigma_value = mx.array(sigma, dtype=mx.float32)
    next_value = mx.array(next_sigma, dtype=mx.float32)
    denoised = x - sigma_value.astype(x.dtype) * velocity
    if step_index < total_steps - 1 and float(next_value) > 0.0:
        if noise is None:
            raise ValueError("production re-noise step requires its cached noise")
        return (
            (1.0 - next_value).astype(x.dtype) * denoised
            + next_value.astype(x.dtype) * noise
        )
    return denoised


def pingpong_trace(
    model_fn,
    initial: mx.array,
    sigmas: mx.array,
    sampler_seed: int,
) -> list[dict[str, np.ndarray | float | None]]:
    """Trace exact ARC sampler states, velocities, and re-noise draws.

    Initial noise is created by the caller from generation seed. Production
    inference gives `sampler_seed` separately (currently generation seed + 1).
    """
    key = mx.random.key(sampler_seed)
    x = initial
    total_steps = sigmas.shape[0] - 1
    trace: list[dict[str, np.ndarray | float | None]] = []
    for index in range(total_steps):
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        t = sigma * mx.ones((x.shape[0],), dtype=x.dtype)
        if t.dtype != mx.float32:
            raise TypeError(
                f"production timestep promotion drift: expected float32, got {t.dtype}"
            )
        velocity = model_fn(x, t)
        mx.eval(x, velocity)
        noise = None
        if index < total_steps - 1 and float(next_sigma) > 0.0:
            key, subkey = mx.random.split(key)
            noise = mx.random.normal(x.shape, dtype=x.dtype, key=subkey)
            mx.eval(noise)
        trace.append(
            {
                "sigma": float(sigma),
                "state": np.asarray(x).astype(np.float32),
                "velocity": np.asarray(velocity).astype(np.float32),
                "noise": None if noise is None else np.asarray(noise).copy(),
            }
        )
        x = pingpong_transition(
            x, velocity, sigma, next_sigma, noise, index, total_steps
        )
        mx.eval(x)
    trace.append(
        {
            "sigma": 0.0,
            "state": np.asarray(x).astype(np.float32),
            "velocity": None,
            "noise": None,
        }
    )
    return trace


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_state_cache_contract(
    runtime_root: Path,
    source_root: Path,
    crop_len: int,
    seconds: float,
    trajectory_steps: int,
) -> dict[str, object]:
    runtime_files = {
        "sampler": runtime_root / "models/defs/sa3_pipeline.py",
        "dit": runtime_root / "models/defs/dit_mlx_medium.py",
        "conditioner": runtime_root / "models/defs/t5gemma_mlx.py",
        "state_builder": source_root / "build_ternary_state_cache.py",
        "condition_builder": source_root / "train_ternary_quality.py",
        "contract": source_root / "ternary_runtime_contract.py",
    }
    try:
        mlx_version = version("mlx")
    except PackageNotFoundError as error:
        raise RuntimeError("MLX package version is unavailable") from error
    return {
        "schema": STATE_CACHE_CONTRACT_SCHEMA,
        "activation_dtype": "float16",
        "timestep_dtype": "float32",
        "fourier_dtype": "float32",
        "sampler_state_dtype": "float16",
        "sampler_transition_dtype": "float16",
        "initial_noise_seed_offset": 0,
        "sampler_reinjection_seed_offset": 1,
        "sampler_reinjection_rng": "sequential_mx.random.split",
        "crop_len": int(crop_len),
        "seconds": float(seconds),
        "trajectory_steps": int(trajectory_steps),
        "mlx_version": mlx_version,
        "runtime_files": {
            name: _file_identity(path) for name, path in runtime_files.items()
        },
    }


def build_teacher_target_contract(
    runtime_root: Path,
    source_root: Path,
    crop_len: int,
    seconds: float,
    trajectory_steps: int = 8,
) -> dict[str, object]:
    """Fingerprint code and numeric conventions that determine teacher targets."""
    state_contract = build_state_cache_contract(
        runtime_root,
        source_root,
        crop_len,
        seconds,
        trajectory_steps=trajectory_steps,
    )
    target_builder = source_root / "prepare_ternary_teacher_targets.py"
    target_trainer = source_root / "train_ternary_window_v6.py"
    return {
        "schema": TARGET_CONTRACT_SCHEMA,
        "state_cache_contract": state_contract,
        "target_storage_dtype": "float16",
        "target_builder": _file_identity(target_builder),
        "target_trainer": _file_identity(target_trainer),
    }

def contract_digest(contract: dict[str, object]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
