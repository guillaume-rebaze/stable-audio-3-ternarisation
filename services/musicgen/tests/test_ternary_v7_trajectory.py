import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from ternary_teacher_targets import file_identity, sha256_file  # noqa: E402
from train_ternary_window_v6 import (  # noqa: E402
    load_trajectory_pair_cache,
    two_step_student_trace,
)


def test_two_step_trace_matches_pingpong_and_keeps_gradient() -> None:
    anchor = mx.array([[[0.5, -0.25]]], dtype=mx.float16)
    noise_first = mx.array([[[0.1, -0.2]]], dtype=mx.float16)
    noise_second = mx.zeros_like(anchor)
    sigmas = (1.0, 0.5, 0.0)

    def constant_velocity(x: mx.array, t: mx.array) -> mx.array:
        assert t.dtype == mx.float32
        return mx.full(x.shape, 0.25, dtype=x.dtype)

    velocity0, state1, velocity1, endpoint = two_step_student_trace(
        constant_velocity,
        anchor,
        sigmas,
        noise_first,
        noise_second,
        pair_start=0,
        total_steps=2,
    )
    mx.eval(velocity0, state1, velocity1, endpoint)
    np.testing.assert_allclose(np.asarray(state1), [[[0.175, -0.35]]], atol=5e-4)
    np.testing.assert_allclose(np.asarray(endpoint), [[[0.05, -0.475]]], atol=5e-4)

    def endpoint_energy(scale: mx.array) -> mx.array:
        _, _, _, result = two_step_student_trace(
            lambda x, _t: x * scale,
            anchor,
            sigmas,
            noise_first,
            noise_second,
            pair_start=0,
            total_steps=2,
        )
        return mx.mean(result.astype(mx.float32) ** 2)

    gradient = mx.grad(endpoint_energy)(mx.array(0.1, dtype=mx.float32))
    mx.eval(gradient)
    assert np.isfinite(float(gradient))
    assert abs(float(gradient)) > 1e-6


def _write_minimal_pair_cache(root: Path) -> tuple[Path, Path, Path, Path]:
    state_cache = root / "state-cache"
    state_cache.mkdir()
    for name in ("manifest.json", "states.npz", "conditions.npz"):
        (state_cache / name).write_bytes(name.encode())
    source_records = root / "records.npz"
    teacher_weights = root / "teacher.npz"
    source_records.write_bytes(b"records")
    teacher_weights.write_bytes(b"teacher")

    pairs_path = root / "pairs.npz"
    count = 14
    latent_shape = (count, 256, 2)
    pair_steps = np.tile(np.arange(7, dtype=np.int8), 2)
    sigma_grid = np.linspace(1.0, 0.0, 9, dtype=np.float32)
    arrays = {
        "anchors": np.zeros(latent_shape, dtype=np.float16),
        "noise_first": np.zeros(latent_shape, dtype=np.float16),
        "noise_second": np.zeros(latent_shape, dtype=np.float16),
        "target_velocity": np.zeros(latent_shape, dtype=np.float16),
        "target_state_one": np.zeros(latent_shape, dtype=np.float16),
        "target_endpoint": np.zeros(latent_shape, dtype=np.float16),
        "sigmas": np.stack(
            [sigma_grid[index : index + 3] for index in pair_steps]
        ),
        "prompt_indices": np.zeros(count, dtype=np.int32),
        "pair_steps": pair_steps,
        "generation_seeds": np.repeat(np.array([1, 2], dtype=np.int64), 7),
        "second_noise_present": pair_steps < 6,
    }
    np.savez_compressed(pairs_path, **arrays)
    manifest_path = pairs_path.with_suffix(".json")
    manifest = {
        "schema": "onus.ternary-quality/v7-trajectory-pairs",
        "status": "prepared",
        "inputs": {
            "state_cache_manifest": file_identity(state_cache / "manifest.json"),
            "states": file_identity(state_cache / "states.npz"),
            "conditions": file_identity(state_cache / "conditions.npz"),
            "source_records": file_identity(source_records),
            "teacher_weights": file_identity(teacher_weights),
        },
        "cache_file": str(pairs_path.resolve()),
        "cache_sha256": sha256_file(pairs_path),
        "arrays_sha256": sha256_file(pairs_path),
        "contract_digest": "test-contract",
        "pair_count": count,
        "prompts": ["test"],
        "sampler": {"sigma_grid": sigma_grid.tolist()},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return pairs_path, state_cache, source_records, teacher_weights


def test_pair_cache_loader_checks_shapes_and_provenance(tmp_path: Path) -> None:
    pairs, state_cache, records, teacher = _write_minimal_pair_cache(tmp_path)
    arrays, metadata = load_trajectory_pair_cache(
        pairs, state_cache, records, teacher, crop_len=2
    )
    assert len(arrays["pair_steps"]) == 14
    assert metadata["contract_digest"] == "test-contract"


def test_pair_cache_loader_rejects_changed_source_records(tmp_path: Path) -> None:
    pairs, state_cache, records, teacher = _write_minimal_pair_cache(tmp_path)
    records.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        load_trajectory_pair_cache(pairs, state_cache, records, teacher, crop_len=2)
