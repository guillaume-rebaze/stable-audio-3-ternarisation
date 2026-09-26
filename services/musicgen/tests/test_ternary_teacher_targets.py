from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from ternary_teacher_targets import load_teacher_targets, save_teacher_targets  # noqa: E402


def test_teacher_target_cache_roundtrip_and_input_provenance(tmp_path: Path) -> None:
    state_cache = tmp_path / "state-cache"
    state_cache.mkdir()
    np.savez(state_cache / "states.npz", states=np.zeros((2, 2), dtype=np.float16))
    np.savez(state_cache / "conditions.npz", global_cond=np.ones((1,), dtype=np.float16))
    (state_cache / "manifest.json").write_text("{}\n", encoding="utf-8")
    teacher_weights = tmp_path / "teacher.npz"
    teacher_weights.write_bytes(b"fixed teacher checkpoint")
    cache_path = tmp_path / "targets.npz"
    targets = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    target_contract = {
        "schema": "onus.ternary-quality/v7-target-contract",
        "timestep_dtype": "float32",
        "mlx_version": "test",
    }

    metadata = save_teacher_targets(
        cache_path, targets, state_cache, teacher_weights, target_contract
    )
    loaded, loaded_metadata = load_teacher_targets(
        cache_path, state_cache, teacher_weights, expected_count=2,
        target_contract=target_contract,
    )

    assert np.array_equal(loaded, targets)
    assert metadata == loaded_metadata
    assert loaded_metadata["target_shape"] == [2, 3, 4]

    wrong_contract = {**target_contract, "timestep_dtype": "float16"}
    with pytest.raises(ValueError, match="provenance"):
        load_teacher_targets(
            cache_path, state_cache, teacher_weights, expected_count=2,
            target_contract=wrong_contract,
        )

    np.savez(state_cache / "conditions.npz", global_cond=np.zeros((1,), dtype=np.float16))
    with pytest.raises(ValueError, match="provenance"):
        load_teacher_targets(
            cache_path, state_cache, teacher_weights, expected_count=2,
            target_contract=target_contract,
        )
