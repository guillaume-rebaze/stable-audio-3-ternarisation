from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from ternary_runtime_contract import timestep_tensor


def test_timestep_is_fp32_for_python_and_mlx_sigmas() -> None:
    for sigma in (0.9943755865097046, mx.array(0.7455465793609619, dtype=mx.float16)):
        result = timestep_tensor(sigma, batch_size=3)
        mx.eval(result)
        assert result.dtype == mx.float32
        assert result.shape == (3,)
        expected = np.float32(sigma)
        assert np.array_equal(np.asarray(result), np.full((3,), expected, dtype=np.float32))


def test_timestep_matches_production_sampler_scalar_promotion() -> None:
    sigma = mx.array(0.7455465793609619, dtype=mx.float32)
    production = sigma * mx.ones((1,), dtype=mx.float16)
    shared = timestep_tensor(sigma)
    mx.eval(production, shared)
    assert production.dtype == mx.float32
    assert np.array_equal(np.asarray(production), np.asarray(shared))


def test_timestep_rejects_invalid_batch_and_non_scalar_sigma() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        timestep_tensor(0.5, batch_size=0)
    with pytest.raises(ValueError, match="scalar"):
        timestep_tensor(mx.ones((2,), dtype=mx.float32))
