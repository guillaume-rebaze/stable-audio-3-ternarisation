from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from ternary_contract import (  # noqa: E402
    codes_from_q,
    dequantize_packed,
    dequantize_record,
    quantize_affine_with_assignment_and_scales,
    quantize_ttq_with_assignment_and_scales,
    quantize_symmetric_weight,
    quantize_symmetric_with_scales,
    quantize_weight,
    unpack_codes,
    validate_ternary_weights,
)


def test_symmetric_contract_roundtrip_and_metadata() -> None:
    rng = np.random.default_rng(123)
    weight = rng.normal(size=(7, 64)).astype(np.float32)
    packed = quantize_symmetric_weight(weight, group_size=64)

    report = validate_ternary_weights(packed)
    assert report["mode"] == "symmetric"
    assert report["codes_valid"]
    assert np.array_equal(packed.biases, -packed.scales)
    assert np.array_equal(packed.q, unpack_codes(packed.packed_codes, 64))

    reconstructed = dequantize_packed(
        packed.packed_codes, packed.scales, packed.biases, group_size=64
    )
    groups = reconstructed.reshape(7, 1, 64)
    for row in range(7):
        expected_levels = np.unique(packed.scales[row, 0].astype(np.float32) * np.array([-1.0, 0.0, 1.0]))
        assert np.all(np.isin(np.unique(groups[row]), expected_levels))


def test_affine_centered_contract_is_not_symmetric_claim() -> None:
    weight = np.array([[2.0 + i * 0.01 for i in range(64)]], dtype=np.float32)
    packed = quantize_weight(weight, group_size=64)
    assert packed.mode == "affine_centered"
    assert validate_ternary_weights(packed)["mode"] == "affine_centered"
    reconstructed = dequantize_packed(
        packed.packed_codes, packed.scales, packed.biases, group_size=64
    )
    assert not np.allclose(reconstructed.mean(), 0.0, atol=1e-4)


def test_invalid_codes_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        codes_from_q(np.array([[[2]]], dtype=np.int8))


def test_learned_scales_still_export_strict_symmetric_levels() -> None:
    weight = np.array([[0.1 * ((index % 7) - 3) for index in range(64)]], dtype=np.float32)
    learned = quantize_symmetric_with_scales(
        weight,
        np.array([[0.2]], dtype=np.float32),
        group_size=64,
    )
    report = validate_ternary_weights(learned)
    assert report["mode"] == "symmetric"
    assert np.array_equal(learned.biases, -learned.scales)
    assert np.all(np.isin(learned.q, (-1, 0, 1)))


def test_learned_affine_assignment_exports_exact_native_affine_levels() -> None:
    rng = np.random.default_rng(125)
    weight = rng.normal(size=(3, 64)).astype(np.float32)
    means = rng.normal(scale=0.05, size=(3, 2)).astype(np.float32)
    assignments = np.full((3, 2), 0.12, dtype=np.float32)
    scales = np.full((3, 2), 0.21, dtype=np.float32)
    packed = quantize_affine_with_assignment_and_scales(
        weight,
        assignments,
        scales,
        means,
        group_size=32,
    )
    assert packed.mode == "affine_centered"
    assert validate_ternary_weights(packed)["mode"] == "affine_centered"
    expected_q = np.clip(
        np.rint(
            (weight.reshape(3, 2, 32) - means[..., None])
            / assignments[..., None]
        ),
        -1,
        1,
    ).astype(np.int8)
    assert np.array_equal(packed.q, expected_q)
    reconstructed = dequantize_packed(
        packed.packed_codes,
        packed.scales,
        packed.biases,
        group_size=32,
    )
    expected = (means[..., None] + scales[..., None] * expected_q).reshape(3, 64)
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-3, atol=2e-3)


def test_ttq_contract_preserves_independent_signed_levels() -> None:
    rng = np.random.default_rng(126)
    weight = rng.normal(size=(2, 64)).astype(np.float32)
    means = rng.normal(scale=0.05, size=(2, 2)).astype(np.float32)
    assignments = np.full((2, 2), 0.12, dtype=np.float32)
    positive = np.array([[0.21, 0.31], [0.17, 0.28]], dtype=np.float32)
    negative = np.array([[0.09, 0.27], [0.24, 0.13]], dtype=np.float32)
    packed = quantize_ttq_with_assignment_and_scales(
        weight,
        assignments,
        positive,
        negative,
        means,
        group_size=32,
    )
    assert packed.mode == "ttq"
    assert validate_ternary_weights(packed)["ttq_branches_present"]
    expected_q = np.clip(
        np.rint(
            (weight.reshape(2, 2, 32) - means[..., None])
            / assignments[..., None]
        ),
        -1,
        1,
    ).astype(np.int8)
    assert np.array_equal(packed.q, expected_q)
    expected = (
        means[..., None]
        + np.maximum(expected_q, 0) * positive[..., None]
        + np.minimum(expected_q, 0) * negative[..., None]
    ).reshape(2, 64)
    np.testing.assert_allclose(dequantize_record(packed), expected, rtol=2e-3, atol=2e-3)
