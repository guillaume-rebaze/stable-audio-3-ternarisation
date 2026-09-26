"""Exact ternary quantization/packing contract shared by QAT, export and reload.

First production lane: direct group-wise ternary affine quantization.
Hadamard is intentionally not implemented here. It needs a dedicated runtime
operator and must not silently share this standard MLX affine format.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TernaryWeights:
    """Packed ternary representation for one dense weight matrix."""

    packed_codes: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    q: np.ndarray
    group_means: np.ndarray
    group_size: int
    mode: str = "affine_centered"
    linear_bias: np.ndarray | None = None
    positive_scales: np.ndarray | None = None
    negative_scales: np.ndarray | None = None

    @property
    def out_dim(self) -> int:
        return int(self.q.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.q.shape[1] * self.group_size)


def hadamard_matrix(group_size: int) -> np.ndarray:
    """Return the normalized Sylvester Hadamard matrix for one quantizer group."""
    size = int(group_size)
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard group size must be a positive power of two")
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(float(size))


def rotate_weight_hadamard(weight: np.ndarray, group_size: int) -> np.ndarray:
    """Rotate each input group of a matrix before ternary quantization."""
    values = np.asarray(weight, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {values.shape}")
    if values.shape[1] % int(group_size):
        raise ValueError(
            f"input dimension {values.shape[1]} is not divisible by group_size={group_size}"
        )
    groups = values.reshape(values.shape[0], values.shape[1] // int(group_size), int(group_size))
    return np.matmul(groups, hadamard_matrix(int(group_size))).reshape(values.shape)


def quantize_weight(weight: np.ndarray, group_size: int = 64, eps: float = 1e-6) -> TernaryWeights:
    """Quantize W with deterministic mean-preserving {-1,0,+1} affine groups.

    MLX affine convention:
        dequantized = scales * code + biases
        code 0 = +1, code 1 = 0, code 2 = -1

    Therefore scales are -s and biases are mean+s. The function is the only
    quantization definition used by the recovery pipeline.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.shape}")
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"input dimension {in_dim} is not divisible by group_size={group_size}")
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16 for 2-bit packing")

    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    means = groups.mean(axis=-1, keepdims=True)
    centered = groups - means
    base = np.mean(np.abs(centered), axis=-1, keepdims=True) + eps
    q = np.clip(np.rint(centered / base), -1, 1).astype(np.int8)

    q2 = np.sum(q.astype(np.float32) ** 2, axis=-1, keepdims=True)
    numerator = np.sum(centered * q.astype(np.float32), axis=-1, keepdims=True)
    scale_positive = numerator / np.maximum(q2, 1.0)
    scale_positive = np.maximum(scale_positive, eps).astype(np.float32)

    scales = -scale_positive[..., 0].astype(np.float16)
    biases = (means[..., 0] + scale_positive[..., 0]).astype(np.float16)
    packed = pack_codes(q)

    return TernaryWeights(
        packed_codes=packed,
        scales=scales,
        biases=biases,
        q=q,
        group_means=means[..., 0].astype(np.float32),
        group_size=group_size,
        mode="affine_centered",
    )


def quantize_symmetric_weight(
    weight: np.ndarray,
    group_size: int = 64,
    eps: float = 1e-6,
) -> TernaryWeights:
    """Quantize W to exactly ``s_g * q`` with q in {-1, 0, +1}.

    MLX's affine 2-bit kernel stores codes as ``c = 1 - q``.  To represent the
    symmetric levels without a learned offset, the serialized affine fields
    are ``scales = -s`` and ``biases = s``.  The bias is derived metadata, not
    an independent affine degree of freedom.  Keeping this representation
    explicit prevents the previous centered ``m + s*q`` quantizer from being
    mislabeled as symmetric ternary.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.shape}")
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"input dimension {in_dim} is not divisible by group_size={group_size}")
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16 for 2-bit packing")

    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    base = np.mean(np.abs(groups), axis=-1, keepdims=True)
    q = np.clip(np.rint(groups / np.maximum(base, eps)), -1, 1).astype(np.int8)
    q2 = np.sum(q.astype(np.float32) ** 2, axis=-1, keepdims=True)
    numerator = np.sum(groups * q.astype(np.float32), axis=-1, keepdims=True)
    scale_positive = numerator / np.maximum(q2, 1.0)
    scale_positive = np.maximum(scale_positive, 0.0).astype(np.float32)
    # All-zero groups must remain exactly zero, not become an arbitrary eps.
    scale_positive = np.where(q2 > 0, scale_positive, 0.0)
    scales = -scale_positive[..., 0].astype(np.float16)
    biases = scale_positive[..., 0].astype(np.float16)
    return TernaryWeights(
        packed_codes=pack_codes(q),
        scales=scales,
        biases=biases,
        q=q,
        group_means=np.zeros((out_dim, in_dim // group_size), dtype=np.float32),
        group_size=group_size,
        mode="symmetric",
    )


def quantize_symmetric_with_scales(
    weight: np.ndarray,
    positive_scales: np.ndarray,
    group_size: int = 64,
    eps: float = 1e-8,
) -> TernaryWeights:
    """Serialize symmetric ternary codes using externally learned scales.

    The scale is the only extra quantizer parameter.  Codes are still selected
    by the same hard ``round/clip`` rule as :func:`quantize_symmetric_weight`,
    and the exported representation remains exactly ``s * q``.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.shape}")
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"input dimension {in_dim} is not divisible by group_size={group_size}")
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16 for 2-bit packing")
    scales = np.asarray(positive_scales, dtype=np.float32)
    expected_shape = (out_dim, in_dim // group_size)
    if scales.shape != expected_shape:
        raise ValueError(f"positive_scales shape {scales.shape} != {expected_shape}")
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("positive_scales must be finite and non-negative")
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    safe_scales = np.maximum(scales[..., None], eps)
    q = np.clip(np.rint(groups / safe_scales), -1, 1).astype(np.int8)
    # A zero learned scale is a deliberate dead group.  It must not acquire a
    # hidden epsilon level in the serialized model.
    q = np.where(scales[..., None] > eps, q, 0).astype(np.int8)
    scales16 = scales.astype(np.float16)
    return TernaryWeights(
        packed_codes=pack_codes(q),
        scales=-scales16,
        biases=scales16,
        q=q,
        group_means=np.zeros(expected_shape, dtype=np.float32),
        group_size=group_size,
        mode="symmetric",
    )


def quantize_symmetric_with_assignment_and_scales(
    weight: np.ndarray,
    assignment_scales: np.ndarray,
    positive_scales: np.ndarray,
    group_size: int = 64,
    eps: float = 1e-8,
) -> TernaryWeights:
    """Serialize the QAT hard assignment with its separate learned levels.

    Some QAT schemes deliberately choose ternary codes with a fixed or
    weight-derived threshold while learning only the reconstruction scale.
    In that case assignment and reconstruction scales are different parts of
    the forward contract and must not be conflated at export.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.shape}")
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"input dimension {in_dim} is not divisible by group_size={group_size}")
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16 for 2-bit packing")
    expected_shape = (out_dim, in_dim // group_size)
    assignments = np.asarray(assignment_scales, dtype=np.float32)
    scales = np.asarray(positive_scales, dtype=np.float32)
    for name, values in (("assignment_scales", assignments), ("positive_scales", scales)):
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape {values.shape} != {expected_shape}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} must be finite and non-negative")

    groups = w.reshape(expected_shape + (group_size,))
    safe_assignments = np.maximum(assignments[..., None], eps)
    q = np.clip(np.rint(groups / safe_assignments), -1, 1).astype(np.int8)
    scales16 = scales.astype(np.float16)
    return TernaryWeights(
        packed_codes=pack_codes(q),
        scales=-scales16,
        biases=scales16,
        q=q,
        group_means=np.zeros(expected_shape, dtype=np.float32),
        group_size=group_size,
        mode="symmetric",
    )


def quantize_affine_with_assignment_and_scales(
    weight: np.ndarray,
    assignment_scales: np.ndarray,
    positive_scales: np.ndarray,
    group_means: np.ndarray,
    group_size: int = 64,
    eps: float = 1e-8,
) -> TernaryWeights:
    """Serialize a learned affine ternary QAT assignment.

    The stored MLX affine fields represent ``mean + scale * q`` with
    ``scales = -scale`` and ``biases = mean + scale``.  The assignment
    threshold and reconstruction scale are deliberately separate: the former
    is learned for the hard code decision, while the latter is the level used
    by the deployable kernel.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.shape}")
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"input dimension {in_dim} is not divisible by group_size={group_size}")
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16 for 2-bit packing")
    expected_shape = (out_dim, in_dim // group_size)
    assignments = np.asarray(assignment_scales, dtype=np.float32)
    scales = np.asarray(positive_scales, dtype=np.float32)
    means = np.asarray(group_means, dtype=np.float32)
    for name, values in (
        ("assignment_scales", assignments),
        ("positive_scales", scales),
        ("group_means", means),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape {values.shape} != {expected_shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
    if np.any(assignments < 0) or np.any(scales < 0):
        raise ValueError("assignment_scales and positive_scales must be non-negative")

    groups = w.reshape(expected_shape + (group_size,))
    safe_assignments = np.maximum(assignments[..., None], eps)
    q = np.clip(
        np.rint((groups - means[..., None]) / safe_assignments), -1, 1
    ).astype(np.int8)
    q = np.where(scales[..., None] > eps, q, 0).astype(np.int8)
    scales16 = scales.astype(np.float16)
    means16 = means.astype(np.float16)
    return TernaryWeights(
        packed_codes=pack_codes(q),
        scales=-scales16,
        biases=(means16 + scales16).astype(np.float16),
        q=q,
        group_means=means,
        group_size=group_size,
        mode="affine_centered",
    )


def quantize_ttq_with_assignment_and_scales(
    weight: np.ndarray,
    assignment_scales: np.ndarray,
    positive_scales: np.ndarray,
    negative_scales: np.ndarray,
    group_means: np.ndarray,
    group_size: int = 64,
    eps: float = 1e-8,
    mode: str = "ttq",
) -> TernaryWeights:
    """Serialize a TTQ group with independent positive/negative levels.

    The deployable levels are ``mean + s_positive`` for ``q=+1``, ``mean``
    for ``q=0`` and ``mean - s_negative`` for ``q=-1``.  The legacy
    ``scales``/``biases`` fields remain populated with the positive branch so
    older metadata readers can inspect the record, while TTQ readers use the
    explicit branch arrays.
    """
    if mode not in {"ttq", "ttq_hadamard"}:
        raise ValueError(f"invalid TTQ mode: {mode}")
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.shape}")
    out_dim, in_dim = w.shape
    if in_dim % group_size:
        raise ValueError(f"input dimension {in_dim} is not divisible by group_size={group_size}")
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16 for 2-bit packing")
    expected_shape = (out_dim, in_dim // group_size)
    assignments = np.asarray(assignment_scales, dtype=np.float32)
    positive = np.asarray(positive_scales, dtype=np.float32)
    negative = np.asarray(negative_scales, dtype=np.float32)
    means = np.asarray(group_means, dtype=np.float32)
    for name, values in (
        ("assignment_scales", assignments),
        ("positive_scales", positive),
        ("negative_scales", negative),
        ("group_means", means),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape {values.shape} != {expected_shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
    if np.any(assignments < 0) or np.any(positive < 0) or np.any(negative < 0):
        raise ValueError("TTQ scales must be non-negative")

    groups = w.reshape(expected_shape + (group_size,))
    safe_assignments = np.maximum(assignments[..., None], eps)
    q = np.clip(
        np.rint((groups - means[..., None]) / safe_assignments), -1, 1
    ).astype(np.int8)
    q = np.where(
        (positive[..., None] > eps) | (negative[..., None] > eps), q, 0
    ).astype(np.int8)
    positive16 = positive.astype(np.float16)
    negative16 = negative.astype(np.float16)
    means16 = means.astype(np.float16)
    return TernaryWeights(
        packed_codes=pack_codes(q),
        scales=-positive16,
        biases=(means16 + positive16).astype(np.float16),
        q=q,
        group_means=means,
        group_size=group_size,
        mode=mode,
        positive_scales=positive16,
        negative_scales=negative16,
    )


def validate_ternary_weights(weights: TernaryWeights) -> dict:
    """Return strict structural checks; raise on invalid serialized content."""
    if weights.mode not in {
        "affine_centered",
        "symmetric",
        "symmetric_hadamard",
        "ttq",
        "ttq_hadamard",
    }:
        raise ValueError(f"unknown ternary mode: {weights.mode}")
    if not np.all(np.isin(weights.q, (-1, 0, 1))):
        raise ValueError("q contains values outside {-1,0,+1}")
    if not np.isfinite(weights.scales).all() or not np.isfinite(weights.biases).all():
        raise ValueError("scales/biases contain NaN or Inf")
    if weights.mode in {"ttq", "ttq_hadamard"}:
        if weights.positive_scales is None or weights.negative_scales is None:
            raise ValueError("TTQ records must contain positive and negative scales")
        expected_shape = weights.q.shape[:2]
        for name, values in (
            ("positive_scales", weights.positive_scales),
            ("negative_scales", weights.negative_scales),
        ):
            values = np.asarray(values)
            if values.shape != expected_shape:
                raise ValueError(f"{name} shape {values.shape} != {expected_shape}")
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        means = np.asarray(weights.group_means)
        if means.shape != expected_shape or not np.isfinite(means).all():
            raise ValueError("TTQ group means are invalid")
    if weights.linear_bias is not None:
        if weights.linear_bias.shape != (weights.out_dim,):
            raise ValueError(
                f"linear bias shape {weights.linear_bias.shape} != {(weights.out_dim,)}"
            )
        if not np.isfinite(weights.linear_bias).all():
            raise ValueError("linear bias contains NaN or Inf")
    if weights.mode in {"symmetric", "symmetric_hadamard"}:
        if not np.array_equal(weights.biases, -weights.scales):
            raise ValueError("symmetric affine metadata must satisfy biases == -scales")
    return {
        "mode": weights.mode,
        "group_size": int(weights.group_size),
        "shape": list(weights.q.shape),
        "codes_valid": True,
        "scale_finite": True,
        "linear_bias_present": weights.linear_bias is not None,
        "ttq_branches_present": (
            weights.mode in {"ttq", "ttq_hadamard"}
            and weights.positive_scales is not None
            and weights.negative_scales is not None
        ),
        "symmetric_metadata": weights.mode not in {"symmetric", "symmetric_hadamard"}
        or bool(np.array_equal(weights.biases, -weights.scales)),
        "code_histogram": {
            str(value): int(np.sum(weights.q == value)) for value in (-1, 0, 1)
        },
    }


def codes_from_q(q: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q)
    if not np.all(np.isin(q_arr, (-1, 0, 1))):
        raise ValueError("ternary q contains values outside {-1,0,+1}")
    return np.where(q_arr == 1, 0, np.where(q_arr == 0, 1, 2)).astype(np.uint32)


def q_from_codes(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes)
    if not np.all((c >= 0) & (c <= 2)):
        raise ValueError("2-bit codes must be in [0,2]")
    return np.where(c == 0, 1, np.where(c == 1, 0, -1)).astype(np.int8)


def pack_codes(q: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q)
    if q_arr.ndim != 3:
        raise ValueError(f"q must have shape [out, groups, group_size], got {q_arr.shape}")
    out_dim, groups, group_size = q_arr.shape
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16")
    codes = codes_from_q(q_arr).reshape(out_dim, groups * group_size)
    packed = np.zeros((out_dim, codes.shape[1] // 16), dtype=np.uint32)
    for bit_index in range(16):
        packed |= codes[:, bit_index::16] << (2 * bit_index)
    return packed


def unpack_codes(packed: np.ndarray, group_size: int) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint32)
    if p.ndim != 2:
        raise ValueError(f"packed codes must be 2-D, got {p.shape}")
    out_dim, packed_cols = p.shape
    if group_size % 16:
        raise ValueError("group_size must be divisible by 16")
    full_codes = np.zeros((out_dim, packed_cols * 16), dtype=np.uint32)
    for bit_index in range(16):
        full_codes[:, bit_index::16] = (p >> (2 * bit_index)) & 0x3
    return q_from_codes(full_codes.reshape(out_dim, -1, group_size))


def dequantize(q: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.int8)
    scale_arr = np.asarray(scales, dtype=np.float32)[..., None]
    bias_arr = np.asarray(biases, dtype=np.float32)[..., None]
    codes = codes_from_q(q_arr)
    return (scale_arr * codes.astype(np.float32) + bias_arr).reshape(
        q_arr.shape[0], q_arr.shape[1] * q_arr.shape[2]
    )


def dequantize_packed(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int) -> np.ndarray:
    q = unpack_codes(packed, group_size)
    return dequantize(q, scales, biases)


def dequantize_ttq(
    q: np.ndarray,
    positive_scales: np.ndarray,
    negative_scales: np.ndarray,
    group_means: np.ndarray,
) -> np.ndarray:
    """Reconstruct independent TTQ levels from unpacked ternary codes."""
    q_arr = np.asarray(q, dtype=np.int8)
    positive = np.asarray(positive_scales, dtype=np.float32)[..., None]
    negative = np.asarray(negative_scales, dtype=np.float32)[..., None]
    means = np.asarray(group_means, dtype=np.float32)[..., None]
    levels = (
        means
        + np.maximum(q_arr.astype(np.float32), 0.0) * positive
        + np.minimum(q_arr.astype(np.float32), 0.0) * negative
    )
    return levels.reshape(q_arr.shape[0], q_arr.shape[1] * q_arr.shape[2])


def dequantize_record(weights: TernaryWeights) -> np.ndarray:
    """Reconstruct a record using the quantizer mode it declares."""
    if weights.mode in {"ttq", "ttq_hadamard"}:
        assert weights.positive_scales is not None
        assert weights.negative_scales is not None
        return dequantize_ttq(
            weights.q,
            weights.positive_scales,
            weights.negative_scales,
            weights.group_means,
        )
    return dequantize(weights.q, weights.scales, weights.biases)


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a32 - b32) / max(np.linalg.norm(a32), eps))


def scope_digest(paths: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(p) for p in paths)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def self_test() -> dict:
    rng = np.random.default_rng(42)
    weight = rng.normal(size=(32, 64)).astype(np.float32)
    packed = quantize_weight(weight, group_size=64)
    symmetric = quantize_symmetric_weight(weight, group_size=64)
    unpacked_q = unpack_codes(packed.packed_codes, group_size=64)
    reconstructed = dequantize_packed(
        packed.packed_codes, packed.scales, packed.biases, group_size=64
    )
    result = {
        "codes_exact": bool(np.array_equal(packed.q, unpacked_q)),
        "codes_valid": bool(np.all(np.isin(packed.q, (-1, 0, 1)))),
        "roundtrip_error": relative_error(
            reconstructed, dequantize(packed.q, packed.scales, packed.biases)
        ),
        "weight_reconstruction_error": relative_error(weight, reconstructed),
        "shape": list(reconstructed.shape),
    }
    if not result["codes_exact"] or result["roundtrip_error"] > 1e-7:
        raise AssertionError(result)
    validate_ternary_weights(symmetric)
    symmetric_reconstructed = dequantize_packed(
        symmetric.packed_codes,
        symmetric.scales,
        symmetric.biases,
        group_size=64,
    )
    if not np.array_equal(symmetric.biases, -symmetric.scales):
        raise AssertionError("symmetric metadata mismatch")
    if not np.all(np.isin(symmetric.q, (-1, 0, 1))):
        raise AssertionError("symmetric codes are not ternary")
    result["symmetric_weight_reconstruction_error"] = relative_error(
        weight, symmetric_reconstructed
    )
    return result


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
