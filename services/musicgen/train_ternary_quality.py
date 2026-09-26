"""Quality-first, reload-verified ternary DiT training pipeline.

This is the recovery path after the failed Bonsai/H128 experiments:
- direct MLX-compatible affine ternary quantization;
- core attention/FFN scope only (7 projections per block);
- hard-quantized prefix between blocks;
- full real-latent/prompt/sigma sampling;
- optional FP16 modulation/project-out velocity polish;
- final artifact and reload audit.

Hadamard is intentionally absent. It needs a different runtime operator.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
import random
import sys
import time
from typing import Iterable

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
THIS_DIR = Path(__file__).resolve().parent
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR), str(THIS_DIR)] + [
    p for p in sys.path if "musicgen" not in p and "abelton" not in p
]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np
import training_checkpoint as tc

from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import (
    apply_prompt_padding,
    build_pingpong_schedule,
    load_conditioner_from_npz,
)
from models.defs.t5gemma_mlx import T5Gemma
from sa3_mlx import T5GEMMA_NPZ_REL
from ternary_contract import (
    TernaryWeights,
    dequantize,
    dequantize_record,
    hadamard_matrix,
    quantize_affine_with_assignment_and_scales,
    quantize_ttq_with_assignment_and_scales,
    quantize_weight,
    rotate_weight_hadamard,
    quantize_symmetric_weight,
    quantize_symmetric_with_assignment_and_scales,
    relative_error,
    scope_digest,
    unpack_codes,
    validate_ternary_weights,
    write_json,
)
from ternary_run_config import CONFIG_KEYS, resolve_config
from ternary_runtime_contract import timestep_tensor
from weights import ensure_local


CORE_NAMES = (
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "ff.ff.0.proj",
    "ff.ff.2",
)

TTQ_MODES = {"ttq", "ttq_hadamard"}
SYMMETRIC_HADAMARD_MODE = "learned_symmetric_hadamard"
LEARNED_MODES = {
    "learned_symmetric",
    SYMMETRIC_HADAMARD_MODE,
    "learned_affine",
    *TTQ_MODES,
}


def hadamard_transform_mx(values: mx.array, group_size: int) -> mx.array:
    """Apply a normalized block Hadamard transform on the input dimension."""
    if values.shape[-1] % int(group_size):
        raise ValueError(
            f"input dimension {values.shape[-1]} is not divisible by group_size={group_size}"
        )
    original_shape = values.shape
    grouped = values.reshape(
        original_shape[:-1]
        + (original_shape[-1] // int(group_size), int(group_size))
    )
    matrix = mx.array(hadamard_matrix(int(group_size)), dtype=mx.float32)
    rotated = mx.matmul(grouped.astype(mx.float32), matrix)
    return rotated.astype(values.dtype).reshape(original_shape)

LEARNED_THRESHOLD_LOG_BOUNDS = (-2.0, 2.0)


def file_fingerprint(path: Path, include_sha256: bool = True) -> dict[str, object]:
    """Return a stable identity for a file without loading it as a tensor."""
    result: dict[str, object] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }
    if include_sha256 and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    else:
        result["sha256"] = None
    return result


def tree_add(left: object, right: object) -> object:
    return tree_map(lambda a, b: a + b, left, right)


def tree_scale(value: object, factor: float) -> object:
    return tree_map(lambda item: item * factor, value)


def memory_snapshot() -> dict[str, float]:
    active_fn = getattr(mx, "get_active_memory", mx.metal.get_active_memory)
    peak_fn = getattr(mx, "get_peak_memory", mx.metal.get_peak_memory)
    return {
        "metal_active_gb": float(active_fn() / (1024**3)),
        "metal_peak_gb": float(peak_fn() / (1024**3)),
    }


def module_at(block: nn.Module, path: str) -> nn.Module:
    current = block
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def set_module_at(block: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent = block
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def core_modules(block: nn.Module) -> dict[str, nn.Module]:
    return {name: module_at(block, name) for name in CORE_NAMES}


def quantize_weight_mx(
    weight: mx.array,
    group_size: int,
    mode: str = "affine_centered",
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """MLX form of the selected ternary_contract quantizer.

    Returns reconstructed weight, q, positive scale and group mean. The numpy
    contract is used again at hard-freeze/export as the serialization authority.
    """
    out_dim, in_dim = map(int, weight.shape)
    if in_dim % group_size:
        raise ValueError(f"{in_dim=} not divisible by {group_size=}")
    groups = weight.astype(mx.float32).reshape(out_dim, in_dim // group_size, group_size)
    if mode in {"symmetric", "learned_symmetric", SYMMETRIC_HADAMARD_MODE}:
        means = mx.zeros(groups.shape[:-1] + (1,), dtype=mx.float32)
        centered = groups
    elif mode == "affine_centered":
        means = mx.mean(groups, axis=-1, keepdims=True)
        centered = groups - means
    else:
        raise ValueError(f"unknown quantizer mode: {mode}")
    mean_abs = mx.mean(mx.abs(centered), axis=-1, keepdims=True)
    base = (
        mean_abs + 1e-6
        if mode == "affine_centered"
        else mx.maximum(mean_abs, mx.array(1e-6, dtype=mx.float32))
    )
    q = mx.clip(mx.round(centered / base), -1.0, 1.0)
    q2 = mx.sum(q * q, axis=-1, keepdims=True)
    numerator = mx.sum(centered * q, axis=-1, keepdims=True)
    scale = mx.maximum(numerator / mx.maximum(q2, 1.0), mx.array(0.0, dtype=mx.float32))
    if mode == "affine_centered":
        scale = mx.maximum(scale, mx.array(1e-6, dtype=mx.float32))
    reconstructed = (means + q * scale).reshape(out_dim, in_dim)
    return reconstructed, q, scale, means


class TernaryQATLinear(nn.Module):
    """Direct ternary QAT layer with learned levels and thresholds.

    hard=True means the dense weight already came from the serialized q/s/b
    representation. It is then used without a second quantization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        group_size: int,
        quantizer_mode: str = "affine_centered",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_size = int(group_size)
        self.use_hadamard = False
        self.quantizer_mode = str(quantizer_mode)
        self.hard = False
        self.surrogate_mode = "smooth"
        # The schedule is runtime control, not a trainable/model-serialized
        # parameter. It lets dense-start QAT begin with a useful surrogate and
        # end with the exact hard ternary forward used for export.
        self.soft_forward = False
        self.quantization_sharpness = 8.0
        self.weight = mx.zeros((out_dim, in_dim), dtype=mx.float32)
        self.log_scales = (
            mx.zeros((out_dim, in_dim // group_size), dtype=mx.float32)
            if self.quantizer_mode in LEARNED_MODES
            else None
        )
        self.log_negative_scales = (
            mx.zeros((out_dim, in_dim // group_size), dtype=mx.float32)
            if self.quantizer_mode in TTQ_MODES
            else None
        )
        self.log_threshold_multiplier = (
            mx.zeros((out_dim, in_dim // group_size), dtype=mx.float32)
            if self.quantizer_mode in LEARNED_MODES
            else None
        )
        self.group_biases = (
            mx.zeros((out_dim, in_dim // group_size), dtype=mx.float32)
            if self.quantizer_mode in {"learned_affine", *TTQ_MODES}
            else None
        )
        self.bias = mx.zeros((out_dim,), dtype=mx.float32) if bias else None

    def initialize_learned_parameters(self) -> None:
        if self.quantizer_mode not in LEARNED_MODES:
            return
        groups = np.asarray(self.weight, dtype=np.float32).reshape(
            self.out_dim, self.in_dim // self.group_size, self.group_size
        )
        if self.quantizer_mode in {"learned_affine", *TTQ_MODES}:
            means = np.mean(groups, axis=-1)
            centered = groups - means[..., None]
        else:
            means = np.zeros(groups.shape[:-1], dtype=np.float32)
            centered = groups
        base = np.maximum(np.mean(np.abs(centered), axis=-1, keepdims=True), 1e-6)
        q = np.clip(np.rint(centered / base), -1, 1).astype(np.float32)
        q2 = np.sum(q * q, axis=-1)
        numerator = np.sum(centered * q, axis=-1)
        initial = np.maximum(numerator / np.maximum(q2, 1.0), 0.0)
        initial = np.where(q2 > 0.0, initial, 0.0)
        self.log_scales = mx.log(
            mx.maximum(mx.array(initial, dtype=mx.float32), mx.array(1e-6))
        )
        if self.quantizer_mode in TTQ_MODES:
            positive_count = np.sum(q > 0.0, axis=-1)
            negative_count = np.sum(q < 0.0, axis=-1)
            positive_sum = np.sum(np.where(q > 0.0, centered, 0.0), axis=-1)
            negative_sum = np.sum(np.where(q < 0.0, centered, 0.0), axis=-1)
            positive = np.maximum(
                positive_sum / np.maximum(positive_count, 1.0), 1e-6
            )
            negative = np.maximum(
                -negative_sum / np.maximum(negative_count, 1.0), 1e-6
            )
            base = np.maximum(np.mean(np.abs(centered), axis=-1), 1e-6)
            positive = np.where(positive_count > 0, positive, base)
            negative = np.where(negative_count > 0, negative, base)
            self.log_scales = mx.log(mx.array(positive, dtype=mx.float32))
            self.log_negative_scales = mx.log(mx.array(negative, dtype=mx.float32))
        if self.quantizer_mode in {"learned_affine", *TTQ_MODES}:
            self.group_biases = mx.array(means, dtype=mx.float32)

    # Compatibility for older callers/tests and previous checkpoints.
    def initialize_learned_scales(self) -> None:
        self.initialize_learned_parameters()

    def set_quantization_schedule(self, soft_forward: bool, sharpness: float) -> None:
        if sharpness <= 0 or not np.isfinite(sharpness):
            raise ValueError("quantization sharpness must be finite and positive")
        self.soft_forward = bool(soft_forward)
        self.quantization_sharpness = float(sharpness)

    def set_surrogate_mode(self, mode: str) -> None:
        if mode not in {"smooth", "identity"}:
            raise ValueError(f"unknown QAT surrogate mode: {mode}")
        self.surrogate_mode = str(mode)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        group_size: int,
        quantizer_mode: str = "affine_centered",
    ) -> TernaryQATLinear:
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        result = cls(
            int(layer.weight.shape[1]),
            int(layer.weight.shape[0]),
            has_bias,
            group_size,
            quantizer_mode,
        )
        dense_weight = np.asarray(layer.weight, dtype=np.float32)
        if quantizer_mode in {"ttq_hadamard", SYMMETRIC_HADAMARD_MODE}:
            dense_weight = rotate_weight_hadamard(dense_weight, group_size)
        result.weight = mx.array(dense_weight, dtype=mx.float32)
        result.initialize_learned_parameters()
        if has_bias:
            result.bias = layer.bias.astype(mx.float32)
        return result

    def __call__(self, x: mx.array) -> mx.array:
        if self.hard:
            effective = self.weight
        elif self.quantizer_mode in LEARNED_MODES:
            groups = self.weight.astype(mx.float32).reshape(
                self.out_dim, self.in_dim // self.group_size, self.group_size
            )
            if self.quantizer_mode in {"learned_affine", *TTQ_MODES}:
                group_biases = self.group_biases
                centered = groups - group_biases[..., None]
            else:
                group_biases = mx.zeros(
                    (self.out_dim, self.in_dim // self.group_size),
                    dtype=mx.float32,
                )
                centered = groups
            log_scales = mx.clip(self.log_scales, -20.0, 20.0)
            positive_scales = mx.exp(log_scales)[..., None]
            threshold_log = mx.clip(
                self.log_threshold_multiplier,
                LEARNED_THRESHOLD_LOG_BOUNDS[0],
                LEARNED_THRESHOLD_LOG_BOUNDS[1],
            )
            threshold_multiplier = mx.exp(threshold_log)[..., None]
            assignment_scale = mx.maximum(
                mx.mean(mx.abs(centered), axis=-1, keepdims=True),
                mx.array(1e-6),
            ) * threshold_multiplier
            z = centered / assignment_scale
            q_hard = mx.clip(mx.round(z), -1.0, 1.0)
            # Smooth surrogate for the three hard bins: -1 below -0.5, 0
            # around zero, +1 above +0.5. Forward remains exactly ternary;
            # the surrogate only supplies gradients to weight and scale.
            if self.surrogate_mode == "identity":
                # LSQ-style straight-through surrogate.  The hard forward
                # remains exactly {-1, 0, +1}, while the identity derivative
                # lets masters cross a bin boundary instead of freezing the
                # source checkpoint's codes near +/-0.5.
                q_soft = mx.clip(z, -1.0, 1.0)
            else:
                sharpness = mx.array(self.quantization_sharpness, dtype=mx.float32)
                q_soft = mx.sigmoid(sharpness * (z - 0.5)) - mx.sigmoid(
                    sharpness * (-z - 0.5)
                )
            q_effective = (
                q_soft
                if self.soft_forward
                else q_hard + q_soft - mx.stop_gradient(q_soft)
            )
            if self.quantizer_mode in TTQ_MODES:
                negative_scales = mx.exp(
                    mx.clip(self.log_negative_scales, -20.0, 20.0)
                )[..., None]
                positive_part = mx.maximum(q_effective, 0.0) * positive_scales
                negative_part = mx.minimum(q_effective, 0.0) * negative_scales
                effective = (
                    group_biases[..., None] + positive_part + negative_part
                ).reshape(self.out_dim, self.in_dim)
            else:
                effective = (
                    group_biases[..., None] + positive_scales * q_effective
                ).reshape(self.out_dim, self.in_dim)
        else:
            reconstructed, _, _, _ = quantize_weight_mx(
                self.weight, self.group_size, self.quantizer_mode
            )
            effective = self.weight + mx.stop_gradient(reconstructed - self.weight)
        input_values = (
            hadamard_transform_mx(x, self.group_size)
            if self.quantizer_mode in {"ttq_hadamard", SYMMETRIC_HADAMARD_MODE}
            else x
        )
        result = input_values @ effective.astype(x.dtype).T
        if self.bias is not None:
            result = result + self.bias.astype(x.dtype)
        return result


class TernaryHadamardLinear(nn.Module):
    """Strict symmetric Bonsai runtime with a per-group Hadamard rotation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        group_size: int,
    ):
        super().__init__()
        if in_dim % group_size:
            raise ValueError("input dimension must be divisible by group_size")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_size = int(group_size)
        self.q = mx.zeros(
            (out_dim, in_dim // group_size, group_size), dtype=mx.int8
        )
        self.scales = mx.zeros(
            (out_dim, in_dim // group_size), dtype=mx.float32
        )
        self.bias = mx.zeros((out_dim,), dtype=mx.float32) if bias else None

    @classmethod
    def from_record(
        cls,
        module: nn.Module,
        record: TernaryWeights,
    ) -> "TernaryHadamardLinear":
        if record.mode != "symmetric_hadamard":
            raise ValueError("Hadamard runtime requires a symmetric_hadamard record")
        has_bias = getattr(module, "bias", None) is not None
        result = cls(
            record.in_dim,
            record.out_dim,
            bias=has_bias,
            group_size=record.group_size,
        )
        result.q = mx.array(record.q, dtype=mx.int8)
        # MLX affine metadata stores symmetric scales as ``-s``; custom
        # Hadamard runtime stores physical positive levels ``s``.
        result.scales = mx.array(-record.scales, dtype=mx.float32)
        if has_bias:
            result.bias = mx.array(
                record.linear_bias
                if record.linear_bias is not None
                else np.asarray(module.bias),
                dtype=mx.float32,
            )
        result.freeze()
        return result

    def __call__(self, x: mx.array) -> mx.array:
        levels = (
            self.q.astype(mx.float32) * self.scales[..., None]
        ).reshape(self.out_dim, self.in_dim)
        input_values = hadamard_transform_mx(x, self.group_size)
        result = input_values @ levels.astype(x.dtype).T
        if self.bias is not None:
            result = result + self.bias.astype(x.dtype)
        return result


class TernaryTTQLinear(nn.Module):
    """Reloadable runtime for TTQ records with independent signed levels.

    MLX's built-in affine 2-bit kernel has one scale per group. TTQ needs one
    level on each side of zero, so this small runtime expands the packed codes
    to a ternary matrix at call time while keeping the artifact itself packed.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        group_size: int,
    ):
        super().__init__()
        if in_dim % group_size:
            raise ValueError("TTQ input dimension must be divisible by group_size")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_size = int(group_size)
        self.q = mx.zeros(
            (out_dim, in_dim // group_size, group_size), dtype=mx.int8
        )
        self.positive_scales = mx.zeros(
            (out_dim, in_dim // group_size), dtype=mx.float32
        )
        self.negative_scales = mx.zeros(
            (out_dim, in_dim // group_size), dtype=mx.float32
        )
        self.group_means = mx.zeros(
            (out_dim, in_dim // group_size), dtype=mx.float32
        )
        self.bias = mx.zeros((out_dim,), dtype=mx.float32) if bias else None

    @classmethod
    def from_record(
        cls,
        module: nn.Module,
        record: TernaryWeights,
    ) -> "TernaryTTQLinear":
        if record.mode not in TTQ_MODES:
            raise ValueError("TTQ runtime requires a TTQ record")
        has_bias = getattr(module, "bias", None) is not None
        result = cls(
            record.in_dim,
            record.out_dim,
            bias=has_bias,
            group_size=record.group_size,
        )
        result.use_hadamard = record.mode == "ttq_hadamard"
        assert record.positive_scales is not None
        assert record.negative_scales is not None
        result.q = mx.array(record.q, dtype=mx.int8)
        result.positive_scales = mx.array(
            record.positive_scales, dtype=mx.float32
        )
        result.negative_scales = mx.array(
            record.negative_scales, dtype=mx.float32
        )
        result.group_means = mx.array(record.group_means, dtype=mx.float32)
        if has_bias:
            result.bias = mx.array(
                record.linear_bias
                if record.linear_bias is not None
                else np.asarray(module.bias),
                dtype=mx.float32,
            )
        result.freeze()
        return result

    def __call__(self, x: mx.array) -> mx.array:
        q = self.q.astype(mx.float32)
        levels = (
            self.group_means[..., None]
            + mx.maximum(q, 0.0) * self.positive_scales[..., None]
            + mx.minimum(q, 0.0) * self.negative_scales[..., None]
        ).reshape(self.out_dim, self.in_dim)
        input_values = (
            hadamard_transform_mx(x, self.group_size) if self.use_hadamard else x
        )
        result = input_values @ levels.astype(x.dtype).T
        if self.bias is not None:
            result = result + self.bias.astype(x.dtype)
        return result

def replace_core_block(
    block: nn.Module,
    group_size: int,
    quantizer_mode: str = "affine_centered",
) -> None:
    for name in CORE_NAMES:
        current = module_at(block, name)
        if isinstance(current, TernaryQATLinear):
            continue
        if not isinstance(current, nn.Linear):
            raise TypeError(f"Expected Linear at {name}, got {type(current)}")
        set_module_at(
            block,
            name,
            TernaryQATLinear.from_linear(current, group_size, quantizer_mode),
        )


def hard_freeze_block(
    block: nn.Module,
    block_index: int,
    group_size: int,
    records: dict[str, TernaryWeights],
    quantizer_mode: str = "affine_centered",
) -> dict[str, float]:
    """Serialize each QAT matrix once, then use its exact affine reconstruction."""
    metrics: list[float] = []
    for name, module in core_modules(block).items():
        if not isinstance(module, TernaryQATLinear):
            raise TypeError(f"Block {block_index} {name} is not QAT")
        mx.eval(module.weight)
        if quantizer_mode in LEARNED_MODES:
            mx.eval(module.log_scales, module.log_threshold_multiplier)
            learned_scales = np.exp(np.clip(np.array(module.log_scales), -20.0, 20.0))
            threshold_log = np.clip(
                np.array(module.log_threshold_multiplier),
                LEARNED_THRESHOLD_LOG_BOUNDS[0],
                LEARNED_THRESHOLD_LOG_BOUNDS[1],
            )
            threshold_multiplier = np.exp(threshold_log)
            weight = np.array(module.weight, dtype=np.float32)
            groups = weight.reshape(
                module.out_dim, module.in_dim // group_size, group_size
            )
            if quantizer_mode == "learned_affine":
                mx.eval(module.group_biases)
                group_biases = np.asarray(module.group_biases, dtype=np.float32)
                centered = groups - group_biases[..., None]
                assignment_scales = np.maximum(
                    np.mean(np.abs(centered), axis=-1), 1e-6
                ) * threshold_multiplier
                quantized = quantize_affine_with_assignment_and_scales(
                    weight,
                    assignment_scales,
                    learned_scales,
                    group_biases,
                    group_size=group_size,
                )
            elif quantizer_mode in TTQ_MODES:
                mx.eval(module.group_biases, module.log_negative_scales)
                group_biases = np.asarray(module.group_biases, dtype=np.float32)
                negative_scales = np.exp(
                    np.clip(np.array(module.log_negative_scales), -20.0, 20.0)
                )
                centered = groups - group_biases[..., None]
                assignment_scales = np.maximum(
                    np.mean(np.abs(centered), axis=-1), 1e-6
                ) * threshold_multiplier
                quantized = quantize_ttq_with_assignment_and_scales(
                    weight,
                    assignment_scales,
                    learned_scales,
                    negative_scales,
                    group_biases,
                    group_size=group_size,
                    mode=quantizer_mode,
                )
            else:
                assignment_scales = np.maximum(
                    np.mean(np.abs(groups), axis=-1), 1e-6
                ) * threshold_multiplier
                quantized = quantize_symmetric_with_assignment_and_scales(
                    weight,
                    assignment_scales,
                    learned_scales,
                    group_size=group_size,
                )
                if quantizer_mode == SYMMETRIC_HADAMARD_MODE:
                    quantized = replace(quantized, mode="symmetric_hadamard")
        elif quantizer_mode in {"symmetric", SYMMETRIC_HADAMARD_MODE}:
            quantized = quantize_symmetric_weight(
                np.array(module.weight), group_size=group_size
            )
            if quantizer_mode == SYMMETRIC_HADAMARD_MODE:
                quantized = replace(quantized, mode="symmetric_hadamard")
        elif quantizer_mode == "affine_centered":
            quantized = quantize_weight(np.array(module.weight), group_size=group_size)
        else:
            raise ValueError(f"unknown quantizer mode: {quantizer_mode}")
        if module.bias is not None:
            mx.eval(module.bias)
            quantized = replace(
                quantized,
                linear_bias=np.asarray(module.bias, dtype=np.float16),
            )
        reconstructed = dequantize_record(quantized)
        dense_error = relative_error(np.array(module.weight), reconstructed)
        validate_ternary_weights(quantized)
        if quantizer_mode == SYMMETRIC_HADAMARD_MODE:
            quantized_linear = TernaryHadamardLinear.from_record(module, quantized)
        elif quantizer_mode in TTQ_MODES:
            quantized_linear = TernaryTTQLinear.from_record(module, quantized)
        else:
            quantized_linear = nn.QuantizedLinear(
                module.in_dim,
                module.out_dim,
                bias=module.bias is not None,
                group_size=group_size,
                bits=2,
                mode="affine",
            )
            quantized_linear.weight = mx.array(quantized.packed_codes)
            quantized_linear.scales = mx.array(quantized.scales)
            quantized_linear.biases = mx.array(quantized.biases)
            if module.bias is not None:
                quantized_linear.bias = mx.array(quantized.linear_bias)
            quantized_linear.freeze()
        set_module_at(block, name, quantized_linear)
        records[f"transformer.layers.{block_index}.{name}"] = quantized
        metrics.append(dense_error)

    block.freeze()
    mx.eval(block.parameters())
    return {
        "block": int(block_index),
        "layers": len(CORE_NAMES),
        "max_dense_reconstruction_error": float(max(metrics or [0.0])),
        "mean_dense_reconstruction_error": float(np.mean(metrics or [0.0])),
    }


def save_records_checkpoint(
    path: Path,
    records: dict[str, TernaryWeights],
    next_block: int,
    group_size: int,
    crop_len: int,
    quantizer_mode: str,
) -> None:
    """Persist only the completed ternary blocks for resumable cascade runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    group_size_by_prefix: dict[str, int] = {}
    quantizer_mode_by_prefix: dict[str, str] = {}
    for prefix, quantized in sorted(records.items()):
        group_size_by_prefix[prefix] = int(quantized.group_size)
        quantizer_mode_by_prefix[prefix] = str(quantized.mode)
        arrays[f"{prefix}.packed_codes"] = quantized.packed_codes
        arrays[f"{prefix}.scales"] = quantized.scales
        arrays[f"{prefix}.biases"] = quantized.biases
        if quantized.mode in TTQ_MODES:
            assert quantized.positive_scales is not None
            assert quantized.negative_scales is not None
            arrays[f"{prefix}.positive_scales"] = quantized.positive_scales
            arrays[f"{prefix}.negative_scales"] = quantized.negative_scales
        if quantized.linear_bias is not None:
            arrays[f"{prefix}.linear_bias"] = quantized.linear_bias
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_path), **arrays)
    os.replace(str(tmp_path), str(path))
    write_json(
        path.with_suffix(".json"),
        {
            "schema": "onus.ternary-quality/v4-records-checkpoint",
            "artifact": str(path),
            "next_block": int(next_block),
            "group_size": int(group_size),
            # Keep legacy top-level group_size for old consumers.  The map is
            # authoritative when a cascade deliberately mixes G16/G32/etc.
            "group_size_by_prefix": group_size_by_prefix,
            "quantizer_mode_by_prefix": quantizer_mode_by_prefix,
            "crop_len": int(crop_len),
            "quantizer_mode": quantizer_mode,
            "scope": sorted(records),
            "scope_digest": scope_digest(records),
            "linear_bias_scope": sorted(
                prefix for prefix, record in records.items()
                if record.linear_bias is not None
            ),
            "linear_bias_count": sum(
                record.linear_bias is not None for record in records.values()
            ),
            "payload_sha256": file_fingerprint(path)["sha256"],
        },
    )


def load_records_checkpoint(
    path: Path,
) -> tuple[dict[str, TernaryWeights], dict]:
    """Load a checkpoint and reconstruct the in-memory quantization records."""
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    if schema not in {
        "onus.ternary-quality/v3-checkpoint",
        "onus.ternary-quality/v4-records-checkpoint",
    }:
        raise ValueError(f"Unsupported checkpoint schema: {metadata.get('schema')!r}")
    if schema == "onus.ternary-quality/v4-records-checkpoint":
        actual_sha = file_fingerprint(path)["sha256"]
        if actual_sha != metadata.get("payload_sha256"):
            raise ValueError("records checkpoint payload checksum mismatch")
    records: dict[str, TernaryWeights] = {}
    group_size_by_prefix = metadata.get("group_size_by_prefix", {})
    if not isinstance(group_size_by_prefix, dict):
        raise ValueError("records checkpoint group_size_by_prefix must be an object")
    quantizer_mode_by_prefix = metadata.get("quantizer_mode_by_prefix", {})
    if not isinstance(quantizer_mode_by_prefix, dict):
        raise ValueError("records checkpoint quantizer_mode_by_prefix must be an object")
    with np.load(path, allow_pickle=False) as arrays:
        observed_bias_scope = sorted(
            prefix for prefix in metadata["scope"]
            if f"{prefix}.linear_bias" in arrays.files
        )
        if schema == "onus.ternary-quality/v4-records-checkpoint" and (
            observed_bias_scope != metadata.get("linear_bias_scope")
            or len(observed_bias_scope) != metadata.get("linear_bias_count")
        ):
            raise ValueError("records checkpoint linear bias inventory mismatch")
        for prefix in metadata["scope"]:
            packed = np.array(arrays[f"{prefix}.packed_codes"], dtype=np.uint32)
            scales = np.array(arrays[f"{prefix}.scales"], dtype=np.float16)
            biases = np.array(arrays[f"{prefix}.biases"], dtype=np.float16)
            linear_bias_key = f"{prefix}.linear_bias"
            linear_bias = (
                np.array(arrays[linear_bias_key], dtype=np.float16)
                if linear_bias_key in arrays.files
                else None
            )
            record_group_size = int(
                group_size_by_prefix.get(prefix, metadata["group_size"])
            )
            q = unpack_codes(packed, record_group_size)
            quantizer_mode = str(
                quantizer_mode_by_prefix.get(prefix, metadata["quantizer_mode"])
            )
            if quantizer_mode in {
                "symmetric",
                "learned_symmetric",
                SYMMETRIC_HADAMARD_MODE,
                "symmetric_hadamard",
            }:
                means = np.zeros_like(scales, dtype=np.float32)
            else:
                means = biases.astype(np.float32) + scales.astype(np.float32)
            positive_scales = None
            negative_scales = None
            if quantizer_mode in TTQ_MODES:
                positive_scales = np.array(
                    arrays[f"{prefix}.positive_scales"], dtype=np.float16
                )
                negative_scales = np.array(
                    arrays[f"{prefix}.negative_scales"], dtype=np.float16
                )
            record = TernaryWeights(
                packed_codes=packed,
                scales=scales,
                biases=biases,
                q=q,
                group_means=means,
                group_size=record_group_size,
                mode=(
                    "symmetric_hadamard"
                    if quantizer_mode in {
                        SYMMETRIC_HADAMARD_MODE,
                        "symmetric_hadamard",
                    }
                    else "symmetric"
                    if quantizer_mode in {"symmetric", "learned_symmetric"}
                    else (
                        quantizer_mode
                        if quantizer_mode in TTQ_MODES
                        else "affine_centered"
                    )
                ),
                linear_bias=linear_bias,
                positive_scales=positive_scales,
                negative_scales=negative_scales,
            )
            validate_ternary_weights(record)
            records[prefix] = record
    if set(group_size_by_prefix) - set(records):
        raise ValueError("records checkpoint group_size_by_prefix has unknown scope")
    if set(quantizer_mode_by_prefix) - set(records):
        raise ValueError("records checkpoint quantizer_mode_by_prefix has unknown scope")
    if scope_digest(records) != metadata["scope_digest"]:
        raise ValueError("Checkpoint scope digest mismatch")
    return records, metadata


def apply_records_to_model(
    model: nn.Module,
    records: dict[str, TernaryWeights],
    group_size: int,
) -> nn.Module:
    """Apply completed block records to a fresh dense model for resume."""
    for prefix, quantized in sorted(records.items()):
        parts = prefix.split(".")
        if len(parts) < 4 or parts[0:2] != ["transformer", "layers"]:
            raise ValueError(f"Unsupported checkpoint path: {prefix}")
        block_index = int(parts[2])
        name = ".".join(parts[3:])
        current = module_at(model.transformer.layers[block_index], name)
        if not isinstance(current, nn.Linear):
            raise TypeError(f"Checkpoint target is not Linear: {prefix}")
        has_bias = hasattr(current, "bias") and current.bias is not None
        if quantized.mode == "symmetric_hadamard":
            quantized_linear = TernaryHadamardLinear.from_record(current, quantized)
        elif quantized.mode in TTQ_MODES:
            quantized_linear = TernaryTTQLinear.from_record(current, quantized)
        else:
            quantized_linear = nn.QuantizedLinear(
                int(current.weight.shape[1]),
                int(current.weight.shape[0]),
                bias=has_bias,
                group_size=quantized.group_size,
                bits=2,
                mode="affine",
            )
            quantized_linear.weight = mx.array(quantized.packed_codes)
            quantized_linear.scales = mx.array(quantized.scales)
            quantized_linear.biases = mx.array(quantized.biases)
            if has_bias:
                quantized_linear.bias = (
                    mx.array(quantized.linear_bias)
                    if quantized.linear_bias is not None
                    else current.bias.astype(mx.float16)
                )
            quantized_linear.freeze()
        set_module_at(model.transformer.layers[block_index], name, quantized_linear)
    model.freeze()
    mx.eval(model.parameters())
    return model


def apply_symmetric_derived_biases(model: nn.Module, scope: Iterable[str]) -> None:
    """Reconstruct the MLX affine bias from one stored symmetric scale."""
    for prefix in scope:
        parts = prefix.split(".")
        if len(parts) < 4 or parts[0:2] != ["transformer", "layers"]:
            raise ValueError(f"Unsupported ternary scope path: {prefix}")
        module = module_at(model.transformer.layers[int(parts[2])], ".".join(parts[3:]))
        if not isinstance(module, nn.QuantizedLinear):
            raise TypeError(f"Expected QuantizedLinear at {prefix}, got {type(module)}")
        module.biases = -module.scales
    mx.eval(model.parameters())


def load_samples(dataset_dir: Path, max_samples: int = 0) -> list[dict]:
    samples: list[dict] = []
    for npy_path in sorted(dataset_dir.glob("*.npy")):
        meta_path = npy_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        latents = np.load(npy_path).astype(np.float16)
        if latents.ndim != 2 or latents.shape[0] != 256:
            raise ValueError(f"Unexpected latent shape in {npy_path}: {latents.shape}")
        samples.append(
            {
                "path": str(npy_path),
                "latents": latents,
                "prompt": str(meta.get("prompt", "dynamic musical piece, rich stereo production")),
                "genre": str(meta.get("genre", "unknown")),
            }
        )
        if max_samples and len(samples) >= max_samples:
            break
    if not samples:
        raise FileNotFoundError(f"No latent .npy files in {dataset_dir}")
    return samples


def cache_conditioning(
    teacher: nn.Module,
    teacher_weights: Path,
    prompts: Iterable[str],
    seconds: float,
) -> tuple[dict[str, mx.array], dict[str, mx.array], mx.array, mx.array]:
    padding_emb, seconds_embedder = load_conditioner_from_npz(
        str(teacher_weights), prefix="cond."
    )
    sec_tok = seconds_embedder(seconds).astype(mx.float16)
    global_cond = sec_tok[:, 0, :]

    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    context_cache: dict[str, mx.array] = {}
    cross_cache: dict[str, mx.array] = {}
    for prompt in sorted(set(prompts)):
        emb, mask = t5.encode([prompt], max_len=256)
        padded = apply_prompt_padding(
            emb.astype(mx.float32),
            mask,
            padding_emb.astype(mx.float32),
        ).astype(mx.float16)
        cross_full = mx.concatenate([padded, sec_tok], axis=1)
        context = teacher.to_cond_embed[2](
            nn.silu(teacher.to_cond_embed[0](cross_full))
        )
        mx.eval(context)
        context_cache[prompt] = context
        cross_cache[prompt] = cross_full

    global_pre = teacher.to_global_embed[2](
        nn.silu(teacher.to_global_embed[0](global_cond))
    )
    mx.eval(global_pre, global_cond, sec_tok)
    del t5, padding_emb, seconds_embedder
    gc.collect()
    mx.clear_cache()
    return context_cache, cross_cache, global_cond, global_pre


def timestep_projection(teacher: nn.Module, global_pre: mx.array, sigma: float) -> mx.array:
    t_arr = timestep_tensor(sigma)
    features = teacher.timestep_features(t_arr)
    features = nn.silu(teacher.to_timestep_embed[0](features))
    global_embed = global_pre + teacher.to_timestep_embed[2](features)
    projected = teacher.transformer.global_cond_embedder[2](
        nn.silu(teacher.transformer.global_cond_embedder[0](global_embed))
    )
    mx.eval(projected)
    return projected


def local_pads(teacher: nn.Module, crop_len: int) -> list[mx.array]:
    # DiT.__call__ creates local-add conditioning with MLX's default float32.
    # Cascade training must use that same dtype; fp16 here created a hidden
    # train/inference mismatch before any ternary error was measured.
    zeros = mx.zeros((1, crop_len, dit_mlx_medium.LOCAL_ADD_COND_DIM), dtype=mx.float32)
    pad = mx.zeros((1, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM), dtype=mx.float32)
    result = []
    for block in teacher.transformer.layers:
        local = block.to_local_embed(zeros)
        result.append(mx.concatenate([pad, local], axis=1))
    mx.eval(*result)
    return result


def noised_latent(
    sample: dict,
    crop_len: int,
    sigma: float,
    key_seed: int,
) -> mx.array:
    latents = sample["latents"]
    if latents.shape[1] >= crop_len:
        start = (key_seed * 7919) % (latents.shape[1] - crop_len + 1)
        crop = latents[:, start : start + crop_len]
    else:
        crop = np.pad(latents, ((0, 0), (0, crop_len - latents.shape[1])))
    x0 = mx.array(crop[None], dtype=mx.float16)
    noise = mx.random.normal(x0.shape, dtype=mx.float16, key=mx.random.key(key_seed))
    return x0 * (1.0 - sigma) + noise * sigma


def model_input(model: nn.Module, x: mx.array) -> mx.array:
    x_lc = x.transpose(0, 2, 1)
    x_pp = model.preprocess_conv(x_lc) + x_lc
    h = model.transformer.project_in(x_pp)
    memory = mx.broadcast_to(
        model.transformer.memory_tokens[None],
        (x.shape[0], dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM),
    )
    return mx.concatenate([memory, h], axis=1)


def block_loss(
    block: nn.Module,
    h_in: mx.array,
    context: mx.array,
    global_cond: mx.array,
    local_pad: mx.array,
    target: mx.array,
) -> mx.array:
    output = block(h_in, context, global_cond, local_pad)
    out_audio = output[:, dit_mlx_medium.NUM_MEMORY_TOKENS :, :].astype(mx.float32)
    target_audio = target[:, dit_mlx_medium.NUM_MEMORY_TOKENS :, :].astype(mx.float32)
    denom = mx.mean(target_audio * target_audio) + 1e-6
    mse = mx.mean((out_audio - target_audio) ** 2) / denom
    cosine = mx.sum(out_audio * target_audio) / (
        mx.sqrt(mx.sum(out_audio * out_audio))
        * mx.sqrt(mx.sum(target_audio * target_audio))
        + 1e-6
    )
    out_mean = mx.mean(out_audio)
    target_mean = mx.mean(target_audio)
    rms_out = mx.sqrt(mx.mean(out_audio * out_audio) + 1e-6)
    rms_target = mx.sqrt(mx.mean(target_audio * target_audio) + 1e-6)
    rms_loss = ((rms_out - rms_target) / rms_target) ** 2
    delta_out = out_audio[:, 1:, :] - out_audio[:, :-1, :]
    delta_target = target_audio[:, 1:, :] - target_audio[:, :-1, :]
    temporal = mx.mean((delta_out - delta_target) ** 2) / (
        mx.mean(delta_target * delta_target) + 1e-6
    )
    return mse + 1.5 * (1.0 - cosine) + 0.25 * rms_loss + 0.10 * temporal


def terminal_block_loss(
    block: nn.Module,
    h_in: mx.array,
    context: mx.array,
    global_cond: mx.array,
    local_pad: mx.array,
    target: mx.array,
    target_velocity: mx.array,
    project_out: nn.Module,
    postprocess_conv: nn.Module,
) -> mx.array:
    """Train the terminal block against hidden state and final velocity.

    The last block is the point where small hidden-state errors become the
    sampler's velocity.  A hidden-only objective can look acceptable there
    while the actual DiT output is already off-trajectory.
    """
    output = block(h_in, context, global_cond, local_pad)
    output_audio = output[:, dit_mlx_medium.NUM_MEMORY_TOKENS :, :].astype(mx.float32)
    target_audio = target[:, dit_mlx_medium.NUM_MEMORY_TOKENS :, :].astype(mx.float32)
    hidden_mse = mx.mean((output_audio - target_audio) ** 2) / (
        mx.mean(target_audio * target_audio) + 1e-6
    )
    hidden_cosine = mx.sum(output_audio * target_audio) / (
        mx.sqrt(mx.sum(output_audio * output_audio))
        * mx.sqrt(mx.sum(target_audio * target_audio))
        + 1e-6
    )
    predicted = project_out(output_audio)
    predicted = postprocess_conv(predicted) + predicted
    target_out = target_velocity.transpose(0, 2, 1).astype(mx.float32)
    velocity_mse = mx.mean((predicted - target_out) ** 2) / (
        mx.mean(target_out * target_out) + 1e-6
    )
    velocity_cosine = mx.sum(predicted * target_out) / (
        mx.sqrt(mx.sum(predicted * predicted))
        * mx.sqrt(mx.sum(target_out * target_out))
        + 1e-6
    )
    return (
        hidden_mse
        + 1.5 * (1.0 - hidden_cosine)
        + 1.25 * velocity_mse
        + (1.0 - velocity_cosine)
    )


def prefix_states(
    teacher: nn.Module,
    student: nn.Module,
    x: mx.array,
    block_index: int,
    context: mx.array,
    global_cond: mx.array,
    teacher_local: list[mx.array],
) -> tuple[mx.array, mx.array]:
    h_teacher = model_input(teacher, x)
    h_student = model_input(student, x)
    for prev in range(block_index):
        h_teacher = teacher.transformer.layers[prev](
            h_teacher, context, global_cond, teacher_local[prev]
        )
        h_student = student.transformer.layers[prev](
            h_student, context, global_cond, teacher_local[prev]
        )
    return h_teacher, h_student


def model_params_for_export(
    student: nn.Module,
    records: dict[str, TernaryWeights],
    group_size: int,
    crop_len: int,
    quantizer_mode: str = "affine_centered",
) -> dict[str, mx.array]:
    target = dit_mlx_medium.DiT(T_lat=crop_len)
    prefixes = set(records)

    def predicate(path: str, module: nn.Module) -> bool:
        return path in prefixes

    nn.quantize(
        target,
        bits=2,
        group_size=group_size,
        mode="affine",
        class_predicate=predicate,
    )
    target_params = dict(tree_flatten(target.parameters()))
    student_params = dict(tree_flatten(student.parameters()))

    for key, value in student_params.items():
        if any(key == f"{prefix}.weight" for prefix in prefixes):
            continue
        if key not in target_params:
            raise KeyError(f"Student parameter missing in export target: {key}")
        # Preserve model and buffer dtypes.  A blanket fp16 cast changed the
        # deterministic Fourier buffer and made reload differ from the model
        # that was evaluated during training.
        target_params[key] = value

    for prefix, quantized in records.items():
        expected = (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases")
        if any(key not in target_params for key in expected):
            available = [key for key in target_params if prefix in key]
            raise KeyError(f"Target was not quantized at {prefix}; available={available}")
        target_params[f"{prefix}.weight"] = mx.array(quantized.packed_codes)
        target_params[f"{prefix}.scales"] = mx.array(quantized.scales)
        target_params[f"{prefix}.biases"] = mx.array(quantized.biases)

    mx.eval(*target_params.values())
    return target_params


def parameter_reload_report(student: nn.Module, reloaded: nn.Module) -> dict:
    """Compare every serialized parameter/buffer after an in-process reload."""
    expected = dict(tree_flatten(student.parameters()))
    actual = dict(tree_flatten(reloaded.parameters()))
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches: list[dict] = []
    for key in sorted(set(expected) & set(actual)):
        left = np.asarray(expected[key])
        right = np.asarray(actual[key])
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            mismatches.append(
                {
                    "key": key,
                    "expected_shape": list(left.shape),
                    "actual_shape": list(right.shape),
                    "expected_dtype": str(left.dtype),
                    "actual_dtype": str(right.dtype),
                    "relative_error": relative_error(left, right),
                }
            )
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "unexpected": unexpected,
        "mismatches": mismatches,
        "exact": not missing and not unexpected and not mismatches,
    }


def export_artifact(
    student: nn.Module,
    records: dict[str, TernaryWeights],
    output_path: Path,
    manifest_path: Path,
    group_size: int,
    crop_len: int,
    config: dict,
    quantizer_mode: str = "affine_centered",
    storage_mode: str | None = None,
) -> dict:
    params = model_params_for_export(
        student, records, group_size, crop_len, quantizer_mode
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symmetric_training = quantizer_mode in {"symmetric", "learned_symmetric"}
    if storage_mode is None:
        storage_mode = (
            "symmetric_compact" if symmetric_training else "full_affine"
        )
    arrays = {
        key: np.array(value)
        for key, value in params.items()
        if not (
            storage_mode == "symmetric_compact"
            and key.endswith(".biases")
        )
    }
    tmp_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_path), **arrays)
    os.replace(str(tmp_path), str(output_path))

    manifest = {
        "schema": "onus.ternary-quality/v3",
        "artifact": str(output_path),
        "created_at_unix": time.time(),
        "model": {
            "family": "stable-audio-3",
            "dit": "medium",
            "base_weights": "models/mlx/dit_medium_f16.npz",
            "crop_len_at_train": crop_len,
            "group_size": group_size,
            "mode": "direct_symmetric" if symmetric_training else "direct_affine",
            "hadamard": False,
            "quantizer": "s_q" if symmetric_training else "affine_centered_m_plus_s_q",
            "quantizer_mode": "symmetric" if symmetric_training else "affine_centered",
            "training_quantizer_mode": quantizer_mode,
            "storage_mode": storage_mode,
        },
        "scope": {
            "kind": "core-ternary",
            "paths": sorted(records),
            "count": len(records),
            "digest": scope_digest(records),
            "excluded": [
                "to_local_embed.seq.0",
                "to_local_embed.seq.2",
                "project_in",
                "project_out",
                "global_cond_embedder",
                "norms",
                "to_scale_shift_gate",
            ],
        },
        "config": config,
        "size_bytes": output_path.stat().st_size,
        "parameter_dtype_policy": "preserve_student_parameters_and_buffers",
    }
    write_json(manifest_path, manifest)
    return manifest


def reload_model(
    artifact: Path,
    records: dict[str, TernaryWeights],
    group_size: int,
    crop_len: int,
    quantizer_mode: str = "affine_centered",
    storage_mode: str = "full_affine",
) -> nn.Module:
    model = dit_mlx_medium.DiT(T_lat=crop_len)
    prefixes = set(records)

    def predicate(path: str, module: nn.Module) -> bool:
        return path in prefixes

    nn.quantize(
        model,
        bits=2,
        group_size=group_size,
        mode="affine",
        class_predicate=predicate,
    )
    model.load_weights(str(artifact), strict=storage_mode != "symmetric_compact")
    if storage_mode == "symmetric_compact":
        apply_symmetric_derived_biases(model, prefixes)
    model.freeze()
    mx.eval(model.parameters())
    return model


def reload_check(
    student: nn.Module,
    reloaded: nn.Module,
    samples: list[dict],
    cross_cache: dict[str, mx.array],
    global_cond: mx.array,
    sigmas: list[float],
    crop_len: int,
    seed: int,
) -> dict:
    values: list[float] = []
    cosines: list[float] = []
    for i, sigma in enumerate(sigmas):
        sample = samples[i % len(samples)]
        x = noised_latent(sample, crop_len, sigma, seed + i)
        cross = cross_cache[sample["prompt"]]
        t = timestep_tensor(sigma)
        expected = student(x, t, cross, global_cond)
        actual = reloaded(x, t, cross, global_cond)
        mx.eval(expected, actual)
        expected_np = np.array(expected)
        actual_np = np.array(actual)
        values.append(relative_error(expected_np, actual_np))
        flat_expected = expected_np.astype(np.float32).ravel()
        flat_actual = actual_np.astype(np.float32).ravel()
        cosines.append(float(np.dot(flat_expected, flat_actual) / (np.linalg.norm(flat_expected) * np.linalg.norm(flat_actual) + 1e-8)))
    return {
        "checked": len(values),
        "max_relative_error": float(max(values or [0.0])),
        "min_cosine": float(min(cosines or [1.0])),
        "relative_errors": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-first direct ternary DiT training")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="strict v6 trainer config; CLI values explicitly supplied here override it",
    )
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--teacher-weights", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument(
        "--quantizer-mode",
        choices=("symmetric", "learned_symmetric", "learned_affine", "affine_centered"),
        default=None,
        help="symmetric=s*q; learned modes learn thresholds/levels; affine_centered is fixed",
    )
    parser.add_argument("--crop-len", type=int, default=None)
    parser.add_argument("--start-block", type=int, default=None)
    parser.add_argument("--end-block", type=int, default=None)
    parser.add_argument("--steps-per-block", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--polish-steps", type=int, default=None)
    parser.add_argument("--polish-learning-rate", type=float, default=None)
    parser.add_argument("--polish-grad-clip", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-end", type=float, default=None)
    parser.add_argument("--optimizer-eps", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument(
        "--sigma-grid",
        type=str,
        default=None,
        help="JSON list, e.g. '[0.95,0.75,0.50,0.25,0.10]'",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="resume completed ternary blocks from records_checkpoint.npz",
    )
    parser.add_argument(
        "--checkpoint-every-blocks",
        type=int,
        default=None,
        help="rolling ternary-record checkpoint cadence",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help="full current-block/optimizer/RNG checkpoint cadence; 0 disables",
    )
    parser.add_argument(
        "--resume-step-checkpoint",
        type=str,
        default=None,
        help="resume a current block from a full optimizer/RNG checkpoint",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="stop after training/checkpoint metrics; do not write the full dense-shaped export",
    )
    args = parser.parse_args()
    overrides = {
        key: getattr(args, key)
        for key in CONFIG_KEYS
        if hasattr(args, key)
    }
    if args.sigma_grid is not None:
        overrides["sigma_grid"] = json.loads(args.sigma_grid)
    pilot_only = args.pilot_only
    resolved_config = resolve_config(args.config, overrides)
    values = dict(resolved_config["values"])
    for key in (
        "dataset_dir",
        "teacher_weights",
        "output_dir",
        "resume_checkpoint",
        "resume_step_checkpoint",
    ):
        values[key] = Path(values[key]) if values[key] is not None else None
    args = argparse.Namespace(**values)
    args.pilot_only = pilot_only
    args.config = Path(resolved_config["source_config"]) if resolved_config["source_config"] else None
    args.resolved_config = resolved_config
    model_block_count = 24
    block_stop = (
        args.end_block + 1 if args.end_block is not None else args.max_blocks
    )
    if args.start_block >= model_block_count:
        raise ValueError(f"start_block {args.start_block} is outside the {model_block_count}-block DiT")
    if block_stop <= args.start_block:
        raise ValueError("the selected block range is empty")
    args.block_stop = min(block_stop, model_block_count)
    if args.group_size not in (32, 64, 128):
        raise ValueError(
            "MLX QuantizedLinear supports group_size 32, 64, or 128; "
            "use a custom kernel before requesting group16"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    config = dict(args.resolved_config["values"])
    config["dataset_dir"] = str(args.dataset_dir)
    config["teacher_weights"] = str(args.teacher_weights)
    config["output_dir"] = str(args.output_dir)
    config["resume_checkpoint"] = str(args.resume_checkpoint) if args.resume_checkpoint else None
    config["resume_step_checkpoint"] = str(args.resume_step_checkpoint) if args.resume_step_checkpoint else None
    config["effective_block_range"] = {
        "start_inclusive": args.start_block,
        "stop_exclusive": args.block_stop,
    }
    config["runtime"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "mlx_runtime_root": str(MLX_RUNTIME_ROOT),
        "teacher_fingerprint": file_fingerprint(args.teacher_weights),
    }
    write_json(args.output_dir / "resolved_config.json", args.resolved_config)
    write_json(args.output_dir / "config.json", config)

    print(
        "=== Ternary quality recovery: "
        f"{args.quantizer_mode}, hard prefix, reload gate ===",
        flush=True,
    )
    print(json.dumps(config, indent=2, default=str), flush=True)
    print(f"[Memory] start {memory_snapshot()}", flush=True)

    samples = load_samples(args.dataset_dir, args.max_samples)
    print(f"[Data] {len(samples)} real latents, {len(set(s['prompt'] for s in samples))} prompts", flush=True)

    scope_paths = [
        f"transformer.layers.{block}.{name}"
        for block in range(args.start_block, args.block_stop)
        for name in CORE_NAMES
    ]
    write_json(
        args.output_dir / "scope.json",
        {
            "schema": "onus.ternary-quality/v6-scope",
            "kind": "core-ternary",
            "paths": scope_paths,
            "count": len(scope_paths),
            "digest": scope_digest(scope_paths),
            "group_size": args.group_size,
            "quantizer_mode": args.quantizer_mode,
        },
    )
    write_json(
        args.output_dir / "baseline_manifest.json",
        {
            "schema": "onus.ternary-quality/v6-baseline",
            "teacher": file_fingerprint(args.teacher_weights),
            "dataset": {
                "directory": str(args.dataset_dir),
                "sample_count": len(samples),
                "prompt_count": len({sample["prompt"] for sample in samples}),
                "samples": [
                    {
                        "path": sample["path"],
                        "file": file_fingerprint(Path(sample["path"])),
                        "prompt": sample["prompt"],
                        "genre": sample["genre"],
                    }
                    for sample in samples
                ],
            },
            "scope": {
                "paths": scope_paths,
                "count": len(scope_paths),
                "digest": scope_digest(scope_paths),
            },
            "config": config,
        },
    )

    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    student = dit_mlx_medium.DiT(T_lat=args.crop_len)
    student.load_weights(str(args.teacher_weights), strict=False)
    records: dict[str, TernaryWeights] = {}
    start_block = args.start_block
    block_metrics: list[dict] = []
    step_resume_state: dict | None = None
    if args.resume_step_checkpoint:
        if not args.resume_checkpoint:
            raise ValueError("--resume-step-checkpoint requires --resume-checkpoint")
        step_resume_state = tc.load_step_checkpoint(args.resume_step_checkpoint)
        step_meta = step_resume_state["metadata"]
        if step_meta.get("quantizer_mode") != args.quantizer_mode:
            raise ValueError("step checkpoint quantizer mode does not match current run")
        if int(step_meta.get("group_size")) != args.group_size:
            raise ValueError("step checkpoint group_size does not match current run")
        if int(step_meta.get("crop_len")) != args.crop_len:
            raise ValueError("step checkpoint crop_len does not match current run")
    if args.resume_checkpoint:
        records, checkpoint = load_records_checkpoint(args.resume_checkpoint)
        if int(checkpoint["group_size"]) != args.group_size:
            raise ValueError("checkpoint group_size does not match current run")
        if int(checkpoint["crop_len"]) != args.crop_len:
            raise ValueError("checkpoint crop_len does not match current run")
        if checkpoint["quantizer_mode"] != args.quantizer_mode:
            raise ValueError("checkpoint quantizer mode does not match current run")
        student = apply_records_to_model(student, records, args.group_size)
        start_block = int(checkpoint["next_block"])
        metrics_path = args.output_dir / "block_metrics.json"
        if metrics_path.exists():
            block_metrics = json.loads(metrics_path.read_text(encoding="utf-8")).get(
                "blocks", []
            )
        print(
            f"[Resume] checkpoint={args.resume_checkpoint} next_block={start_block} "
            f"records={len(records)}",
            flush=True,
        )
    print(f"[Memory] teacher+student {memory_snapshot()}", flush=True)

    context_cache, cross_cache, global_cond, global_pre = cache_conditioning(
        teacher, args.teacher_weights, [s["prompt"] for s in samples], args.seconds
    )
    teacher_local = local_pads(teacher, args.crop_len)
    sigmas = [float(s) for s in args.sigma_grid]
    rng = random.Random(args.seed)
    if step_resume_state:
        rng = tc.restore_rngs(step_resume_state, rng)
        resume_block = int(step_resume_state["metadata"]["block_index"])
        if start_block != resume_block:
            raise ValueError(
                f"records checkpoint next_block={start_block} does not match "
                f"step checkpoint block={resume_block}"
            )

    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint cadence must be zero or positive")
    data_order_digest = scope_digest(s["path"] for s in samples)
    if args.checkpoint_every_steps:
        # A step checkpoint must have a matching completed-block snapshot even
        # for block 0, where that snapshot is intentionally empty.
        save_records_checkpoint(
            args.output_dir / "records_checkpoint.npz",
            records,
            start_block,
            args.group_size,
            args.crop_len,
            args.quantizer_mode,
        )
    if args.checkpoint_every_blocks <= 0:
        raise ValueError("checkpoint cadence must be positive")

    for block_index in range(
        start_block, min(args.block_stop, len(student.transformer.layers))
    ):
        started = time.time()
        block_base_records = (
            args.output_dir / "checkpoints" / f"block_{block_index:02d}_base_records.npz"
        )
        save_records_checkpoint(
            block_base_records,
            records,
            block_index,
            args.group_size,
            args.crop_len,
            args.quantizer_mode,
        )
        block = student.transformer.layers[block_index]
        replace_core_block(block, args.group_size, args.quantizer_mode)
        student.freeze()
        # v3 trains only the declared ternary matrices.  Excluded modulators,
        # norms and projections remain the teacher checkpoint, making a block
        # boundary checkpoint sufficient for deterministic resume.
        for module in core_modules(block).values():
            module.unfreeze()
        optimizer = optim.AdamW(
            learning_rate=optim.cosine_decay(
                args.learning_rate,
                args.steps_per_block,
                end=args.learning_rate_end,
            ),
            eps=args.optimizer_eps,
            weight_decay=args.weight_decay,
        )
        optimizer.init(block.trainable_parameters())
        step_start = 0
        if step_resume_state and block_index == int(
            step_resume_state["metadata"]["block_index"]
        ):
            block.update(step_resume_state["model_state"])
            optimizer.state = step_resume_state["optimizer_state"]
            optimizer.init(block.trainable_parameters())
            step_start = int(step_resume_state["metadata"]["step_next"])
            print(
                f"[ResumeStep] block={block_index} next_step={step_start}",
                flush=True,
            )
        terminal = block_index == len(student.transformer.layers) - 1

        def terminal_loss(model, h_in, context, g, local, target, target_velocity):
            return terminal_block_loss(
                model,
                h_in,
                context,
                g,
                local,
                target,
                target_velocity,
                student.transformer.project_out,
                student.postprocess_conv,
            )

        value_grad = nn.value_and_grad(block, terminal_loss if terminal else block_loss)
        latest = None

        for step in range(step_start, args.steps_per_block):
            accumulated_grads = None
            micro_losses: list[float] = []
            sigma = sigmas[(step * args.gradient_accumulation + block_index) % len(sigmas)]
            for micro in range(args.gradient_accumulation):
                sample = samples[rng.randrange(len(samples))]
                sigma = sigmas[
                    (step * args.gradient_accumulation + micro + block_index)
                    % len(sigmas)
                ]
                x = noised_latent(
                    sample,
                    args.crop_len,
                    sigma,
                    args.seed
                    + block_index * 100000
                    + step * args.gradient_accumulation
                    + micro,
                )
                context = context_cache[sample["prompt"]]
                g = timestep_projection(teacher, global_pre, sigma)
                h_teacher, h_student = prefix_states(
                    teacher, student, x, block_index, context, g, teacher_local
                )
                target = teacher.transformer.layers[block_index](
                    h_teacher, context, g, teacher_local[block_index]
                )
                if terminal:
                    t = timestep_tensor(sigma)
                    target_velocity = teacher(
                        x, t, cross_cache[sample["prompt"]], global_cond
                    )
                    mx.eval(h_student, target, target_velocity)
                    loss, grads = value_grad(
                        block,
                        h_student,
                        context,
                        g,
                        teacher_local[block_index],
                        target,
                        target_velocity,
                    )
                else:
                    mx.eval(h_student, target)
                    loss, grads = value_grad(
                        block, h_student, context, g, teacher_local[block_index], target
                    )
                mx.eval(loss, grads)
                loss_value = float(loss)
                if not np.isfinite(loss_value):
                    raise RuntimeError(
                        f"non-finite block loss at block={block_index} "
                        f"step={step + 1} micro={micro + 1}"
                    )
                accumulated_grads = (
                    grads
                    if accumulated_grads is None
                    else tree_add(accumulated_grads, grads)
                )
                micro_losses.append(loss_value)
            grads = tree_scale(
                accumulated_grads,
                1.0 / float(args.gradient_accumulation),
            )
            grads, _ = optim.clip_grad_norm(grads, args.gradient_clip)
            optimizer.update(block, grads)
            mx.eval(block.parameters(), optimizer.state)
            latest = float(np.mean(micro_losses))
            if (
                args.checkpoint_every_steps
                and (step + 1) % args.checkpoint_every_steps == 0
            ):
                step_path = (
                    args.output_dir
                    / "checkpoints"
                    / f"block_{block_index:02d}_step_{step + 1:04d}.npz"
                )
                tc.save_step_checkpoint(
                    step_path,
                    block.trainable_parameters(),
                    optimizer.state,
                    {
                        "block_index": block_index,
                        "step_next": step + 1,
                        "group_size": args.group_size,
                        "crop_len": args.crop_len,
                        "quantizer_mode": args.quantizer_mode,
                        "seed": args.seed,
                        "gradient_accumulation": args.gradient_accumulation,
                        "data_order_digest": data_order_digest,
                        "records_checkpoint": str(block_base_records),
                    },
                    rng.getstate(),
                    np.random.get_state(),
                    list(getattr(mx.random, "state", [])),
                )
                print(f"[StepCheckpoint] {step_path}", flush=True)
            if step == 0 or (step + 1) % max(1, args.steps_per_block // 4) == 0:
                print(
                    f"[Block {block_index:02d}] step {step + 1}/{args.steps_per_block} "
                    f"loss={latest:.5f} sigma={sigma:.4f} "
                    f"microbatches={args.gradient_accumulation}",
                    flush=True,
                )

        mx.eval(block.parameters())
        hard_metrics = hard_freeze_block(
            block,
            block_index,
            args.group_size,
            records,
            args.quantizer_mode,
        )
        block_metric = {
            **hard_metrics,
            "steps": args.steps_per_block,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "learning_rate_end": args.learning_rate_end,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "last_loss": latest,
            "seconds": time.time() - started,
        }
        block_metrics.append(block_metric)
        write_json(args.output_dir / "block_metrics.json", {"blocks": block_metrics})
        if (
            (block_index + 1) % args.checkpoint_every_blocks == 0
            or block_index + 1 == min(args.block_stop, len(student.transformer.layers))
        ):
            checkpoint_path = args.output_dir / "records_checkpoint.npz"
            save_records_checkpoint(
                checkpoint_path,
                records,
                block_index + 1,
                args.group_size,
                args.crop_len,
                args.quantizer_mode,
            )
            print(f"[Checkpoint] {checkpoint_path} records={len(records)}", flush=True)
        print(f"[Block {block_index:02d}] hard-frozen {hard_metrics} memory={memory_snapshot()}", flush=True)
        del optimizer, value_grad
        gc.collect()
        mx.clear_cache()
        step_resume_state = None

    if args.start_block == 0 and args.block_stop == len(student.transformer.layers) and args.polish_steps:
        print(f"[Polish] {args.polish_steps} velocity steps on FP16 modulators/project_out", flush=True)
        student.freeze()
        for block in student.transformer.layers:
            block.unfreeze(recurse=False, keys=["to_scale_shift_gate"])
            block.pre_norm.unfreeze()
            block.cross_attend_norm.unfreeze()
            block.ff_norm.unfreeze()
        student.transformer.project_out.unfreeze()
        polish_opt = optim.AdamW(
            learning_rate=optim.cosine_decay(
                args.polish_learning_rate,
                args.polish_steps,
                end=max(args.polish_learning_rate * 0.1, 1e-8),
            ),
            # MLX casts eps to the gradient dtype.  1e-8 underflows to zero
            # for FP16 modulators and turns zero-gradient entries into 0/0.
            eps=1e-4,
            weight_decay=1e-6,
        )
        polish_opt.init(student.trainable_parameters())

        def polish_loss(model, x, t, cross, global_raw, target):
            prediction = model(x, t, cross, global_raw)
            p = prediction.astype(mx.float32)
            q = target.astype(mx.float32)
            mse = mx.mean((p - q) ** 2) / (mx.mean(q * q) + 1e-6)
            cosine = mx.sum(p * q) / (
                mx.sqrt(mx.sum(p * p)) * mx.sqrt(mx.sum(q * q)) + 1e-6
            )
            return mse + 3.0 * (1.0 - cosine)

        polish_grad = nn.value_and_grad(student, polish_loss)
        for step in range(args.polish_steps):
            sample = samples[rng.randrange(len(samples))]
            sigma = sigmas[(step * 3) % len(sigmas)]
            x = noised_latent(sample, args.crop_len, sigma, args.seed + 500000 + step)
            cross = cross_cache[sample["prompt"]]
            t = timestep_tensor(sigma)
            target = teacher(x, t, cross, global_cond)
            mx.eval(target)
            loss, grads = polish_grad(
                student, x, t, cross, global_cond, target
            )
            mx.eval(loss, grads)
            loss_value = float(loss)
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Polish produced non-finite loss at step {step + 1}")
            bad_grads = [
                (key, list(np.asarray(value).shape), str(np.asarray(value).dtype))
                for (key, value) in tree_flatten(grads)
                if not np.isfinite(np.asarray(value)).all()
            ]
            if bad_grads:
                raise RuntimeError(
                    f"Polish produced non-finite gradients at step {step + 1}: "
                    f"{bad_grads[:8]}"
                )
            grads, _ = optim.clip_grad_norm(grads, args.polish_grad_clip)
            polish_opt.update(student, grads)
            mx.eval(student.parameters(), polish_opt.state, loss)
            parameter_values = [
                np.asarray(value)
                for _, value in tree_flatten(student.parameters())
            ]
            bad_parameters = [
                (key, list(np.asarray(value).shape), str(np.asarray(value).dtype))
                for (key, value) in tree_flatten(student.parameters())
                if not np.isfinite(np.asarray(value)).all()
            ]
            if bad_parameters:
                raise RuntimeError(
                    f"Polish produced non-finite parameters at step {step + 1}: "
                    f"{bad_parameters[:8]}"
                )
            if step == 0 or (step + 1) % max(1, args.polish_steps // 4) == 0:
                print(
                    f"[Polish] step {step + 1}/{args.polish_steps} "
                    f"loss={float(loss):.5f} sigma={sigma:.4f}",
                    flush=True,
                )
        del polish_opt, polish_grad

    if args.pilot_only:
        write_json(
            args.output_dir / "run_summary.json",
            {
                "status": "pilot_only_no_export",
                "records_checkpoint": str(args.output_dir / "records_checkpoint.npz"),
                "blocks_trained": min(args.block_stop, len(student.transformer.layers)) - args.start_block,
                "block_metrics": block_metrics,
                "elapsed_seconds": time.time() - run_started,
                "memory": memory_snapshot(),
            },
        )
        print(
            f"[PilotOnly] checkpoint={args.output_dir / 'records_checkpoint.npz'} "
            f"blocks={len(block_metrics)} memory={memory_snapshot()}",
            flush=True,
        )
        return

    artifact = args.output_dir / (
        f"dit_medium_ternary_quality_{args.quantizer_mode}_group{args.group_size}_core.npz"
    )
    manifest_path = artifact.with_suffix(".json")
    manifest = export_artifact(
        student,
        records,
        artifact,
        manifest_path,
        args.group_size,
        args.crop_len,
        config,
        args.quantizer_mode,
        "symmetric_compact"
        if args.quantizer_mode in {"symmetric", "learned_symmetric"}
        else "full_affine",
    )
    print(f"[Export] {artifact} bytes={artifact.stat().st_size} scope={manifest['scope']['count']}", flush=True)
    reloaded = reload_model(
        artifact,
        records,
        args.group_size,
        args.crop_len,
        args.quantizer_mode,
        "symmetric_compact"
        if args.quantizer_mode in {"symmetric", "learned_symmetric"}
        else "full_affine",
    )
    mx.eval(reloaded.parameters())
    print(f"[Reload] success memory={memory_snapshot()}", flush=True)
    parameter_metrics = parameter_reload_report(student, reloaded)
    print(f"[Reload] parameter parity {parameter_metrics}", flush=True)
    if not parameter_metrics["exact"]:
        raise RuntimeError(f"Parameter/buffer reload mismatch: {parameter_metrics}")
    reload_metrics = reload_check(
        student,
        reloaded,
        samples,
        cross_cache,
        global_cond,
        [0.95, 0.50, 0.10],
        args.crop_len,
        args.seed + 900000,
    )
    print(f"[Reload] metrics {reload_metrics}", flush=True)
    if reload_metrics["max_relative_error"] > 1e-6 or reload_metrics["min_cosine"] < 0.999999:
        raise RuntimeError(f"Reload gate failed: {reload_metrics}")
    write_json(
        args.output_dir / "run_summary.json",
        {
            "status": "exported_reloadable",
            "artifact": str(artifact),
            "manifest": str(manifest_path),
            "blocks_trained": min(args.block_stop, len(student.transformer.layers)) - args.start_block,
            "block_metrics": block_metrics,
            "parameter_reload": parameter_metrics,
            "reload_metrics": reload_metrics,
            "elapsed_seconds": time.time() - run_started,
            "memory": memory_snapshot(),
        },
    )


if __name__ == "__main__":
    main()
