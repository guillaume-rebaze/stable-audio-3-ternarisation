from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from render_ternary_quality import (  # noqa: E402
    audio_mono,
    audio_stereo,
    check_metal_memory,
)
from train_ternary_quality import (  # noqa: E402
    CORE_NAMES,
    TernaryHadamardLinear,
    TernaryQATLinear,
    TernaryTTQLinear,
    dequantize,
    dequantize_record,
    hard_freeze_block,
)
from train_ternary_window_v6 import (  # noqa: E402
    checkpointed_module_call,
    quantized_linear_to_qat,
    split_scale_parameter_tree,
)
from ternary_contract import (  # noqa: E402
    quantize_affine_with_assignment_and_scales,
    quantize_symmetric_weight,
    quantize_symmetric_with_assignment_and_scales,
    quantize_ttq_with_assignment_and_scales,
    rotate_weight_hadamard,
)


def test_audio_gate_preserves_stereo_and_uses_mid_for_scalar_metrics() -> None:
    left = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    right = -left
    decoded = mx.array(np.stack([left, right], axis=0)[None])
    stereo = audio_stereo(decoded)
    assert stereo.shape == (32, 2)
    assert np.allclose(stereo[:, 0], left)
    assert np.allclose(stereo[:, 1], right)
    assert np.allclose(audio_mono(stereo), 0.0)


def test_audio_gate_accepts_time_major_decoder_output() -> None:
    values = np.zeros((32, 2), dtype=np.float32)
    values[:, 0] = 1.0
    values[:, 1] = 2.0
    stereo = audio_stereo(mx.array(values[None]))
    assert stereo.shape == (32, 2)
    assert np.allclose(stereo, values)


def test_audio_gate_enforces_metal_memory_limit(monkeypatch) -> None:
    snapshot = {"metal_active_gb": 0.5, "metal_peak_gb": 1.0}
    monkeypatch.setattr("render_ternary_quality.tq.memory_snapshot", lambda: snapshot)

    assert check_metal_memory(2 * 1024**3, "test") == snapshot
    with pytest.raises(RuntimeError, match="Metal memory guard"):
        check_metal_memory(512 * 1024**2, "test")
    with pytest.raises(ValueError, match="positive"):
        check_metal_memory(0, "test")


def test_learned_symmetric_qat_forward_matches_serialized_assignment() -> None:
    rng = np.random.default_rng(709)
    weight = rng.normal(size=(3, 64)).astype(np.float32)
    positive_scales = np.array([[0.21, 0.37], [0.13, 0.29], [0.44, 0.18]], dtype=np.float32)
    threshold_logs = np.array([[0.25, -0.25], [0.15, -0.15], [0.30, -0.30]], dtype=np.float32)
    layer = TernaryQATLinear(64, 3, bias=False, group_size=32, quantizer_mode="learned_symmetric")
    layer.weight = mx.array(weight)
    layer.log_scales = mx.log(mx.array(positive_scales))
    layer.log_threshold_multiplier = mx.array(threshold_logs)

    groups = weight.reshape(3, 2, 32)
    assignment_scales = (
        np.maximum(np.mean(np.abs(groups), axis=-1), 1e-6)
        * np.exp(threshold_logs)
    )
    serialized = quantize_symmetric_with_assignment_and_scales(
        weight, assignment_scales, positive_scales, group_size=32
    )
    qat_output = np.asarray(layer(mx.eye(64)), dtype=np.float32)
    exported_weight = dequantize(serialized.q, serialized.scales, serialized.biases)
    fixed_q = np.clip(
        np.rint(
            groups
            / np.maximum(np.mean(np.abs(groups), axis=-1, keepdims=True), 1e-6)
        ),
        -1,
        1,
    )

    assert np.array_equal(
        serialized.q,
        np.clip(np.rint(groups / assignment_scales[..., None]), -1, 1).astype(np.int8),
    )
    assert np.any(serialized.q != fixed_q.astype(np.int8))
    assert np.allclose(qat_output, exported_weight.T, atol=1e-3, rtol=1e-3)


def test_learned_affine_qat_forward_matches_serialized_assignment() -> None:
    rng = np.random.default_rng(710)
    weight = rng.normal(size=(3, 64)).astype(np.float32)
    means = rng.normal(scale=0.05, size=(3, 2)).astype(np.float32)
    positive_scales = np.array(
        [[0.21, 0.37], [0.13, 0.29], [0.44, 0.18]], dtype=np.float32
    )
    threshold_logs = np.array(
        [[0.25, -0.25], [0.15, -0.15], [0.30, -0.30]], dtype=np.float32
    )
    layer = TernaryQATLinear(
        64, 3, bias=False, group_size=32, quantizer_mode="learned_affine"
    )
    layer.weight = mx.array(weight)
    layer.group_biases = mx.array(means)
    layer.log_scales = mx.log(mx.array(positive_scales))
    layer.log_threshold_multiplier = mx.array(threshold_logs)

    groups = weight.reshape(3, 2, 32)
    assignment_scales = (
        np.maximum(np.mean(np.abs(groups - means[..., None]), axis=-1), 1e-6)
        * np.exp(threshold_logs)
    )
    serialized = quantize_affine_with_assignment_and_scales(
        weight, assignment_scales, positive_scales, means, group_size=32
    )
    qat_output = np.asarray(layer(mx.eye(64)), dtype=np.float32)
    exported_weight = dequantize(serialized.q, serialized.scales, serialized.biases)
    assert np.allclose(qat_output, exported_weight.T, atol=1e-3, rtol=1e-3)


def test_ttq_qat_and_runtime_forward_match_independent_signed_levels() -> None:
    rng = np.random.default_rng(711)
    weight = rng.normal(size=(3, 64)).astype(np.float32)
    means = rng.normal(scale=0.05, size=(3, 2)).astype(np.float32)
    positive_scales = np.array(
        [[0.21, 0.37], [0.13, 0.29], [0.44, 0.18]], dtype=np.float32
    )
    negative_scales = np.array(
        [[0.11, 0.31], [0.23, 0.19], [0.35, 0.27]], dtype=np.float32
    )
    threshold_logs = np.array(
        [[0.25, -0.25], [0.15, -0.15], [0.30, -0.30]], dtype=np.float32
    )
    layer = TernaryQATLinear(64, 3, bias=False, group_size=32, quantizer_mode="ttq")
    layer.weight = mx.array(weight)
    layer.group_biases = mx.array(means)
    layer.log_scales = mx.log(mx.array(positive_scales))
    layer.log_negative_scales = mx.log(mx.array(negative_scales))
    layer.log_threshold_multiplier = mx.array(threshold_logs)

    groups = weight.reshape(3, 2, 32)
    assignment_scales = (
        np.maximum(np.mean(np.abs(groups - means[..., None]), axis=-1), 1e-6)
        * np.exp(threshold_logs)
    )
    serialized = quantize_ttq_with_assignment_and_scales(
        weight,
        assignment_scales,
        positive_scales,
        negative_scales,
        means,
        group_size=32,
    )
    identity = mx.eye(64)
    qat_output = np.asarray(layer(identity), dtype=np.float32)
    runtime = TernaryTTQLinear.from_record(SimpleNamespace(bias=None), serialized)
    runtime_output = np.asarray(runtime(identity), dtype=np.float32)
    np.testing.assert_allclose(qat_output, runtime_output, atol=1e-3, rtol=1e-3)


def test_ttq_runtime_matches_mlx_affine_when_signed_levels_are_equal() -> None:
    rng = np.random.default_rng(712)
    weight = rng.normal(size=(4, 64)).astype(np.float32)
    means = rng.normal(scale=0.05, size=(4, 2)).astype(np.float32)
    positive = np.array(
        [[0.21, 0.37], [0.13, 0.29], [0.44, 0.18], [0.17, 0.31]],
        dtype=np.float32,
    )
    assignment = np.maximum(
        np.mean(np.abs(weight.reshape(4, 2, 32) - means[..., None]), axis=-1),
        1e-6,
    )
    record = quantize_ttq_with_assignment_and_scales(
        weight, assignment, positive, positive, means, group_size=32
    )
    mlx_runtime = nn.QuantizedLinear(
        64, 4, bias=False, group_size=32, bits=2, mode="affine"
    )
    mlx_runtime.weight = mx.array(record.packed_codes)
    mlx_runtime.scales = mx.array(record.scales)
    mlx_runtime.biases = mx.array(record.biases)
    mlx_runtime.freeze()
    ttq_runtime = TernaryTTQLinear.from_record(SimpleNamespace(bias=None), record)
    inputs = mx.array(rng.normal(size=(3, 64)).astype(np.float16))
    expected = mlx_runtime(inputs)
    actual = ttq_runtime(inputs)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-3, atol=3e-3
    )


def test_ttq_hadamard_qat_and_runtime_forward_match() -> None:
    rng = np.random.default_rng(713)
    dense_weight = rng.normal(size=(3, 64)).astype(np.float32)
    rotated_weight = rotate_weight_hadamard(dense_weight, 32)
    means = rng.normal(scale=0.05, size=(3, 2)).astype(np.float32)
    positive = np.array(
        [[0.21, 0.37], [0.13, 0.29], [0.44, 0.18]], dtype=np.float32
    )
    negative = np.array(
        [[0.11, 0.31], [0.23, 0.19], [0.35, 0.27]], dtype=np.float32
    )
    threshold_logs = np.zeros((3, 2), dtype=np.float32)
    layer = TernaryQATLinear(
        64, 3, bias=False, group_size=32, quantizer_mode="ttq_hadamard"
    )
    layer.weight = mx.array(rotated_weight)
    layer.group_biases = mx.array(means)
    layer.log_scales = mx.log(mx.array(positive))
    layer.log_negative_scales = mx.log(mx.array(negative))
    layer.log_threshold_multiplier = mx.array(threshold_logs)
    groups = rotated_weight.reshape(3, 2, 32)
    assignment = np.maximum(
        np.mean(np.abs(groups - means[..., None]), axis=-1), 1e-6
    )
    record = quantize_ttq_with_assignment_and_scales(
        rotated_weight,
        assignment,
        positive,
        negative,
        means,
        group_size=32,
        mode="ttq_hadamard",
    )
    identity = mx.eye(64)
    qat_output = np.asarray(layer(identity), dtype=np.float32)
    runtime = TernaryTTQLinear.from_record(SimpleNamespace(bias=None), record)
    runtime_output = np.asarray(runtime(identity), dtype=np.float32)
    np.testing.assert_allclose(qat_output, runtime_output, atol=1e-3, rtol=1e-3)


def test_reopened_ternary_linear_preserves_serialized_forward() -> None:
    rng = np.random.default_rng(923)
    weight = rng.normal(size=(4, 64)).astype(np.float32)
    bias = rng.normal(size=(4,)).astype(np.float32)
    record = quantize_symmetric_weight(weight, group_size=32)
    quantized = nn.QuantizedLinear(
        64, 4, bias=True, group_size=32, bits=2, mode="affine"
    )
    quantized.weight = mx.array(record.packed_codes)
    quantized.scales = mx.array(record.scales)
    quantized.biases = mx.array(record.biases)
    quantized.bias = mx.array(bias, dtype=mx.float16)
    quantized.freeze()

    reopened = quantized_linear_to_qat(quantized, record, 32, "symmetric")
    inputs = mx.array(rng.normal(size=(3, 64)).astype(np.float16))
    expected = quantized(inputs)
    actual = reopened(inputs)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(
        np.asarray(reopened.weight),
        dequantize(record.q, record.scales, record.biases),
        rtol=0,
        atol=0,
    )


def test_reopened_symmetric_hadamard_preserves_rotated_basis() -> None:
    rng = np.random.default_rng(925)
    dense_weight = rng.normal(size=(4, 64)).astype(np.float32)
    rotated_weight = rotate_weight_hadamard(dense_weight, 32)
    record = replace(
        quantize_symmetric_weight(rotated_weight, group_size=32),
        mode="symmetric_hadamard",
    )
    runtime = TernaryHadamardLinear.from_record(SimpleNamespace(bias=None), record)

    reopened = quantized_linear_to_qat(
        runtime, record, 32, "learned_symmetric_hadamard"
    )
    inputs = mx.array(rng.normal(size=(3, 64)).astype(np.float16))
    expected = runtime(inputs)
    actual = reopened(inputs)
    mx.eval(expected, actual)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(
        np.asarray(reopened.weight),
        dequantize_record(record),
        rtol=0,
        atol=0,
    )


def test_reopened_biasless_ternary_linear() -> None:
    rng = np.random.default_rng(924)
    record = quantize_symmetric_weight(
        rng.normal(size=(4, 64)).astype(np.float32), group_size=32
    )
    quantized = nn.QuantizedLinear(
        64, 4, bias=False, group_size=32, bits=2, mode="affine"
    )
    quantized.weight = mx.array(record.packed_codes)
    quantized.scales = mx.array(record.scales)
    quantized.biases = mx.array(record.biases)
    quantized.freeze()

    reopened = quantized_linear_to_qat(quantized, record, 32, "symmetric")
    assert reopened.bias is None


def test_learned_threshold_gradient_reaches_optimizer_update() -> None:
    rng = np.random.default_rng(923)
    layer = TernaryQATLinear(
        64, 3, bias=False, group_size=32, quantizer_mode="learned_symmetric"
    )
    layer.weight = mx.array(rng.normal(size=(3, 64)).astype(np.float32))
    layer.log_scales = mx.log(mx.ones((3, 2), dtype=mx.float32) * 0.25)
    value_grad = nn.value_and_grad(
        layer, lambda model, inputs: mx.mean(model(inputs) ** 2)
    )
    loss, grads = value_grad(layer, mx.eye(64))
    mx.eval(loss, grads)
    threshold_grad = np.asarray(grads["log_threshold_multiplier"], dtype=np.float32)
    assert np.isfinite(threshold_grad).all()
    assert float(np.linalg.norm(threshold_grad)) > 0.0

    _, quantizer_params = split_scale_parameter_tree(layer.trainable_parameters())
    _, quantizer_grads = split_scale_parameter_tree(grads)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    optimizer.init(quantizer_params)
    updates = optimizer.apply_gradients(quantizer_grads, quantizer_params)
    layer.update(updates)
    mx.eval(layer.log_threshold_multiplier)
    assert np.any(np.asarray(layer.log_threshold_multiplier) != 0.0)


def test_checkpointed_qat_forward_retains_threshold_gradient() -> None:
    rng = np.random.default_rng(924)
    layer = TernaryQATLinear(
        64, 3, bias=False, group_size=32, quantizer_mode="learned_symmetric"
    )
    layer.weight = mx.array(rng.normal(size=(3, 64)).astype(np.float32))
    layer.log_scales = mx.log(mx.ones((3, 2), dtype=mx.float32) * 0.25)

    def checkpointed_loss(model, inputs):
        return mx.mean(checkpointed_module_call(model, inputs) ** 2)

    value_grad = nn.value_and_grad(layer, checkpointed_loss)
    loss, grads = value_grad(layer, mx.eye(64))
    mx.eval(loss, grads)
    threshold_grad = np.asarray(grads["log_threshold_multiplier"], dtype=np.float32)
    weight_grad = np.asarray(grads["weight"], dtype=np.float32)
    assert np.isfinite(threshold_grad).all()
    assert np.isfinite(weight_grad).all()
    assert float(np.linalg.norm(threshold_grad)) > 0.0
    assert float(np.linalg.norm(weight_grad)) > 0.0


def test_hard_freeze_preserves_learned_threshold_forward_for_every_core_matrix() -> None:
    rng = np.random.default_rng(812)
    modules = {}
    for name in CORE_NAMES:
        layer = TernaryQATLinear(
            32, 32, bias=False, group_size=32, quantizer_mode="learned_symmetric"
        )
        layer.weight = mx.array(rng.normal(size=(32, 32)).astype(np.float32))
        layer.log_scales = mx.log(
            mx.array(np.full((32, 1), 0.3, dtype=np.float32))
        )
        layer.log_threshold_multiplier = mx.array(
            rng.uniform(-0.5, 0.5, size=(32, 1)).astype(np.float32)
        )
        modules[name] = layer

    block = SimpleNamespace(
        self_attn=SimpleNamespace(
            to_qkv=modules["self_attn.to_qkv"], to_out=modules["self_attn.to_out"]
        ),
        cross_attn=SimpleNamespace(
            to_q=modules["cross_attn.to_q"],
            to_kv=modules["cross_attn.to_kv"],
            to_out=modules["cross_attn.to_out"],
        ),
        ff=SimpleNamespace(
            ff=[
                SimpleNamespace(proj=modules["ff.ff.0.proj"]),
                SimpleNamespace(),
                modules["ff.ff.2"],
            ]
        ),
        freeze=lambda: None,
        parameters=lambda: {},
    )
    identity = mx.eye(32)
    qat_outputs = {
        name: np.asarray(layer(identity), dtype=np.float32)
        for name, layer in modules.items()
    }
    records = {}

    hard_freeze_block(block, 0, 32, records, "learned_symmetric")

    for name in CORE_NAMES:
        record = records[f"transformer.layers.0.{name}"]
        exported_weight = dequantize(record.q, record.scales, record.biases)
        assert np.allclose(qat_outputs[name], exported_weight.T, atol=1e-3, rtol=1e-3)


def test_optimizer_tree_splits_threshold_and_reconstruction_parameters() -> None:
    tree = {
        "transformer": {
            "layers": [
                {
                    "weight": mx.array([1.0]),
                    "log_scales": mx.array([2.0]),
                    "log_negative_scales": mx.array([2.5]),
                    "log_threshold_multiplier": mx.array([3.0]),
                    "group_biases": mx.array([4.0]),
                }
            ]
        }
    }
    weights, quantizer = split_scale_parameter_tree(tree)
    assert "weight" in weights["transformer"]["layers"][0]
    assert "log_scales" not in weights["transformer"]["layers"][0]
    assert "log_negative_scales" not in weights["transformer"]["layers"][0]
    assert "log_threshold_multiplier" not in weights["transformer"]["layers"][0]
    assert "group_biases" not in weights["transformer"]["layers"][0]
    assert "log_scales" in quantizer["transformer"]["layers"][0]
    assert "log_negative_scales" in quantizer["transformer"]["layers"][0]
    assert "log_threshold_multiplier" in quantizer["transformer"]["layers"][0]
    assert "group_biases" in quantizer["transformer"]["layers"][0]
