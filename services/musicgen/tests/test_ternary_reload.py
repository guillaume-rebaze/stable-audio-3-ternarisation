from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from audit_ternary_quality import validate_artifact_scope  # noqa: E402
import train_ternary_quality as tq  # noqa: E402
from train_ternary_quality import parameter_reload_report  # noqa: E402
from ternary_contract import quantize_symmetric_weight, scope_digest  # noqa: E402


class _FakeModel:
    def __init__(self, values: dict[str, mx.array]):
        self._values = values

    def parameters(self):
        return self._values


def test_parameter_reload_report_requires_exact_dtype_and_values() -> None:
    expected = _FakeModel(
        {
            "weight": mx.array([[1.0, 2.0]], dtype=mx.float16),
            "timestep_features.freqs": mx.array([1.0, 3.0], dtype=mx.float32),
        }
    )
    same = _FakeModel(
        {
            "weight": mx.array([[1.0, 2.0]], dtype=mx.float16),
            "timestep_features.freqs": mx.array([1.0, 3.0], dtype=mx.float32),
        }
    )
    assert parameter_reload_report(expected, same)["exact"]

    wrong_dtype = _FakeModel(
        {
            "weight": mx.array([[1.0, 2.0]], dtype=mx.float16),
            "timestep_features.freqs": mx.array([1.0, 3.0], dtype=mx.float16),
        }
    )
    report = parameter_reload_report(expected, wrong_dtype)
    assert not report["exact"]
    assert report["mismatches"][0]["key"] == "timestep_features.freqs"


def test_parameter_reload_report_detects_scope_drift() -> None:
    expected = _FakeModel({"a": mx.array([1.0], dtype=mx.float32)})
    actual = _FakeModel({"a": mx.array([1.0], dtype=mx.float32), "b": mx.array([2.0])})
    report = parameter_reload_report(expected, actual)
    assert report["unexpected"] == ["b"]
    assert not report["exact"]


def test_artifact_scope_validation_checks_compact_symmetric_arrays(tmp_path) -> None:
    scope = "transformer.layers.0.self_attn.to_qkv"
    quantized = quantize_symmetric_weight(
        np.arange(64, dtype=np.float32).reshape(2, 32), group_size=32
    )
    artifact = tmp_path / "compact.npz"
    np.savez(
        artifact,
        **{
            f"{scope}.weight": quantized.packed_codes,
            f"{scope}.scales": quantized.scales,
        },
    )
    manifest = {
        "schema": "onus.ternary-quality/v3",
        "model": {
            "group_size": 32,
            "quantizer_mode": "symmetric",
            "storage_mode": "symmetric_compact",
        },
        "scope": {"paths": [scope], "digest": scope_digest([scope])},
    }
    report = validate_artifact_scope(artifact, manifest)
    assert report["codes_and_metadata_valid"]
    assert report["scope_count"] == 1


def test_artifact_scope_validation_rejects_reserved_ternary_code(tmp_path) -> None:
    scope = "transformer.layers.0.self_attn.to_qkv"
    quantized = quantize_symmetric_weight(
        np.arange(64, dtype=np.float32).reshape(2, 32), group_size=32
    )
    packed = quantized.packed_codes.copy()
    packed[0, 0] |= np.uint32(3 << 6)
    artifact = tmp_path / "invalid.npz"
    np.savez(
        artifact,
        **{
            f"{scope}.weight": packed,
            f"{scope}.scales": quantized.scales,
        },
    )
    manifest = {
        "schema": "onus.ternary-quality/v3",
        "model": {
            "group_size": 32,
            "quantizer_mode": "symmetric",
            "storage_mode": "symmetric_compact",
        },
        "scope": {"paths": [scope], "digest": scope_digest([scope])},
    }
    with pytest.raises(ValueError, match=r"\[0,2\]"):
        validate_artifact_scope(artifact, manifest)


def test_records_roundtrip_restores_trained_linear_bias(tmp_path) -> None:
    def qat_linear():
        module = tq.TernaryQATLinear(32, 2, True, 32, "symmetric")
        weight = np.zeros((2, 32), dtype=np.float32)
        weight[0, :2] = [63.0, 1.0]
        weight[1, :2] = [-63.0, -1.0]
        module.weight = mx.array(weight)
        module.bias = mx.array([0.125, -0.25], dtype=mx.float32)
        return module

    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.to_qkv = qat_linear()
    block.self_attn.to_out = qat_linear()
    block.cross_attn = nn.Module()
    block.cross_attn.to_q = qat_linear()
    block.cross_attn.to_kv = qat_linear()
    block.cross_attn.to_out = qat_linear()
    block.ff = nn.Module()
    block.ff.ff = [nn.Module(), nn.Module(), nn.Module()]
    block.ff.ff[0].proj = qat_linear()
    block.ff.ff[2] = qat_linear()
    inputs = mx.arange(32, dtype=mx.float16).reshape(1, 32) / 32
    qat_expected = np.asarray(block.ff.ff[2](inputs))
    records = {}
    tq.hard_freeze_block(block, 0, 32, records, "symmetric")
    prefix = "transformer.layers.0.ff.ff.2"
    expected = np.asarray(block.ff.ff[2](inputs))
    checkpoint = tmp_path / "records.npz"
    tq.save_records_checkpoint(checkpoint, records, 1, 32, 128, "symmetric")
    records, metadata = tq.load_records_checkpoint(checkpoint)

    model = nn.Module()
    model.transformer = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.to_qkv = nn.Linear(32, 2, bias=True)
    layer.self_attn.to_out = nn.Linear(32, 2, bias=True)
    layer.cross_attn = nn.Module()
    layer.cross_attn.to_q = nn.Linear(32, 2, bias=True)
    layer.cross_attn.to_kv = nn.Linear(32, 2, bias=True)
    layer.cross_attn.to_out = nn.Linear(32, 2, bias=True)
    layer.ff = nn.Module()
    layer.ff.ff = [nn.Module(), nn.Module(), nn.Linear(32, 2, bias=True)]
    layer.ff.ff[0].proj = nn.Linear(32, 2, bias=True)
    for module in tq.core_modules(layer).values():
        module.weight = mx.zeros((2, 32), dtype=mx.float16)
        module.weight = mx.array(
            [[63.0, 1.0] + [0.0] * 30, [-63.0, -1.0] + [0.0] * 30],
            dtype=mx.float16,
        )
        module.bias = mx.zeros((2,), dtype=mx.float16)
    model.transformer.layers = [layer]
    tq.apply_records_to_model(model, records, group_size=32)

    actual = np.asarray(
        model.transformer.layers[0].ff.ff[2](inputs)
    )
    bias_output = np.asarray(
        model.transformer.layers[0].ff.ff[2](mx.zeros((1, 32), dtype=mx.float16))
    )
    assert metadata["linear_bias_count"] == len(records) == 7
    assert np.array_equal(records[prefix].linear_bias, np.array([0.125, -0.25], dtype=np.float16))
    assert np.allclose(qat_expected, expected, rtol=0.0, atol=1e-3)
    assert np.array_equal(actual, expected)
    assert np.array_equal(bias_output, np.array([[0.125, -0.25]], dtype=np.float16))


def test_records_roundtrip_supports_mixed_group_sizes(tmp_path) -> None:
    rng = np.random.default_rng(19)
    records = {}
    for index, name in enumerate(tq.CORE_NAMES):
        weight = rng.normal(size=(2, 64)).astype(np.float32)
        group_size = 64 if index == len(tq.CORE_NAMES) - 1 else 32
        record = quantize_symmetric_weight(
            weight, group_size=group_size
        )
        if index == len(tq.CORE_NAMES) - 1:
            record = replace(record, mode="symmetric_hadamard")
        records[f"transformer.layers.0.{name}"] = record

    checkpoint = tmp_path / "mixed-records.npz"
    tq.save_records_checkpoint(checkpoint, records, 1, 32, 128, "symmetric")
    restored, metadata = tq.load_records_checkpoint(checkpoint)

    assert metadata["group_size"] == 32
    assert metadata["group_size_by_prefix"][
        "transformer.layers.0.ff.ff.2"
    ] == 64
    assert metadata["quantizer_mode_by_prefix"][
        "transformer.layers.0.ff.ff.2"
    ] == "symmetric_hadamard"
    assert {record.group_size for record in restored.values()} == {32, 64}
    assert restored["transformer.layers.0.ff.ff.2"].mode == "symmetric_hadamard"

    model = nn.Module()
    model.transformer = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.to_qkv = nn.Linear(64, 2, bias=False)
    layer.self_attn.to_out = nn.Linear(64, 2, bias=False)
    layer.cross_attn = nn.Module()
    layer.cross_attn.to_q = nn.Linear(64, 2, bias=False)
    layer.cross_attn.to_kv = nn.Linear(64, 2, bias=False)
    layer.cross_attn.to_out = nn.Linear(64, 2, bias=False)
    layer.ff = nn.Module()
    layer.ff.ff = [nn.Module(), nn.Module(), nn.Linear(64, 2, bias=False)]
    layer.ff.ff[0].proj = nn.Linear(64, 2, bias=False)
    model.transformer.layers = [layer]

    tq.apply_records_to_model(model, restored, group_size=32)
    assert model.transformer.layers[0].self_attn.to_qkv.group_size == 32
    assert model.transformer.layers[0].ff.ff[2].group_size == 64
    assert type(model.transformer.layers[0].ff.ff[2]).__name__ == "TernaryHadamardLinear"
