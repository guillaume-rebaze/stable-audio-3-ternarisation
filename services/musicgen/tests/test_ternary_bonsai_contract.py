from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ternary_bonsai_contract import (
    BonsaiContractError,
    TensorHeader,
    build_experiment_contract,
    build_weight_scope,
    inspect_npz_headers,
    pack_bonsai_codes,
    storage_report,
    unpack_bonsai_codes,
    validate_experiment_contract,
    validate_strict_groups,
)


def synthetic_headers(block_count: int = 24) -> list[TensorHeader]:
    headers = [
        TensorHeader(
            name=f"transformer.layers.{block}.{suffix}",
            shape=(128, 128),
            dtype="<f2",
        )
        for block in range(block_count)
        for suffix in (
            "self_attn.to_qkv.weight",
            "self_attn.to_out.weight",
            "cross_attn.to_q.weight",
            "cross_attn.to_kv.weight",
            "cross_attn.to_out.weight",
            "ff.ff.0.proj.weight",
            "ff.ff.2.weight",
        )
    ]
    headers += [
        TensorHeader("transformer.layers.0.norm.weight", (128,), "<f4"),
        TensorHeader("transformer.layers.0.norm.bias", (128,), "<f4"),
    ]
    return headers


def test_scope_is_exact_and_supports_are_native() -> None:
    scope = build_weight_scope(synthetic_headers())
    assert scope["core_tensor_count"] == 168
    assert scope["support_tensor_count"] == 2
    assert scope["core_numel"] == 168 * 128 * 128
    assert scope["core_fraction"] > 0.99
    assert all(item["role"] == "core" for item in scope["tensors"] if item["name"].endswith("to_qkv.weight"))
    assert all(item["role"] == "support_native" for item in scope["tensors"] if "norm" in item["name"])


def test_scope_rejects_missing_and_unexpected_core_tensor() -> None:
    headers = synthetic_headers()
    headers = [
        header
        for header in headers
        if header.name != "transformer.layers.23.ff.ff.2.weight"
    ]
    with pytest.raises(BonsaiContractError, match="missing scoped core"):
        build_weight_scope(headers)

    headers = synthetic_headers()
    headers.append(TensorHeader("transformer.layers.24.self_attn.to_qkv.weight", (128, 128), "<f2"))
    with pytest.raises(BonsaiContractError, match="unexpected tensors"):
        build_weight_scope(headers)


def test_storage_report_uses_two_bit_codes_and_native_supports() -> None:
    scope = build_weight_scope(synthetic_headers())
    report = storage_report(scope, 128)
    expected_codes = 168 * 128 * 128 // 4
    expected_scales = 168 * 128 * 2
    expected_support = 2 * 128 * 4
    assert report["core_code_bytes"] == expected_codes
    assert report["core_scale_bytes"] == expected_scales
    assert report["support_native_bytes"] == expected_support
    assert report["packed_payload_bytes"] == expected_codes + expected_scales + expected_support


def test_packed_codes_round_trip_and_reserved_code_rejected() -> None:
    q = np.array(
        [
            [[1, 0, -1, 1] * 4, [0, -1, 1, 0] * 4],
            [[-1, 1, 0, -1] * 4, [1, 1, 0, 0] * 4],
        ],
        dtype=np.int8,
    )
    packed = pack_bonsai_codes(q)
    restored = unpack_bonsai_codes(packed, out_dim=2, group_count=2, group_size=16)
    np.testing.assert_array_equal(restored, q)

    broken = packed.copy()
    broken[0, 0] |= np.uint32(3)
    with pytest.raises(BonsaiContractError, match="reserved packed code"):
        unpack_bonsai_codes(broken, out_dim=2, group_count=2, group_size=16)


def test_strict_groups_reject_affine_like_invalid_state() -> None:
    q = np.zeros((2, 3, 16), dtype=np.int8)
    scales = np.ones((2, 3), dtype=np.float16)
    assert validate_strict_groups(q, scales)["dead_group_count"] == 0
    with pytest.raises(BonsaiContractError, match="non-negative"):
        validate_strict_groups(q, np.full((2, 3), -1, dtype=np.float16))
    q[0, 0, 0] = 1
    with pytest.raises(BonsaiContractError, match="zero-scale"):
        validate_strict_groups(q, np.zeros((2, 3), dtype=np.float16))


def test_npz_inventory_is_header_only_and_contract_digest_catches_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.npz"
    np.savez(checkpoint, **{"transformer.layers.0.norm.weight": np.ones(4, dtype=np.float32)})
    headers = inspect_npz_headers(checkpoint)
    assert headers[0].name == "transformer.layers.0.norm.weight"
    assert headers[0].shape == (4,)
    source = tmp_path / "source.py"
    source.write_text("v1\n", encoding="utf-8")
    scope = {"tensors": [], "core_tensor_count": 0, "support_tensor_count": 0}
    storage = {"group_size": 128, "packed_payload_bytes": 0}
    contract = build_experiment_contract(
        scope=scope,
        storage=storage,
        files=[(source, "source")],
        teacher={"checkpoint": "teacher.npz"},
    )
    validate_experiment_contract(contract)
    mutated = json.loads(json.dumps(contract))
    mutated["teacher"]["checkpoint"] = "changed.npz"
    with pytest.raises(BonsaiContractError, match="contract_digest"):
        validate_experiment_contract(mutated, verify_files=False)
