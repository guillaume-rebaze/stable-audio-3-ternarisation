"""Bonsai-style ternary contract used by the V9 recovery pipeline.

This module deliberately stays independent from MLX and the legacy quantizers.
It is the small, auditable boundary between a dense Stable Audio 3 checkpoint
and a deployable Bonsai-style package:

* the seven attention/FFN matrices in each DiT block are ternary;
* every other tensor remains native precision;
* each ternary group is ``W = scale * q`` with ``q in {-1, 0, +1}``;
* packed codes reserve the fourth two-bit value and reject it on decode.

The contract is intentionally strict.  Training code may experiment behind it,
but an export cannot silently change the scope, representation, or provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONTRACT_SCHEMA = "onus.ternary-quality/v9-bonsai-contract"
EXPECTED_BLOCKS = 24
CORE_SUFFIXES: tuple[str, ...] = (
    "self_attn.to_qkv.weight",
    "self_attn.to_out.weight",
    "cross_attn.to_q.weight",
    "cross_attn.to_kv.weight",
    "cross_attn.to_out.weight",
    "ff.ff.0.proj.weight",
    "ff.ff.2.weight",
)
SUPPORTED_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)
PACK_BITS = 2
PACK_VALUES = 4
TERNARY_VALUES = frozenset((-1, 0, 1))


class BonsaiContractError(ValueError):
    """Raised when a checkpoint or package violates the P0 contract."""


@dataclass(frozen=True)
class TensorHeader:
    """Header-only description of one NPZ member."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def native_bytes(self) -> int:
        return self.numel * int(np.dtype(self.dtype).itemsize)

    def to_dict(self, role: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
            "native_bytes": self.native_bytes,
            "role": role,
        }


def normalize_tensor_name(name: str) -> str:
    """Normalize an NPZ member name without changing its logical key."""

    normalized = name.replace("\\", "/")
    if normalized.endswith(".npy"):
        normalized = normalized[:-4]
    return normalized


def expected_core_names(block_count: int = EXPECTED_BLOCKS) -> frozenset[str]:
    return frozenset(
        f"transformer.layers.{block}.{suffix}"
        for block in range(block_count)
        for suffix in CORE_SUFFIXES
    )


def _core_candidate(name: str) -> bool:
    """Whether a name looks like one of the explicitly scoped core linears."""

    match = re.fullmatch(r"transformer\.layers\.(\d+)\.(.+)", name)
    return bool(match and any(match.group(2) == suffix for suffix in CORE_SUFFIXES))


def inspect_npz_headers(path: str | Path) -> list[TensorHeader]:
    """Read NPZ/Numpy headers only; no tensor payload is materialized."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    headers: list[TensorHeader] = []
    with zipfile.ZipFile(checkpoint) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith(".npy"):
                continue
            with archive.open(member, "r") as stream:
                version = np.lib.format.read_magic(stream)
                shape, _fortran_order, dtype = np.lib.format._read_array_header(  # type: ignore[attr-defined]
                    stream, version
                )
            headers.append(
                TensorHeader(
                    name=normalize_tensor_name(member.filename),
                    shape=tuple(int(dim) for dim in shape),
                    dtype=np.dtype(dtype).str,
                )
            )

    headers.sort(key=lambda item: item.name)
    return headers


def build_weight_scope(
    headers: Sequence[TensorHeader],
    *,
    block_count: int = EXPECTED_BLOCKS,
) -> dict[str, Any]:
    """Build and validate the exact Bonsai core/support inventory."""

    normalized: list[TensorHeader] = []
    seen: set[str] = set()
    for header in headers:
        name = normalize_tensor_name(header.name)
        if name in seen:
            raise BonsaiContractError(f"duplicate tensor name: {name}")
        seen.add(name)
        normalized.append(
            TensorHeader(name=name, shape=tuple(header.shape), dtype=np.dtype(header.dtype).str)
        )

    expected = expected_core_names(block_count)
    present = {header.name for header in normalized}
    missing = sorted(expected - present)
    if missing:
        raise BonsaiContractError(f"missing scoped core tensors ({len(missing)}): {missing[:4]}")

    unexpected_candidates = sorted(
        name for name in present if _core_candidate(name) and name not in expected
    )
    if unexpected_candidates:
        raise BonsaiContractError(
            "core scope contains unexpected tensors: " + ", ".join(unexpected_candidates[:4])
        )

    entries = [
        header.to_dict("core" if header.name in expected else "support_native")
        for header in sorted(normalized, key=lambda item: item.name)
    ]
    core_entries = [entry for entry in entries if entry["role"] == "core"]
    support_entries = [entry for entry in entries if entry["role"] != "core"]
    core_numel = sum(int(entry["numel"]) for entry in core_entries)
    total_numel = sum(int(entry["numel"]) for entry in entries)
    return {
        "block_count": block_count,
        "core_suffixes": list(CORE_SUFFIXES),
        "core_tensor_count": len(core_entries),
        "support_tensor_count": len(support_entries),
        "core_numel": core_numel,
        "total_numel": total_numel,
        "core_fraction": core_numel / total_numel if total_numel else 0.0,
        "tensors": entries,
    }


def storage_report(scope: Mapping[str, Any], group_size: int = 128) -> dict[str, Any]:
    """Calculate payload bytes for packed 2-bit core plus native supports."""

    if group_size not in SUPPORTED_GROUP_SIZES:
        raise BonsaiContractError(f"unsupported group size: {group_size}")

    core_codes_bytes = 0
    core_scale_count = 0
    padded_elements = 0
    core_matrix_count = 0
    support_native_bytes = 0
    for tensor in scope["tensors"]:
        shape = tuple(int(dim) for dim in tensor["shape"])
        if tensor["role"] != "core":
            support_native_bytes += int(tensor["native_bytes"])
            continue
        if len(shape) != 2:
            raise BonsaiContractError(f"core tensor is not a matrix: {tensor['name']} {shape}")
        out_dim, in_dim = shape
        groups_per_row = (in_dim + group_size - 1) // group_size
        padded_in = groups_per_row * group_size
        groups = out_dim * groups_per_row
        core_codes_bytes += groups * group_size * PACK_BITS // 8
        core_scale_count += groups
        padded_elements += out_dim * (padded_in - in_dim)
        core_matrix_count += 1

    scale_bytes = core_scale_count * np.dtype(np.float16).itemsize
    code_bytes = int(core_codes_bytes)
    packed_payload_bytes = code_bytes + scale_bytes + support_native_bytes
    return {
        "group_size": group_size,
        "core_matrix_count": core_matrix_count,
        "core_code_bits": PACK_BITS,
        "core_code_bytes": code_bytes,
        "core_scale_count": int(core_scale_count),
        "core_scale_dtype": "float16",
        "core_scale_bytes": int(scale_bytes),
        "support_native_bytes": int(support_native_bytes),
        "padded_core_elements": int(padded_elements),
        "packed_payload_bytes": int(packed_payload_bytes),
        "packed_payload_mib": packed_payload_bytes / (1024 * 1024),
        "package_envelope_bytes": 650_000_000,
        "within_package_envelope": packed_payload_bytes <= 650_000_000,
    }


def _validate_q_array(q: np.ndarray) -> np.ndarray:
    values = np.asarray(q)
    if values.ndim != 3:
        raise BonsaiContractError(f"q must be [out, groups, group], got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise BonsaiContractError(f"q must be integer, got {values.dtype}")
    if not np.isin(values, list(TERNARY_VALUES)).all():
        raise BonsaiContractError("q contains a value outside {-1, 0, +1}")
    return values.astype(np.int8, copy=False)


def validate_strict_groups(q: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    """Validate strict ``W = scale * q`` groups and return compact statistics."""

    codes = _validate_q_array(q)
    scale_values = np.asarray(scales)
    if scale_values.shape != codes.shape[:2]:
        raise BonsaiContractError(
            f"scale shape {scale_values.shape} != group shape {codes.shape[:2]}"
        )
    if not np.issubdtype(scale_values.dtype, np.floating):
        raise BonsaiContractError(f"scales must be floating point, got {scale_values.dtype}")
    if not np.isfinite(scale_values).all() or (scale_values < 0).any():
        raise BonsaiContractError("scales must be finite and non-negative")
    dead_groups = scale_values == 0
    if dead_groups.any() and np.any(codes[dead_groups] != 0):
        raise BonsaiContractError("zero-scale groups must contain only zero q values")
    return {
        "out_dim": int(codes.shape[0]),
        "group_count": int(codes.shape[0] * codes.shape[1]),
        "group_size": int(codes.shape[2]),
        "zero_fraction": float(np.mean(codes == 0)),
        "dead_group_count": int(dead_groups.sum()),
    }


def pack_bonsai_codes(q: np.ndarray) -> np.ndarray:
    """Pack ternary q values into two-bit words; code 3 is never emitted."""

    values = _validate_q_array(q)
    group_size = values.shape[2]
    if group_size % 16:
        raise BonsaiContractError("group size must be divisible by 16 for uint32 packing")
    # +1 -> 0, 0 -> 1, -1 -> 2; 3 remains reserved and is never valid.
    encoded = np.where(values == 1, 0, np.where(values == 0, 1, 2)).astype(np.uint32)
    encoded = encoded.reshape(values.shape[0], -1)
    packed = np.zeros((encoded.shape[0], encoded.shape[1] // 16), dtype=np.uint32)
    for slot in range(16):
        packed |= encoded[:, slot::16] << np.uint32(slot * PACK_BITS)
    return packed


def unpack_bonsai_codes(
    packed: np.ndarray,
    *,
    out_dim: int,
    group_count: int,
    group_size: int,
) -> np.ndarray:
    """Unpack codes and reject the reserved two-bit value 3."""

    if group_size % 16:
        raise BonsaiContractError("group size must be divisible by 16 for uint32 packing")
    packed_values = np.asarray(packed, dtype=np.uint32)
    flat_code_count = group_count * group_size
    expected_words = (flat_code_count + 15) // 16
    if packed_values.shape != (out_dim, expected_words):
        raise BonsaiContractError(
            f"packed shape {packed_values.shape} != {(out_dim, expected_words)}"
        )
    decoded = np.empty((out_dim, expected_words * 16), dtype=np.int8)
    for slot in range(16):
        code = (packed_values >> np.uint32(slot * PACK_BITS)) & np.uint32(3)
        if np.any(code == 3):
            raise BonsaiContractError("reserved packed code 3 encountered")
        decoded[:, slot::16] = np.where(code == 0, 1, np.where(code == 1, 0, -1))
    return decoded[:, :flat_code_count].reshape(out_dim, group_count, group_size)


def dequantize_strict_groups(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    validate_strict_groups(q, scales)
    return np.asarray(q, dtype=np.float32) * np.asarray(scales, dtype=np.float32)[..., None]


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    candidate = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, role: str) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "role": role,
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(candidate),
    }


def build_experiment_contract(
    *,
    scope: Mapping[str, Any],
    storage: Mapping[str, Any],
    files: Iterable[tuple[str | Path, str]],
    teacher: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    sampler: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the immutable, hashable P0 contract payload."""

    payload: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": {
            "core_values": [-1, 0, 1],
            "representation": "W = scale * q",
            "code_bits": PACK_BITS,
            "reserved_code": 3,
            "support_policy": "native_precision",
        },
        "scope": dict(scope),
        "storage": dict(storage),
        "files": [file_identity(path, role=role) for path, role in files],
        "teacher": dict(teacher or {}),
        "data": dict(data or {}),
        "sampler": dict(sampler or {}),
        "resources": dict(resources or {}),
    }
    payload["contract_digest"] = canonical_json_digest(payload)
    return payload


def validate_experiment_contract(contract: Mapping[str, Any], *, verify_files: bool = True) -> None:
    """Validate schema, digest, and optionally all declared file identities."""

    if contract.get("schema") != CONTRACT_SCHEMA:
        raise BonsaiContractError(f"unexpected schema: {contract.get('schema')!r}")
    supplied_digest = contract.get("contract_digest")
    if not isinstance(supplied_digest, str):
        raise BonsaiContractError("missing contract_digest")
    without_digest = dict(contract)
    without_digest.pop("contract_digest", None)
    if canonical_json_digest(without_digest) != supplied_digest:
        raise BonsaiContractError("contract_digest does not match contract contents")
    if verify_files:
        for item in contract.get("files", []):
            path = Path(item["path"])
            if not path.is_file():
                raise BonsaiContractError(f"declared file is missing: {path}")
            if int(item["size_bytes"]) != path.stat().st_size:
                raise BonsaiContractError(f"declared file size changed: {path}")
            if sha256_file(path) != item["sha256"]:
                raise BonsaiContractError(f"declared file hash changed: {path}")


def write_contract(contract: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(contract, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return output
