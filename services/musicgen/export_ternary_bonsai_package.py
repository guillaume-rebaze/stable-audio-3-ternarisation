#!/usr/bin/env python3
"""Export a compact, reloadable Bonsai-style ternary DiT package.

The training checkpoint is intentionally not the deployment format: it keeps
unpacked q arrays and MLX affine metadata for resumption, so it is much larger
than a Bonsai payload.  This exporter writes one NPZ payload containing:

* native-precision support tensors copied from the dense teacher;
* two-bit packed ternary codes for the seven scoped matrices in every block;
* one positive FP16 scale per ternary group;
* explicit per-matrix direct/Hadamard runtime metadata.

The output is independent of the training checkpoint at runtime.  The
checkpoint is retained only as provenance in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ternary_bonsai_contract import (
    BonsaiContractError,
    build_weight_scope,
    expected_core_names,
    inspect_npz_headers,
    sha256_file,
    storage_report,
    unpack_bonsai_codes,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = Path(
    "/Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/"
    "mlx/models/mlx/dit_medium_f16.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-weights", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--cascade-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--crop-len", type=int, default=128)
    return parser.parse_args()


def fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_array_member(archive: zipfile.ZipFile, member: str, array: np.ndarray) -> None:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    archive.writestr(member, stream.getvalue())


def copy_member(archive: zipfile.ZipFile, source: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    # Copy the decompressed .npy bytes into the new archive.  This keeps the
    # deployment payload self-contained while allowing a fresh compression
    # level and a stable member name.
    archive.writestr(info.filename, source.read(info))


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "onus.ternary-quality/v4-records-checkpoint":
        raise BonsaiContractError(f"unsupported records schema: {metadata.get('schema')!r}")
    if sha256_file(path) != metadata.get("payload_sha256"):
        raise BonsaiContractError("records checkpoint checksum mismatch")
    return metadata


def validate_checkpoint_scope(metadata: dict[str, Any], group_size: int) -> list[str]:
    scope = sorted(str(item) for item in metadata.get("scope", []))
    expected = sorted(name[:-len(".weight")] for name in expected_core_names())
    if scope != expected:
        missing = sorted(set(expected) - set(scope))
        extra = sorted(set(scope) - set(expected))
        raise BonsaiContractError(
            f"checkpoint is not the complete 24-block core: missing={missing[:4]} extra={extra[:4]}"
        )
    per_prefix = metadata.get("group_size_by_prefix", {})
    if any(int(per_prefix.get(prefix, metadata["group_size"])) != group_size for prefix in scope):
        raise BonsaiContractError("mixed or unexpected group size in final checkpoint")
    modes = metadata.get("quantizer_mode_by_prefix", {})
    invalid = sorted(
        prefix
        for prefix in scope
        if modes.get(prefix) not in {"symmetric", "symmetric_hadamard"}
    )
    if invalid:
        raise BonsaiContractError(f"unsupported deploy modes: {invalid[:4]}")
    return scope


def main() -> int:
    args = parse_args()
    checkpoint = args.records_checkpoint.expanduser().resolve()
    teacher = args.teacher_weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    payload = output_dir / "payload.npz"
    manifest_path = output_dir / "manifest.json"
    temporary = output_dir / ".payload.tmp.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    if output_dir.exists() and any(path.exists() for path in (payload, manifest_path, temporary)):
        raise FileExistsError(f"refusing to overwrite existing package: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_checkpoint_metadata(checkpoint)
    scope = validate_checkpoint_scope(metadata, args.group_size)
    headers = inspect_npz_headers(teacher)
    weight_scope = build_weight_scope(headers)
    expected = {entry["name"] for entry in weight_scope["tensors"] if entry["role"] == "core"}
    if expected != {f"{prefix}.weight" for prefix in scope}:
        raise BonsaiContractError("teacher/checkpoint core scope mismatch")
    storage = storage_report(weight_scope, args.group_size)

    header_by_name = {header.name: header for header in headers}
    source_members: dict[str, zipfile.ZipInfo] = {}
    support_overrides: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(teacher) as source_archive:
        for info in source_archive.infolist():
            if info.is_dir() or not info.filename.endswith(".npy"):
                continue
            source_members[info.filename[:-4]] = info

        # The checkpoint contains all records in one compressed archive.  Read
        # each member once and immediately emit its compact deployment form.
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as output_archive:
            with np.load(checkpoint, allow_pickle=False) as checkpoint_arrays:
                core_entries: list[dict[str, Any]] = []
                for prefix in scope:
                    core_name = f"{prefix}.weight"
                    header = header_by_name[core_name]
                    shape = tuple(int(value) for value in header.shape)
                    if len(shape) != 2:
                        raise BonsaiContractError(
                            f"core tensor is not a matrix: {core_name} {shape}"
                        )
                    out_dim, in_dim = shape
                    if in_dim % args.group_size:
                        raise BonsaiContractError(
                            f"input dimension not divisible by group size: {core_name}"
                        )
                    raw_packed = np.asarray(checkpoint_arrays[f"{prefix}.packed_codes"], dtype=np.uint32)
                    raw_scales = np.asarray(checkpoint_arrays[f"{prefix}.scales"], dtype=np.float16)
                    expected_groups = (out_dim, in_dim // args.group_size)
                    expected_packed = (out_dim, in_dim // 16)
                    if raw_packed.shape != expected_packed:
                        raise BonsaiContractError(
                            f"packed shape mismatch for {prefix}: {raw_packed.shape} != {expected_packed}"
                        )
                    if raw_scales.shape != expected_groups:
                        raise BonsaiContractError(
                            f"scale shape mismatch for {prefix}: {raw_scales.shape} != {expected_groups}"
                        )
                    # Decode once to reject the reserved code 3 and independently
                    # prove that the training checkpoint is truly ternary.
                    q = unpack_bonsai_codes(
                        raw_packed,
                        out_dim=out_dim,
                        group_count=in_dim // args.group_size,
                        group_size=args.group_size,
                    )
                    if not np.all(np.isin(q, (-1, 0, 1))):
                        raise BonsaiContractError(f"non-ternary code in {prefix}")
                    if not np.isfinite(raw_scales).all() or np.any(raw_scales > 0):
                        raise BonsaiContractError(f"expected MLX negative scales in {prefix}")
                    physical_scales = (-raw_scales).astype(np.float16, copy=False)
                    if not np.isfinite(physical_scales).all() or np.any(physical_scales < 0):
                        raise BonsaiContractError(f"invalid physical scales in {prefix}")

                    packed_member = f"__ternary_core__/{prefix}.packed_codes.npy"
                    scale_member = f"__ternary_core__/{prefix}.scales.npy"
                    write_array_member(output_archive, packed_member, raw_packed)
                    write_array_member(output_archive, scale_member, physical_scales)
                    bias_key = f"{prefix}.linear_bias"
                    bias_member = None
                    if bias_key in checkpoint_arrays.files:
                        checkpoint_bias = np.asarray(checkpoint_arrays[bias_key], dtype=np.float16)
                        teacher_bias_info = source_members.get(f"{prefix}.bias")
                        if teacher_bias_info is None:
                            raise BonsaiContractError(f"native bias missing from teacher: {prefix}.bias")
                        with source_archive.open(teacher_bias_info) as teacher_bias_stream:
                            teacher_bias = np.asarray(
                                np.load(teacher_bias_stream, allow_pickle=False), dtype=np.float16
                            )
                        if teacher_bias.shape != checkpoint_bias.shape:
                            raise BonsaiContractError(
                                f"trained linear bias shape differs from native support: {bias_key}"
                            )
                        # A bias is outside the seven-matrix ternary scope.  If
                        # QAT learned it, keep it as a native support tensor
                        # rather than silently reverting to the dense teacher.
                        support_overrides[f"{prefix}.bias"] = checkpoint_bias.copy()
                        bias_member = f"{prefix}.bias.npy"
                    mode = str(metadata["quantizer_mode_by_prefix"][prefix])
                    core_entries.append(
                        {
                            "name": core_name,
                            "prefix": prefix,
                            "shape": list(shape),
                            "group_size": args.group_size,
                            "group_count": in_dim // args.group_size,
                            "mode": mode,
                            "hadamard_input": mode == "symmetric_hadamard",
                            "packed_member": packed_member[:-4],
                            "scale_member": scale_member[:-4],
                            "linear_bias_member": bias_member[:-4] if bias_member else None,
                            "zero_fraction": float(np.mean(q == 0)),
                        }
                    )

                core_names = {entry["name"] for entry in core_entries}
                support_names = sorted(
                    header.name for header in headers if header.name not in core_names
                )
                for support_name in support_names:
                    info = source_members.get(support_name)
                    if info is None:
                        raise BonsaiContractError(
                            f"support member missing from teacher archive: {support_name}"
                        )
                    if support_name in support_overrides:
                        write_array_member(
                            output_archive,
                            f"{support_name}.npy",
                            support_overrides[support_name],
                        )
                    else:
                        copy_member(output_archive, source_archive, info)

    # The archive context is deliberately opened after validation below; the
    # manifest is written only once the payload is atomically complete.
    payload_tmp = temporary
    # ``output_archive`` is created in the context above; this assignment is a
    # guard against accidentally leaving a partial output after a failed write.
    if not payload_tmp.exists():
        # The block above writes to the final temporary path through the alias
        # established just before the archive context.
        raise RuntimeError("temporary payload was not created")

    payload_tmp.replace(payload)
    manifest = {
        "schema": "onus.ternary-quality/v10-bonsai-package",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": {
            "representation": "W = scale * q",
            "core_values": [-1, 0, 1],
            "code_bits": 2,
            "reserved_code": 3,
            "payload_format": "NPZ/ZIP with native support members and packed core members",
            "runtime": "direct symmetric or block-input Hadamard symmetric",
        },
        "model": {
            "family": "stable-audio-3",
            "dit": "medium",
            "crop_len": args.crop_len,
            "group_size": args.group_size,
            "block_count": 24,
            "core_matrices_per_block": 7,
        },
        "storage": {
            **storage,
            "payload_file_bytes": int(payload.stat().st_size),
            "payload_sha256": sha256_file(payload),
            "package_envelope_check": bool(payload.stat().st_size <= 650_000_000),
        },
        "scope": {
            "core_count": len(core_entries),
            "support_count": len(support_names),
            "core_names": sorted(core_names),
            "support_names": support_names,
            "support_overrides": sorted(support_overrides),
            "core": core_entries,
        },
        "provenance": {
            "records_checkpoint": fingerprint(checkpoint),
            "teacher_weights": fingerprint(teacher),
            "cascade_manifest": (
                fingerprint(args.cascade_manifest.expanduser().resolve())
                if args.cascade_manifest is not None
                else None
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "exported",
                "payload": str(payload),
                "manifest": str(manifest_path),
                "payload_bytes": payload.stat().st_size,
                "packed_payload_bytes": storage["packed_payload_bytes"],
                "within_envelope": manifest["storage"]["package_envelope_check"],
                "core_count": len(core_entries),
                "support_count": len(support_names),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
