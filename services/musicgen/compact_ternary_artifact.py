"""Drop derivable symmetric affine biases from an exported ternary NPZ."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import train_ternary_quality as tq
from ternary_contract import TernaryWeights, unpack_codes, validate_ternary_weights, write_json


def records_from_artifact(artifact: Path, manifest: dict) -> dict[str, TernaryWeights]:
    group_size = int(manifest["model"]["group_size"])
    mode = str(manifest["model"].get("quantizer_mode", "symmetric"))
    records: dict[str, TernaryWeights] = {}
    with np.load(artifact, allow_pickle=False) as arrays:
        for prefix in manifest["scope"]["paths"]:
            packed = np.array(arrays[f"{prefix}.weight"], dtype=np.uint32)
            scales = np.array(arrays[f"{prefix}.scales"], dtype=np.float16)
            biases = np.array(arrays[f"{prefix}.biases"], dtype=np.float16)
            q = unpack_codes(packed, group_size)
            record = TernaryWeights(
                packed_codes=packed,
                scales=scales,
                biases=biases,
                q=q,
                group_means=np.zeros_like(scales, dtype=np.float32),
                group_size=group_size,
                mode=mode,
            )
            validate_ternary_weights(record)
            records[prefix] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact a symmetric ternary artifact")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    source_manifest_path = args.source.with_suffix(".json")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest["model"].get("quantizer_mode") != "symmetric":
        raise ValueError("Only symmetric artifacts can use derived-bias compaction")
    records = records_from_artifact(args.source, manifest)
    arrays: dict[str, np.ndarray] = {}
    with np.load(args.source, allow_pickle=False) as source:
        for key in source.files:
            if key.endswith(".biases"):
                continue
            arrays[key] = np.array(source[key])
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.destination.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), **arrays)
    os.replace(str(tmp), str(args.destination))

    compact_manifest = dict(manifest)
    compact_manifest["artifact"] = str(args.destination)
    compact_manifest["size_bytes"] = args.destination.stat().st_size
    compact_manifest["model"] = dict(manifest["model"])
    compact_manifest["model"]["storage_mode"] = "symmetric_compact"
    compact_manifest["compaction"] = {
        "removed": "biases",
        "reconstruction": "biases = -scales at load time",
        "scope_count": len(records),
    }
    destination_manifest = args.destination.with_suffix(".json")
    write_json(destination_manifest, compact_manifest)

    reference = tq.reload_model(
        args.source,
        records,
        int(manifest["model"]["group_size"]),
        int(manifest["model"]["crop_len_at_train"]),
        "symmetric",
        "full_affine",
    )
    compact = tq.reload_model(
        args.destination,
        records,
        int(manifest["model"]["group_size"]),
        int(manifest["model"]["crop_len_at_train"]),
        "symmetric",
        "symmetric_compact",
    )
    parity = tq.parameter_reload_report(reference, compact)
    write_json(
        args.destination.with_name("compact_validation.json"),
        {
            "status": "validated" if parity["exact"] else "failed",
            "source": str(args.source),
            "destination": str(args.destination),
            "parameter_parity": parity,
            "source_bytes": args.source.stat().st_size,
            "destination_bytes": args.destination.stat().st_size,
        },
    )
    if not parity["exact"]:
        raise SystemExit(f"compact reload mismatch: {parity}")
    print(json.dumps({
        "status": "validated",
        "source_bytes": args.source.stat().st_size,
        "destination_bytes": args.destination.stat().st_size,
        "saved_bytes": args.source.stat().st_size - args.destination.stat().st_size,
        "parameter_exact": parity["exact"],
    }, indent=2))


if __name__ == "__main__":
    main()
