"""Extract a cumulative ternary records checkpoint from an audited artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_ternary_quality as tq
from ternary_contract import TernaryWeights, unpack_codes, validate_ternary_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ternary records from an artifact")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    group_size = int(manifest["model"]["group_size"])
    quantizer_mode = str(manifest["model"].get("quantizer_mode", "symmetric"))
    if quantizer_mode not in {"symmetric", "affine_centered"}:
        raise ValueError(f"unsupported serialized quantizer mode: {quantizer_mode}")
    records: dict[str, TernaryWeights] = {}
    with np.load(args.artifact, allow_pickle=False) as arrays:
        storage_mode = str(manifest["model"].get("storage_mode", "full_affine"))
        for prefix in manifest["scope"]["paths"]:
            packed = np.array(arrays[f"{prefix}.weight"], dtype=np.uint32)
            scales = np.array(arrays[f"{prefix}.scales"], dtype=np.float16)
            biases = (
                -scales
                if storage_mode == "symmetric_compact"
                else np.array(arrays[f"{prefix}.biases"], dtype=np.float16)
            )
            q = unpack_codes(packed, group_size)
            linear_bias = (
                np.array(arrays[f"{prefix}.bias"], dtype=np.float16)
                if f"{prefix}.bias" in arrays.files
                else None
            )
            record = TernaryWeights(
                packed_codes=packed,
                scales=scales,
                biases=biases,
                q=q,
                group_means=np.zeros_like(scales, dtype=np.float32),
                group_size=group_size,
                mode=quantizer_mode,
                linear_bias=linear_bias,
            )
            validate_ternary_weights(record)
            records[prefix] = record
    next_block = max((int(path.split(".")[2]) for path in records), default=-1) + 1
    tq.save_records_checkpoint(
        args.output,
        records,
        next_block,
        group_size,
        int(manifest["model"]["crop_len_at_train"]),
        "symmetric" if quantizer_mode == "symmetric" else "affine_centered",
    )
    print({"records": len(records), "next_block": next_block, "output": str(args.output)})


if __name__ == "__main__":
    main()
