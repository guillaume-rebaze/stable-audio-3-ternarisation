"""Verify trained hard forwards survive record or artifact reload in a fresh process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_ternary_quality as tq
from audit_ternary_quality import load_manifest, load_student

mx = tq.mx


def compare_forward(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    left = np.asarray(actual, dtype=np.float32).ravel()
    right = np.asarray(expected, dtype=np.float32).ravel()
    delta = left - right
    return {
        "relative_l2": float(
            np.linalg.norm(delta) / max(float(np.linalg.norm(right)), 1e-12)
        ),
        "cosine": float(np.dot(left, right) / (
            np.linalg.norm(left) * np.linalg.norm(right) + 1e-12
        )),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite roundtrip report: {args.output}")

    records, record_metadata = tq.load_records_checkpoint(args.records)
    if int(record_metadata["group_size"]) != args.group_size:
        raise ValueError("record checkpoint group size does not match verifier")
    if args.artifact is not None:
        manifest_path = args.manifest or args.artifact.with_suffix(".json")
        manifest = load_manifest(manifest_path)
        if manifest["scope"]["digest"] != record_metadata["scope_digest"]:
            raise ValueError("artifact and record checkpoint scopes differ")
        model = load_student(args.artifact, manifest, args.crop_len)
        with np.load(args.artifact, allow_pickle=False) as arrays:
            for prefix, record in records.items():
                if not np.array_equal(arrays[f"{prefix}.weight"], record.packed_codes):
                    raise ValueError(f"packed codes changed during export: {prefix}")
                if not np.array_equal(arrays[f"{prefix}.scales"], record.scales):
                    raise ValueError(f"scales changed during export: {prefix}")
                if f"{prefix}.biases" in arrays.files and not np.array_equal(
                    arrays[f"{prefix}.biases"], record.biases
                ):
                    raise ValueError(f"affine metadata changed during export: {prefix}")
                if record.linear_bias is not None and not np.array_equal(
                    arrays[f"{prefix}.bias"], record.linear_bias
                ):
                    raise ValueError(f"trained linear bias changed during export: {prefix}")
        reload_source = "exported_artifact"
    else:
        model = tq.dit_mlx_medium.DiT(T_lat=args.crop_len)
        model.load_weights(str(args.teacher_weights), strict=False)
        model = tq.apply_records_to_model(model, records, args.group_size)
        reload_source = "records_checkpoint"
    memory = tq.memory_snapshot()
    if int(memory["metal_peak_gb"] * (1024**3)) > args.max_metal_bytes:
        raise RuntimeError(f"roundtrip load exceeded Metal memory guard: {memory}")

    forward_results = []
    with np.load(args.fixture, allow_pickle=False) as fixture:
        sample_count = int(fixture["sample_count"])
        for index in range(sample_count):
            actual = model(
                mx.array(fixture[f"x_{index}"], dtype=mx.float16),
                mx.array(fixture[f"t_{index}"], dtype=mx.float32),
                mx.array(fixture[f"cross_{index}"]),
                mx.array(fixture["global_cond"]),
            )
            mx.eval(actual)
            metrics = compare_forward(
                np.asarray(actual), fixture[f"expected_{index}"]
            )
            forward_results.append({"sample": index, **metrics})
            memory = tq.memory_snapshot()
            if int(memory["metal_peak_gb"] * (1024**3)) > args.max_metal_bytes:
                raise RuntimeError(
                    f"roundtrip forward exceeded Metal memory guard: {memory}"
                )

    passed = all(
        item["relative_l2"] <= 1e-3 and item["cosine"] >= 0.99999
        for item in forward_results
    )
    report = {
        "schema": "onus.ternary-quality/v7-cross-process-roundtrip",
        "status": "pass" if passed else "fail",
        "reload_source": reload_source,
        "artifact": str(args.artifact) if args.artifact else None,
        "records": str(args.records),
        "scope_count": len(records),
        "scope_digest": record_metadata["scope_digest"],
        "linear_bias_count": record_metadata.get("linear_bias_count"),
        "forward_tolerances": {
            "relative_l2_max": 1e-3,
            "cosine_min": 0.99999,
        },
        "forwards": forward_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
