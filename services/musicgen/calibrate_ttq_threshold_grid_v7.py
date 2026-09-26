"""Build block-0 TTQ records for an activation-agnostic threshold grid.

This is a cheap P1 calibration lane.  It deliberately does not train masters:
it recomputes the signed TTQ levels from the dense teacher for several group
threshold multipliers, so behavioral auditing can select a starting point
before expensive QAT.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

import train_ternary_quality as tq


CORE_NAMES = tuple(tq.CORE_NAMES)


def ttq_record(weight: np.ndarray, bias: np.ndarray | None, group_size: int, factor: float):
    values = np.asarray(weight, dtype=np.float32)
    out_dim, in_dim = values.shape
    groups = values.reshape(out_dim, in_dim // group_size, group_size)
    means = np.mean(groups, axis=-1)
    centered = groups - means[..., None]
    base = np.maximum(np.mean(np.abs(centered), axis=-1), 1e-6)
    assignment = base * float(factor)
    normalized = centered / assignment[..., None]
    q = np.clip(np.rint(normalized), -1, 1).astype(np.int8)

    positive_count = np.sum(q > 0, axis=-1)
    negative_count = np.sum(q < 0, axis=-1)
    positive_sum = np.sum(np.where(q > 0, centered, 0.0), axis=-1)
    negative_sum = np.sum(np.where(q < 0, centered, 0.0), axis=-1)
    positive = np.where(
        positive_count > 0,
        positive_sum / np.maximum(positive_count, 1),
        base,
    )
    negative = np.where(
        negative_count > 0,
        -negative_sum / np.maximum(negative_count, 1),
        base,
    )
    positive = np.maximum(positive, 1e-6).astype(np.float32)
    negative = np.maximum(negative, 1e-6).astype(np.float32)
    record = tq.quantize_ttq_with_assignment_and_scales(
        values,
        assignment,
        positive,
        negative,
        means,
        group_size=group_size,
        mode="ttq",
    )
    if bias is not None:
        record = replace(record, linear_bias=np.asarray(bias, dtype=np.float16))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30],
    )
    args = parser.parse_args()
    if args.group_size % 16:
        raise ValueError("group-size must be divisible by 16")
    if any(not np.isfinite(value) or value <= 0 for value in args.factors):
        raise ValueError("factors must be finite and positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    with np.load(args.teacher_weights, allow_pickle=False) as archive:
        dense: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        for name in CORE_NAMES:
            prefix = f"transformer.layers.0.{name}"
            weight = np.asarray(archive[f"{prefix}.weight"], dtype=np.float32)
            bias_key = f"{prefix}.bias"
            bias = (
                np.asarray(archive[bias_key], dtype=np.float32)
                if bias_key in archive.files
                else None
            )
            dense[prefix] = (weight, bias)

        for factor in args.factors:
            records = {}
            errors: list[float] = []
            code_histogram = {-1: 0, 0: 0, 1: 0}
            for prefix, (weight, bias) in dense.items():
                record = ttq_record(weight, bias, args.group_size, factor)
                records[prefix] = record
                reconstructed = tq.dequantize_record(record)
                errors.append(tq.relative_error(weight, reconstructed))
                for value in (-1, 0, 1):
                    code_histogram[value] += int(np.sum(record.q == value))

            label = f"factor-{factor:.4f}".replace(".", "p")
            run_dir = args.output_dir / label
            checkpoint = run_dir / "records_checkpoint.npz"
            tq.save_records_checkpoint(
                checkpoint,
                records,
                next_block=1,
                group_size=args.group_size,
                crop_len=128,
                quantizer_mode="ttq",
            )
            reports.append(
                {
                    "factor": float(factor),
                    "records": str(checkpoint),
                    "mean_relative_error": float(np.mean(errors)),
                    "max_relative_error": float(np.max(errors)),
                    "code_histogram": {str(key): value for key, value in code_histogram.items()},
                    "code_count": int(sum(code_histogram.values())),
                }
            )

    payload = {
        "schema": "onus.ternary-quality/v7-ttq-threshold-grid",
        "teacher_weights": str(args.teacher_weights.resolve()),
        "group_size": int(args.group_size),
        "scope": [f"transformer.layers.0.{name}" for name in CORE_NAMES],
        "reports": reports,
    }
    (args.output_dir / "grid_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
