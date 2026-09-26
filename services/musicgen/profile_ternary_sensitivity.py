"""Profile strict symmetric ternary error over the complete DiT core scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ternary_contract import dequantize, quantize_symmetric_weight, relative_error


CORE_NAMES = (
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "ff.ff.0.proj",
    "ff.ff.2",
)


def is_core_weight(key: str) -> bool:
    return key.startswith("transformer.layers.") and any(
        key.endswith(f".{name}.weight") for name in CORE_NAMES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ternary core sensitivity")
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path.home() / ".cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-v3/g3-sensitivity.json"),
    )
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()

    started = time.time()
    records: list[dict] = []
    with np.load(args.teacher_weights, allow_pickle=False) as archive:
        keys = sorted(key for key in archive.files if is_core_weight(key))
        for key in keys:
            weight = np.asarray(archive[key], dtype=np.float32)
            layer = int(key.split(".")[2])
            name = key.split(f"transformer.layers.{layer}.", 1)[1].rsplit(".weight", 1)[0]
            item = {
                "path": key,
                "layer": layer,
                "name": name,
                "shape": list(weight.shape),
                "groups": {},
            }
            for group_size in args.group_sizes:
                quantized = quantize_symmetric_weight(weight, group_size=group_size)
                reconstructed = dequantize(
                    quantized.q, quantized.scales, quantized.biases
                )
                item["groups"][str(group_size)] = {
                    "relative_error": relative_error(weight, reconstructed),
                    "zero_fraction": float(np.mean(quantized.q == 0)),
                    "positive_fraction": float(np.mean(quantized.q == 1)),
                    "negative_fraction": float(np.mean(quantized.q == -1)),
                    "scale_mean": float(np.mean(-quantized.scales.astype(np.float32))),
                    "scale_min": float(np.min(-quantized.scales.astype(np.float32))),
                    "scale_max": float(np.max(-quantized.scales.astype(np.float32))),
                }
            records.append(item)

    summaries = {}
    for group_size in args.group_sizes:
        values = [r["groups"][str(group_size)]["relative_error"] for r in records]
        summaries[str(group_size)] = {
            "matrix_count": len(values),
            "mean_relative_error": float(np.mean(values)),
            "median_relative_error": float(np.median(values)),
            "max_relative_error": float(np.max(values)),
            "p95_relative_error": float(np.quantile(values, 0.95)),
            "mean_zero_fraction": float(
                np.mean([r["groups"][str(group_size)]["zero_fraction"] for r in records])
            ),
        }

    by_layer: dict[str, dict] = {}
    for layer in sorted({r["layer"] for r in records}):
        layer_records = [r for r in records if r["layer"] == layer]
        by_layer[str(layer)] = {
            str(group_size): {
                "mean_relative_error": float(
                    np.mean(
                        [
                            r["groups"][str(group_size)]["relative_error"]
                            for r in layer_records
                        ]
                    )
                ),
                "max_relative_error": float(
                    np.max(
                        [
                            r["groups"][str(group_size)]["relative_error"]
                            for r in layer_records
                        ]
                    )
                ),
            }
            for group_size in args.group_sizes
        }

    payload = {
        "schema": "onus.ternary-quality-v3/g3-sensitivity",
        "teacher_weights": str(args.teacher_weights),
        "scope": "7 core projections per transformer block",
        "records": records,
        "by_group_size": summaries,
        "by_layer": by_layer,
        "selection_rule": "Prefer the largest group that passes behavioral gates; weight error alone cannot release a model.",
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "profiled", "groups": summaries}, indent=2))


if __name__ == "__main__":
    main()
