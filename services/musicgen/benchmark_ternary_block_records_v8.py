"""Score reloaded ternary records on cached teacher block inputs/outputs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import train_ternary_quality as tq
from models.defs import dit_mlx_medium
from ternary_provenance_v8 import sha256_file


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float32).ravel()
    b = right.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _score_model(model, calibration: dict[str, np.ndarray]) -> dict[str, Any]:
    cosines: list[float] = []
    nmses: list[float] = []
    rms_ratios: list[float] = []
    per_sigma: dict[str, list[float]] = {}
    for index in range(len(calibration["h_in"])):
        h_in = mx.array(calibration["h_in"][index][None], dtype=mx.float16)
        context = mx.array(calibration["context"][index][None], dtype=mx.float16)
        global_cond = mx.array(calibration["global_cond"][index][None], dtype=mx.float16)
        local_padded = mx.array(calibration["local_padded"][index][None], dtype=mx.float16)
        prediction = model.transformer.layers[0](h_in, context, global_cond, local_padded)
        mx.eval(prediction)
        prediction_np = np.asarray(prediction)[0].astype(np.float32)
        target = calibration["target"][index].astype(np.float32)
        cosine = _cosine(prediction_np, target)
        error = prediction_np - target
        nmses.append(float(np.mean(error * error) / (np.mean(target * target) + 1e-8)))
        rms_ratios.append(float(np.sqrt(np.mean(prediction_np * prediction_np) + 1e-8) / (np.sqrt(np.mean(target * target) + 1e-8))))
        cosines.append(cosine)
        sigma_key = f"{float(calibration['sigmas'][index]):.8f}"
        per_sigma.setdefault(sigma_key, []).append(cosine)
    return {
        "sample_count": len(cosines),
        "block_output_mean_cosine": float(np.mean(cosines)),
        "block_output_min_cosine": float(np.min(cosines)),
        "block_output_mean_nmse": float(np.mean(nmses)),
        "block_output_mean_rms_ratio": float(np.mean(rms_ratios)),
        "per_sigma_mean_cosine": {
            sigma: float(np.mean(values)) for sigma, values in sorted(per_sigma.items(), key=lambda item: float(item[0]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration_metadata = json.loads(
        args.calibration.with_name("block0_calibration.json").read_text(encoding="utf-8")
    )
    with np.load(args.calibration, allow_pickle=False) as arrays:
        calibration = {name: np.array(arrays[name]) for name in arrays.files}
    reports: list[dict[str, Any]] = []
    group_size: int | None = None
    for record_path in args.records:
        records, metadata = tq.load_records_checkpoint(record_path)
        current_group_size = int(metadata["group_size"])
        if group_size is None:
            group_size = current_group_size
        if current_group_size != group_size:
            raise ValueError("all records must use one group size")
        model = dit_mlx_medium.DiT(T_lat=int(calibration["h_in"].shape[2] - dit_mlx_medium.NUM_MEMORY_TOKENS))
        model.load_weights(str(args.teacher_weights), strict=False)
        tq.apply_records_to_model(model, records, current_group_size)
        model.freeze()
        mx.eval(model.parameters())
        score = _score_model(model, calibration)
        score.update({
            "records": str(record_path),
            "records_sha256": sha256_file(record_path),
            "scope_count": len(records),
            "scope_digest": metadata["scope_digest"],
        })
        reports.append(score)
        print(json.dumps(score, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        mx.clear_cache()

    payload = {
        "schema": "onus.ternary-quality/v8-block-record-benchmark",
        "teacher_weights": {"path": str(args.teacher_weights), "sha256": sha256_file(args.teacher_weights)},
        "calibration": calibration_metadata,
        "reports": reports,
        "memory": tq.memory_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "audited", "reports": len(reports), "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
