#!/usr/bin/env python3
"""Fresh-process structural and forward verification for a Bonsai package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
RUNTIME_SCRIPTS = RUNTIME_ROOT / "scripts"
THIS_DIR = Path(__file__).resolve().parent


def runtime_sys_path() -> None:
    sys.path = [str(RUNTIME_ROOT), str(RUNTIME_SCRIPTS), str(THIS_DIR)] + [
        item for item in sys.path if "musicgen" not in item and "abelton" not in item
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    left32 = np.asarray(left, dtype=np.float32)
    right32 = np.asarray(right, dtype=np.float32)
    return float(np.linalg.norm(left32 - right32) / max(np.linalg.norm(left32), 1e-8))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64).ravel()
    right64 = np.asarray(right, dtype=np.float64).ravel()
    return float(np.dot(left64, right64) / max(np.linalg.norm(left64) * np.linalg.norm(right64), 1e-12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_sys_path()
    import mlx.core as mx

    from models.defs import dit_mlx_medium
    import train_ternary_quality as tq
    from ternary_bonsai_contract import unpack_bonsai_codes
    from ternary_contract import TernaryWeights, validate_ternary_weights
    from ternary_runtime_contract import timestep_tensor

    package_dir = args.package_dir.expanduser().resolve()
    manifest_path = package_dir / "manifest.json"
    payload = package_dir / "payload.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "onus.ternary-quality/v10-bonsai-package":
        raise ValueError(f"unexpected package schema: {manifest.get('schema')!r}")
    expected_hash = manifest["storage"]["payload_sha256"]
    actual_hash = sha256_file(payload)
    if actual_hash != expected_hash:
        raise ValueError("payload sha256 mismatch")
    if payload.stat().st_size > 650_000_000:
        raise ValueError("payload exceeds the Bonsai package envelope")

    group_size = int(manifest["model"]["group_size"])
    crop_len = int(manifest["model"]["crop_len"])
    core_entries = manifest["scope"]["core"]
    support_names = set(manifest["scope"]["support_names"])
    records: dict[str, TernaryWeights] = {}
    code_count = 0
    with np.load(payload, allow_pickle=False) as arrays:
        members = set(arrays.files)
        missing_support = sorted(name for name in support_names if name not in members)
        if missing_support:
            raise ValueError(f"missing support members: {missing_support[:4]}")
        for entry in core_entries:
            packed = np.asarray(arrays[entry["packed_member"]], dtype=np.uint32)
            scales_positive = np.asarray(arrays[entry["scale_member"]], dtype=np.float16)
            out_dim, in_dim = (int(value) for value in entry["shape"])
            q = unpack_bonsai_codes(
                packed,
                out_dim=out_dim,
                group_count=int(entry["group_count"]),
                group_size=group_size,
            )
            if q.shape != (out_dim, in_dim // group_size, group_size):
                raise ValueError(f"decoded q shape mismatch for {entry['name']}: {q.shape}")
            if not np.isfinite(scales_positive).all() or np.any(scales_positive < 0):
                raise ValueError(f"invalid positive scale in {entry['name']}")
            code_count += int(q.size)
            linear_bias = None
            bias_member = entry.get("linear_bias_member")
            if bias_member is not None:
                linear_bias = np.asarray(arrays[bias_member], dtype=np.float16)
            record = TernaryWeights(
                packed_codes=packed,
                scales=-scales_positive,
                biases=scales_positive,
                q=q,
                group_means=np.zeros(scales_positive.shape, dtype=np.float32),
                group_size=group_size,
                mode=str(entry["mode"]),
                linear_bias=linear_bias,
            )
            validate_ternary_weights(record)
            records[str(entry["prefix"])] = record

    expected_core = int(manifest["scope"]["core_count"])
    if len(records) != expected_core:
        raise ValueError(f"core count mismatch: {len(records)} != {expected_core}")
    if code_count == 0 or not all(np.all(np.isin(record.q, (-1, 0, 1))) for record in records.values()):
        raise ValueError("package is not strictly ternary")

    # Load only native support tensors from the package.  The custom core
    # members are deliberately not model parameter names and are ignored by
    # MLX's non-strict loader; they are installed below by the runtime adapter.
    package_model = dit_mlx_medium.DiT(T_lat=crop_len)
    package_model.load_weights(str(payload), strict=False)
    package_model = tq.apply_records_to_model(package_model, records, group_size)

    # Independent reference: dense teacher + the exact package records.  This
    # catches missing support tensors, wrong bias precedence, and a direct vs
    # Hadamard runtime mismatch in one fresh process.
    reference_model = dit_mlx_medium.DiT(T_lat=crop_len)
    reference_model.load_weights(str(args.teacher_weights.expanduser().resolve()), strict=False)
    reference_model = tq.apply_records_to_model(reference_model, records, group_size)
    parameter_report = tq.parameter_reload_report(reference_model, package_model)
    if not parameter_report["exact"]:
        raise RuntimeError(f"package parameter reload mismatch: {parameter_report}")

    samples = tq.load_samples(args.dataset_dir, max_samples=0)
    sample = samples[args.sample_index % len(samples)]
    _, cross_cache, global_cond, _ = tq.cache_conditioning(
        package_model,
        payload,
        [sample["prompt"]],
        12.0,
    )
    x = tq.noised_latent(sample, crop_len, args.sigma, args.seed)
    timestep = timestep_tensor(args.sigma)
    reference_output = reference_model(x, timestep, cross_cache[sample["prompt"]], global_cond)
    package_output = package_model(x, timestep, cross_cache[sample["prompt"]], global_cond)
    mx.eval(reference_output, package_output)
    reference_np = np.asarray(reference_output, dtype=np.float32)
    package_np = np.asarray(package_output, dtype=np.float32)
    forward = {
        "relative_l2": relative_error(reference_np, package_np),
        "cosine": cosine(reference_np, package_np),
        "shape": list(package_np.shape),
        "finite": bool(np.isfinite(package_np).all()),
    }
    result = {
        "schema": "onus.ternary-quality/v10-bonsai-package-verification",
        "status": "pass",
        "payload": str(payload),
        "payload_bytes": int(payload.stat().st_size),
        "payload_sha256": actual_hash,
        "core_count": len(records),
        "support_count": len(support_names),
        "ternary_code_count": code_count,
        "modes": sorted({record.mode for record in records.values()}),
        "parameter_reload": parameter_report,
        "forward": forward,
        "thresholds": {"relative_l2_max": 1e-3, "cosine_min": 0.99999},
    }
    report_path = package_dir / "reload_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if forward["relative_l2"] > 1e-3 or forward["cosine"] < 0.99999:
        raise RuntimeError("package forward parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
