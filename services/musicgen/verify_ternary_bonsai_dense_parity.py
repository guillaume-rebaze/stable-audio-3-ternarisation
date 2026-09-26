#!/usr/bin/env python3
"""Run the P0 dense identity canary in two fresh processes.

The check deliberately uses the production DiT, conditioning path, timestep
promotion, latent axis convention, and one real latent.  It does not load any
ternary artifact and therefore cannot hide a model/runtime mismatch behind a
quantizer result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
RUNTIME_SCRIPTS = RUNTIME_ROOT / "scripts"


def _runtime_sys_path() -> None:
    sys.path = [str(RUNTIME_ROOT), str(RUNTIME_SCRIPTS), str(THIS_DIR)] + [
        path for path in sys.path if "musicgen" not in path and "abelton" not in path
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(args: argparse.Namespace) -> int:
    _runtime_sys_path()
    import mlx.core as mx

    from models.defs import dit_mlx_medium
    from train_ternary_quality import cache_conditioning, load_samples, noised_latent
    from ternary_runtime_contract import timestep_tensor

    teacher_path = args.teacher_weights.expanduser().resolve()
    samples = load_samples(args.dataset_dir, max_samples=0)
    if not samples:
        raise RuntimeError("dense parity has no samples")
    sample = samples[args.sample_index % len(samples)]
    model = dit_mlx_medium.DiT(T_lat=args.crop_len)
    model.load_weights(str(teacher_path), strict=False)
    model.freeze()
    context_cache, cross_cache, global_cond, _global_pre = cache_conditioning(
        model,
        teacher_path,
        [sample["prompt"]],
        args.seconds,
    )
    x = noised_latent(sample, args.crop_len, args.sigma, args.seed)
    timestep = timestep_tensor(args.sigma)
    output = model(x, timestep, cross_cache[sample["prompt"]], global_cond)
    mx.eval(x, output)
    result = np.asarray(output, dtype=np.float32)
    np.save(args.output, result)
    report = {
        "pid": os.getpid(),
        "sample_path": sample["path"],
        "prompt": sample["prompt"],
        "input_shape": list(x.shape),
        "output_shape": list(result.shape),
        "timestep_dtype": str(timestep.dtype),
        "output_dtype": str(result.dtype),
        "output_sha256": _sha256(args.output),
        "output_l2": float(np.linalg.norm(result)),
        "output_finite": bool(np.isfinite(result).all()),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.output is None:
            raise ValueError("--worker requires --output")
        return _worker(args)

    with tempfile.TemporaryDirectory(prefix="ternary-bonsai-dense-") as temp_dir:
        temp = Path(temp_dir)
        worker_outputs: list[Path] = []
        worker_reports: list[dict] = []
        for index in range(2):
            output = temp / f"output-{index}.npy"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--teacher-weights",
                str(args.teacher_weights),
                "--dataset-dir",
                str(args.dataset_dir),
                "--crop-len",
                str(args.crop_len),
                "--seconds",
                str(args.seconds),
                "--sigma",
                str(args.sigma),
                "--seed",
                str(args.seed),
                "--sample-index",
                str(args.sample_index),
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            worker_reports.append(json.loads(lines[-1]))
            worker_outputs.append(output)

        first = np.load(worker_outputs[0])
        second = np.load(worker_outputs[1])
        if first.shape != second.shape:
            raise RuntimeError(f"dense process shapes differ: {first.shape} != {second.shape}")
        delta = first.astype(np.float64) - second.astype(np.float64)
        relative_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(first), 1e-12))
        cosine = float(
            np.sum(first.astype(np.float64) * second.astype(np.float64))
            / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)
        )
        result = {
            "schema": "onus.ternary-quality/v9-dense-parity",
            "process_count": 2,
            "relative_l2": relative_l2,
            "cosine": cosine,
            "shape": list(first.shape),
            "finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
            "worker_reports": worker_reports,
            "thresholds": {"relative_l2_max": 1e-3, "cosine_min": 0.99999},
            "pass": bool(relative_l2 <= 1e-3 and cosine >= 0.99999),
        }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit("dense parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

