#!/usr/bin/env python3
"""Exercise production dense cache serialization across 16 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
RUNTIME_SCRIPTS = RUNTIME_ROOT / "scripts"


def _runtime_sys_path() -> None:
    import sys

    sys.path = [str(RUNTIME_ROOT), str(RUNTIME_SCRIPTS), str(THIS_DIR)] + [
        path for path in sys.path if "musicgen" not in path and "abelton" not in path
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


def main() -> int:
    _runtime_sys_path()
    import mlx.core as mx

    from models.defs import dit_mlx_medium
    from train_ternary_quality import cache_conditioning, load_samples, noised_latent
    from ternary_runtime_contract import timestep_tensor

    args = parse_args()
    samples = load_samples(args.dataset_dir, max_samples=0)
    sigmas = (0.995, 0.90, 0.75, 0.60, 0.50, 0.35, 0.20, 0.10)
    lengths = (64, 128)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ternary-bonsai-cache-") as temp_dir:
        temp = Path(temp_dir)
        for length in lengths:
            model = dit_mlx_medium.DiT(T_lat=length)
            model.load_weights(str(args.teacher_weights), strict=False)
            model.freeze()
            sample = samples[0]
            context_cache, cross_cache, global_cond, _global_pre = cache_conditioning(
                model, args.teacher_weights, [sample["prompt"]], args.seconds
            )
            for state_index, sigma in enumerate(sigmas):
                x = noised_latent(sample, length, sigma, args.seed + state_index + length)
                cache_path = temp / f"state-{length}-{state_index}.npz"
                np.savez(
                    cache_path,
                    x=np.asarray(x),
                    sigma=np.asarray([sigma], dtype=np.float32),
                )
                with np.load(cache_path, allow_pickle=False) as cached:
                    replay_x = mx.array(cached["x"])
                    replay_sigma = float(cached["sigma"][0])
                original = model(
                    x,
                    timestep_tensor(sigma),
                    cross_cache[sample["prompt"]],
                    global_cond,
                )
                replay = model(
                    replay_x,
                    timestep_tensor(replay_sigma),
                    cross_cache[sample["prompt"]],
                    global_cond,
                )
                mx.eval(original, replay)
                left = np.asarray(original, dtype=np.float32)
                right = np.asarray(replay, dtype=np.float32)
                delta = left.astype(np.float64) - right.astype(np.float64)
                rel = float(np.linalg.norm(delta) / max(np.linalg.norm(left), 1e-12))
                cosine = float(
                    np.sum(left.astype(np.float64) * right.astype(np.float64))
                    / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)
                )
                rows.append(
                    {
                        "length": length,
                        "sigma": sigma,
                        "input_shape": list(x.shape),
                        "output_shape": list(left.shape),
                        "input_dtype": str(x.dtype),
                        "replay_input_dtype": str(replay_x.dtype),
                        "timestep_dtype": str(timestep_tensor(sigma).dtype),
                        "relative_l2": rel,
                        "cosine": cosine,
                        "cache_sha256": _sha256(cache_path),
                        "pass": bool(rel <= 1e-3 and cosine >= 0.99999),
                    }
                )
            del model
            mx.clear_cache()

    result = {
        "schema": "onus.ternary-quality/v9-cache-replay",
        "state_count": len(rows),
        "lengths": list(lengths),
        "sigmas": list(sigmas),
        "rows": rows,
        "pass": bool(len(rows) == 16 and all(row["pass"] for row in rows)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"state_count": len(rows), "pass": result["pass"]}, sort_keys=True))
    if not result["pass"]:
        raise SystemExit("cache replay gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

