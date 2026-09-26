"""Precompute frozen teacher velocities so QAT runs need not keep a teacher loaded."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np

import train_ternary_window_v6 as window_v6
import train_ternary_quality as tq
from ternary_teacher_targets import save_teacher_targets
from ternary_runtime_contract import (
    build_teacher_target_contract,
    timestep_tensor,
)

mx = tq.mx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-metal-bytes",
        type=int,
        default=11_000_000_000,
        help="Stop before writing the cache if the observed Metal peak exceeds this guard.",
    )
    args = parser.parse_args()
    started = time.time()
    states, sigmas, prompt_indices, _sources, cross_cache, global_cond, cache_manifest = (
        window_v6.load_state_cache(args.state_cache)
    )
    cache_config = cache_manifest["cache"]
    target_contract = build_teacher_target_contract(
        tq.MLX_RUNTIME_ROOT,
        Path(__file__).resolve().parent,
        int(cache_config["crop_len"]),
        float(cache_config["seconds"]),
        int(cache_config["trajectory_steps"]),
    )
    teacher = tq.dit_mlx_medium.DiT(
        T_lat=int(cache_manifest["cache"].get("crop_len", 128))
    )
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()

    outputs: list[np.ndarray] = []
    for index in range(len(states)):
        x = mx.array(states[index][None], dtype=mx.float16)
        t = timestep_tensor(float(sigmas[index]))
        cross = cross_cache[int(prompt_indices[index])]
        target = teacher(x, t, cross, global_cond)
        mx.eval(target)
        outputs.append(np.asarray(target[0], dtype=np.float16))
        peak_bytes = int(tq.memory_snapshot()["metal_peak_gb"] * (1024**3))
        if peak_bytes > args.max_metal_bytes:
            raise RuntimeError(
                f"teacher target precompute exceeded memory guard: "
                f"{peak_bytes} > {args.max_metal_bytes} bytes"
            )
        if index == 0 or (index + 1) % max(1, len(states) // 10) == 0:
            print(
                f"[TeacherTargets] {index + 1}/{len(states)} "
                f"shape={outputs[-1].shape} memory={tq.memory_snapshot()}",
                flush=True,
            )

    targets = np.stack(outputs, axis=0).astype(np.float16, copy=False)
    metadata = save_teacher_targets(
        args.output, targets, args.state_cache, args.teacher_weights,
        target_contract,
    )
    del teacher, outputs, targets
    gc.collect()
    mx.clear_cache()
    print(
        {
            "status": "cached",
            "output": str(args.output),
            "shape": metadata["target_shape"],
            "dtype": metadata["target_dtype"],
            "target_contract": metadata["inputs"]["target_contract"],
            "elapsed_seconds": time.time() - started,
            "memory": tq.memory_snapshot(),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
