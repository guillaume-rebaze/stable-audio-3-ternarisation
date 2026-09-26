"""Prepare matched two-step teacher targets from a ternary student's train rollouts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import train_ternary_window_v6 as window_v6
import train_ternary_quality as tq
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_runtime_contract import pingpong_trace, pingpong_transition, timestep_tensor
from ternary_teacher_targets import file_identity

mx = tq.mx


def digest_json(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def check_memory(max_bytes: int, where: str) -> dict[str, float]:
    memory = tq.memory_snapshot()
    peak_bytes = int(memory["metal_peak_gb"] * (1024**3))
    if peak_bytes > max_bytes:
        raise RuntimeError(
            f"{where} exceeded Metal guard: {peak_bytes} > {max_bytes} bytes"
        )
    return memory


def student_trace_pairs(
    records_path: Path,
    teacher_weights: Path,
    prompts: list[str],
    cross_cache: list[mx.array],
    global_cond: mx.array,
    crop_len: int,
    steps: int,
    seed: int,
    max_metal_bytes: int,
) -> dict[str, np.ndarray]:
    records, records_metadata = tq.load_records_checkpoint(records_path)
    group_size = int(records_metadata["group_size"])
    student = tq.dit_mlx_medium.DiT(T_lat=crop_len)
    student.load_weights(str(teacher_weights), strict=False)
    student = tq.apply_records_to_model(student, records, group_size)
    student.freeze()
    mx.eval(student.parameters())

    schedule = build_pingpong_schedule(
        steps, sigma_max=1.0, use_logsnr_shift=True
    )
    sigma_values = np.asarray(schedule, dtype=np.float32)
    anchors: list[np.ndarray] = []
    noise_first: list[np.ndarray] = []
    noise_second: list[np.ndarray] = []
    sigma_triplets: list[np.ndarray] = []
    prompt_indices: list[int] = []
    pair_steps: list[int] = []
    generation_seeds: list[int] = []
    second_noise_present: list[bool] = []

    for prompt_index, _prompt in enumerate(prompts):
        for repeat in range(2):
            generation_seed = seed + prompt_index * 2 + repeat
            initial = mx.random.normal(
                (1, 256, crop_len),
                dtype=mx.float16,
                key=mx.random.key(generation_seed),
            )
            trace = pingpong_trace(
                lambda x, t: student(
                    x, t, cross_cache[prompt_index], global_cond
                ),
                initial,
                schedule,
                sampler_seed=generation_seed + 1,
            )
            if len(trace) != steps + 1:
                raise RuntimeError(
                    f"student trace length {len(trace)} != expected {steps + 1}"
                )
            for pair_start in range(steps - 1):
                first_record = trace[pair_start]
                second_record = trace[pair_start + 1]
                first_noise = first_record["noise"]
                if first_noise is None:
                    raise RuntimeError("non-terminal pair is missing its first re-noise")
                next_noise = second_record["noise"]
                anchors.append(
                    np.asarray(first_record["state"], dtype=np.float16)[0].copy()
                )
                noise_first.append(np.asarray(first_noise, dtype=np.float16)[0].copy())
                noise_second.append(
                    np.zeros((256, crop_len), dtype=np.float16)
                    if next_noise is None
                    else np.asarray(next_noise, dtype=np.float16)[0].copy()
                )
                sigma_triplets.append(sigma_values[pair_start : pair_start + 3].copy())
                prompt_indices.append(prompt_index)
                pair_steps.append(pair_start)
                generation_seeds.append(generation_seed)
                second_noise_present.append(next_noise is not None)
            check_memory(max_metal_bytes, "student rollout capture")
        print(
            f"[TrajectoryPairs] student prompts={prompt_index + 1}/{len(prompts)} "
            f"pairs={len(anchors)} memory={tq.memory_snapshot()}",
            flush=True,
        )

    del student, records, initial, trace
    gc.collect()
    mx.clear_cache()
    return {
        "anchors": np.stack(anchors).astype(np.float16, copy=False),
        "noise_first": np.stack(noise_first).astype(np.float16, copy=False),
        "noise_second": np.stack(noise_second).astype(np.float16, copy=False),
        "sigmas": np.stack(sigma_triplets).astype(np.float32, copy=False),
        "prompt_indices": np.asarray(prompt_indices, dtype=np.int32),
        "pair_steps": np.asarray(pair_steps, dtype=np.int8),
        "generation_seeds": np.asarray(generation_seeds, dtype=np.int64),
        "second_noise_present": np.asarray(second_noise_present, dtype=np.bool_),
    }


def teacher_branch_targets(
    pairs: dict[str, np.ndarray],
    teacher_weights: Path,
    cross_cache: list[mx.array],
    global_cond: mx.array,
    crop_len: int,
    steps: int,
    max_metal_bytes: int,
) -> dict[str, np.ndarray]:
    teacher = tq.dit_mlx_medium.DiT(T_lat=crop_len)
    teacher.load_weights(str(teacher_weights), strict=False)
    teacher.freeze()
    mx.eval(teacher.parameters())

    velocities: list[np.ndarray] = []
    state_one: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    count = len(pairs["pair_steps"])
    for index in range(count):
        anchor = mx.array(pairs["anchors"][index][None], dtype=mx.float16)
        noise_first = mx.array(pairs["noise_first"][index][None], dtype=mx.float16)
        noise_second = mx.array(pairs["noise_second"][index][None], dtype=mx.float16)
        sigma0, sigma1, sigma2 = (float(value) for value in pairs["sigmas"][index])
        pair_start = int(pairs["pair_steps"][index])
        cross = cross_cache[int(pairs["prompt_indices"][index])]

        velocity0 = teacher(
            anchor, timestep_tensor(sigma0, anchor.shape[0]), cross, global_cond
        )
        mx.eval(velocity0)
        first_state = pingpong_transition(
            anchor,
            velocity0,
            sigma0,
            sigma1,
            noise_first,
            pair_start,
            steps,
        )
        velocity1 = teacher(
            first_state,
            timestep_tensor(sigma1, first_state.shape[0]),
            cross,
            global_cond,
        )
        mx.eval(velocity1)
        if bool(pairs["second_noise_present"][index]):
            second_noise_arg = noise_second
        else:
            second_noise_arg = None
        endpoint = pingpong_transition(
            first_state,
            velocity1,
            sigma1,
            sigma2,
            second_noise_arg,
            pair_start + 1,
            steps,
        )
        mx.eval(endpoint)
        velocities.append(np.asarray(velocity0[0], dtype=np.float16).copy())
        state_one.append(np.asarray(first_state[0], dtype=np.float16).copy())
        endpoints.append(np.asarray(endpoint[0], dtype=np.float16).copy())
        check_memory(max_metal_bytes, "teacher pair-target preparation")
        if index == 0 or (index + 1) % 28 == 0 or index + 1 == count:
            print(
                f"[TrajectoryPairs] teacher targets={index + 1}/{count} "
                f"memory={tq.memory_snapshot()}",
                flush=True,
            )

    del teacher, anchor, noise_first, noise_second, velocity0, velocity1
    gc.collect()
    mx.clear_cache()
    return {
        **pairs,
        "target_velocity": np.stack(velocities).astype(np.float16, copy=False),
        "target_state_one": np.stack(state_one).astype(np.float16, copy=False),
        "target_endpoint": np.stack(endpoints).astype(np.float16, copy=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".json")
    temporary_path = args.output.with_name(f".{args.output.stem}.tmp.npz")
    if any(path.exists() for path in (args.output, metadata_path, temporary_path)):
        raise FileExistsError(f"refusing to overwrite trajectory-pair cache: {args.output}")
    if not args.source_records.is_file() or not args.teacher_weights.is_file():
        raise FileNotFoundError("source records or teacher weights do not exist")

    started = time.time()
    (
        _states,
        _sigmas,
        _prompt_indices,
        _sources,
        cross_cache,
        global_cond,
        cache_manifest,
    ) = window_v6.load_state_cache(args.state_cache)
    cache_config = cache_manifest["cache"]
    prompts = list(cache_config["prompt_index"])
    steps = int(cache_config["trajectory_steps"])
    crop_len = int(cache_config["crop_len"])
    if len(prompts) != 16 or steps != 8:
        raise ValueError(
            f"P3 requires 16 train prompts x 8 steps, got {len(prompts)} x {steps}"
        )
    pairs = student_trace_pairs(
        args.source_records,
        args.teacher_weights,
        prompts,
        cross_cache,
        global_cond,
        crop_len,
        steps,
        args.seed,
        args.max_metal_bytes,
    )
    pairs = teacher_branch_targets(
        pairs,
        args.teacher_weights,
        cross_cache,
        global_cond,
        crop_len,
        steps,
        args.max_metal_bytes,
    )
    expected_count = len(prompts) * 2 * (steps - 1)
    if len(pairs["pair_steps"]) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} trajectory pairs, got {len(pairs['pair_steps'])}"
        )
    for name, array in pairs.items():
        if not np.isfinite(array).all():
            raise ValueError(f"trajectory-pair array {name} contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary_path, **pairs)
    os.replace(temporary_path, args.output)
    inputs = {
        "state_cache_manifest": file_identity(args.state_cache / "manifest.json"),
        "states": file_identity(args.state_cache / "states.npz"),
        "conditions": file_identity(args.state_cache / "conditions.npz"),
        "source_records": file_identity(args.source_records),
        "teacher_weights": file_identity(args.teacher_weights),
    }
    cache_file = file_identity(args.output)
    contract = {
        "schema": "onus.ternary-quality/v7-trajectory-pairs",
        "inputs": inputs,
        "sampler": {
            "steps": steps,
            "sigma_grid": [
                float(value)
                for value in build_pingpong_schedule(
                    steps, sigma_max=1.0, use_logsnr_shift=True
                )
            ],
            "generation_seed_formula": "base_seed + prompt_index * 2 + repeat",
            "initial_noise_offset": 0,
            "sampler_reinjection_seed_offset": 1,
            "rng": "sequential_mx.random.split",
            "transition": "ARC sample_flow_pingpong",
            "pair_rule": "all seven adjacent step pairs, including first and terminal-reaching pair",
        },
        "sampling": {
            "prompt_count": len(prompts),
            "seeds_per_prompt": 2,
            "pair_count": expected_count,
            "pair_step_counts": {
                str(step): int(np.sum(pairs["pair_steps"] == step))
                for step in range(steps - 1)
            },
        },
        "runtime_contract": cache_manifest["runtime_contract"],
    }
    manifest = {
        **contract,
        "status": "prepared",
        "created_at_unix": time.time(),
        "cache_file": cache_file["path"],
        "cache_bytes": cache_file["bytes"],
        "cache_sha256": cache_file["sha256"],
        "arrays_sha256": cache_file["sha256"],
        "contract_digest": digest_json(contract),
        "pair_count": expected_count,
        "prompts": prompts,
        "array_shapes": {name: list(array.shape) for name, array in pairs.items()},
        "array_dtypes": {name: str(array.dtype) for name, array in pairs.items()},
        "elapsed_seconds": time.time() - started,
        "memory": tq.memory_snapshot(),
    }
    write_json_atomic(metadata_path, manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "pairs": expected_count,
                "pair_step_counts": contract["sampling"]["pair_step_counts"],
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "elapsed_seconds": manifest["elapsed_seconds"],
                "memory": manifest["memory"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
