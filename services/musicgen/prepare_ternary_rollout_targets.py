"""Prepare exact on-policy full-rollout targets for V7 terminal training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import train_ternary_quality as tq
import train_ternary_window_v6 as window_v6
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_runtime_contract import pingpong_trace
from ternary_teacher_targets import file_identity, sha256_file

mx = tq.mx


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_memory(max_metal_bytes: int, label: str) -> dict[str, float]:
    memory = tq.memory_snapshot()
    peak = int(memory["metal_peak_gb"] * (1024**3))
    if peak > max_metal_bytes:
        raise RuntimeError(
            f"{label} exceeded Metal guard: {peak} > {max_metal_bytes} bytes"
        )
    return memory


def collect_student_rollouts(
    records_path: Path,
    teacher_weights: Path,
    prompts: list[str],
    cross_cache: list[mx.array],
    global_cond: mx.array,
    crop_len: int,
    steps: int,
    repeat_counts: list[int],
    seed: int,
    max_metal_bytes: int,
    seed_by_prompt: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    records, metadata = tq.load_records_checkpoint(records_path)
    group_size = int(metadata["group_size"])
    student = tq.dit_mlx_medium.DiT(T_lat=crop_len)
    student.load_weights(str(teacher_weights), strict=False)
    student = tq.apply_records_to_model(student, records, group_size)
    student.freeze()
    mx.eval(student.parameters())

    initial_states: list[np.ndarray] = []
    source_states: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    prompt_indices: list[int] = []
    generation_seeds: list[int] = []
    schedule = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    for prompt_index, prompt in enumerate(prompts):
        for repeat in range(repeat_counts[prompt_index]):
            generation_seed = (
                int(seed_by_prompt[prompt]) + repeat
                if seed_by_prompt is not None and prompt in seed_by_prompt
                else seed + prompt_index * max(repeat_counts) + repeat
            )
            initial = mx.random.normal(
                (1, 256, crop_len),
                dtype=mx.float16,
                key=mx.random.key(generation_seed),
            )
            trace = pingpong_trace(
                lambda x, t: student(x, t, cross_cache[prompt_index], global_cond),
                initial,
                schedule,
                sampler_seed=generation_seed + 1,
            )
            if len(trace) != steps + 1:
                raise RuntimeError("student rollout trace has an unexpected length")
            trace_noises = [
                np.asarray(record["noise"], dtype=np.float16)[0].copy()
                for record in trace[:-1]
                if record["noise"] is not None
            ]
            if len(trace_noises) != steps - 1:
                raise RuntimeError("student rollout trace has an unexpected noise count")
            initial_states.append(np.asarray(trace[0]["state"], dtype=np.float16)[0])
            source_states.append(
                np.stack(
                    [np.asarray(record["state"], dtype=np.float16)[0] for record in trace]
                )
            )
            noises.append(np.stack(trace_noises).astype(np.float16, copy=False))
            prompt_indices.append(prompt_index)
            generation_seeds.append(generation_seed)
        print(
            f"[RolloutTargets] student prompts={prompt_index + 1}/{len(prompts)} "
            f"memory={tq.memory_snapshot()}",
            flush=True,
        )
        check_memory(max_metal_bytes, "student rollout capture")

    del student, records, initial, trace
    tq.gc.collect()
    mx.clear_cache()
    return {
        "initial_states": np.stack(initial_states).astype(np.float16, copy=False),
        "source_states": np.stack(source_states).astype(np.float16, copy=False),
        "noises": np.stack(noises).astype(np.float16, copy=False),
        "prompt_indices": np.asarray(prompt_indices, dtype=np.int32),
        "generation_seeds": np.asarray(generation_seeds, dtype=np.int64),
    }


def collect_teacher_rollouts(
    teacher_weights: Path,
    prompts: list[str],
    cross_cache: list[mx.array],
    global_cond: mx.array,
    crop_len: int,
    steps: int,
    repeat_counts: list[int],
    seed: int,
    max_metal_bytes: int,
    seed_by_prompt: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Capture teacher states as stable anchors for short-window distillation."""
    teacher = tq.dit_mlx_medium.DiT(T_lat=crop_len)
    teacher.load_weights(str(teacher_weights), strict=False)
    teacher.freeze()
    mx.eval(teacher.parameters())

    initial_states: list[np.ndarray] = []
    source_states: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    target_velocity: list[np.ndarray] = []
    target_terminal: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    prompt_indices: list[int] = []
    generation_seeds: list[int] = []
    schedule = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    for prompt_index, prompt in enumerate(prompts):
        for repeat in range(repeat_counts[prompt_index]):
            generation_seed = (
                int(seed_by_prompt[prompt]) + repeat
                if seed_by_prompt is not None and prompt in seed_by_prompt
                else seed + prompt_index * max(repeat_counts) + repeat
            )
            initial = mx.random.normal(
                (1, 256, crop_len),
                dtype=mx.float16,
                key=mx.random.key(generation_seed),
            )
            trace = pingpong_trace(
                lambda x, t: teacher(x, t, cross_cache[prompt_index], global_cond),
                initial,
                schedule,
                sampler_seed=generation_seed + 1,
            )
            if len(trace) != steps + 1:
                raise RuntimeError("teacher rollout trace has an unexpected length")
            trace_noises = [
                np.asarray(record["noise"], dtype=np.float16)[0].copy()
                for record in trace[:-1]
                if record["noise"] is not None
            ]
            if len(trace_noises) != steps - 1:
                raise RuntimeError("teacher rollout trace has an unexpected noise count")
            trace_states = np.stack(
                [np.asarray(record["state"], dtype=np.float16)[0] for record in trace]
            )
            initial_states.append(trace_states[0].copy())
            source_states.append(trace_states.copy())
            noises.append(np.stack(trace_noises).astype(np.float16, copy=False))
            target_velocity.append(
                np.asarray(trace[0]["velocity"], dtype=np.float16)[0].copy()
            )
            target_terminal.append(trace_states[-1].copy())
            target_states.append(trace_states.copy())
            prompt_indices.append(prompt_index)
            generation_seeds.append(generation_seed)
        print(
            f"[RolloutTargets] teacher anchors={prompt_index + 1}/{len(prompts)} "
            f"memory={tq.memory_snapshot()}",
            flush=True,
        )
        check_memory(max_metal_bytes, "teacher anchor preparation")

    del teacher, initial, trace
    tq.gc.collect()
    mx.clear_cache()
    return {
        "initial_states": np.stack(initial_states).astype(np.float16, copy=False),
        "source_states": np.stack(source_states).astype(np.float16, copy=False),
        "noises": np.stack(noises).astype(np.float16, copy=False),
        "target_velocity": np.stack(target_velocity).astype(np.float16, copy=False),
        "target_terminal": np.stack(target_terminal).astype(np.float16, copy=False),
        "target_states": np.stack(target_states).astype(np.float16, copy=False),
        "prompt_indices": np.asarray(prompt_indices, dtype=np.int32),
        "generation_seeds": np.asarray(generation_seeds, dtype=np.int64),
    }


def collect_teacher_endpoints(
    rollout: dict[str, np.ndarray],
    teacher_weights: Path,
    cross_cache: list[mx.array],
    global_cond: mx.array,
    prompts: list[str],
    crop_len: int,
    steps: int,
    max_metal_bytes: int,
) -> dict[str, np.ndarray]:
    teacher = tq.dit_mlx_medium.DiT(T_lat=crop_len)
    teacher.load_weights(str(teacher_weights), strict=False)
    teacher.freeze()
    mx.eval(teacher.parameters())
    sigma_schedule = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    target_velocity: list[np.ndarray] = []
    target_terminal: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    for index, prompt_index in enumerate(rollout["prompt_indices"]):
        initial = mx.array(rollout["initial_states"][index][None], dtype=mx.float16)
        generation_seed = int(rollout["generation_seeds"][index])
        trace = pingpong_trace(
            lambda x, t: teacher(x, t, cross_cache[int(prompt_index)], global_cond),
            initial,
            sigma_schedule,
            sampler_seed=generation_seed + 1,
        )
        target_velocity.append(
            np.asarray(trace[0]["velocity"], dtype=np.float16)[0].copy()
        )
        target_terminal.append(
            np.asarray(trace[-1]["state"], dtype=np.float16)[0].copy()
        )
        target_states.append(
            np.stack(
                [np.asarray(record["state"], dtype=np.float16)[0] for record in trace]
            )
        )
        check_memory(max_metal_bytes, "teacher rollout target preparation")
        if index == 0 or (index + 1) % max(1, len(rollout["prompt_indices"]) // 4) == 0:
            print(
                f"[RolloutTargets] teacher targets={index + 1}/"
                f"{len(rollout['prompt_indices'])} memory={tq.memory_snapshot()}",
                flush=True,
            )

    del teacher, initial, trace
    tq.gc.collect()
    mx.clear_cache()
    return {
        **rollout,
        "target_velocity": np.stack(target_velocity).astype(np.float16, copy=False),
        "target_terminal": np.stack(target_terminal).astype(np.float16, copy=False),
        "target_states": np.stack(target_states).astype(np.float16, copy=False),
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
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--prompt-counts",
        type=str,
        default=None,
        help="JSON list of per-prompt rollout counts; enables explicit weighted sampling",
    )
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--rollout-source-mode",
        choices=("student", "teacher"),
        default="student",
        help="use source-record student states (on-policy) or teacher states (stable anchors)",
    )
    parser.add_argument(
        "--audit-contract",
        type=Path,
        default=None,
        help="assign seed+sorted-dataset-index exactly as audit_ternary_quality does",
    )
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite rollout target cache: {args.output}")

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
    cache = cache_manifest["cache"]
    prompts = list(cache["prompt_index"])
    steps = int(cache["trajectory_steps"])
    crop_len = int(cache["crop_len"])
    if steps != 8:
        raise ValueError("full rollout targets currently require the production 8-step grid")
    if not args.source_records.is_file() or not args.teacher_weights.is_file():
        raise FileNotFoundError("source records or teacher weights do not exist")

    if args.prompt_counts is None:
        repeat_counts = [args.repeats] * len(prompts)
    else:
        repeat_counts = [int(value) for value in json.loads(args.prompt_counts)]
        if len(repeat_counts) != len(prompts) or any(value <= 0 for value in repeat_counts):
            raise ValueError(
                "prompt-counts must contain one positive count per cache prompt"
            )

    seed_by_prompt = None
    audit_contract_identity = None
    if args.audit_contract is not None:
        contract = json.loads(args.audit_contract.read_text(encoding="utf-8"))
        selected = contract.get("dataset", {}).get("selected", [])
        if not isinstance(selected, list) or not selected:
            raise ValueError("audit contract has no selected dataset samples")
        ordered = sorted(selected, key=lambda item: str(item.get("path", "")))
        seed_by_prompt = {}
        for index, item in enumerate(ordered):
            prompt = str(item.get("prompt", ""))
            if not prompt or prompt in seed_by_prompt:
                raise ValueError("audit contract has duplicate or empty prompts")
            seed_by_prompt[prompt] = args.seed + index
        audit_contract_identity = file_identity(args.audit_contract)

    if args.rollout_source_mode == "teacher":
        arrays = collect_teacher_rollouts(
            args.teacher_weights,
            prompts,
            cross_cache,
            global_cond,
            crop_len,
            steps,
            repeat_counts,
            args.seed,
            args.max_metal_bytes,
            seed_by_prompt,
        )
    else:
        rollout = collect_student_rollouts(
            args.source_records,
            args.teacher_weights,
            prompts,
            cross_cache,
            global_cond,
            crop_len,
            steps,
            repeat_counts,
            args.seed,
            args.max_metal_bytes,
            seed_by_prompt,
        )
        arrays = collect_teacher_endpoints(
            rollout,
            args.teacher_weights,
            cross_cache,
            global_cond,
            prompts,
            crop_len,
            steps,
            args.max_metal_bytes,
        )
    count = len(arrays["prompt_indices"])
    expected_shape = (count, 256, crop_len)
    for name in (
        "initial_states",
        "target_velocity",
        "target_terminal",
    ):
        if arrays[name].shape != expected_shape or arrays[name].dtype != np.float16:
            raise ValueError(f"invalid rollout array {name}")
    if arrays["source_states"].shape != (count, steps + 1, 256, crop_len):
        raise ValueError("invalid source state array shape")
    if arrays["target_states"].shape != (count, steps + 1, 256, crop_len):
        raise ValueError("invalid target state array shape")
    if arrays["source_states"].dtype != np.float16 or arrays["target_states"].dtype != np.float16:
        raise ValueError("rollout state arrays must be float16")
    if arrays["noises"].shape != (count, steps - 1, 256, crop_len):
        raise ValueError("invalid rollout noise array shape")
    if arrays["noises"].dtype != np.float16:
        raise ValueError("rollout noises must be float16")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(temporary, output)
    sigma_grid = [float(value) for value in build_pingpong_schedule(
        steps, sigma_max=1.0, use_logsnr_shift=True
    )]
    inputs = {
        "state_cache_manifest": file_identity(args.state_cache / "manifest.json"),
        "states": file_identity(args.state_cache / "states.npz"),
        "conditions": file_identity(args.state_cache / "conditions.npz"),
        "source_records": file_identity(args.source_records),
        "teacher_weights": file_identity(args.teacher_weights),
    }
    if audit_contract_identity is not None:
        inputs["audit_contract"] = audit_contract_identity
    metadata = {
        "schema": "onus.ternary-quality/v7-full-rollout-targets",
        "status": "cached",
        "cache_file": str(output),
        "cache_sha256": sha256_file(output),
        "arrays_sha256": sha256_file(output),
        "inputs": inputs,
        "contract_digest": digest_json(inputs),
        "prompts": prompts,
        "prompt_count": len(prompts),
        "repeats": args.repeats if len(set(repeat_counts)) == 1 else None,
        "pair_count": count,
        "sampling_policy": (
            "balanced" if len(set(repeat_counts)) == 1 else "weighted"
        ),
        "prompt_counts": repeat_counts,
        "sampling_description": (
            "equal repeats per prompt"
            if len(set(repeat_counts)) == 1
            else "explicit per-prompt counts supplied by --prompt-counts"
        ),
        "crop_len": crop_len,
        "trajectory_steps": steps,
        "sampler": {
            "sigma_grid": sigma_grid,
            "initial_noise_seed_offset": 0,
            "sampler_reinjection_seed_offset": 1,
            "rng": "sequential_mx.random.split",
            "source_state_mode": args.rollout_source_mode,
            "seed_policy": (
                "audit_contract_sorted_dataset_paths"
                if audit_contract_identity is not None
                else "prompt_index"
            ),
        },
        "source_records": tq.load_records_checkpoint(args.source_records)[1],
        "elapsed_seconds": time.time() - started,
        "memory": tq.memory_snapshot(),
    }
    write_json_atomic(output.with_suffix(".json"), metadata)
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "output": str(output),
                "pair_count": count,
                "prompt_count": len(prompts),
                "steps": steps,
                "memory": metadata["memory"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
