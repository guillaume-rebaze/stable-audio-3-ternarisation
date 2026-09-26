"""Build the bounded V7 distillation-state cache.

The cache stores model inputs, not teacher outputs. Teacher targets are
recomputed separately so the cache cannot silently become a second model
checkpoint. Half the states are real-latent interpolations; half come from
either teacher or explicitly selected student sampler trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

import train_ternary_quality as tq
from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_runtime_contract import (
    build_state_cache_contract,
    pingpong_trace,
)
from ternary_provenance_v8 import validate_contract

mx = tq.mx


def parent_id(meta: dict, path: Path) -> str:
    source = str(meta.get("src_relpath") or meta.get("path") or path.name)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def stratified_prompt_subset(samples: list[dict], limit: int) -> list[str]:
    """Select prompts round-robin across broad musical families."""
    prompt_genre: dict[str, str] = {}
    for sample in samples:
        prompt_genre.setdefault(
            sample["prompt"], str(sample.get("genre", "unknown")).strip() or "unknown"
        )
    prompts_by_family: dict[str, list[str]] = {}
    for prompt, genre in prompt_genre.items():
        prompts_by_family.setdefault(genre_family(genre), []).append(prompt)
    for prompts in prompts_by_family.values():
        prompts.sort()
    selected: list[str] = []
    depth = 0
    families = sorted(prompts_by_family)
    while len(selected) < min(limit, len(prompt_genre)):
        added = False
        for family in families:
            prompts = prompts_by_family[family]
            if depth < len(prompts):
                selected.append(prompts[depth])
                added = True
                if len(selected) == min(limit, len(prompt_genre)):
                    break
        if not added:
            break
        depth += 1
    return selected


def audit_prompt_subset(samples: list[dict], limit: int) -> list[str]:
    """Select prompts in the same first-seen order used by the quality audit."""
    selected: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        prompt = str(sample["prompt"])
        if prompt in seen:
            continue
        seen.add(prompt)
        selected.append(prompt)
        if len(selected) == limit:
            break
    return selected


def genre_family(genre: str) -> str:
    """Collapse narrow metadata labels so one subgenre cannot dominate a pilot."""
    label = genre.strip().lower()
    if label.startswith("voice") or label == "vocal_pop":
        return "vocal"
    if "classical" in label:
        return "classical"
    if "hiphop" in label:
        return "hiphop"
    if "rock" in label:
        return "rock"
    if "jazz" in label or "funk" in label or "soul" in label:
        return "jazz_funk_soul"
    if "ambient" in label or label == "oneohtrix":
        return "ambient"
    if "world" in label or label == "dub":
        return "world_dub"
    return "electronic"


def model_rollout_states(
    model,
    cross,
    global_cond,
    latent_len: int,
    steps: int,
    seed: int,
) -> list[tuple[np.ndarray, float]]:
    sigmas = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    initial = mx.random.normal(
        (1, 256, latent_len), dtype=mx.float16, key=mx.random.key(seed)
    )
    trace = pingpong_trace(
        lambda x, t: model(x, t, cross, global_cond),
        initial,
        sigmas,
        sampler_seed=seed + 1,
    )
    return [
        (np.asarray(record["state"]).astype(np.float16), float(record["sigma"]))
        for record in trace[:-1]
    ]


def check_metal_memory(max_metal_bytes: int, label: str) -> dict[str, float]:
    snapshot = tq.memory_snapshot()
    peak_bytes = int(snapshot["metal_peak_gb"] * (1024**3))
    if peak_bytes > max_metal_bytes:
        raise RuntimeError(
            f"{label} exceeded Metal memory guard: {peak_bytes} > {max_metal_bytes} bytes"
        )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded V7 ternary state cache")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-expanded-v3/train"),
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument(
        "--rollout-checkpoint",
        type=Path,
        default=None,
        help="records checkpoint whose student rollouts should seed trajectory states",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-v7-20260924/state-cache-512-fp32-balanced-v2"),
    )
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--trajectory-steps", type=int, default=8)
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument(
        "--prompt-selection",
        choices=("stratified", "audit"),
        default="stratified",
        help="prompt order; audit matches audit_ternary_quality first-seen selection",
    )
    parser.add_argument(
        "--audit-aligned-states",
        action="store_true",
        help="include the exact four deterministic states used by the 4-step audit",
    )
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--dataset-contract",
        type=Path,
        default=None,
        help="V8 dataset contract; when supplied, train paths and hashes are checked before loading MLX",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.states < 2 or args.states % 2:
        raise ValueError("states must be an even number >= 2")
    if args.trajectory_steps < 2:
        raise ValueError("trajectory_steps must be >= 2")
    sigma_schedule = build_pingpong_schedule(
        args.trajectory_steps, sigma_max=1.0, use_logsnr_shift=True
    )
    mx.eval(sigma_schedule)
    sigmas = [float(value) for value in sigma_schedule[:-1]]
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite state cache: {args.output_dir}")

    samples = tq.load_samples(args.dataset_dir, 0)
    selection_manifest_path = args.dataset_dir.parent / "selection_manifest.json"
    if not selection_manifest_path.is_file():
        raise FileNotFoundError(
            f"V7 training data requires its selection manifest: {selection_manifest_path}"
        )
    dataset_contract_report = None
    if args.dataset_contract is not None:
        dataset_contract_report = validate_contract(
            args.dataset_contract, args.project_root
        )
        contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
        declared_train = (
            args.project_root / contract["splits"]["train"]["directory"]
        ).resolve()
        if args.dataset_dir.resolve() != declared_train:
            raise ValueError(
                "dataset directory does not match V8 contract train split: "
                f"{args.dataset_dir.resolve()} != {declared_train}"
            )
    if args.prompt_count <= 0:
        raise ValueError("prompt_count must be positive")
    prompts = (
        stratified_prompt_subset(samples, args.prompt_count)
        if args.prompt_selection == "stratified"
        else audit_prompt_subset(samples, args.prompt_count)
    )
    if len(prompts) != args.prompt_count:
        raise ValueError(
            f"requested {args.prompt_count} prompts but selected only {len(prompts)}"
        )
    if len(sigmas) != args.trajectory_steps:
        raise ValueError("production sigma grid and trajectory step count diverged")
    half = args.states // 2
    states_per_prompt = len(sigmas)
    if half % (len(prompts) * states_per_prompt):
        raise ValueError(
            "state count must equal two balanced halves of prompt x sigma x seed; "
            f"half={half}, prompts={len(prompts)}, sigmas={len(sigmas)}"
        )
    seeds_per_prompt = half // (len(prompts) * states_per_prompt)
    audit_sigmas = [
        float(value)
        for value in build_pingpong_schedule(4, sigma_max=1.0, use_logsnr_shift=True)[:-1]
    ]
    audit_sigma_indices = {
        sigma_index: audit_index
        for sigma_index, sigma in enumerate(sigmas)
        for audit_index, audit_sigma in enumerate(audit_sigmas)
        if np.isclose(sigma, audit_sigma, rtol=0.0, atol=1e-7)
    }
    samples_by_prompt: dict[str, list[dict]] = {prompt: [] for prompt in prompts}
    for sample in samples:
        if sample["prompt"] in samples_by_prompt:
            samples_by_prompt[sample["prompt"]].append(sample)
    for members in samples_by_prompt.values():
        members.sort(key=lambda sample: sample["path"])
    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    _, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher, args.teacher_weights, prompts, args.seconds
    )
    check_metal_memory(args.max_metal_bytes, "teacher conditioning")
    trajectory_source = "teacher_trajectory"
    rollout_fingerprint = None
    if args.rollout_checkpoint is not None:
        records, rollout_metadata = tq.load_records_checkpoint(args.rollout_checkpoint)
        if int(rollout_metadata["crop_len"]) != args.crop_len:
            raise ValueError(
                "rollout checkpoint crop_len does not match requested cache: "
                f"{rollout_metadata['crop_len']} != {args.crop_len}"
            )
        teacher = tq.apply_records_to_model(
            teacher, records, int(rollout_metadata["group_size"])
        )
        trajectory_source = "student_trajectory"
        rollout_fingerprint = {
            "checkpoint": tq.file_fingerprint(args.rollout_checkpoint),
            "scope_digest": rollout_metadata["scope_digest"],
            "scope_count": len(records),
            "group_size": int(rollout_metadata["group_size"]),
            "quantizer_mode": str(rollout_metadata["quantizer_mode"]),
        }
    prompt_index = {prompt: index for index, prompt in enumerate(prompts)}

    states: list[np.ndarray] = []
    state_sigmas: list[float] = []
    state_prompt_indices: list[int] = []
    state_sources: list[str] = []
    state_parents: list[str] = []
    selected_source_inventory: dict[str, dict[str, object]] = {}
    for seed_index in range(seeds_per_prompt):
        for sigma_index, sigma in enumerate(sigmas):
            for prompt in prompts:
                members = samples_by_prompt[prompt]
                sample_index = (
                    0
                    if args.audit_aligned_states and seed_index == 0
                    else (args.seed + seed_index) % len(members)
                )
                sample = members[sample_index]
                metadata_path = Path(sample["path"]).with_suffix(".json")
                metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.exists()
                    else {}
                )
                source_path = str(Path(sample["path"]).resolve())
                if source_path not in selected_source_inventory:
                    selected_source_inventory[source_path] = {
                        "latent": tq.file_fingerprint(Path(sample["path"])),
                        "metadata": (
                            tq.file_fingerprint(metadata_path)
                            if metadata_path.is_file()
                            else None
                        ),
                        "prompt": sample["prompt"],
                        "genre": sample["genre"],
                    }
                if (
                    args.audit_aligned_states
                    and seed_index == 0
                    and sigma_index in audit_sigma_indices
                ):
                    sample_seed = (
                        args.seed
                        + prompt_index[prompt] * 1000
                        + audit_sigma_indices[sigma_index]
                    )
                else:
                    sample_seed = (
                        args.seed
                        + seed_index * len(prompts) * len(sigmas)
                        + sigma_index * len(prompts)
                        + prompt_index[prompt]
                    )
                x = tq.noised_latent(sample, args.crop_len, sigma, sample_seed)
                mx.eval(x)
                states.append(np.array(x[0]).astype(np.float16))
                state_sigmas.append(sigma)
                state_prompt_indices.append(prompt_index[prompt])
                state_sources.append("real_latent_noised")
                state_parents.append(parent_id(metadata, Path(sample["path"])))

    for seed_index in range(seeds_per_prompt):
        for prompt in prompts:
            trajectory_seed = (
                args.seed + 100000 + seed_index * len(prompts) + prompt_index[prompt]
            )
            trajectory = model_rollout_states(
                teacher,
                cross_cache[prompt],
                global_cond,
                args.crop_len,
                args.trajectory_steps,
                trajectory_seed,
            )
            check_metal_memory(args.max_metal_bytes, "teacher trajectory cache")
            for x, sigma in trajectory:
                states.append(x[0])
                state_sigmas.append(sigma)
                state_prompt_indices.append(prompt_index[prompt])
                state_sources.append(trajectory_source)
                state_parents.append(
                    f"prompt:{hashlib.sha256(prompt.encode()).hexdigest()[:24]}"
                )

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "states.npz",
        states=np.stack(states).astype(np.float16),
        sigmas=np.asarray(state_sigmas, dtype=np.float32),
        prompt_indices=np.asarray(state_prompt_indices, dtype=np.int32),
        sources=np.asarray(state_sources),
        parents=np.asarray(state_parents),
    )
    condition_arrays = {f"cross_{index:04d}": np.array(cross_cache[prompt]) for prompt, index in prompt_index.items()}
    condition_arrays["global_cond"] = np.array(global_cond)
    np.savez_compressed(args.output_dir / "conditions.npz", **condition_arrays)
    manifest = {
        "schema": "onus.ternary-quality/v7-state-cache",
        "status": "prepared",
        "created_at_unix": time.time(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "teacher": tq.file_fingerprint(args.teacher_weights),
        "text_conditioner": tq.file_fingerprint(
            tq.ensure_local(tq.T5GEMMA_NPZ_REL)
        ),
        "runtime_contract": build_state_cache_contract(
            tq.MLX_RUNTIME_ROOT,
            Path(__file__).resolve().parent,
            args.crop_len,
            args.seconds,
            args.trajectory_steps,
        ),
        "dataset": {
            "directory": str(args.dataset_dir),
            "sample_count": len(samples),
            "prompt_count": len(prompts),
            "prompt_selection": args.prompt_selection,
            "audit_aligned_states": bool(args.audit_aligned_states),
            "selection_manifest": tq.file_fingerprint(selection_manifest_path),
            "dataset_contract": dataset_contract_report,
            "selected_prompts": prompts,
            "selected_prompt_families": [
                {
                    "prompt": prompt,
                    "family": genre_family(
                        next(
                            sample["genre"]
                            for sample in samples
                            if sample["prompt"] == prompt
                        )
                    ),
                }
                for prompt in prompts
            ],
            "selected_source_inventory": list(selected_source_inventory.values()),
            "authorization": "user_authorized_personal_study",
        },
        "cache": {
            "states_file": str(args.output_dir / "states.npz"),
            "conditions_file": str(args.output_dir / "conditions.npz"),
            "state_count": len(states),
            "real_latent_noised": state_sources.count("real_latent_noised"),
            "teacher_trajectory": state_sources.count("teacher_trajectory"),
            "student_trajectory": state_sources.count("student_trajectory"),
            "trajectory_source": trajectory_source,
            "rollout_model": rollout_fingerprint,
            "production_sigma_grid": sigmas,
            "prompt_count": len(prompts),
            "seeds_per_prompt": seeds_per_prompt,
            "seed": args.seed,
            "rng_contract": {
                "initial_noise_seed": "trajectory_seed",
                "sampler_reinjection_seed": "trajectory_seed + 1",
                "sampler_transition": "ARC sample_flow_pingpong",
            },
            "trajectory_steps": args.trajectory_steps,
            "crop_len": args.crop_len,
            "seconds": args.seconds,
            "prompt_index": prompts,
            "parent_count": len(set(state_parents)),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "states": len(states),
        "real_latent_noised": state_sources.count("real_latent_noised"),
        "teacher_trajectory": state_sources.count("teacher_trajectory"),
        "student_trajectory": state_sources.count("student_trajectory"),
        "prompts": len(prompts),
        "parent_count": len(set(state_parents)),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
