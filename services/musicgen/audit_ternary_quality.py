"""Independent velocity audit for a reloaded ternary DiT candidate.

This script deliberately loads the exported NPZ from disk.  It does not compare
the in-memory training object with itself, and it never normalizes an output to
make a collapsed model look healthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import train_ternary_quality as tq
from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_contract import (
    TernaryWeights,
    scope_digest,
    unpack_codes,
    validate_ternary_weights,
)
from ternary_runtime_contract import pingpong_trace, timestep_tensor
from ternary_provenance_v8 import validate_contract

mx = tq.mx
nn = tq.nn


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float32).ravel()
    bf = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-8))


def check_metal_memory(max_metal_bytes: int, stage: str) -> dict[str, float]:
    snapshot = tq.memory_snapshot()
    peak = int(snapshot["metal_peak_gb"] * (1024**3))
    if peak > max_metal_bytes:
        raise RuntimeError(
            f"{stage} exceeded Metal memory guard: {peak} > {max_metal_bytes} bytes"
        )
    return snapshot


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = str(manifest.get("schema", ""))
    if not schema.startswith("onus.ternary-quality/"):
        raise ValueError(f"Unsupported ternary manifest schema: {schema!r}")
    paths = manifest.get("scope", {}).get("paths", [])
    if not paths:
        raise ValueError("Manifest has empty ternary scope")
    expected = manifest.get("scope", {}).get("digest")
    actual = scope_digest(paths)
    if expected and expected != actual:
        raise ValueError(f"Scope digest mismatch: manifest={expected}, actual={actual}")
    return manifest


def load_split_manifest(path: Path, samples: list[dict], role: str) -> tuple[list[dict], dict]:
    """Select only files named by a provenance manifest and verify membership."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    roles = payload.get("roles")
    paths = roles.get(role) if isinstance(roles, dict) else None
    if paths is None and role == "validation":
        paths = payload.get("validation")
    if paths is None and role == "test":
        paths = payload.get("test")
    if isinstance(paths, dict):
        sources = paths.get("sources")
        if isinstance(sources, list):
            staged_paths = [
                source.get("staged_latent")
                for source in sources
                if isinstance(source, dict)
            ]
            if len(staged_paths) != len(sources) or any(
                not isinstance(value, str) or not value for value in staged_paths
            ):
                raise ValueError(
                    f"split manifest has invalid staged source paths for {role!r}: {path}"
                )
            paths = staged_paths
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"split manifest has no non-empty role {role!r}: {path}")
    expected = {str(Path(value).resolve()) for value in paths}
    indexed = {str(Path(sample["path"]).resolve()): sample for sample in samples}
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(
            f"split manifest references {len(missing)} samples outside dataset-dir; "
            f"first={missing[:3]}"
        )
    selected = [indexed[value] for value in sorted(expected)]
    parents = {
        str(sample.get("parent_id") or Path(sample["path"]).stem)
        for sample in selected
    }
    return selected, {
        "path": str(path),
        "role": role,
        "verified": True,
        "sample_count": len(selected),
        "parent_count": len(parents),
        "manifest_status": payload.get("status"),
    }


def validate_dataset_contract(
    contract_path: Path,
    project_root: Path,
    dataset_dir: Path,
    split_role: str,
) -> dict:
    report = validate_contract(contract_path, project_root)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_role = "train" if split_role == "debug_train_seen" else split_role
    declared = (project_root / contract["splits"][contract_role]["directory"]).resolve()
    actual = dataset_dir.resolve()
    if actual != declared:
        raise ValueError(
            f"dataset directory does not match V8 contract {contract_role} split: "
            f"{actual} != {declared}"
        )
    report["role"] = contract_role
    report["dataset_directory"] = str(actual)
    report["dataset_digest"] = contract["dataset_digest"]
    return report


def validate_artifact_scope(artifact: Path, manifest: dict) -> dict:
    """Validate serialized ternary arrays before handing them to MLX."""
    scope = sorted(set(manifest["scope"]["paths"]))
    group_size = int(manifest["model"]["group_size"])
    quantizer_mode = str(manifest["model"].get("quantizer_mode", "affine_centered"))
    storage_mode = str(manifest["model"].get("storage_mode", "full_affine"))
    if group_size not in {32, 64, 128}:
        raise ValueError(f"unsupported ternary group_size={group_size}")
    if quantizer_mode not in {"symmetric", "affine_centered"}:
        raise ValueError(f"unsupported quantizer_mode={quantizer_mode!r}")
    if storage_mode not in {"symmetric_compact", "full_affine"}:
        raise ValueError(f"unsupported storage_mode={storage_mode!r}")
    if quantizer_mode == "symmetric" and storage_mode not in {
        "symmetric_compact",
        "full_affine",
    }:
        raise ValueError("symmetric quantizer has incompatible storage mode")

    validated = 0
    with np.load(artifact, allow_pickle=False) as arrays:
        available = set(arrays.files)
        for prefix in scope:
            required = {f"{prefix}.weight", f"{prefix}.scales"}
            if storage_mode == "full_affine":
                required.add(f"{prefix}.biases")
            missing = sorted(required - available)
            if missing:
                raise ValueError(f"artifact missing ternary arrays: {missing}")
            if storage_mode == "symmetric_compact" and f"{prefix}.biases" in available:
                raise ValueError(f"compact artifact stores derivable biases: {prefix}")

            packed = np.asarray(arrays[f"{prefix}.weight"])
            scales = np.asarray(arrays[f"{prefix}.scales"])
            if packed.dtype != np.uint32:
                raise ValueError(f"{prefix}.weight must be uint32, got {packed.dtype}")
            q = unpack_codes(packed, group_size)
            if scales.shape != q.shape[:2]:
                raise ValueError(
                    f"{prefix}.scales shape {scales.shape} does not match q {q.shape}"
                )
            if storage_mode == "symmetric_compact":
                biases = -scales
            else:
                biases = np.asarray(arrays[f"{prefix}.biases"])
            if biases.shape != scales.shape:
                raise ValueError(f"{prefix}.biases shape {biases.shape} != scales {scales.shape}")
            if quantizer_mode == "symmetric":
                if np.any(scales > 0) or np.any(biases < 0):
                    raise ValueError(f"{prefix} symmetric scales/biases have invalid signs")
            validate_ternary_weights(
                TernaryWeights(
                    packed_codes=packed,
                    scales=scales,
                    biases=biases,
                    q=q,
                    group_means=np.zeros_like(scales, dtype=np.float32),
                    group_size=group_size,
                    mode=quantizer_mode,
                )
            )
            validated += 1
    return {
        "artifact": str(artifact),
        "scope_count": validated,
        "scope_digest": scope_digest(scope),
        "group_size": group_size,
        "quantizer_mode": quantizer_mode,
        "storage_mode": storage_mode,
        "codes_and_metadata_valid": True,
    }


def load_student(artifact: Path, manifest: dict, crop_len: int) -> nn.Module:
    validate_artifact_scope(artifact, manifest)
    model = dit_mlx_medium.DiT(T_lat=crop_len)
    scope = set(manifest["scope"]["paths"])
    group_size = int(manifest["model"]["group_size"])
    storage_mode = str(manifest["model"].get("storage_mode", "full_affine"))

    def predicate(path: str, module: nn.Module) -> bool:
        return path in scope

    nn.quantize(
        model,
        bits=2,
        group_size=group_size,
        mode="affine",
        class_predicate=predicate,
    )
    model.load_weights(str(artifact), strict=storage_mode != "symmetric_compact")
    if storage_mode == "symmetric_compact":
        tq.apply_symmetric_derived_biases(model, scope)
    model.freeze()
    mx.eval(model.parameters())
    return model


def one_sample_per_prompt(samples: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for sample in samples:
        if sample["prompt"] in seen:
            continue
        seen.add(sample["prompt"])
        selected.append(sample)
        if limit and len(selected) >= limit:
            break
    return selected


def audit_teacher_reference(
    teacher: nn.Module,
    samples: list[dict],
    cross_cache: dict[str, mx.array],
    global_cond: mx.array,
    crop_len: int,
    sigmas: list[float],
    seed: int,
) -> dict:
    records: list[dict] = []
    max_repeat_error = 0.0
    for sample_index, sample in enumerate(samples):
        cross = cross_cache[sample["prompt"]]
        for sigma_index, sigma in enumerate(sigmas):
            key_seed = seed + sample_index * 1000 + sigma_index
            x = tq.noised_latent(sample, crop_len, sigma, key_seed)
            t = timestep_tensor(sigma)
            first = teacher(x, t, cross, global_cond)
            second = teacher(x, t, cross, global_cond)
            mx.eval(first, second)
            first_np = np.array(first)
            second_np = np.array(second)
            repeat_error = tq.relative_error(first_np, second_np)
            max_repeat_error = max(max_repeat_error, repeat_error)
            records.append(
                {
                    "prompt": sample["prompt"],
                    "sigma": sigma,
                    "output_rms": float(np.sqrt(np.mean(first_np.astype(np.float32) ** 2))),
                    "repeat_relative_error": repeat_error,
                }
            )
    return {
        "samples": len(samples),
        "sigmas": sigmas,
        "records": records,
        "max_repeat_relative_error": max_repeat_error,
        "deterministic_pass": max_repeat_error < 1e-6,
    }


def audit_velocity(
    teacher: nn.Module,
    student: nn.Module,
    samples: list[dict],
    cross_cache: dict[str, mx.array],
    global_cond: mx.array,
    crop_len: int,
    sigmas: list[float],
    seed: int,
) -> dict:
    records: list[dict] = []
    for sample_index, sample in enumerate(samples):
        cross = cross_cache[sample["prompt"]]
        for sigma_index, sigma in enumerate(sigmas):
            key_seed = seed + sample_index * 1000 + sigma_index
            x = tq.noised_latent(sample, crop_len, sigma, key_seed)
            t = timestep_tensor(sigma)
            teacher_out = teacher(x, t, cross, global_cond)
            student_out = student(x, t, cross, global_cond)
            mx.eval(teacher_out, student_out)
            teacher_np = np.array(teacher_out).astype(np.float32)
            student_np = np.array(student_out).astype(np.float32)
            records.append(
                {
                    "prompt": sample["prompt"],
                    "sigma": sigma,
                    "relative_error": tq.relative_error(teacher_np, student_np),
                    "cosine": cosine(teacher_np, student_np),
                    "teacher_rms": float(np.sqrt(np.mean(teacher_np * teacher_np))),
                    "student_rms": float(np.sqrt(np.mean(student_np * student_np))),
                    "rms_ratio": float(
                        np.sqrt(np.mean(student_np * student_np))
                        / (np.sqrt(np.mean(teacher_np * teacher_np)) + 1e-8)
                    ),
                }
            )

    by_sigma: dict[str, dict] = {}
    for sigma in sigmas:
        bucket = [r for r in records if r["sigma"] == sigma]
        by_sigma[str(sigma)] = {
            "count": len(bucket),
            "mean_cosine": float(np.mean([r["cosine"] for r in bucket])),
            "min_cosine": float(np.min([r["cosine"] for r in bucket])),
            "mean_relative_error": float(np.mean([r["relative_error"] for r in bucket])),
            "mean_rms_ratio": float(np.mean([r["rms_ratio"] for r in bucket])),
        }

    cosines = [r["cosine"] for r in records]
    return {
        "samples": len(samples),
        "sigmas": sigmas,
        "records": records,
        "by_sigma": by_sigma,
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "mean_relative_error": float(np.mean([r["relative_error"] for r in records])),
        "candidate_velocity_pass": bool(np.mean(cosines) >= 0.90 and np.min(cosines) >= 0.80),
        "release_velocity_pass": bool(np.mean(cosines) >= 0.93 and np.min(cosines) >= 0.85),
    }


def rollout_trace(
    model: nn.Module,
    cross: mx.array,
    global_cond: mx.array,
    latent_len: int,
    steps: int,
    seed: int,
) -> list[dict[str, np.ndarray | float | None]]:
    """Run the same stochastic sampler path and retain every visited state."""
    sigmas = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    init = mx.random.normal(
        (1, 256, latent_len), dtype=mx.float16, key=mx.random.key(seed)
    )
    return pingpong_trace(
        lambda x, t: model(x, t, cross, global_cond),
        init,
        sigmas,
        sampler_seed=seed + 1,
    )


def audit_trajectory(
    teacher: nn.Module,
    student: nn.Module,
    samples: list[dict],
    cross_cache: dict[str, mx.array],
    global_cond: mx.array,
    latent_len: int,
    steps: int,
    seed: int,
) -> dict:
    records: list[dict] = []
    terminal_records: list[dict] = []
    on_policy_velocity_cosines: list[float] = []
    for sample_index, sample in enumerate(samples):
        teacher_trace = rollout_trace(
            teacher, cross_cache[sample["prompt"]], global_cond,
            latent_len, steps, seed + sample_index,
        )
        student_trace = rollout_trace(
            student, cross_cache[sample["prompt"]], global_cond,
            latent_len, steps, seed + sample_index,
        )
        for step_index, (teacher_step, student_step) in enumerate(
            zip(teacher_trace, student_trace)
        ):
            teacher_state = teacher_step["state"]
            student_state = student_step["state"]
            teacher_velocity = teacher_step["velocity"]
            student_velocity = student_step["velocity"]
            state_record = {
                "prompt": sample["prompt"],
                "sample_index": sample_index,
                "step": step_index,
                "sigma": teacher_step["sigma"],
                "terminal": teacher_velocity is None,
                "state_relative_error": tq.relative_error(teacher_state, student_state),
                "state_cosine": cosine(teacher_state, student_state),
                "state_rms_ratio": float(
                    np.sqrt(np.mean(student_state * student_state))
                    / (np.sqrt(np.mean(teacher_state * teacher_state)) + 1e-8)
                ),
            }
            if teacher_velocity is not None and student_velocity is not None:
                t = timestep_tensor(float(teacher_step["sigma"]))
                teacher_on_student = teacher(
                    mx.array(student_state, dtype=mx.float16),
                    t,
                    cross_cache[sample["prompt"]],
                    global_cond,
                )
                mx.eval(teacher_on_student)
                teacher_on_student_np = np.array(teacher_on_student).astype(np.float32)
                on_policy_cosine = cosine(teacher_on_student_np, student_velocity)
                on_policy_velocity_cosines.append(on_policy_cosine)
                state_record.update(
                    {
                        "velocity_relative_error": tq.relative_error(
                            teacher_velocity, student_velocity
                        ),
                        "velocity_cosine": cosine(teacher_velocity, student_velocity),
                        "teacher_on_student_velocity_relative_error": tq.relative_error(
                            teacher_on_student_np, student_velocity
                        ),
                        "teacher_on_student_velocity_cosine": on_policy_cosine,
                    }
                )
                records.append(state_record)
            else:
                terminal_records.append(state_record)
    all_state_records = records + terminal_records
    state_cosines = [record["state_cosine"] for record in all_state_records]
    velocity_cosines = [record["velocity_cosine"] for record in records]
    return {
        "samples": len(samples),
        "steps": steps,
        "records": records,
        "terminal_records": terminal_records,
        "terminal_state_mean_cosine": float(
            np.mean([record["state_cosine"] for record in terminal_records] or [1.0])
        ),
        "terminal_state_min_cosine": float(
            np.min([record["state_cosine"] for record in terminal_records] or [1.0])
        ),
        "state_mean_cosine": float(np.mean(state_cosines or [1.0])),
        "state_min_cosine": float(np.min(state_cosines or [1.0])),
        "velocity_mean_cosine": float(np.mean(velocity_cosines or [1.0])),
        "velocity_min_cosine": float(np.min(velocity_cosines or [1.0])),
        "teacher_on_student_velocity_mean_cosine": float(
            np.mean(on_policy_velocity_cosines or [1.0])
        ),
        "teacher_on_student_velocity_min_cosine": float(
            np.min(on_policy_velocity_cosines or [1.0])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a ternary artifact or record checkpoint")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--records-checkpoint",
        type=Path,
        default=None,
        help="audit a V7 record checkpoint directly, without exporting dense-rest weights",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"))
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="required to claim validation/test provenance; selects only listed files",
    )
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    parser.add_argument(
        "--dataset-contract",
        type=Path,
        default=None,
        help="V8 dataset contract to verify against the selected dataset directory",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--require-dataset-contract",
        action="store_true",
        help="fail if a V8 dataset contract is not supplied",
    )
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=8,
        help="sampler states for rollout diagnostics; 0 disables",
    )
    parser.add_argument(
        "--sampler-steps",
        type=int,
        default=8,
        help="production sampler step count used for pointwise velocity audits",
    )
    parser.add_argument(
        "--split-role",
        choices=("debug_train_seen", "validation", "test"),
        default="debug_train_seen",
        help="Provenance label; does not turn train-seen data into held-out data.",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Write metrics and return zero even when an audit gate fails.",
    )
    args = parser.parse_args()

    if args.require_dataset_contract and args.dataset_contract is None:
        raise ValueError("--require-dataset-contract requires --dataset-contract")

    if (args.artifact is None) == (args.records_checkpoint is None):
        raise ValueError("provide exactly one of --artifact or --records-checkpoint")
    if args.records_checkpoint is not None and args.manifest is not None:
        raise ValueError("--manifest applies only to --artifact")

    manifest_path = (
        args.manifest or args.artifact.with_suffix(".json")
        if args.artifact is not None
        else None
    )
    output_dir = args.output_dir or (
        args.artifact.parent
        if args.artifact is not None
        else args.records_checkpoint.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.max_metal_bytes <= 0:
        raise ValueError("max-metal-bytes must be positive")

    records = None
    if args.artifact is not None:
        manifest = load_manifest(manifest_path)
        contract_report = validate_artifact_scope(args.artifact, manifest)
        scope_paths = manifest["scope"]["paths"]
        group_size = int(manifest["model"]["group_size"])
    else:
        records, records_metadata = tq.load_records_checkpoint(
            args.records_checkpoint
        )
        group_size = int(records_metadata["group_size"])
        scope_paths = sorted(records)
        manifest = {
            "schema": "onus.ternary-quality/v7-records-audit",
            "model": {
                "group_size": group_size,
                "quantizer_mode": records_metadata["quantizer_mode"],
                "storage_mode": "record_checkpoint",
            },
            "scope": {
                "paths": scope_paths,
                "count": len(scope_paths),
                "digest": records_metadata["scope_digest"],
            },
        }
        contract_report = {
            "records_checkpoint": str(args.records_checkpoint),
            "payload_valid": True,
            "scope_count": len(scope_paths),
            "scope_digest": records_metadata["scope_digest"],
            "linear_bias_count": records_metadata.get("linear_bias_count"),
        }
    samples = tq.load_samples(args.dataset_dir, 0)
    dataset_contract_report = None
    if args.dataset_contract is not None:
        dataset_contract_report = validate_dataset_contract(
            args.dataset_contract,
            args.project_root,
            args.dataset_dir,
            args.split_role,
        )
    split_report = {
        "path": None,
        "role": args.split_role,
        "verified": False,
        "sample_count": len(samples),
        "parent_count": None,
        "reason": "no split manifest supplied",
    }
    if args.split_manifest is not None:
        samples, split_report = load_split_manifest(
            args.split_manifest, samples, args.split_role
        )
    selected = one_sample_per_prompt(samples, args.max_prompts)
    if args.sampler_steps <= 0:
        raise ValueError("sampler-steps must be positive")
    sigma_schedule = build_pingpong_schedule(
        args.sampler_steps, sigma_max=1.0, use_logsnr_shift=True
    )
    mx.eval(sigma_schedule)
    sigmas = [float(value) for value in sigma_schedule[:-1]]
    print(f"[Audit] {len(selected)} prompts x {len(sigmas)} sigmas", flush=True)

    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    context_cache, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher,
        args.teacher_weights,
        [s["prompt"] for s in selected],
        seconds=12.0,
    )
    del context_cache
    check_metal_memory(args.max_metal_bytes, "teacher conditioning")
    if args.artifact is not None:
        student = load_student(args.artifact, manifest, args.crop_len)
        loaded_source = "artifact"
    else:
        student = dit_mlx_medium.DiT(T_lat=args.crop_len)
        student.load_weights(str(args.teacher_weights), strict=False)
        student = tq.apply_records_to_model(student, records, group_size)
        loaded_source = "records_checkpoint"
    print(f"[Audit] loaded {loaded_source}; memory={tq.memory_snapshot()}", flush=True)
    check_metal_memory(args.max_metal_bytes, "candidate load")

    teacher_metrics = audit_teacher_reference(
        teacher, selected, cross_cache, global_cond, args.crop_len, sigmas, args.seed
    )
    velocity_metrics = audit_velocity(
        teacher, student, selected, cross_cache, global_cond, args.crop_len, sigmas, args.seed
    )
    trajectory_metrics = None
    if args.trajectory_steps:
        trajectory_metrics = audit_trajectory(
            teacher,
            student,
            selected,
            cross_cache,
            global_cond,
            args.crop_len,
            args.trajectory_steps,
            args.seed + 700000,
        )
    tq.write_json(output_dir / "teacher_metrics.json", teacher_metrics)
    tq.write_json(output_dir / "velocity_metrics.json", velocity_metrics)
    if trajectory_metrics is not None:
        tq.write_json(output_dir / "trajectory_metrics.json", trajectory_metrics)
    memory = check_metal_memory(args.max_metal_bytes, "quality audit")
    summary = {
        "status": "audited",
        "artifact": str(args.artifact) if args.artifact else None,
        "records_checkpoint": (
            str(args.records_checkpoint) if args.records_checkpoint else None
        ),
        "manifest": str(manifest_path) if manifest_path else None,
        "scope_digest": manifest["scope"]["digest"],
        "scope_count": manifest["scope"]["count"],
        "contract": contract_report,
        "dataset_contract": dataset_contract_report,
        "split_role": args.split_role,
        "split": split_report,
        "heldout": args.split_role in {"validation", "test"} and split_report["verified"],
        "teacher": teacher_metrics,
        "velocity": velocity_metrics,
        "trajectory": trajectory_metrics,
        "elapsed_seconds": time.time() - started,
        "memory": memory,
    }
    tq.write_json(output_dir / "audit_summary.json", summary)
    print(json.dumps({
        "teacher_deterministic": teacher_metrics["deterministic_pass"],
        "velocity_mean_cosine": velocity_metrics["mean_cosine"],
        "velocity_min_cosine": velocity_metrics["min_cosine"],
        "candidate_velocity_pass": velocity_metrics["candidate_velocity_pass"],
        "release_velocity_pass": velocity_metrics["release_velocity_pass"],
        "trajectory_state_mean_cosine": (
            trajectory_metrics["state_mean_cosine"] if trajectory_metrics else None
        ),
        "trajectory_state_min_cosine": (
            trajectory_metrics["state_min_cosine"] if trajectory_metrics else None
        ),
        "trajectory_terminal_state_mean_cosine": (
            trajectory_metrics["terminal_state_mean_cosine"]
            if trajectory_metrics
            else None
        ),
        "trajectory_teacher_on_student_velocity_mean_cosine": (
            trajectory_metrics["teacher_on_student_velocity_mean_cosine"]
            if trajectory_metrics
            else None
        ),
    }, indent=2), flush=True)
    if not args.allow_fail and (
        not teacher_metrics["deterministic_pass"]
        or not velocity_metrics["candidate_velocity_pass"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
