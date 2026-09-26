"""Read-only V6 artifact diagnostics; no training or checkpoint modification.

Compare serialized code transitions, then optionally isolate timestep precision
on cached TRAIN states. Reports are new files, never replacements for V6 logs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np


LINEAGE = (
    "cascade-g32-symmetric-checkpointfix/window-02-03",
    "joint-refine-g32-symmetric-checkpointfix/window-00-03-250",
    "joint-refine-g32-symmetric-expanded-cache-2048/window-00-03-250",
    "joint-refine-g32-symmetric-sftvoices-v1/window-00-03-250",
    "joint-refine-g32-symmetric-sftvoices-student-v1/window-00-03-250",
    "joint-refine-g32-symmetric-sftvoices-student16-low-sigma-v2/window-00-03-250",
)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def packed_transition_count(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    if a.dtype != np.uint32 or b.dtype != np.uint32 or a.shape != b.shape:
        raise ValueError("expected identically shaped uint32 code arrays")
    if not a.size:
        raise ValueError("empty code array is not evidence of zero transitions")
    delta = np.bitwise_xor(a, b)
    changed = 0
    for shift in range(0, 32, 2):
        if np.any(((a >> shift) & 3) == 3) or np.any(((b >> shift) & 3) == 3):
            raise ValueError("reserved code 3")
        changed += int(np.count_nonzero((delta >> shift) & 3))
    return changed, a.size * 16


def compare_records(a: Path, b: Path) -> dict:
    rows = []
    with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
        keys = sorted(key for key in za.files if key.endswith(".packed_codes"))
        if not keys or keys != sorted(key for key in zb.files if key.endswith(".packed_codes")):
            raise ValueError("missing or unequal scopes")
        for key in keys:
            changed, count = packed_transition_count(za[key], zb[key])
            prefix = key.removesuffix(".packed_codes")
            sa = za[prefix + ".scales"].astype(np.float64)
            sb = zb[prefix + ".scales"].astype(np.float64)
            if sa.shape != sb.shape or not np.isfinite(sa).all() or not np.isfinite(sb).all():
                raise ValueError("invalid scales")
            rows.append({"layer": prefix, "changed_codes": changed, "code_count": count,
                         "changed_scales": int(np.count_nonzero(sa != sb)),
                         "scale_count": sa.size,
                         "scale_relative_l2": float(np.linalg.norm(sb-sa) / max(np.linalg.norm(sa), 1e-30))})
    total = sum(row["code_count"] for row in rows)
    changed = sum(row["changed_codes"] for row in rows)
    return {"source": fingerprint(a), "destination": fingerprint(b), "scope_count": len(rows),
            "code_count": total, "changed_codes": changed, "code_change_fraction": changed / total,
            "layers_with_code_changes": sum(row["changed_codes"] > 0 for row in rows),
            "scale_change_fraction": sum(row["changed_scales"] for row in rows) / sum(row["scale_count"] for row in rows),
            "layers": rows}


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a = reference.astype(np.float64).ravel()
    b = candidate.astype(np.float64).ravel()
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"cosine": float(np.dot(a, b) / max(an*bn, 1e-30)),
            "relative_l2": float(np.linalg.norm(b-a) / max(an, 1e-30)),
            "rms_ratio": bn / max(an, 1e-30)}


def bias_probe(root: Path, scratch: Path) -> dict:
    import train_ternary_quality as tq

    mx, nn = tq.mx, tq.nn
    model = nn.Module()
    model.transformer = nn.Module()
    block = nn.Module()
    block.ff = nn.Module()
    block.ff.ff = [None, None, nn.Linear(32, 2, bias=True)]
    model.transformer.layers = [block]
    linear = block.ff.ff[2]
    linear.weight = mx.full((2, 32), 0.25, dtype=mx.float32)
    linear.bias = mx.zeros((2,), dtype=mx.float32)
    record = tq.quantize_symmetric_weight(np.asarray(linear.weight), group_size=32)
    prefix = "transformer.layers.0.ff.ff.2"
    # Controlled bias-only update. It is not a claimed value from a real run.
    linear.bias = mx.full((2,), 0.125, dtype=mx.float32)
    record = replace(record, linear_bias=np.asarray(linear.bias, dtype=np.float16))
    expected = np.asarray(linear(mx.zeros((1, 32))), dtype=np.float32)
    scratch.mkdir(parents=True, exist_ok=False)
    checkpoint = scratch / "records_checkpoint.npz"
    tq.save_records_checkpoint(checkpoint, {prefix: record}, 1, 32, 128, "symmetric")
    records, _ = tq.load_records_checkpoint(checkpoint)
    # The records materializer starts from the unchanged dense teacher bias.
    linear.bias = mx.zeros((2,), dtype=mx.float32)
    tq.apply_records_to_model(model, records, 32)
    actual = np.asarray(model.transformer.layers[0].ff.ff[2](mx.zeros((1, 32))), dtype=np.float32)
    with np.load(checkpoint, allow_pickle=False) as saved:
        keys = saved.files

    run = root / LINEAGE[-1]
    artifact = run / "core_g32_blocks0-3_sftvoices_student16.npz"
    teacher = tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    rows = []
    with np.load(teacher, allow_pickle=False) as a, np.load(artifact, allow_pickle=False) as b:
        for block_id in range(4):
            for name in ("ff.ff.0.proj.bias", "ff.ff.2.bias"):
                key = f"transformer.layers.{block_id}.{name}"
                av, bv = a[key], b[key]
                rows.append({"key": key, "elements": av.size,
                             "equal_to_teacher": bool(np.array_equal(av, bv))})
    return {"synthetic_update_not_real_training": True,
            "synthetic_before": expected.tolist(), "synthetic_records_reload": actual.tolist(),
            "synthetic_max_abs_gap": float(np.max(np.abs(expected-actual))),
            "saved_keys": keys, "latest_artifact_ffn_biases": rows,
            "limitation": "Real bias update magnitudes are unrecoverable: only records were saved."}


def timestep_probe(root: Path, max_metal_bytes: int) -> dict:
    import train_ternary_quality as tq
    from audit_ternary_quality import load_student
    from train_ternary_window_v6 import load_state_cache

    mx = tq.mx
    cache = root / "state-cache-sftvoices-2048-seed-9042027"
    states, sigmas, indices, _sources, crosses, global_cond, manifest = load_state_cache(cache)
    with np.load(cache / "states.npz", allow_pickle=False) as archive:
        sources = archive["sources"]
    prompts = manifest["cache"]["prompt_index"]
    chosen = sorted({0, len(prompts)//2, len(prompts)-1})
    selected = []
    for prompt_id in chosen:
        candidates = np.flatnonzero((sources == "teacher_trajectory") & (indices == prompt_id))
        if len(candidates) < 8:
            raise ValueError("missing complete eight-step train trajectory")
        selected.extend(candidates[:8].tolist())

    teacher_path = tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    teacher = tq.dit_mlx_medium.DiT(T_lat=128)
    teacher.load_weights(str(teacher_path), strict=False)
    teacher.freeze()
    run = root / LINEAGE[-1]
    artifact = run / "core_g32_blocks0-3_sftvoices_student16.npz"
    artifact_manifest = json.loads(artifact.with_suffix(".json").read_text())
    student = load_student(artifact, artifact_manifest, 128)

    def predict(model, x, sigma, dtype, cross):
        result = model(x, mx.array([sigma], dtype=dtype), cross, global_cond)
        mx.eval(result)
        if mx.get_peak_memory() > max_metal_bytes:
            raise RuntimeError("timestep probe exceeded memory guard")
        return np.asarray(result, dtype=np.float32)

    rows = []
    for index in selected:
        x = mx.array(states[index][None], dtype=mx.float16)
        x32 = np.asarray(x, dtype=np.float32)
        sigma = float(sigmas[index])
        cross = crosses[int(indices[index])]
        teacher32 = predict(teacher, x, sigma, mx.float32, cross)
        teacher16 = predict(teacher, x, sigma, mx.float16, cross)
        student32 = predict(student, x, sigma, mx.float32, cross)
        student16 = predict(student, x, sigma, mx.float16, cross)
        d_teacher = x32 - np.float32(sigma)*teacher32
        d_student = x32 - np.float32(sigma)*student32
        tf32 = teacher.timestep_features(mx.array([sigma], dtype=mx.float32))
        tf16 = teacher.timestep_features(mx.array([sigma], dtype=mx.float16))
        mx.eval(tf32, tf16)
        row = {"cache_index": index, "prompt": prompts[int(indices[index])],
               "sigma_fp32": sigma, "sigma_fp16": float(np.float16(sigma)),
               "fourier_precision_only": metrics(np.asarray(tf32), np.asarray(tf16)),
               "teacher_precision_only": metrics(teacher32, teacher16),
               "student_vs_teacher_fp32_t": metrics(teacher32, student32),
               "student_vs_teacher_fp16_t": metrics(teacher16, student16),
               "denoised_student_vs_teacher_fp32_t": metrics(d_teacher, d_student),
               "denoised_error_amplification": float(sigma*np.linalg.norm(teacher32)/max(np.linalg.norm(d_teacher), 1e-30))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"scope": "24 cached TRAIN states; 3 prompts; not a generalization or listening test",
            "metric_arithmetic": "float32 denoising, float64 reductions; not an audio score",
            "cache_manifest": fingerprint(cache / "manifest.json"),
            "teacher": fingerprint(teacher_path), "student": fingerprint(artifact),
            "runtime": fingerprint(tq.MLX_RUNTIME_ROOT / "models/defs/sa3_pipeline.py"),
            "rows": rows, "memory": tq.memory_snapshot()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/sample-expertise-pilot/ternary-quality-v6-20260923"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-timesteps", action="store_true")
    parser.add_argument("--probe-biases", action="store_true")
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    started = time.time()
    rows = [compare_records(args.root/a/"records_checkpoint.npz", args.root/b/"records_checkpoint.npz")
            for a, b in zip(LINEAGE, LINEAGE[1:])]
    report = {"schema": "onus.ternary/v7-forensics", "created_at_unix": time.time(),
              "diagnostic_script": fingerprint(Path(__file__)),
              "limitations": ["Endpoint code equality does not rule out intermediate flips followed by reversals.",
                              "V6 master weights and optimizer states were not preserved by the window trainer."],
              "code_transitions": rows}
    if args.probe_timesteps:
        report["timestep_probe"] = timestep_probe(args.root, args.max_metal_bytes)
    if args.probe_biases:
        report["bias_probe"] = bias_probe(args.root, args.output.parent / (args.output.stem + "-bias-fixture"))
    report["elapsed_seconds"] = time.time()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
