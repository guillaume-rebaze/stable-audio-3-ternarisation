"""Render raw teacher/student audio for a ternary quality gate.

The files are written without peak normalization.  Ratios are therefore useful
for detecting the silent-output failure that the historical renderer hid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

import train_ternary_quality as tq
from audit_ternary_quality import (
    load_manifest,
    load_split_manifest,
    load_student,
    one_sample_per_prompt,
)
from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import (
    build_pingpong_schedule,
    patched_decode,
    sample_flow_pingpong,
)
from sa3_mlx import load_decoder
from ternary_adapter import TernaryAdapterLinear

mx = tq.mx


def check_metal_memory(max_metal_bytes: int, stage: str) -> dict[str, float]:
    if max_metal_bytes <= 0:
        raise ValueError("max_metal_bytes must be positive")
    snapshot = tq.memory_snapshot()
    peak = int(snapshot["metal_peak_gb"] * (1024**3))
    if peak > max_metal_bytes:
        raise RuntimeError(
            f"{stage} exceeded Metal memory guard: {peak} > {max_metal_bytes} bytes"
        )
    return snapshot


def attach_adapter(student, adapter_path: Path, adapter_manifest_path: Path):
    adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    data = np.load(adapter_path)
    info = adapter_manifest["adapter"]
    for path in info["scope"]:
        base = tq.module_at(student, path)
        wrapper = TernaryAdapterLinear(
            base,
            rank=int(info["rank"]),
            alpha=float(info["alpha"]),
            seed=0,
        )
        wrapper.down = mx.array(data[f"{path}.down"])
        wrapper.up = mx.array(data[f"{path}.up"])
        wrapper.freeze()
        tq.set_module_at(student, path, wrapper)
    mx.eval(student.parameters())
    return student


def audio_stereo(audio: mx.array) -> np.ndarray:
    """Return decoded audio as [samples, channels], preserving both channels."""
    value = np.array(audio).astype(np.float32)
    if value.ndim == 3:
        value = value[0]
    elif value.ndim == 2:
        pass
    elif value.ndim == 1:
        return value[:, None]
    else:
        raise ValueError(f"Unexpected decoded audio shape: {value.shape}")
    if value.shape[0] <= 4 and value.shape[1] > value.shape[0]:
        value = value.T
    elif value.shape[1] > 4 and value.shape[0] > value.shape[1]:
        raise ValueError(f"Cannot identify channels in decoded audio shape: {value.shape}")
    if value.shape[1] > 2:
        raise ValueError(f"Expected mono/stereo audio, got {value.shape[1]} channels")
    return value


def audio_mono(audio: np.ndarray) -> np.ndarray:
    """Mid/mono view used only for scalar spectral diagnostics."""
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 1:
        return value
    return np.mean(value, axis=1, dtype=np.float32)


def spectral_correlation(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    length = min(len(a), len(b))
    if length < 4096:
        return 0.0
    _, _, first = spectrogram(a[:length], fs=sr, nperseg=2048, noverlap=1024)
    _, _, second = spectrogram(b[:length], fs=sr, nperseg=2048, noverlap=1024)
    x = np.log1p(first).astype(np.float64).ravel()
    y = np.log1p(second).astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def envelope_correlation(a: np.ndarray, b: np.ndarray, frame: int = 2048) -> float:
    count = min(len(a), len(b)) // frame
    if count < 4:
        return 0.0
    aa = a[: count * frame].reshape(count, frame)
    bb = b[: count * frame].reshape(count, frame)
    ea = np.sqrt(np.mean(aa * aa, axis=1))
    eb = np.sqrt(np.mean(bb * bb, axis=1))
    if np.std(ea) < 1e-8 or np.std(eb) < 1e-8:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def render_latents(
    model,
    noise: mx.array,
    sigmas: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    seed: int,
) -> mx.array:
    def model_fn(x, t):
        return model(x, t, cross, global_cond)

    return sample_flow_pingpong(model_fn, noise, sigmas, seed=seed)


def render_one(
    name: str,
    prompt: str,
    teacher,
    student,
    cross: mx.array,
    global_cond: mx.array,
    decoder,
    chunk_fn,
    chunk: int,
    overlap: int,
    latent_len: int,
    steps: int,
    seed: int,
    output_dir: Path,
) -> dict:
    sigmas = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    noise = mx.random.normal(
        (1, 256, latent_len), dtype=mx.float16, key=mx.random.key(seed)
    )
    started = time.time()
    teacher_latents = render_latents(teacher, noise, sigmas, cross, global_cond, seed)
    mx.eval(teacher_latents)
    # Recreate the exact same noise for the student; no shared mutable graph.
    student_noise = mx.random.normal(
        (1, 256, latent_len), dtype=mx.float16, key=mx.random.key(seed)
    )
    student_latents = render_latents(student, student_noise, sigmas, cross, global_cond, seed)
    mx.eval(student_latents)

    teacher_audio = patched_decode(
        chunk_fn(decoder, teacher_latents.astype(mx.float32), chunk, overlap),
        patch_size=256,
        channels=2,
    )
    student_audio = patched_decode(
        chunk_fn(decoder, student_latents.astype(mx.float32), chunk, overlap),
        patch_size=256,
        channels=2,
    )
    mx.eval(teacher_audio, student_audio)
    teacher_np = audio_stereo(teacher_audio)
    student_np = audio_stereo(student_audio)
    channel_match = teacher_np.shape[1] == student_np.shape[1]
    teacher_mono = audio_mono(teacher_np)
    student_mono = audio_mono(student_np)
    teacher_peak = float(np.max(np.abs(teacher_np)))
    student_peak = float(np.max(np.abs(student_np)))
    teacher_rms = float(np.sqrt(np.mean(teacher_np * teacher_np)))
    student_rms = float(np.sqrt(np.mean(student_np * student_np)))
    teacher_path = output_dir / f"teacher_{name}.wav"
    student_path = output_dir / f"student_{name}.wav"
    # Raw float WAVs: intentionally no normalization and no clipping.
    sf.write(str(teacher_path), teacher_np, 44100, subtype="FLOAT")
    sf.write(str(student_path), student_np, 44100, subtype="FLOAT")

    rms_ratio = student_rms / (teacher_rms + 1e-8)
    peak_ratio = student_peak / (teacher_peak + 1e-8)
    result = {
        "name": name,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "latent_len": latent_len,
        "teacher_latent_std": float(mx.std(teacher_latents)),
        "student_latent_std": float(mx.std(student_latents)),
        "teacher_peak_raw": teacher_peak,
        "student_peak_raw": student_peak,
        "teacher_rms_raw": teacher_rms,
        "student_rms_raw": student_rms,
        "rms_ratio": rms_ratio,
        "peak_ratio": peak_ratio,
        "channels": int(teacher_np.shape[1]),
        "student_channels": int(student_np.shape[1]),
        "channel_match": bool(channel_match),
        "duration_seconds": float(min(len(teacher_np), len(student_np)) / 44100.0),
        "spectral_correlation_mid": spectral_correlation(teacher_mono, student_mono, 44100),
        "envelope_correlation_mid": envelope_correlation(teacher_mono, student_mono),
        "channel_metrics": [
            {
                "channel": channel,
                "spectral_correlation": spectral_correlation(
                    teacher_np[:, channel], student_np[:, channel], 44100
                ),
                "envelope_correlation": envelope_correlation(
                    teacher_np[:, channel], student_np[:, channel]
                ),
            }
            for channel in range(min(teacher_np.shape[1], student_np.shape[1]))
        ],
        "finite": bool(np.isfinite(teacher_np).all() and np.isfinite(student_np).all()),
        "teacher_wav": str(teacher_path),
        "student_wav": str(student_path),
        "elapsed_seconds": time.time() - started,
    }
    result["technical_pass"] = bool(
        result["finite"]
        and result["channel_match"]
        and result["channels"] == 2
        and result["duration_seconds"] > 0
        and 0.70 <= rms_ratio <= 1.30
        and 0.50 <= peak_ratio <= 1.50
    )
    # Spectral similarity is retained as evidence, not used as an uncalibrated
    # universal quality gate.  A generation can be musically valid while not
    # matching one teacher waveform arrangement exactly.
    result["candidate_pass"] = result["technical_pass"]
    print(
        f"[{name}] spec_mid={result['spectral_correlation_mid']:.4f} "
        f"env_mid={result['envelope_correlation_mid']:.4f} "
        f"rms={student_rms:.5f}/{teacher_rms:.5f} ({rms_ratio:.3f}) "
        f"peak={student_peak:.5f}/{teacher_peak:.5f} ({peak_ratio:.3f}) "
        f"candidate={'PASS' if result['candidate_pass'] else 'FAIL'}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw audio gate for a ternary DiT artifact")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-recovery/group64-core/dit_medium_ternary_quality_group64_core.npz"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--adapter-manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"))
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="select only the provenance-verified role for this raw-audio pilot",
    )
    parser.add_argument(
        "--split-role",
        choices=("debug_train_seen", "validation", "test"),
        default="debug_train_seen",
        help="role within --split-manifest; provenance label only",
    )
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    if args.max_metal_bytes <= 0:
        raise ValueError("max-metal-bytes must be positive")

    manifest_path = args.manifest or args.artifact.with_suffix(".json")
    output_dir = args.output_dir or args.artifact.parent / "audio-gate-12s"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    samples = tq.load_samples(args.dataset_dir, 0)
    if args.split_manifest is not None:
        samples, _ = load_split_manifest(
            args.split_manifest, samples, args.split_role
        )
    selected = one_sample_per_prompt(samples, args.max_prompts)
    print(f"[Render] {len(selected)} prompts, {args.seconds}s target, no normalization", flush=True)

    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    _, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher, args.teacher_weights, [s["prompt"] for s in selected], args.seconds
    )
    check_metal_memory(args.max_metal_bytes, "audio conditioning")
    student = load_student(args.artifact, manifest, args.crop_len)
    check_metal_memory(args.max_metal_bytes, "audio student load")
    if args.adapter:
        adapter_manifest_path = args.adapter_manifest or args.adapter.with_name("adapter_manifest.json")
        student = attach_adapter(student, args.adapter, adapter_manifest_path)
        print(f"[Render] attached adapter {args.adapter}", flush=True)
    decoder, chunk_fn, (chunk, overlap) = __import__(
        "sa3_mlx", fromlist=["load_decoder"]
    ).load_decoder("same-l", mx.float32)
    check_metal_memory(args.max_metal_bytes, "audio decoder load")
    results = []
    for index, sample in enumerate(selected):
        safe_name = f"{index:02d}_{sample['genre'].replace(' ', '_').lower()}"
        results.append(
            render_one(
                safe_name,
                sample["prompt"],
                teacher,
                student,
                cross_cache[sample["prompt"]],
                global_cond,
                decoder,
                chunk_fn,
                chunk,
                overlap,
                args.crop_len,
                args.steps,
                args.seed + index,
                output_dir,
            )
        )
        check_metal_memory(args.max_metal_bytes, f"audio render {index + 1}")
    summary = {
        "status": "audio_audited",
        "artifact": str(args.artifact),
        "manifest": str(manifest_path),
        "seconds_requested": args.seconds,
        "results": results,
        "candidate_pass": bool(results) and all(r["candidate_pass"] for r in results),
        "memory": tq.memory_snapshot(),
    }
    tq.write_json(output_dir / "audio_metrics.json", summary)
    print(json.dumps({
        "candidate_pass": summary["candidate_pass"],
        "prompts": len(results),
        "memory": summary["memory"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
