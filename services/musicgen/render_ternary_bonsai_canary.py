#!/usr/bin/env python3
"""Render one matched teacher/student audio canary from a record checkpoint.

This is a technical audio gate, not a claim of perceptual equivalence.  The
teacher and student run in separate processes so the DiT and SAME-L decoder do
not silently exceed the local memory budget together.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


THIS_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
RUNTIME_SCRIPTS = RUNTIME_ROOT / "scripts"


def _runtime_sys_path() -> None:
    sys.path = [str(RUNTIME_ROOT), str(RUNTIME_SCRIPTS), str(THIS_DIR)] + [
        path for path in sys.path if "musicgen" not in path and "abelton" not in path
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--worker", choices=("teacher", "student"))
    parser.add_argument("--output-wav", type=Path)
    parser.add_argument("--output-latent", type=Path)
    return parser.parse_args()


def worker(args: argparse.Namespace) -> int:
    _runtime_sys_path()
    import mlx.core as mx

    from models.defs import dit_mlx_medium
    from models.defs.sa3_pipeline import (
        apply_prompt_padding,
        build_pingpong_schedule,
        load_conditioner_from_npz,
        patched_decode,
        sample_flow_pingpong,
    )
    from models.defs.t5gemma_mlx import T5Gemma
    from sa3_mlx import T5GEMMA_NPZ_REL, load_decoder, save_wav
    from train_ternary_quality import load_samples
    from ternary_runtime_contract import timestep_tensor
    from weights import ensure_local

    if args.worker is None or args.output_wav is None or args.output_latent is None:
        raise ValueError("worker mode requires --worker, --output-wav and --output-latent")
    samples = load_samples(args.dataset_dir, max_samples=0)
    sample = samples[args.sample_index % len(samples)]
    model = dit_mlx_medium.DiT(T_lat=args.crop_len)
    model.load_weights(str(args.teacher_weights), strict=False)
    if args.worker == "student":
        import train_ternary_quality as tq

        records, metadata = tq.load_records_checkpoint(args.records_checkpoint)
        model = tq.apply_records_to_model(model, records, int(metadata["group_size"]))
    model.freeze()

    padding_emb, seconds_embedder = load_conditioner_from_npz(
        str(args.teacher_weights), prefix="cond."
    )
    sec_tok = seconds_embedder(args.seconds).astype(mx.float16)
    global_cond = sec_tok[:, 0, :]
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    emb, mask = t5.encode([sample["prompt"]], max_len=256)
    padded = apply_prompt_padding(
        emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)
    ).astype(mx.float16)
    cross = mx.concatenate([padded, sec_tok], axis=1)
    mx.eval(global_cond, cross)

    noise = mx.random.normal(
        (1, 256, args.crop_len), dtype=mx.float16, key=mx.random.key(args.seed)
    )
    sigmas = build_pingpong_schedule(args.steps, sigma_max=1.0, use_logsnr_shift=True)
    mx.eval(noise, sigmas)

    def model_fn(x: mx.array, t: mx.array) -> mx.array:
        # Keep the explicit FP32 timestep contract for the pointwise canary;
        # the sampler itself uses its production transition arithmetic.
        return model(x, timestep_tensor(float(t)), cross, global_cond)

    latents = sample_flow_pingpong(model_fn, noise, sigmas, seed=args.seed)
    mx.eval(latents)
    decoder, chunk_fn, (chunk, overlap) = load_decoder("same-l", mx.float32)
    patches = chunk_fn(decoder, latents.astype(mx.float32), chunk, overlap)
    audio = patched_decode(patches, patch_size=256, channels=2)
    mx.eval(audio)
    audio_np = np.asarray(audio[0], dtype=np.float32)
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    args.output_latent.parent.mkdir(parents=True, exist_ok=True)
    save_wav(str(args.output_wav), audio_np)
    np.save(args.output_latent, np.asarray(latents[0], dtype=np.float32))
    report = {
        "role": args.worker,
        "prompt": sample["prompt"],
        "input_shape": [1, 256, args.crop_len],
        "latent_shape": list(np.asarray(latents).shape),
        "audio_shape": list(audio_np.shape),
        "sample_rate": 44100,
        "audio_finite": bool(np.isfinite(audio_np).all()),
        "peak": float(np.max(np.abs(audio_np))),
        "rms": float(np.sqrt(np.mean(audio_np * audio_np))),
        "duration_seconds": float(audio_np.shape[-1] / 44100),
        "timestep_dtype": str(timestep_tensor(float(sigmas[0])).dtype),
        "wav": str(args.output_wav),
        "latent": str(args.output_latent),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker is not None:
        return worker(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ternary-bonsai-audio-") as temp_dir:
        temp = Path(temp_dir)
        reports: dict[str, dict] = {}
        for role in ("teacher", "student"):
            wav = args.out_dir / f"{role}.wav"
            latent = temp / f"{role}.npy"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                role,
                "--records-checkpoint",
                str(args.records_checkpoint),
                "--teacher-weights",
                str(args.teacher_weights),
                "--dataset-dir",
                str(args.dataset_dir),
                "--out-dir",
                str(args.out_dir),
                "--crop-len",
                str(args.crop_len),
                "--seconds",
                str(args.seconds),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--sample-index",
                str(args.sample_index),
                "--output-wav",
                str(wav),
                "--output-latent",
                str(latent),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            reports[role] = json.loads(lines[-1])
            reports[role]["stdout_tail"] = completed.stdout[-2000:]

        teacher_latent = np.load(temp / "teacher.npy")
        student_latent = np.load(temp / "student.npy")
        teacher_audio, teacher_sr = sf.read(args.out_dir / "teacher.wav", dtype="float32", always_2d=True)
        student_audio, student_sr = sf.read(args.out_dir / "student.wav", dtype="float32", always_2d=True)
        latent_delta = teacher_latent.astype(np.float64) - student_latent.astype(np.float64)
        audio_length = min(len(teacher_audio), len(student_audio))
        audio_delta = teacher_audio[:audio_length] - student_audio[:audio_length]
        teacher_flat = teacher_audio[:audio_length].reshape(-1).astype(np.float64)
        student_flat = student_audio[:audio_length].reshape(-1).astype(np.float64)
        audio_cosine = float(
            np.dot(teacher_flat, student_flat)
            / max(np.linalg.norm(teacher_flat) * np.linalg.norm(student_flat), 1e-12)
        )
        result = {
            "schema": "onus.ternary-quality/v9-audio-canary",
            "records_checkpoint": str(args.records_checkpoint),
            "sample_rate_match": teacher_sr == student_sr == 44100,
            "teacher": reports["teacher"],
            "student": reports["student"],
            "latent_relative_l2": float(
                np.linalg.norm(latent_delta) / max(np.linalg.norm(teacher_latent), 1e-12)
            ),
            "audio_relative_l2": float(
                np.linalg.norm(audio_delta.astype(np.float64))
                / max(np.linalg.norm(teacher_audio[:audio_length].astype(np.float64)), 1e-12)
            ),
            "audio_cosine": audio_cosine,
            "finite": bool(np.isfinite(teacher_audio).all() and np.isfinite(student_audio).all()),
            "technical_pass": bool(
                teacher_sr == student_sr == 44100
                and np.isfinite(teacher_audio).all()
                and np.isfinite(student_audio).all()
                and teacher_audio.shape == student_audio.shape
            ),
        }
    report_path = args.out_dir / "audio_canary.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["technical_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

