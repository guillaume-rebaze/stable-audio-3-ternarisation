"""Render 30s audio evaluations for Bonsai Pure Ternary DiT Medium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz, sample_flow_pingpong, patched_decode
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local
from sa3_mlx import T5GEMMA_NPZ_REL, load_decoder, save_wav


PROMPTS = {
    "piano": "intimate solo acoustic upright piano, melancholic chords, warm room reverb",
    "funk": "70s funk groove with slap bass, wah-wah guitar, punchy drums",
    "ambient": "cinematic ambient drone with lush shimmering pads, deep sub-bass, slow evolving textures"
}


def compute_spectral_correlation(audio1: np.ndarray, audio2: np.ndarray, sr: int = 44100) -> float:
    min_len = min(len(audio1), len(audio2))
    a1, a2 = audio1[:min_len], audio2[:min_len]
    _, _, s1 = spectrogram(a1, fs=sr, nperseg=2048, noverlap=1024)
    _, _, s2 = spectrogram(a2, fs=sr, nperseg=2048, noverlap=1024)
    l1 = np.log1p(s1).flatten()
    l2 = np.log1p(s2).flatten()
    l1 = l1 - np.mean(l1)
    l2 = l2 - np.mean(l2)
    norm1 = np.sqrt(np.sum(l1**2))
    norm2 = np.sqrt(np.sum(l2**2))
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.sum(l1 * l2) / (norm1 * norm2))


def main():
    parser = argparse.ArgumentParser(description="Render audio with Bonsai pure ternary model.")
    parser.add_argument("--model-path", type=str, default="output/sample-expertise-pilot/universal-models/dit_medium_bonsai_ternary_456mb.npz")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, default="output/sample-expertise-pilot/bonsai-ternary-renders")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"

    print("=" * 70)
    print("RENDER EVALUATION: BONSAI PURE TERNARY {-1, 0, +1} DiT")
    print(f"Model:    {args.model_path}")
    print(f"Duration: {args.seconds}s, Steps: {args.steps}")
    print("=" * 70)

    # 1. Load conditioning
    print("[1/4] Loading conditioner & text prompt encoder...")
    padding_emb, secs_embedder = load_conditioner_from_npz(str(teacher_path), prefix="cond.")
    sec_tok = secs_embedder(args.seconds).astype(mx.float16)
    global_cond_val = sec_tok[:, 0, :]

    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    prompt_embs = {}
    for genre, p_text in PROMPTS.items():
        emb, mask = t5.encode([p_text], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)).astype(mx.float16)
        prompt_embs[genre] = mx.concatenate([padded, sec_tok], axis=1)
    del t5

    # 2. Load Decoder
    print("[2/4] Loading VAE decoder...")
    decoder, chunk_fn, (chunk, ovl) = load_decoder("same-l", mx.float32)

    # 3. Load Bonsai Pure Ternary Model
    print("[3/4] Loading Bonsai pure ternary DiT...")
    T_lat = int(round(args.seconds * 44100 / 4096))
    student = dit_mlx_medium.DiT(T_lat=T_lat)

    # Apply 2-bit affine quantization for all layers with group_size 128
    def pred_ternary(path, mod):
        if hasattr(mod, "weight") and mod.weight.ndim == 2 and mod.weight.shape[1] % 128 == 0:
            if "layers." in path:
                return True
        return False

    nn.quantize(student, bits=2, group_size=128, mode="affine", class_predicate=pred_ternary)
    student.load_weights(args.model_path, strict=True)
    mx.eval(student.parameters())
    print("  ✓ Model loaded into MLX QuantizedLinear 2-bit successfully.")

    # 4. Generate audio per genre
    print("[4/4] Generating 30s audio clips...")
    sigmas = build_pingpong_schedule(args.steps, sigma_max=1.0, use_logsnr_shift=True)
    results = {}

    for genre, prompt_text in PROMPTS.items():
        print(f"\n--- Generating: {genre.upper()} ({args.seconds}s, {args.steps} steps) ---")
        t0 = time.time()
        cross_full = prompt_embs[genre]

        def sample_fn(x, t):
            return student(x, t, cross_full, global_cond_val)

        noise = mx.random.normal((1, 256, T_lat), dtype=mx.float16, key=mx.random.key(42))
        latents = sample_flow_pingpong(sample_fn, noise, sigmas, seed=42)
        mx.eval(latents)

        lat_std = float(mx.std(latents))
        print(f"  Latent std: {lat_std:.4f}")

        # Decode
        patches = chunk_fn(decoder, latents.astype(mx.float32), chunk, ovl)
        audio = patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)
        gen_time = time.time() - t0

        audio_np = np.array(audio[0, 0]).astype(np.float32)
        peak_raw = float(np.max(np.abs(audio_np)))
        rms_raw = float(np.sqrt(np.mean(audio_np**2)))

        # Mastering peak normalization
        if peak_raw > 1e-4:
            audio_norm = audio_np * (0.85 / peak_raw)
        else:
            audio_norm = audio_np

        wav_path = out_dir / f"bonsai_ternary_{genre}_30s.wav"
        sf.write(str(wav_path), audio_norm, 44100)

        # Compare vs Teacher Base
        teacher_wav_path = Path(f"output/sample-expertise-pilot/teacher_base_{genre}_30s.wav")
        spec_corr = 0.0
        if teacher_wav_path.exists():
            t_audio, _ = sf.read(str(teacher_wav_path))
            spec_corr = compute_spectral_correlation(audio_norm, t_audio)

        print(f"  Raw Peak:   {peak_raw:.4f}")
        print(f"  Raw RMS:    {rms_raw:.4f}")
        print(f"  SpecCorr:   {spec_corr:.4f} vs Teacher FP16")
        print(f"  Time:       {gen_time:.1f}s")
        print(f"  Saved:      {wav_path}")

        results[genre] = {
            "peak_raw": peak_raw,
            "rms_raw": rms_raw,
            "spectral_correlation": spec_corr,
            "gen_time": gen_time,
            "wav_path": str(wav_path)
        }

    summary_path = out_dir / "bonsai_ternary_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
