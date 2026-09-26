"""Encode sftminimal audio tracks into 12s SAME-L latents for distillation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import numpy as np
import soundfile as sf

from models.defs.audio_encoding import encode_audio
from sa3_mlx import load_encoder

SAMPLE_RATE = 44100
TARGET_SAMPLES = 528384  # exactly 129 latents (128 * 4096 + 4096 = 528384)


def load_audio_resampled(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = data.T
    y = np.stack([y[0], y[0]]) if y.shape[0] == 1 else y[:2]
    if sr != SAMPLE_RATE:
        in_len = y.shape[-1]
        new_len = int(round(in_len * SAMPLE_RATE / sr))
        scale = in_len / new_len
        pos = np.clip((np.arange(new_len) + 0.5) * scale - 0.5, 0, in_len - 1)
        y = np.stack([np.interp(pos, np.arange(in_len), ch) for ch in y])
    return np.ascontiguousarray(y, dtype=np.float32)


def main():
    audio_dir = Path("sftminimal")
    output_dir = Path("output/sample-expertise-pilot/sftminimal/latents-12s")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = sorted([p for p in audio_dir.glob("*.mp3") if not p.name.startswith("._")])
    print(f"Found {len(tracks)} tracks in {audio_dir}")

    print("Loading SAME-L encoder...")
    encoder, pad_mod = load_encoder("same-l", mx.float32)

    clip_count = 0
    t0 = time.time()

    for idx, track_path in enumerate(tracks, start=1):
        try:
            audio = load_audio_resampled(track_path)
            total_samples = audio.shape[-1]
            if total_samples < TARGET_SAMPLES:
                continue

            # Extract 2 distinct 12s segments: at 35% and 65% of track duration (groove sections)
            pos_1 = int(total_samples * 0.35)
            pos_2 = int(total_samples * 0.65)
            cuts = [pos_1, pos_2]

            clean_name = track_path.stem.split("[")[0].strip().replace(" ", "-").lower()

            for cut_idx, start_sample in enumerate(cuts, start=1):
                end_sample = min(start_sample + TARGET_SAMPLES, total_samples)
                chunk = audio[:, start_sample:end_sample]
                if chunk.shape[-1] < TARGET_SAMPLES:
                    chunk = np.pad(chunk, ((0, 0), (0, TARGET_SAMPLES - chunk.shape[-1])))

                clip_id = f"{idx:03d}_{cut_idx}_{clean_name[:30]}"
                npy_path = output_dir / f"{clip_id}.npy"
                json_path = output_dir / f"{clip_id}.json"

                if npy_path.is_file() and json_path.is_file():
                    clip_count += 1
                    continue

                encoded = encode_audio(
                    encoder,
                    chunk[None, ...],
                    valid_sample_lengths=[TARGET_SAMPLES],
                    pad_modulo=16,
                    chunked=False,
                )
                latents_np = np.asarray(encoded.latents.astype(mx.float32))[0]
                padding_mask = [int(v) for v in np.asarray(encoded.padding_mask)[0]]

                np.save(str(npy_path), latents_np)

                meta = {
                    "path": str(track_path.resolve()),
                    "relpath": f"{clip_id}.npy",
                    "src_relpath": track_path.name,
                    "seconds_total": round(TARGET_SAMPLES / SAMPLE_RATE, 3),
                    "seconds_start": round(start_sample / SAMPLE_RATE, 3),
                    "audio_samples": TARGET_SAMPLES,
                    "latent_shape": list(latents_np.shape),
                    "padding_mask": padding_mask,
                    "prompt": f"sftberlin; techno, minimal-techno; 127 BPM; minimal groove, clean kick, sharp percussions, {clean_name}",
                    "genre": "minimal-techno",
                    "sft_trigger": "sftberlin",
                    "source_annotation_status": "provided_unreviewed",
                    "training_intent": "style_and_mix_reference",
                    "bpm": "127",
                }
                json_path.write_text(json.dumps(meta, indent=2))
                clip_count += 1
                print(f"  [{clip_count:02d}] Encoded {clip_id} -> {latents_np.shape}")

        except Exception as e:
            print(f"  Error on {track_path.name}: {e}")

    dt = time.time() - t0
    print(f"\n[Done] Encoded {clip_count} clips in {dt:.1f}s -> {output_dir}")


if __name__ == "__main__":
    main()
