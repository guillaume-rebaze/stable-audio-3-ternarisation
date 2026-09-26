"""Generate 30-second studio audio samples using distilled Pattern 1 model (< 500 MB)."""

import argparse
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
from scipy.signal import welch

from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz, sample_flow_pingpong, patched_decode
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local
from sa3_mlx import T5GEMMA_NPZ_REL, load_decoder, save_wav

PROMPTS = [
    ("piano", "A beautiful acoustic grand piano melody, emotive classical piece, concert hall reverb"),
    ("funk", "70s funk groove with slap bass, wah-wah guitar, punchy acoustic drums and brass section"),
    ("ambient", "Deep cinematic ambient soundscape, evolving analog synth pads, ethereal reverb, floating melody")
]

def main():
    out_dir = Path("output/sample-expertise-pilot/universal-models/final-audios-30s")
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    padding_emb, secs_embedder = load_conditioner_from_npz(str(teacher_path), prefix="cond.")
    sec_tok = secs_embedder(30.0).astype(mx.float16)
    global_cond_val = sec_tok[:, 0, :]

    print("Loading T5Gemma...")
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))

    conds = {}
    for name, prompt in PROMPTS:
        emb, mask = t5.encode([prompt], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)).astype(mx.float16)
        cross_full = mx.concatenate([padded, sec_tok], axis=1)
        conds[name] = cross_full
    del t5

    print("Loading Distilled Pattern 1 model...")
    model = dit_mlx_medium.DiT(T_lat=323)
    int4_blocks = set(range(0, 6)) | set(range(18, 24))

    def pred_int2(p, l):
        if not (hasattr(l, "weight") and l.weight.ndim == 2 and l.weight.shape[1] % 128 == 0): return False
        if "ff.ff" in p or "cross_attn" in p or "to_local_embed" in p: return True
        if "self_attn" in p:
            b_idx = int(p.split(".")[2]) if "layers." in p else -1
            return b_idx not in int4_blocks
        return False

    def pred_int4(p, l):
        if not (hasattr(l, "weight") and l.weight.ndim == 2 and l.weight.shape[1] % 128 == 0): return False
        if "self_attn" in p:
            b_idx = int(p.split(".")[2]) if "layers." in p else -1
            return b_idx in int4_blocks
        return False

    nn.quantize(model, bits=2, group_size=128, mode="affine", class_predicate=pred_int2)
    nn.quantize(model, bits=4, group_size=128, mode="affine", class_predicate=pred_int4)
    model.load_weights("output/sample-expertise-pilot/universal-models/dit_medium_pattern1_distilled_492mb.npz", strict=True)
    model.freeze()

    sigmas = build_pingpong_schedule(24, sigma_max=1.0, use_logsnr_shift=True)
    decoder, chunk_fn, (chunk, ovl) = load_decoder("same-l", mx.float32)

    results = []
    for name, prompt in PROMPTS:
        print(f"\n[Synthesizing] {name.upper()} (30s, seed=42)...")
        cross_full = conds[name]
        key = mx.random.key(42)
        noise = mx.random.normal((1, 256, 323), dtype=mx.float16, key=key)

        def model_fn(x, t):
            return model(x, t, cross_full, global_cond_val)

        t0 = time.time()
        latents = sample_flow_pingpong(model_fn, noise, sigmas, seed=42)
        
        # Check latent std and calibrate to exact SAME-L nominal range
        raw_std = float(mx.std(latents))
        raw_rms = float(mx.sqrt(mx.mean(latents**2)))
        calib_factor = 0.7535 / raw_rms
        latents = latents * calib_factor
        mx.eval(latents)

        patches = chunk_fn(decoder, latents.astype(mx.float32), chunk, ovl)
        audio = patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)

        audio_np = np.array(audio[0])[:, :1323000]
        out_wav = str(out_dir / f"pattern1_distilled_{name}_30s.wav")
        save_wav(out_wav, audio_np)
        
        peak = float(np.max(np.abs(audio_np)))
        rms = float(np.sqrt(np.mean(audio_np**2)))
        crest = peak / (rms + 1e-6)
        dur = time.time() - t0

        print(f"  -> Generated in {dur:.1f}s: peak={peak:.3f}, rms={rms:.3f}, crest={crest:.2f}")
        results.append((name, out_wav, peak, rms, crest, raw_std))

    # Evaluate piano against teacher
    teacher_wav, _ = sf.read("output/sample-expertise-pilot/universal-models/final-audios-30s/teacher_base_piano_30s.wav")
    stud_wav, _ = sf.read(str(out_dir / "pattern1_distilled_piano_30s.wav"))
    min_l = min(len(teacher_wav), len(stud_wav))
    t_w = teacher_wav[:min_l]
    s_w = stud_wav[:min_l]
    f, P_t = welch(t_w[:, 0], 44100, nperseg=4096)
    _, P_s = welch(s_w[:, 0], 44100, nperseg=4096)
    spec_corr = float(np.corrcoef(np.log1p(P_t), np.log1p(P_s))[0, 1])

    print("\n" + "="*60)
    print("FINAL SYNTHESIS RESULTS (Distilled Pattern 1, 492 MB):")
    for name, path, peak, rms, crest, raw_std in results:
        print(f"  - {name:8s}: peak={peak:.3f} | rms={rms:.3f} | crest={crest:.2f} | latent_std={raw_std:.3f}")
    print(f"  - Spectrogram Correlation (Piano vs Teacher FP16): {spec_corr:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
