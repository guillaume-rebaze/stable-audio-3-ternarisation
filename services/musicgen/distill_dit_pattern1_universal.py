"""Universal High-Fidelity Distillation for DiT Medium Pattern 1 (< 500 MB).

Architecture:
- FF (24 layers): INT2 affine (group 128)
- Cross-Attn (24 layers): INT2 affine (group 128)
- Local-Embed: INT2 affine (group 128)
- Self-Attn: INT4 affine on boundary blocks (0..5, 18..23), INT2 on central blocks (6..17)
- Outer & Modulators: FP16
Total disk size: 492.13 MB (< 500 MB).

Distillation targets:
- Flow matching velocity loss (MSE + Cosine Direction + Norm Calibration)
- Trained on 193 real music tracks across all genres.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
from pathlib import Path
import random
import sys
import time

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz, sample_flow_pingpong, patched_decode
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local
from sa3_mlx import T5GEMMA_NPZ_REL, load_decoder, save_wav


def load_dataset(dataset_dir: Path, max_tracks: int = 193):
    npy_files = sorted(glob.glob(str(dataset_dir / "*.npy")))[:max_tracks]
    print(f"Loading {len(npy_files)} audio latents from {dataset_dir}...")
    items = []
    for f in npy_files:
        npy_path = Path(f)
        json_path = npy_path.with_suffix(".json")
        prompt = "dynamic musical piece, studio production"
        if json_path.exists():
            with open(json_path) as jf:
                data = json.load(jf)
                prompt = data.get("prompt", data.get("genre", prompt))
        lat = np.load(npy_path)
        items.append({"path": npy_path, "prompt": prompt, "latent": lat})
    return items


def preencode_prompts(items, teacher_path: Path):
    print("Pre-encoding prompts with T5Gemma...")
    padding_emb, secs_embedder = load_conditioner_from_npz(str(teacher_path), prefix="cond.")
    sec_tok = secs_embedder(12.0).astype(mx.float16)
    global_cond_val = sec_tok[:, 0, :]
    
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    cached = []
    for i, item in enumerate(items):
        emb, mask = t5.encode([item["prompt"]], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)).astype(mx.float16)
        cross_full = mx.concatenate([padded, sec_tok], axis=1)
        lat_arr = mx.array(item["latent"]).astype(mx.float16)
        if lat_arr.ndim == 2:
            lat_arr = lat_arr[None]
        cached.append((cross_full, global_cond_val, lat_arr))
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            print(f"  Encoded {i + 1}/{len(items)} prompts")
    
    del t5
    gc.collect()
    print("T5Gemma evicted from memory.")
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out-model", type=str, default="output/sample-expertise-pilot/universal-models/dit_medium_pattern1_distilled_492mb.npz")
    args = parser.parse_args()

    dataset_dir = Path("output/sample-expertise-pilot/universal-dataset/latents-12s")
    items = load_dataset(dataset_dir)

    teacher_path = MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    cached_data = preencode_prompts(items, teacher_path)

    # 1. Load Teacher (Frozen FP16)
    print("Loading Teacher model (dit_medium_f16.npz)...")
    teacher = dit_mlx_medium.DiT(T_lat=129)
    teacher.load_weights(str(teacher_path), strict=False)
    teacher.freeze()

    # 2. Load Student Pattern 1
    print("Loading Student Pattern 1 model...")
    student = dit_mlx_medium.DiT(T_lat=129)
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

    nn.quantize(student, bits=2, group_size=128, mode="affine", class_predicate=pred_int2)
    nn.quantize(student, bits=4, group_size=128, mode="affine", class_predicate=pred_int4)
    student.load_weights("output/sample-expertise-pilot/universal-models/dit_medium_pattern1_492mb.npz", strict=True)

    # 3. Setup optimizer & gradient filtering for continuous parameters
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    def filter_grads(g):
        filtered = {}
        for k, v in g.items():
            if isinstance(v, dict):
                sub = filter_grads(v)
                if sub: filtered[k] = sub
            elif any(w in k for w in ["to_scale_shift_gate", "project_in", "project_out", "preprocess_conv", "postprocess_conv", "bias", "norm"]):
                filtered[k] = v
        return filtered

    print(f"\nStarting universal velocity distillation ({args.steps} steps)...")
    t_start = time.time()

    for step in range(args.steps):
        sample_idx = step % len(cached_data)
        cross_full, global_cond_val, x_0 = cached_data[sample_idx]
        
        if x_0.shape[-1] != 129:
            x_0 = x_0[:, :, :129]

        t_val = float(0.05 + 0.90 * random.random())
        t = mx.array([t_val], dtype=mx.float16)

        key = mx.random.key(step * 17 + 42)
        noise = mx.random.normal(x_0.shape, dtype=mx.float16, key=key)
        x_t = (1.0 - t) * x_0 + t * noise

        vt = teacher(x_t, t, cross_full, global_cond_val)
        mx.eval(vt)

        norm_vt = mx.sqrt(mx.sum(vt**2)) + 1e-6

        def loss_fn(model):
            vs = model(x_t, t, cross_full, global_cond_val)
            norm_vs = mx.sqrt(mx.sum(vs**2)) + 1e-6
            
            mse = mx.mean((vs - vt)**2)
            cos = mx.sum(vs * vt) / (norm_vs * norm_vt)
            cos_loss = 1.0 - cos
            norm_loss = ((norm_vs - norm_vt) / norm_vt)**2
            
            total_loss = mse + 3.0 * cos_loss + 2.0 * norm_loss
            return total_loss, (mse, cos, norm_vs / norm_vt)

        loss_and_grad = nn.value_and_grad(student, loss_fn)
        (loss, (mse, cos, norm_ratio)), grads = loss_and_grad(student)
        fg = filter_grads(grads)
        optimizer.update(student, fg)
        mx.eval(student.parameters(), optimizer.state)

        if step % 15 == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"Step {step:3d}/{args.steps} | Loss: {float(loss):.4f} | MSE: {float(mse):.4f} | CosSim: {float(cos):.4f} | NormRatio: {float(norm_ratio):.4f} | Elapsed: {elapsed:.1f}s")

    # 4. Save Final Distilled Model
    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat_params = dict(tree_flatten(student.parameters()))
    mx.savez(str(out_path), **flat_params)
    size_mb = out_path.stat().st_size / (1024**2)
    print(f"\n[Saved] Distilled Model -> {out_path} ({size_mb:.2f} MB)")
    assert size_mb < 500.0, f"Model size {size_mb} MB exceeds 500 MB limit!"

    json_path = out_path.with_suffix(".json")
    manifest = {
        "model": "dit_medium_pattern1_distilled_492mb",
        "size_mb": size_mb,
        "format": "hybrid_int2_int4_affine_mlx",
        "steps": args.steps,
        "lr": args.lr,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Saved] Manifest -> {json_path}")


if __name__ == "__main__":
    main()
