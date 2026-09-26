"""Progressive High-Fidelity QAT Distillation for DiT Medium (< 500 MB).

Real QAT with FP32 master weights & Straight-Through Estimator (STE) across all
1.425 billion transformer linear parameters.
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
from mlx.utils import tree_flatten, tree_map_with_path
import numpy as np

from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz, sample_flow_pingpong, patched_decode
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local
from sa3_mlx import T5GEMMA_NPZ_REL, load_decoder, save_wav


class INT2QATLinear(nn.Module):
    """BitNet 1.58-bit / affine ternary linear with FP32 master weights and STE."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False, group_size: int = 128):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.group_size = group_size
        self.weight = mx.zeros((out_d, in_d), dtype=mx.float32)
        self.bias = mx.zeros((out_d,), dtype=mx.float32) if bias else None

    @classmethod
    def from_linear(cls, layer: nn.Linear, group_size: int = 128) -> INT2QATLinear:
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        mod = cls(layer.weight.shape[1], layer.weight.shape[0], bias=has_bias, group_size=group_size)
        mod.weight = layer.weight.astype(mx.float32)
        if has_bias:
            mod.bias = layer.bias.astype(mx.float32)
        return mod

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        out_d, in_d = w.shape
        g = w.reshape(out_d, -1, self.group_size)
        g_mean = mx.mean(g, axis=-1, keepdims=True)
        g_cent = g - g_mean
        scale = mx.mean(mx.abs(g_cent), axis=-1, keepdims=True) + 1e-5
        q = mx.clip(mx.round(g_cent / scale), -1.0, 1.0)
        
        sum_g2 = mx.sum(g_cent**2, axis=-1, keepdims=True)
        sum_q2 = mx.sum(q**2, axis=-1, keepdims=True) + 1e-5
        s_norm = mx.sqrt(sum_g2 / sum_q2) * mx.sign(mx.sum(g_cent * q, axis=-1, keepdims=True) + 1e-8)
        w_q = (g_mean + q * s_norm).reshape(out_d, in_d)
        
        # STE
        w_eff = w + mx.stop_gradient(w_q - w)
        y = x @ w_eff.astype(x.dtype).T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


class INT4QATLinear(nn.Module):
    """4-bit affine linear with FP32 master weights and STE for sensitive attention blocks."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False, group_size: int = 128):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.group_size = group_size
        self.weight = mx.zeros((out_d, in_d), dtype=mx.float32)
        self.bias = mx.zeros((out_d,), dtype=mx.float32) if bias else None

    @classmethod
    def from_linear(cls, layer: nn.Linear, group_size: int = 128) -> INT4QATLinear:
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        mod = cls(layer.weight.shape[1], layer.weight.shape[0], bias=has_bias, group_size=group_size)
        mod.weight = layer.weight.astype(mx.float32)
        if has_bias:
            mod.bias = layer.bias.astype(mx.float32)
        return mod

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        out_d, in_d = w.shape
        g = w.reshape(out_d, -1, self.group_size)
        g_min = mx.min(g, axis=-1, keepdims=True)
        g_max = mx.max(g, axis=-1, keepdims=True)
        scale = (g_max - g_min) / 15.0 + 1e-5
        q = mx.clip(mx.round((g - g_min) / scale), 0.0, 15.0)
        w_q = (g_min + q * scale).reshape(out_d, in_d)
        
        # STE
        w_eff = w + mx.stop_gradient(w_q - w)
        y = x @ w_eff.astype(x.dtype).T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


def load_dataset(dataset_dir: Path):
    npy_files = sorted(glob.glob(str(dataset_dir / "*.npy")))
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


def convert_block_to_qat(block: nn.Module, block_idx: int, group_size: int = 128):
    """Convert linear layers in a block to QAT layers (INT4 for boundary self-attn, INT2 elsewhere)."""
    int4_blocks = set(range(0, 6)) | set(range(18, 24))
    
    def convert(path: str, layer: nn.Module) -> nn.Module:
        if isinstance(layer, nn.Linear) and layer.weight.shape[1] % group_size == 0:
            if "self_attn" in path and block_idx in int4_blocks:
                return INT4QATLinear.from_linear(layer, group_size=group_size)
            else:
                return INT2QATLinear.from_linear(layer, group_size=group_size)
        return layer

    leaves = tree_map_with_path(convert, block.leaf_modules(), is_leaf=nn.Module.is_module)
    block.update_modules(leaves)


def export_final_model(student: nn.Module, out_path: Path, group_size: int = 128):
    """Pack trained master FP32 weights into MLX QuantizedLinear format (< 500 MB)."""
    int4_blocks = set(range(0, 6)) | set(range(18, 24))
    target_dit = dit_mlx_medium.DiT(T_lat=129)

    def pred_int2(p, l):
        if not (hasattr(l, "weight") and l.weight.ndim == 2 and l.weight.shape[1] % group_size == 0): return False
        if "ff.ff" in p or "cross_attn" in p or "to_local_embed" in p: return True
        if "self_attn" in p:
            b_idx = int(p.split(".")[2]) if "layers." in p else -1
            return b_idx not in int4_blocks
        return False

    def pred_int4(p, l):
        if not (hasattr(l, "weight") and l.weight.ndim == 2 and l.weight.shape[1] % group_size == 0): return False
        if "self_attn" in p:
            b_idx = int(p.split(".")[2]) if "layers." in p else -1
            return b_idx in int4_blocks
        return False

    nn.quantize(target_dit, bits=2, group_size=group_size, mode="affine", class_predicate=pred_int2)
    nn.quantize(target_dit, bits=4, group_size=group_size, mode="affine", class_predicate=pred_int4)
    final_params = dict(tree_flatten(target_dit.parameters()))

    flat_student = dict(tree_flatten(student.parameters()))
    for k, v in flat_student.items():
        # Handle QAT linear layers
        if k.endswith(".weight") and v.ndim == 2 and "layers." in k and v.shape[1] % group_size == 0:
            prefix = k[:-7]
            b_idx = int(k.split(".")[2])
            out_d, in_d = v.shape
            g = v.reshape(out_d, -1, group_size).astype(mx.float32)

            if "self_attn" in k and b_idx in int4_blocks:
                # 4-bit packing
                g_min = mx.min(g, axis=-1, keepdims=True)
                g_max = mx.max(g, axis=-1, keepdims=True)
                scale = (g_max - g_min) / 15.0 + 1e-5
                q = mx.clip(mx.round((g - g_min) / scale), 0.0, 15.0).astype(mx.uint32)
                
                scales = scale.astype(mx.float16).reshape(out_d, in_d // group_size)
                biases = g_min.astype(mx.float16).reshape(out_d, in_d // group_size)
                
                # Pack 8 values (4-bit) into each uint32
                q_arr = np.array(q).reshape(out_d, in_d // 8, 8)
                packed = np.zeros((out_d, in_d // 8), dtype=np.uint32)
                for i in range(8):
                    packed |= (q_arr[:, :, i].astype(np.uint32) << (4 * i))
                
                final_params[prefix + ".weight"] = mx.array(packed)
                final_params[prefix + ".scales"] = scales
                final_params[prefix + ".biases"] = biases
            else:
                # 2-bit packing
                g_mean = mx.mean(g, axis=-1, keepdims=True)
                g_cent = g - g_mean
                scale = mx.mean(mx.abs(g_cent), axis=-1, keepdims=True) + 1e-5
                q = mx.clip(mx.round(g_cent / scale), -1.0, 1.0)
                sum_g2 = mx.sum(g_cent**2, axis=-1, keepdims=True)
                sum_q2 = mx.sum(q**2, axis=-1, keepdims=True) + 1e-5
                s_norm = mx.sqrt(sum_g2 / sum_q2) * mx.sign(mx.sum(g_cent * q, axis=-1, keepdims=True) + 1e-8)
                
                scales = -s_norm.astype(mx.float16).reshape(out_d, in_d // group_size)
                biases = (g_mean + s_norm).astype(mx.float16).reshape(out_d, in_d // group_size)
                
                q_code = mx.where(q == 1.0, 0, mx.where(q == 0.0, 1, 2)).astype(mx.uint32).reshape(out_d, in_d)
                np_codes = np.array(q_code).reshape(out_d, in_d // 16, 16)
                packed = np.zeros((out_d, in_d // 16), dtype=np.uint32)
                for i in range(16):
                    packed |= (np_codes[:, :, i].astype(np.uint32) << (2 * i))
                
                final_params[prefix + ".weight"] = mx.array(packed)
                final_params[prefix + ".scales"] = scales
                final_params[prefix + ".biases"] = biases
        else:
            final_params[k] = v.astype(mx.float16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    mx.savez(str(tmp), **final_params)
    os.replace(str(tmp), str(out_path))
    size_mb = out_path.stat().st_size / (1024**2)
    print(f"\n[Export SUCCESS] {out_path} ({size_mb:.2f} MB)")
    assert size_mb < 500.0, f"Error: exported size {size_mb} MB exceeds 500 MB limit!"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-stage", type=int, default=300)
    parser.add_argument("--smooth-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--out-model", type=str, default="output/sample-expertise-pilot/universal-models/dit_medium_progressive_qat_492mb.npz")
    args = parser.parse_args()

    dataset_dir = Path("output/sample-expertise-pilot/universal-dataset/latents-12s")
    items = load_dataset(dataset_dir)

    teacher_path = MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    cached_data = preencode_prompts(items, teacher_path)

    print("Loading Teacher model (dit_medium_f16.npz)...")
    teacher = dit_mlx_medium.DiT(T_lat=129)
    teacher.load_weights(str(teacher_path), strict=False)
    teacher.freeze()

    print("Initializing Student model from teacher weights...")
    student = dit_mlx_medium.DiT(T_lat=129)
    student.load_weights(str(teacher_path), strict=False)

    # Stages of 4 blocks
    stages = [
        ("Stage 1: Central blocks 6..9", range(6, 10)),
        ("Stage 2: Central blocks 10..13", range(10, 14)),
        ("Stage 3: Central blocks 14..17", range(14, 18)),
        ("Stage 4: Input boundary blocks 0..3", range(0, 4)),
        ("Stage 5: Input transition blocks 4..7", range(4, 8)),
        ("Stage 6: Output boundary blocks 18..23", range(18, 24))
    ]

    t_global_start = time.time()

    for s_idx, (stage_name, block_indices) in enumerate(stages):
        print(f"\n{'='*70}\n[Stage {s_idx+1}/{len(stages)}] {stage_name}\n{'='*70}")
        
        # Convert target blocks to QAT layers
        for b in block_indices:
            convert_block_to_qat(student.transformer.layers[b], b, group_size=128)

        # Freeze everything except current stage blocks
        student.freeze()
        for b in block_indices:
            student.transformer.layers[b].unfreeze()

        lr_sched = optim.cosine_decay(args.lr, args.steps_per_stage, end=args.lr * 0.1)
        opt = optim.AdamW(learning_rate=lr_sched, weight_decay=1e-4)

        t_stage_start = time.time()
        for step in range(1, args.steps_per_stage + 1):
            sample_idx = (step + s_idx * 50) % len(cached_data)
            cross_full, global_cond_val, x_0 = cached_data[sample_idx]
            if x_0.shape[-1] != 129:
                x_0 = x_0[:, :, :129]

            # Timestep distribution focused on intermediate noise levels
            u = random.random()
            t_val = float(1.0 / (1.0 + math.exp(-1.5 * math.log(u / (1.0 - u + 1e-6) + 1e-6))))
            t_val = min(max(t_val, 0.05), 0.95)
            t = mx.array([t_val], dtype=mx.float16)

            key = mx.random.key(step * 31 + s_idx * 1000 + 42)
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
                norm_err = ((norm_vs - norm_vt) / norm_vt)**2
                return mse + 3.0 * (1.0 - cos) + 1.0 * norm_err, (mse, cos, norm_vs / norm_vt)

            (loss, (mse, cos, norm_ratio)), grads = nn.value_and_grad(student, loss_fn)(student)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(student, grads)
            mx.eval(student.parameters(), opt.state)

            if step % 25 == 0 or step == args.steps_per_stage:
                dt = time.time() - t_stage_start
                sec_step = dt / step
                print(f"  Step {step:03d}/{args.steps_per_stage:03d} | Loss: {float(loss):.4f} | CosSim: {float(cos):.4f} | NormRatio: {float(norm_ratio):.4f} | {sec_step:.2f}s/step")

        del opt
        gc.collect()
        mx.clear_cache()

    # Stage 7: Final End-to-End Smoothing
    print(f"\n{'='*70}\n[Stage 7/7] Final End-to-End Smoothing ({args.smooth_steps} steps)\n{'='*70}")
    student.freeze()
    # Unfreeze upper blocks 18..23 + project_out + modulateurs
    for b in range(18, 24):
        student.transformer.layers[b].unfreeze()
    student.transformer.project_out.unfreeze()
    student.postprocess_conv.unfreeze()

    smooth_lr = optim.cosine_decay(args.lr * 0.5, args.smooth_steps, end=args.lr * 0.05)
    smooth_opt = optim.AdamW(learning_rate=smooth_lr, weight_decay=1e-4)
    t_smooth_start = time.time()

    for step in range(1, args.smooth_steps + 1):
        sample_idx = (step * 7) % len(cached_data)
        cross_full, global_cond_val, x_0 = cached_data[sample_idx]
        if x_0.shape[-1] != 129:
            x_0 = x_0[:, :, :129]

        t_val = float(0.05 + 0.90 * random.random())
        t = mx.array([t_val], dtype=mx.float16)

        key = mx.random.key(step * 53 + 777)
        noise = mx.random.normal(x_0.shape, dtype=mx.float16, key=key)
        x_t = (1.0 - t) * x_0 + t * noise

        vt = teacher(x_t, t, cross_full, global_cond_val)
        mx.eval(vt)
        norm_vt = mx.sqrt(mx.sum(vt**2)) + 1e-6

        def smooth_loss_fn(model):
            vs = model(x_t, t, cross_full, global_cond_val)
            norm_vs = mx.sqrt(mx.sum(vs**2)) + 1e-6
            mse = mx.mean((vs - vt)**2)
            cos = mx.sum(vs * vt) / (norm_vs * norm_vt)
            norm_err = ((norm_vs - norm_vt) / norm_vt)**2
            return mse + 3.0 * (1.0 - cos) + 1.0 * norm_err, (mse, cos, norm_vs / norm_vt)

        (loss, (mse, cos, norm_ratio)), grads = nn.value_and_grad(student, smooth_loss_fn)(student)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        smooth_opt.update(student, grads)
        mx.eval(student.parameters(), smooth_opt.state)

        if step % 25 == 0 or step == args.smooth_steps:
            dt = time.time() - t_smooth_start
            print(f"  Smooth Step {step:03d}/{args.smooth_steps:03d} | Loss: {float(loss):.4f} | CosSim: {float(cos):.4f} | NormRatio: {float(norm_ratio):.4f} | {dt/step:.2f}s/step")

    # Export final model
    out_path = Path(args.out_model)
    export_final_model(student, out_path, group_size=128)
    total_time = (time.time() - t_global_start) / 60.0
    print(f"\n[ALL COMPLETE] Progressive QAT finished in {total_time:.1f} minutes!")


if __name__ == "__main__":
    main()
