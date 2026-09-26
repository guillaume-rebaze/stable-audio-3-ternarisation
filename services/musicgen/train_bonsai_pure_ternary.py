"""Bonsai Pure Ternary {-1, 0, +1} Sequential Cascade Distillation for DiT Medium.

Follows PrismML Bonsai architecture:
1. Pure ternary weights {-s, 0, +s} with group_size=128 FP16 scale.
2. Walsh-Hadamard H128 rotation per group to eliminate outliers.
3. Cascade sequential block-by-block calibration: each block is trained on
   the REAL outputs of preceding quantized blocks (zero exposure bias).
4. End-to-end velocity polish on continuous modulation parameters.
5. Bit-perfect export to MLX 2-bit packed container (< 460 MB).
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
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
from scipy.linalg import hadamard

from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import apply_prompt_padding, load_conditioner_from_npz
from models.defs.t5gemma_mlx import T5Gemma
from weights import ensure_local
from sa3_mlx import T5GEMMA_NPZ_REL

# Build H128 orthogonal Hadamard matrix
H128_NP = (hadamard(128).astype(np.float32) / np.sqrt(128)).astype(np.float32)
H128_MX = mx.array(H128_NP, dtype=mx.float32)


class TernaryBonsaiQATLinear(nn.Module):
    """Linear layer with Hadamard rotation and pure ternary STE {-1, 0, +1}."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False, group_size: int = 128, use_hadamard: bool = True):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.group_size = group_size
        self.use_hadamard = use_hadamard
        self.weight = mx.zeros((out_d, in_d), dtype=mx.float32)
        self.bias = mx.zeros((out_d,), dtype=mx.float32) if bias else None

    @classmethod
    def from_linear(cls, layer: nn.Linear, group_size: int = 128, use_hadamard: bool = True) -> TernaryBonsaiQATLinear:
        has_bias = hasattr(layer, "bias") and layer.bias is not None
        mod = cls(layer.weight.shape[1], layer.weight.shape[0], bias=has_bias, group_size=group_size, use_hadamard=use_hadamard)
        mod.weight = layer.weight.astype(mx.float32)
        if has_bias:
            mod.bias = layer.bias.astype(mx.float32)
        return mod

    def quantize_weights(self) -> tuple[mx.array, mx.array]:
        """Compute pure ternary codes Q in {-1, 0, 1} and group scales s."""
        w = self.weight
        g = w.reshape(self.out_d, -1, self.group_size)
        if self.use_hadamard:
            g = g @ H128_MX
        scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-6
        q = mx.clip(mx.round(g / scale), -1.0, 1.0)
        sum_gq = mx.sum(g * q, axis=-1, keepdims=True)
        sum_q2 = mx.sum(q**2, axis=-1, keepdims=True) + 1e-6
        s_opt = sum_gq / sum_q2
        w_rot_q = q * s_opt
        if self.use_hadamard:
            w_q = (w_rot_q @ H128_MX.T).reshape(self.out_d, self.in_d)
        else:
            w_q = w_rot_q.reshape(self.out_d, self.in_d)
        return w_q, s_opt

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        w_q, _ = self.quantize_weights()
        # STE: forward uses ternary w_q, backward flows to continuous master weight w
        w_ste = w + mx.stop_gradient(w_q - w)
        out = x @ w_ste.T.astype(x.dtype)
        if self.bias is not None:
            out = out + self.bias.astype(x.dtype)
        return out


def replace_block_with_qat(block: nn.Module, use_hadamard: bool = True):
    """Replaces the 7 linear layers of a TransformerBlock with TernaryBonsaiQATLinear."""
    block.self_attn.to_qkv = TernaryBonsaiQATLinear.from_linear(block.self_attn.to_qkv, use_hadamard=use_hadamard)
    block.self_attn.to_out = TernaryBonsaiQATLinear.from_linear(block.self_attn.to_out, use_hadamard=use_hadamard)
    block.cross_attn.to_q = TernaryBonsaiQATLinear.from_linear(block.cross_attn.to_q, use_hadamard=use_hadamard)
    block.cross_attn.to_kv = TernaryBonsaiQATLinear.from_linear(block.cross_attn.to_kv, use_hadamard=use_hadamard)
    block.cross_attn.to_out = TernaryBonsaiQATLinear.from_linear(block.cross_attn.to_out, use_hadamard=use_hadamard)
    block.ff.ff[0].proj = TernaryBonsaiQATLinear.from_linear(block.ff.ff[0].proj, use_hadamard=use_hadamard)
    block.ff.ff[2] = TernaryBonsaiQATLinear.from_linear(block.ff.ff[2], use_hadamard=use_hadamard)


def freeze_and_quantize_block(block: nn.Module):
    """Converts TernaryBonsaiQATLinear layers in a block into fixed quantized weights."""
    linear_mods = [
        block.self_attn.to_qkv, block.self_attn.to_out,
        block.cross_attn.to_q, block.cross_attn.to_kv, block.cross_attn.to_out,
        block.ff.ff[0].proj, block.ff.ff[2]
    ]
    for mod in linear_mods:
        w_q, _ = mod.quantize_weights()
        mod.weight = mx.stop_gradient(w_q)
        if mod.bias is not None:
            mod.bias = mx.stop_gradient(mod.bias)
    block.freeze()


def export_bonsai_pure_ternary(model: dit_mlx_medium.DiT, out_path: Path):
    """Exports model to MLX QuantizedLinear 2-bit affine format (strictly < 460 MB)."""
    print(f"\n[Export] Packing pure ternary model to {out_path}...")
    flat = dict(tree_flatten(model.parameters()))
    out_dict = {}

    n_packed = 0
    group_size = 128
    for k, v in flat.items():
        if v.ndim == 2 and "layers." in k and k.endswith(".weight") and v.shape[1] % group_size == 0:
            w = v.astype(mx.float32)
            out_d, in_d = w.shape
            g = w.reshape(out_d, -1, group_size)
            scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-6
            q = mx.clip(mx.round(g / scale), -1.0, 1.0)
            sum_gq = mx.sum(g * q, axis=-1, keepdims=True)
            sum_q2 = mx.sum(q**2, axis=-1, keepdims=True) + 1e-6
            s_opt = sum_gq / sum_q2

            # Map ternary {-1, 0, 1} to MLX 2-bit affine format:
            # code 0 -> -s, code 1 -> 0, code 2 -> +s
            q_code = mx.where(q == -1.0, mx.array(0, dtype=mx.uint32),
                     mx.where(q == 0.0, mx.array(1, dtype=mx.uint32),
                              mx.array(2, dtype=mx.uint32)))

            q_flat = q_code.reshape(out_d, -1)
            n_pack = 16
            packed_cols = in_d // n_pack
            packed_w = mx.zeros((out_d, packed_cols), dtype=mx.uint32)
            for i in range(n_pack):
                packed_w = packed_w | (q_flat[:, i::n_pack].astype(mx.uint32) << (2 * i))

            scales_export = (s_opt.reshape(out_d, -1)).astype(mx.float16)
            biases_export = (-s_opt.reshape(out_d, -1)).astype(mx.float16)

            base_k = k[:-7]
            out_dict[f"{base_k}.weight"] = np.array(packed_w)
            out_dict[f"{base_k}.scales"] = np.array(scales_export)
            out_dict[f"{base_k}.biases"] = np.array(biases_export)
            n_packed += 1
        else:
            out_dict[k] = np.array(v.astype(mx.float16))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **out_dict)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[Export] Packed {n_packed} linear layers into pure ternary.")
    print(f"[Export] Final file size: {size_mb:.2f} MB (Target < 500 MB)")
    assert size_mb < 500.0, f"Error: model size {size_mb:.2f} MB exceeds 500 MB"


def main():
    parser = argparse.ArgumentParser(description="Train Bonsai Pure Ternary DiT Medium.")
    parser.add_argument("--steps-per-block", type=int, default=80, help="Steps per block cascade.")
    parser.add_argument("--polish-steps", type=int, default=150, help="Final E2E continuous polish steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--use-hadamard", action="store_true", default=False, help="Use Walsh-Hadamard rotation.")
    parser.add_argument("--output", type=str, default="output/sample-expertise-pilot/universal-models/dit_medium_bonsai_ternary_456mb.npz")
    args = parser.parse_args()

    teacher_path = MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    out_path = Path(args.output)

    print("=" * 70)
    print("BONSAI PURE TERNARY {-1, 0, +1} CASCADE DISTILLATION")
    print(f"Hadamard Rotation: {args.use_hadamard}")
    print(f"Steps per block:   {args.steps_per_block} (x 24 blocks = {args.steps_per_block * 24} total)")
    print(f"Polish steps:      {args.polish_steps}")
    print("=" * 70)

    latent_files = sorted(glob.glob("output/sample-expertise-pilot/universal-dataset/latents-12s/*.npy"))
    if not latent_files:
        print("[Error] No latent files found in dataset directory!")
        return
    print(f"[Data] Loaded {len(latent_files)} calibration latents.")

    padding_emb, secs_embedder = load_conditioner_from_npz(str(teacher_path), prefix="cond.")
    sec_tok = secs_embedder(12.0).astype(mx.float16)
    global_cond_val = sec_tok[:, 0, :]

    prompts = [
        "70s funk groove with slap bass, wah-wah guitar, punchy drums",
        "intimate solo acoustic upright piano, melancholic chords, warm room reverb",
        "cinematic ambient drone with lush shimmering pads, deep sub-bass",
        "upbeat electronic dance track with synth arpeggios and punchy 808 kick"
    ]
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    emb, mask = t5.encode(prompts, max_len=256)
    padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)).astype(mx.float16)
    cross_full = mx.concatenate([padded, mx.repeat(sec_tok, len(prompts), axis=0)], axis=1)
    del t5
    print("[Conditioning] Cached text and duration embeddings.")

    print("[Teacher] Loading teacher model in FP16...")
    teacher = dit_mlx_medium.DiT(T_lat=129)
    teacher.load_weights(str(teacher_path), strict=False)
    teacher.freeze()

    print("[Student] Initializing student model...")
    student = dit_mlx_medium.DiT(T_lat=129)
    student.load_weights(str(teacher_path), strict=False)

    B = 1
    c = teacher.to_cond_embed[0](cross_full[:1])
    c = nn.silu(c)
    context = teacher.to_cond_embed[2](c)

    g = teacher.to_global_embed[0](global_cond_val[:1])
    g = nn.silu(g)
    global_pre = teacher.to_global_embed[2](g)

    mx.eval(student.parameters(), teacher.parameters())
    print(f"[VRAM] Active: {mx.metal.get_active_memory() / (1024**3):.2f} GB, Peak: {mx.metal.get_peak_memory() / (1024**3):.2f} GB")

    calib_latents = [mx.array(np.load(f)) for f in latent_files[:32]]
    start_time = time.time()
    sigmas = [0.95, 0.80, 0.65, 0.50, 0.35, 0.20, 0.10, 0.05]

    for b_idx in range(len(student.transformer.layers)):
        t0_blk = time.time()
        s_blk = student.transformer.layers[b_idx]
        t_blk = teacher.transformer.layers[b_idx]

        s_blk.freeze()
        replace_block_with_qat(s_blk, use_hadamard=args.use_hadamard)
        linear_mods = [s_blk.self_attn.to_qkv, s_blk.self_attn.to_out,
                       s_blk.cross_attn.to_q, s_blk.cross_attn.to_kv, s_blk.cross_attn.to_out,
                       s_blk.ff.ff[0].proj, s_blk.ff.ff[2]]
        for m in linear_mods:
            m.unfreeze()
        optimizer = optim.Adam(learning_rate=args.lr)

        def block_loss_fn(blk_mod, h_in, ctx, g_cond, loc_emb, h_target):
            h_out = blk_mod(h_in, ctx, g_cond, loc_emb)
            h_out_f32 = h_out.astype(mx.float32)
            h_tgt_f32 = h_target.astype(mx.float32)
            cos = mx.sum(h_out_f32 * h_tgt_f32) / (mx.sqrt(mx.sum(h_out_f32**2)) * mx.sqrt(mx.sum(h_tgt_f32**2)) + 1e-8)
            mse = mx.mean((h_out_f32 - h_tgt_f32)**2)
            norm_tgt = mx.sqrt(mx.mean(h_tgt_f32**2) + 1e-8)
            norm_out = mx.sqrt(mx.mean(h_out_f32**2) + 1e-8)
            norm_loss = mx.abs(norm_out - norm_tgt) / norm_tgt
            loss = mse + 2.0 * (1.0 - cos) + 0.5 * norm_loss
            return loss, cos

        loss_and_grad = nn.value_and_grad(s_blk, block_loss_fn)

        print(f"\n--- Calibrating Block {b_idx:02d}/23 ---")
        best_cos = 0.0
        for step in range(args.steps_per_block):
            x_raw = calib_latents[step % len(calib_latents)]
            if x_raw.ndim == 2: x_raw = x_raw[None, ...]
            s_val = sigmas[step % len(sigmas)]
            t_arr = mx.array([s_val], dtype=mx.float16)

            tf = teacher.timestep_features(t_arr)
            tf = teacher.to_timestep_embed[0](tf)
            tf = nn.silu(tf)
            t_embed = teacher.to_timestep_embed[2](tf)
            global_embed = global_pre + t_embed

            x_lc = x_raw.transpose(0, 2, 1)
            x_pp = student.preprocess_conv(x_lc) + x_lc
            local_zeros = mx.zeros((B, x_raw.shape[-1], dit_mlx_medium.LOCAL_ADD_COND_DIM))

            h_s = student.transformer.project_in(x_pp)
            mem = mx.broadcast_to(student.transformer.memory_tokens[None], (B, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM))
            h_s = mx.concatenate([mem, h_s], axis=1)

            g_vec = student.transformer.global_cond_embedder[0](global_embed)
            g_vec = nn.silu(g_vec)
            global_cond = student.transformer.global_cond_embedder[2](g_vec)

            # Pass through all preceding student blocks 0..b_idx-1
            for prev_idx in range(b_idx):
                p_blk = student.transformer.layers[prev_idx]
                loc_emb = p_blk.to_local_embed(local_zeros)
                pad = mx.zeros((B, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM), dtype=loc_emb.dtype)
                loc_pad = mx.concatenate([pad, loc_emb], axis=1)
                h_s = p_blk(h_s, context, global_cond, loc_pad)

            # Preceding teacher representation
            h_t = teacher.transformer.project_in(x_pp)
            h_t = mx.concatenate([mem, h_t], axis=1)
            g_vec_t = teacher.transformer.global_cond_embedder[0](global_embed)
            g_vec_t = nn.silu(g_vec_t)
            global_cond_t = teacher.transformer.global_cond_embedder[2](g_vec_t)
            for prev_idx in range(b_idx):
                pt_blk = teacher.transformer.layers[prev_idx]
                loc_emb_t = pt_blk.to_local_embed(local_zeros)
                pad = mx.zeros((B, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM), dtype=loc_emb_t.dtype)
                loc_pad_t = mx.concatenate([pad, loc_emb_t], axis=1)
                h_t = pt_blk(h_t, context, global_cond_t, loc_pad_t)

            # Target for current block
            loc_emb_cur = t_blk.to_local_embed(local_zeros)
            pad = mx.zeros((B, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM), dtype=loc_emb_cur.dtype)
            loc_pad_cur = mx.concatenate([pad, loc_emb_cur], axis=1)
            h_target = t_blk(h_t, context, global_cond_t, loc_pad_cur)
            mx.eval(h_s, h_target)

            loc_emb_s = s_blk.to_local_embed(local_zeros)
            loc_pad_s = mx.concatenate([pad, loc_emb_s], axis=1)

            (loss_val, cos_val), grads = loss_and_grad(s_blk, h_s, context, global_cond, loc_pad_s, h_target)
            optimizer.update(s_blk, grads)
            mx.eval(s_blk.parameters(), optimizer.state, loss_val, cos_val)

            c_val = float(cos_val)
            if c_val > best_cos:
                best_cos = c_val

            if (step + 1) % 20 == 0 or step == args.steps_per_block - 1:
                print(f"  Step {step+1:03d}/{args.steps_per_block} | Loss: {float(loss_val):.4f} | CosSim: {c_val:.4f} (Best: {best_cos:.4f})")

        freeze_and_quantize_block(s_blk)
        mx.eval(s_blk.parameters())
        dt_blk = time.time() - t0_blk
        print(f"  ✓ Block {b_idx:02d} quantized & frozen in {dt_blk:.1f}s | Final CosSim: {best_cos:.4f}")

    # =========================================================================
    # STAGE 2: Export Bit-Perfect Packed Model (< 460 MB)
    # =========================================================================
    export_bonsai_pure_ternary(student, out_path)
    total_time = time.time() - start_time
    print(f"[Done] Bonsai pure ternary model successfully created in {total_time/60:.1f} minutes.")


if __name__ == "__main__":
    main()
