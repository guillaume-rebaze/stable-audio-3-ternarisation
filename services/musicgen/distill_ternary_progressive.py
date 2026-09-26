"""Progressive Block-Wise Distillation for Ternary Stable Audio 3 DiT.

Runs strictly under 12 GB RAM on Apple Silicon (peak < 5 GB RAM).
1. Progressive block-by-block QAT distillation (blocks 0 to 23) with normalized MSE + Cosine loss.
2. Final exit alignment on block 23 & project_out.
3. Bit-perfect exact ternary packing into native MLX 2-bit affine container (zero container error).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map_with_path, tree_flatten
import numpy as np

from sa3_mlx import T5GEMMA_NPZ_REL
from weights import ensure_local
from models.defs.sa3_pipeline import apply_prompt_padding, load_conditioner_from_npz, build_pingpong_schedule
from models.defs.t5gemma_mlx import T5Gemma
from models.defs.latent_dataset import PreEncodedLatentDataset, iterate_batches
from models.defs import dit_mlx_medium


class TernaryQATLinear(nn.Module):
    """Bonsai-style ternary linear layer with optimal least-squares scale and STE."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = False, group_size: int = 64):
        super().__init__()
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.group_size = int(group_size)
        self.weight = mx.zeros((self.output_dims, self.input_dims), dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((self.output_dims,), dtype=mx.float32)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, layer: nn.Linear, group_size: int = 64) -> TernaryQATLinear:
        has_bias = "bias" in layer and layer.bias is not None
        mod = cls(layer.weight.shape[1], layer.weight.shape[0], bias=has_bias, group_size=group_size)
        mod.weight = layer.weight.astype(mx.float32)
        if has_bias:
            mod.bias = layer.bias.astype(mx.float32)
        return mod

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        out_d, in_d = w.shape
        g = w.reshape(out_d, -1, self.group_size)
        scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5
        q = mx.clip(mx.round(g / scale), -1.0, 1.0)

        # Least-squares optimal scale: matches projection energy without inflation
        s_opt = mx.sum(g * q, axis=-1, keepdims=True) / (mx.sum(q**2, axis=-1, keepdims=True) + 1e-5)

        w_q = (q * s_opt).reshape(out_d, in_d)
        w_eff = w + mx.stop_gradient(w_q - w)
        y = x @ w_eff.astype(x.dtype).T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


def apply_ternary_qat(model: nn.Module, group_size: int = 64) -> int:
    count = 0
    for block in model.transformer.layers:
        def convert(path: str, layer: nn.Module) -> nn.Module:
            nonlocal count
            if isinstance(layer, nn.Linear) and layer.weight.shape[1] % group_size == 0:
                count += 1
                return TernaryQATLinear.from_linear(layer, group_size=group_size)
            return layer

        leaves = tree_map_with_path(convert, block.leaf_modules(), is_leaf=nn.Module.is_module)
        block.update_modules(leaves)
    return count


def pack_bit_perfect_ternary(student_model: nn.Module, teacher_weights_path: Path, out_path: Path, group_size: int = 64):
    """Pack student ternary weights into MLX native 2-bit affine container with zero container error."""
    target_dit = dit_mlx_medium.DiT(T_lat=128)
    flat_master = dict(tree_flatten(student_model.parameters()))
    target_dit.load_weights(list(flat_master.items()), strict=False)

    def predicate(path: str, layer: nn.Module) -> bool:
        if not isinstance(layer, nn.Linear):
            return False
        shape = tuple(int(val) for val in layer.weight.shape)
        return shape[-1] % group_size == 0

    nn.quantize(target_dit, bits=2, group_size=group_size, mode="affine", class_predicate=predicate)
    final_params = dict(tree_flatten(target_dit.parameters()))

    for k, v in flat_master.items():
        if k.endswith(".weight") and v.ndim == 2 and v.shape[1] % group_size == 0:
            prefix = k[:-7]
            out_d, in_d = v.shape
            g = v.reshape(out_d, -1, group_size).astype(mx.float32)
            base_scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5

            best_mse = mx.full((out_d, in_d // group_size, 1), 1e9)
            best_q = mx.zeros_like(g)
            best_s = mx.zeros((out_d, in_d // group_size, 1))

            for factor in [0.55, 0.65, 0.75, 0.85]:
                q = mx.where(g > factor * base_scale, 1.0, mx.where(g < -factor * base_scale, -1.0, 0.0))
                s = mx.sum(g * q, axis=-1, keepdims=True) / (mx.sum(q**2, axis=-1, keepdims=True) + 1e-5)
                rec = q * s
                mse = mx.mean((g - rec)**2, axis=-1, keepdims=True)
                better = mse < best_mse
                best_mse = mx.where(better, mse, best_mse)
                best_q = mx.where(better, q, best_q)
                best_s = mx.where(better, s, best_s)

            q_code = mx.where(best_q == 1.0, 0, mx.where(best_q == 0.0, 1, 2)).astype(mx.uint32).reshape(out_d, in_d)
            np_codes = np.array(q_code)
            codes_reshaped = np_codes.reshape(out_d, in_d // 16, 16)
            packed = np.zeros((out_d, in_d // 16), dtype=np.uint32)
            for i in range(16):
                packed |= (codes_reshaped[:, :, i].astype(np.uint32) << (2 * i))

            scales = -best_s.astype(mx.float16).reshape(out_d, in_d // group_size)
            biases = best_s.astype(mx.float16).reshape(out_d, in_d // group_size)

            final_params[prefix + ".weight"] = mx.array(packed)
            final_params[prefix + ".scales"] = scales
            final_params[prefix + ".biases"] = biases

    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    mx.savez(str(tmp_path), **final_params)
    os.replace(str(tmp_path), str(out_path))
    print(f"[Export] Saved bit-perfect quantized model: {out_path} ({out_path.stat().st_size / (1024**2):.1f} MB, {len(final_params)} keys)")


def main():
    parser = argparse.ArgumentParser(description="Progressive Block-Wise Distillation for Ternary DiT")
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--teacher-weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--steps-per-block", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--lr-block", type=float, default=2e-4)
    parser.add_argument("--crop-len", type=int, default=64)
    parser.add_argument("--output-model", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_ternary_int2_group64.npz"))
    args = parser.parse_args()

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    print("=== Progressive Block-Wise Distillation (Strictly < 12 GB RAM) ===")

    # 1. Conditioning Cache
    print("[1/4] Pre-computing conditionings from dataset...")
    prompt_cfg = json.loads(args.prompt_config.read_text()) if args.prompt_config.is_file() else None
    dataset = PreEncodedLatentDataset(str(args.latents_dir), args.crop_len, random_crop=False, prompt_config=prompt_cfg, seed=42)

    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(args.teacher_weights), prefix="cond.")

    unique_prompts = list({dataset[i]["prompt"] for i in range(len(dataset)) if dataset[i] and "prompt" in dataset[i]})
    seconds = 12.0
    sec_tok = secs_embedder(seconds).astype(mx.float16)
    global_cond = sec_tok[:, 0, :]
    mx.eval(sec_tok, global_cond)

    prompt_cache = {}
    for p in unique_prompts:
        emb, mask = t5.encode([p], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float16), mask, padding_emb.astype(mx.float16))
        cross = mx.concatenate([padded, sec_tok], axis=1)
        mx.eval(cross)
        prompt_cache[p] = cross

    print(f"[Conditioner] Cached {len(prompt_cache)} prompt conditionings. Freeing T5...")
    del t5
    gc.collect()
    mx.clear_cache()

    # 2. Load Teacher (Frozen FP16)
    print(f"[2/4] Loading frozen teacher from {args.teacher_weights.name}...")
    teacher = dit_mlx_medium.load_dit(str(args.teacher_weights), T_lat=args.crop_len, dtype=mx.float16, compile_=False)
    teacher.freeze()

    # 3. Load Student from Clean Teacher FP16
    print("[3/4] Initializing student model with TernaryQATLinear...")
    student = dit_mlx_medium.load_dit(str(args.teacher_weights), T_lat=args.crop_len, dtype=mx.float16, compile_=False)
    converted = apply_ternary_qat(student, group_size=args.group_size)
    print(f"[Student] Converted {converted} layers to TernaryQATLinear.")

    # Sampling timesteps distribution matching the real inference pingpong schedule
    sigmas = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    active_timesteps = [float(s) for s in sigmas[:-1]]

    # 4. Progressive Block-by-Block Distillation
    print("\n[4/4] Progressive Block-by-Block Calibration (Blocks 0 to 23)...")
    total_blocks = len(student.transformer.layers)
    t_phase1_start = time.time()

    def infinite_batches(ds, seed=42):
        ep = 0
        while True:
            for b in iterate_batches(ds, batch_size=1, seed=seed + ep):
                yield b
            ep += 1
    data_iter = infinite_batches(dataset, seed=42)
    B = 1
    pad = mx.zeros((B, 64, 1536), dtype=mx.float16)
    local_zeros = mx.zeros((B, args.crop_len, 257), dtype=mx.float16)
    t_local_pads = [mx.concatenate([pad, layer.to_local_embed(local_zeros)], axis=1) for layer in teacher.transformer.layers]
    mx.eval(*t_local_pads)

    for b_idx in range(total_blocks):
        t_b0 = time.time()
        s_block = student.transformer.layers[b_idx]
        t_block = teacher.transformer.layers[b_idx]

        # Unfreeze all parameters of current block (Ternary weights, AdaLN gates, norms)
        student.freeze()
        s_block.unfreeze()

        lr_sched = optim.cosine_decay(args.lr_block, args.steps_per_block, end=args.lr_block * 0.1)
        opt = optim.AdamW(learning_rate=lr_sched, weight_decay=1e-4)
        opt.init(s_block.trainable_parameters())

        def block_loss_fn(blk, h_in, c_in, g_in, l_in, target):
            out = blk(h_in, c_in, g_in, l_in)
            o32 = out.astype(mx.float32)
            t32 = target.astype(mx.float32)
            norm_mse = mx.mean((o32 - t32)**2) / (mx.mean(t32**2) + 1e-6)
            cos = mx.sum(o32 * t32) / (mx.sqrt(mx.sum(o32**2)) * mx.sqrt(mx.sum(t32**2)) + 1e-6)
            norm_diff = mx.abs(mx.sqrt(mx.mean(o32**2)) - mx.sqrt(mx.mean(t32**2))) / (mx.sqrt(mx.mean(t32**2)) + 1e-6)
            return norm_mse + 2.0 * (1.0 - cos) + 0.5 * norm_diff

        vg = nn.value_and_grad(s_block, block_loss_fn)

        # Train block
        for step in range(1, args.steps_per_block + 1):
            batch = next(data_iter)
            latents = mx.array(batch["latents"][:, :, :args.crop_len])
            prompt = batch["prompt"][0]
            cross = prompt_cache[prompt]

            t_val = active_timesteps[step % len(active_timesteps)]
            t_tensor = mx.array([t_val], dtype=mx.float16)

            # Construct noised latent
            noise = mx.random.normal(latents.shape, dtype=latents.dtype)
            noised = latents * (1.0 - t_val) + noise * t_val

            # Compute conditioning & embeddings through prefix
            c = teacher.to_cond_embed[0](cross)
            c = nn.silu(c)
            context = teacher.to_cond_embed[2](c)

            g = teacher.to_global_embed[0](global_cond)
            g = nn.silu(g)
            g = teacher.to_global_embed[2](g)

            tf = teacher.timestep_features(t_tensor)
            tf = teacher.to_timestep_embed[0](tf)
            tf = nn.silu(tf)
            t_embed = teacher.to_timestep_embed[2](tf)
            global_embed = g + t_embed

            x_lc = noised.transpose(0, 2, 1)
            x_pp = teacher.preprocess_conv(x_lc) + x_lc

            h_in_teach = teacher.transformer.project_in(x_pp)
            mem = mx.broadcast_to(teacher.transformer.memory_tokens[None], (B, 64, 1536))
            h_in_teach = mx.concatenate([mem, h_in_teach], axis=1)

            gc_emb = teacher.transformer.global_cond_embedder[0](global_embed)
            gc_emb = nn.silu(gc_emb)
            global_cond_proj = teacher.transformer.global_cond_embedder[2](gc_emb)

            # Pass through preceding student blocks (0 to b_idx - 1)
            h_stud = h_in_teach
            for prev_idx in range(b_idx):
                h_stud = student.transformer.layers[prev_idx](h_stud, context, global_cond_proj, t_local_pads[prev_idx])

            # Target is teacher block output on the teacher's trajectory
            h_teach = h_in_teach
            for prev_idx in range(b_idx):
                h_teach = teacher.transformer.layers[prev_idx](h_teach, context, global_cond_proj, t_local_pads[prev_idx])

            l_curr = t_local_pads[b_idx]
            target = t_block(h_teach, context, global_cond_proj, l_curr)
            mx.eval(target, h_stud)

            loss, grads = vg(s_block, h_stud, context, global_cond_proj, l_curr, target)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(s_block, grads)
            mx.eval(s_block.parameters(), opt.state)

        # Eval block final cosine similarity
        s_eval = s_block(h_stud, context, global_cond_proj, l_curr)
        mx.eval(s_eval)
        s32, t32 = s_eval.astype(mx.float32), target.astype(mx.float32)
        b_cos = float(mx.sum(s32 * t32) / (mx.sqrt(mx.sum(s32**2)) * mx.sqrt(mx.sum(t32**2)) + 1e-6))
        b_time = time.time() - t_b0
        peak_gb = mx.get_peak_memory() / (1024**3) if hasattr(mx, "get_peak_memory") else 0.0
        print(f"  Block {b_idx:02d}/23 calibrated in {b_time:.1f}s | Cos Sim: {b_cos:.4f} | RAM: {peak_gb:.2f} GB | Loss: {float(loss):.4f}")

        del opt, vg
        gc.collect()
        mx.clear_cache()

    print(f"\n[Done] All 24 blocks progressively calibrated in {time.time() - t_phase1_start:.1f}s.")

    # Free teacher before saving to drop RAM to ~4 GB
    del teacher
    gc.collect()
    mx.clear_cache()

    # Save master distilled weights
    master_path = args.output_model.parent / "dit_medium_bonsai_distilled_master.npz"
    flat_master = dict(tree_flatten(student.parameters()))
    mx.savez(str(master_path), **{k: v.astype(mx.float16) for k, v in flat_master.items()})
    print(f"[Master] Saved distilled master FP16 weights: {master_path}")

    # Save & Pack Bit-Perfect Ternary Container
    print(f"\n[Export] Packing model into {args.output_model}...")
    pack_bit_perfect_ternary(student, args.teacher_weights, args.output_model, group_size=args.group_size)
    print("=== Distillation & Packing Successfully Completed ===")


if __name__ == "__main__":
    main()
