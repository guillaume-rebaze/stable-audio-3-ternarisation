"""Teacher-Forced Block-Wise Distillation for Ternary Stable Audio 3 DiT.

CORRECTION vs ancien script:
- Chaque bloc student est entraîné avec les ACTIVATIONS DU TEACHER en entrée
  (pas les activations student accumulées qui dérivent).
- Cela élimine l'accumulation d'erreur inter-blocs tout en gardant RAM < 8 GB.
- 200 steps/bloc, LR cosine decay, AdamW.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [
    p for p in sys.path if "musicgen" not in p and "abelton" not in p
]

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


# ---------------------------------------------------------------------------
# TernaryQATLinear
# ---------------------------------------------------------------------------

class TernaryQATLinear(nn.Module):
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
    def from_linear(cls, layer: nn.Linear, group_size: int = 64) -> "TernaryQATLinear":
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
        scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5
        q = mx.clip(mx.round(g / scale), -1.0, 1.0)
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


# ---------------------------------------------------------------------------
# Bit-perfect packing
# ---------------------------------------------------------------------------

def pack_bit_perfect_ternary(student_model: nn.Module, out_path: Path, group_size: int = 64):
    target_dit = dit_mlx_medium.DiT(T_lat=128)
    flat_master = dict(tree_flatten(student_model.parameters()))
    target_dit.load_weights(list(flat_master.items()), strict=False)

    def predicate(path: str, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Linear) and tuple(int(v) for v in layer.weight.shape)[-1] % group_size == 0

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

            for factor in [0.50, 0.60, 0.70, 0.80, 0.90]:
                q = mx.where(g > factor * base_scale, 1.0, mx.where(g < -factor * base_scale, -1.0, 0.0))
                s = mx.sum(g * q, axis=-1, keepdims=True) / (mx.sum(q**2, axis=-1, keepdims=True) + 1e-5)
                rec = q * s
                mse = mx.mean((g - rec) ** 2, axis=-1, keepdims=True)
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
    print(f"[Export] {out_path} ({out_path.stat().st_size / (1024**2):.1f} MB, {len(final_params)} keys)")


# ---------------------------------------------------------------------------
# Main distillation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--teacher-weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--steps-per-block", type=int, default=200)
    parser.add_argument("--lr-block", type=float, default=2e-4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--crop-len", type=int, default=64)
    parser.add_argument("--output-model", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_ternary_int2_group64.npz"))
    parser.add_argument("--start-block", type=int, default=0, help="Resume from this block")
    args = parser.parse_args()

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    master_path = args.output_model.parent / "dit_medium_bonsai_distilled_master.npz"

    print("=== Teacher-Forced Block-Wise Distillation ===")
    print(f"Steps/bloc: {args.steps_per_block} | LR: {args.lr_block} | Start block: {args.start_block}")

    # 1. Conditioning cache
    print("[1/4] Pre-computing conditionings...")
    prompt_cfg = json.loads(args.prompt_config.read_text()) if args.prompt_config.is_file() else None
    dataset = PreEncodedLatentDataset(str(args.latents_dir), args.crop_len,
                                       random_crop=True, prompt_config=prompt_cfg, seed=42)

    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(args.teacher_weights), prefix="cond.")

    unique_prompts = list({dataset[i]["prompt"] for i in range(min(len(dataset), 200))
                           if dataset[i] and "prompt" in dataset[i]})
    seconds = 12.0
    sec_tok = secs_embedder(seconds).astype(mx.float16)
    global_cond_base = sec_tok[:, 0, :]
    mx.eval(sec_tok, global_cond_base)

    prompt_cache: dict[str, mx.array] = {}
    for p in unique_prompts:
        emb, mask = t5.encode([p], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float16), mask, padding_emb.astype(mx.float16))
        cross = mx.concatenate([padded, sec_tok], axis=1)
        mx.eval(cross)
        prompt_cache[p] = cross

    print(f"  Cached {len(prompt_cache)} prompts. Freeing T5...")
    del t5
    gc.collect()
    mx.clear_cache()

    # 2. Teacher (frozen)
    print("[2/4] Loading frozen teacher...")
    teacher = dit_mlx_medium.load_dit(str(args.teacher_weights), T_lat=args.crop_len,
                                       dtype=mx.float16, compile_=False)
    teacher.freeze()
    mx.eval(teacher.parameters())

    # 3. Student
    print("[3/4] Initializing student...")
    if master_path.exists() and args.start_block > 0:
        print(f"  Resuming from master: {master_path} (block {args.start_block})")
        student = dit_mlx_medium.load_dit(str(master_path), T_lat=args.crop_len,
                                           dtype=mx.float16, compile_=False)
    else:
        print("  Init from teacher FP16 (clean start)")
        student = dit_mlx_medium.load_dit(str(args.teacher_weights), T_lat=args.crop_len,
                                           dtype=mx.float16, compile_=False)

    n_ternary = apply_ternary_qat(student, group_size=args.group_size)
    print(f"  Converted {n_ternary} transformer layers to TernaryQATLinear.")

    sigmas = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    active_timesteps = [float(s) for s in sigmas[:-1]]

    # 4. Teacher-Forced Block-Wise Distillation
    print(f"\n[4/4] Teacher-Forced Block-Wise Calibration (blocks {args.start_block}..23)...")
    total_blocks = len(student.transformer.layers)
    t_total_start = time.time()

    def infinite_batches(ds, seed=0):
        ep = 0
        while True:
            for b in iterate_batches(ds, batch_size=1, seed=seed + ep):
                yield b
            ep += 1

    data_iter = infinite_batches(dataset, seed=42)

    # Pre-compute local embed pads (teacher's, constant)
    B = 1
    zeros_local = mx.zeros((B, args.crop_len, 257), dtype=mx.float16)
    t_local_pads = []
    for layer in teacher.transformer.layers:
        pad = mx.zeros((B, 64, 1536), dtype=mx.float16)
        local_emb = layer.to_local_embed(zeros_local)
        t_local_pads.append(mx.concatenate([pad, local_emb], axis=1))
    mx.eval(*t_local_pads)

    for b_idx in range(args.start_block, total_blocks):
        t_b0 = time.time()
        s_block = student.transformer.layers[b_idx]
        t_block = teacher.transformer.layers[b_idx]

        # Unfreeze ONLY this block
        student.freeze()
        s_block.unfreeze()

        lr_sched = optim.cosine_decay(args.lr_block, args.steps_per_block, end=args.lr_block * 0.05)
        opt = optim.AdamW(learning_rate=lr_sched, weight_decay=1e-4)
        opt.init(s_block.trainable_parameters())

        def block_loss_fn(blk, h_in, c_in, g_in, l_in, target):
            out = blk(h_in, c_in, g_in, l_in)
            o32 = out.astype(mx.float32)
            t32 = target.astype(mx.float32)
            norm_mse = mx.mean((o32 - t32) ** 2) / (mx.mean(t32 ** 2) + 1e-6)
            cos = mx.sum(o32 * t32) / (mx.sqrt(mx.sum(o32 ** 2)) * mx.sqrt(mx.sum(t32 ** 2)) + 1e-6)
            norm_diff = mx.abs(mx.sqrt(mx.mean(o32**2)) - mx.sqrt(mx.mean(t32**2))) / (mx.sqrt(mx.mean(t32**2)) + 1e-6)
            return norm_mse + 2.0 * (1.0 - cos) + 0.5 * norm_diff

        vg = nn.value_and_grad(s_block, block_loss_fn)

        for step in range(1, args.steps_per_block + 1):
            batch = next(data_iter)
            latents = mx.array(batch["latents"][:, :, :args.crop_len]).astype(mx.float16)
            prompt = batch["prompt"][0]
            cross = prompt_cache[prompt]

            t_val = active_timesteps[(step - 1) % len(active_timesteps)]
            t_tensor = mx.array([t_val], dtype=mx.float16)
            noise = mx.random.normal(latents.shape, dtype=latents.dtype)
            noised = latents * (1.0 - t_val) + noise * t_val

            # Compute teacher conditioning (frozen)
            c_raw = teacher.to_cond_embed[0](cross)
            c_raw = nn.silu(c_raw)
            context = teacher.to_cond_embed[2](c_raw)

            tf = teacher.timestep_features(t_tensor)
            tf = teacher.to_timestep_embed[0](tf)
            tf = nn.silu(tf)
            t_embed = teacher.to_timestep_embed[2](tf)

            g_raw = teacher.to_global_embed[0](global_cond_base)
            g_raw = nn.silu(g_raw)
            g_pre = teacher.to_global_embed[2](g_raw)
            global_embed = g_pre + t_embed

            x_lc = noised.transpose(0, 2, 1)
            x_pp = teacher.preprocess_conv(x_lc) + x_lc
            h_in = teacher.transformer.project_in(x_pp)
            mem = mx.broadcast_to(teacher.transformer.memory_tokens[None], (B, 64, 1536))
            h_in = mx.concatenate([mem, h_in], axis=1)

            gc_emb = teacher.transformer.global_cond_embedder[0](global_embed)
            gc_emb = nn.silu(gc_emb)
            g_proj = teacher.transformer.global_cond_embedder[2](gc_emb)

            # *** KEY FIX: Run TEACHER blocks 0..b_idx-1 to get clean teacher activations ***
            h_teach = h_in
            for prev_idx in range(b_idx):
                h_teach = teacher.transformer.layers[prev_idx](
                    h_teach, context, g_proj, t_local_pads[prev_idx]
                )

            # Target: teacher block b_idx output on teacher-activated inputs
            target = t_block(h_teach, context, g_proj, t_local_pads[b_idx])
            mx.eval(target, h_teach)

            # Train student block b_idx on the same teacher-activated inputs
            loss, grads = vg(s_block, h_teach, context, g_proj, t_local_pads[b_idx], target)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(s_block, grads)
            mx.eval(s_block.parameters(), opt.state)

        # Eval block final cosine
        s_eval = s_block(h_teach, context, g_proj, t_local_pads[b_idx])
        mx.eval(s_eval)
        s32 = s_eval.astype(mx.float32)
        t32 = target.astype(mx.float32)
        b_cos = float(mx.sum(s32 * t32) / (mx.sqrt(mx.sum(s32**2)) * mx.sqrt(mx.sum(t32**2)) + 1e-6))
        b_time = time.time() - t_b0
        peak_gb = mx.get_peak_memory() / (1024**3) if hasattr(mx, "get_peak_memory") else 0.0
        print(f"  Bloc {b_idx:02d}/23 | Cos: {b_cos:.4f} | Loss: {float(loss):.4f} | RAM: {peak_gb:.1f}GB | {b_time:.0f}s")

        del opt, vg, target, h_teach, h_in
        gc.collect()
        mx.clear_cache()

        # Save master after each block (atomic)
        if (b_idx + 1) % 4 == 0 or b_idx == total_blocks - 1:
            flat_master = dict(tree_flatten(student.parameters()))
            tmp_m = master_path.with_suffix(".tmp.npz")
            mx.savez(str(tmp_m), **{k: v.astype(mx.float16) for k, v in flat_master.items()})
            os.replace(str(tmp_m), str(master_path))
            print(f"  [Checkpoint] Master saved après bloc {b_idx} ({master_path.stat().st_size/(1024**2):.0f} MB)")

    print(f"\n[Done] 24 blocs calibrés en {time.time()-t_total_start:.0f}s")

    # 5. Pack ternary
    del teacher
    gc.collect()
    mx.clear_cache()
    print("[5] Packing bit-perfect ternary...")
    pack_bit_perfect_ternary(student, args.output_model, group_size=args.group_size)
    print("=== Complete ===")


if __name__ == "__main__":
    main()
