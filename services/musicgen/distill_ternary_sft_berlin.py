"""Teacher-Forced Block-Wise Distillation of SFT Berlin into Bonsai Ternary INT2.

Fixes applied:
1. Exact conditioning: cross_attn includes the 257th token (seconds_embedder).
2. Exact global_cond: seconds_embedder[:, 0, :] fed into to_global_embed.
3. Natural project_out scale (no artificial booster that triggers SAME-L boundary clicks).
4. Full SFT Berlin LoRA merged into Teacher in memory.
5. Bit-perfect INT2 container with 926 keys matching run_quantized_generation.py.
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
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map_with_path
import numpy as np

from models.defs import dit_mlx_medium, lora_merge
from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz
from models.defs.t5gemma_mlx import T5Gemma
from models.defs.latent_dataset import PreEncodedLatentDataset, iterate_batches
from sa3_mlx import T5GEMMA_NPZ_REL
from weights import ensure_local


class TernaryQATLinear(nn.Module):
    """Group-wise ternary quantization {-s, 0, +s} with STE."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = False, group_size: int = 64):
        super().__init__()
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.group_size = int(group_size)
        self.weight = mx.zeros((self.output_dims, self.input_dims), dtype=mx.float32)
        self.bias = mx.zeros((self.output_dims,), dtype=mx.float32) if bias else None

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


def export_bit_perfect_int2(student_model: nn.Module, out_path: Path, group_size: int = 64):
    """Export hybrid BitNet/Bonsai INT2 model: 192 layers in 1.58-bit ternary, outer layers in FP16."""
    flat_master = dict(tree_flatten(student_model.parameters()))
    target_dit = dit_mlx_medium.DiT(T_lat=128)

    def predicate(path: str, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Linear) and "layers" in path and tuple(int(v) for v in layer.weight.shape)[-1] % group_size == 0

    nn.quantize(target_dit, bits=2, group_size=group_size, mode="affine", class_predicate=predicate)
    final_params = dict(tree_flatten(target_dit.parameters()))

    for k, v in flat_master.items():
        if k.endswith(".weight") and v.ndim == 2 and "layers" in k and v.shape[1] % group_size == 0:
            prefix = k[:-7]
            out_d, in_d = v.shape
            # Bit-perfect ternary {-s, 0, +s}
            g = v.reshape(out_d, -1, group_size).astype(mx.float32)
            scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5
            q = mx.clip(mx.round(g / scale), -1.0, 1.0)
            s_opt = mx.sum(g * q, axis=-1, keepdims=True) / (mx.sum(q**2, axis=-1, keepdims=True) + 1e-5)

            q_code = mx.where(q == 1.0, 0, mx.where(q == 0.0, 1, 2)).astype(mx.uint32).reshape(out_d, in_d)
            np_codes = np.array(q_code)
            codes_reshaped = np_codes.reshape(out_d, in_d // 16, 16)
            packed = np.zeros((out_d, in_d // 16), dtype=np.uint32)
            for i in range(16):
                packed |= (codes_reshaped[:, :, i].astype(np.uint32) << (2 * i))

            scales = -s_opt.astype(mx.float16).reshape(out_d, in_d // group_size)
            biases = s_opt.astype(mx.float16).reshape(out_d, in_d // group_size)

            final_params[prefix + ".weight"] = mx.array(packed)
            final_params[prefix + ".scales"] = scales
            final_params[prefix + ".biases"] = biases
        else:
            final_params[k] = v.astype(mx.float16)

    tmp = out_path.with_suffix(".tmp.npz")
    mx.savez(str(tmp), **final_params)
    os.replace(str(tmp), str(out_path))
    print(f"[Export] {out_path} ({out_path.stat().st_size / (1024**2):.1f} MB, {len(final_params)} keys)")


def main():
    parser = argparse.ArgumentParser(description="Distill SFT Berlin into Bonsai Ternary INT2")
    parser.add_argument("--steps-per-block", type=int, default=200)
    parser.add_argument("--lr-block", type=float, default=2e-4)
    parser.add_argument("--crop-len", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--teacher-weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--lora-path", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-runs/sftberlin-medium-lora-r8-spectral-body-safe-audio-native-hd-clean-content010-tmax085-temporal010-1e-4-20260910-retry2/5477321e/checkpoints/sftberlin-medium-lora-r8-spectral-body-safe-audio-native-hd-clean-content010-tmax085-temporal010-1e-4-20260910-retry2-step=50-epoch=0.safetensors"))
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/quantized-models"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_path = args.output_dir / "dit_medium_bonsai_distilled_master.npz"
    final_int2_path = args.output_dir / "dit_medium_bonsai_ternary_int2_group64.npz"

    print("=== SFT Berlin Student-Forced Ternary Distillation (Self-Correcting) ===")
    print(f"Steps/bloc: {args.steps_per_block} | LR: {args.lr_block} | Start block: {args.start_block}")

    # 1. Dataset & Prompt Cache + Seconds Conditioning
    print("[1/4] Pre-computing conditionings on Berlin tracks...")
    prompt_cfg = json.loads(args.prompt_config.read_text()) if args.prompt_config.is_file() else None
    dataset = PreEncodedLatentDataset(str(args.latents_dir), args.crop_len, random_crop=True, prompt_config=prompt_cfg, seed=42)

    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(args.teacher_weights), prefix="cond.")

    # Compute seconds token for 12s clips (the latent training duration)
    sec_tok = secs_embedder(12.0).astype(mx.float16)  # (1, 1, 768)
    global_cond_val = sec_tok[:, 0, :]                # (1, 768)
    mx.eval(sec_tok, global_cond_val)

    unique_prompts = list({it["prompt"] for it in (dataset[i] for i in range(len(dataset))) if it and "prompt" in it})
    prompt_cache = {}
    for p in unique_prompts:
        emb, mask = t5.encode([p], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)).astype(mx.float16)
        # Full 257 tokens: prompt (256) + seconds (1)
        cross_full = mx.concatenate([padded, sec_tok], axis=1)
        mx.eval(cross_full)
        prompt_cache[p] = cross_full

    print(f"  Cached {len(prompt_cache)} Berlin prompts with 257-token time conditioning. Freeing T5...")
    del t5, padding_emb, secs_embedder
    gc.collect()
    mx.clear_cache()

    # 2. Frozen Teacher with SFT Berlin LoRA Merged in Memory
    print("[2/4] Loading and merging SFT Berlin Teacher...")
    t_weights = dict(mx.load(str(args.teacher_weights)))
    if args.lora_path.is_file():
        stats = lora_merge.merge_loras_into_weights(t_weights, [str(args.lora_path)], strength=1.0)
        print(f"  Merged SFT Berlin adapter: {stats['merged']} layers merged into Teacher.")

    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(list(t_weights.items()), strict=False)
    teacher.freeze()
    mx.eval(teacher.parameters())

    # 3. Student Initialization
    print("[3/4] Initializing Ternary Student from SFT Teacher...")
    student = dit_mlx_medium.DiT(T_lat=args.crop_len)
    student.load_weights(list(t_weights.items()), strict=False)
    del t_weights
    gc.collect()

    n_ternary = apply_ternary_qat(student, group_size=args.group_size)
    print(f"  Converted {n_ternary} layers to TernaryQATLinear.")

    sigmas = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    active_timesteps = [float(s) for s in sigmas[:-1]]

    # 4. Student-Forced (Self-Correcting) Block-Wise Distillation
    print(f"\n[4/4] Student-Forced Distillation on Berlin Tracks (blocks {args.start_block}..23)...")
    total_blocks = len(student.transformer.layers)
    t_total_start = time.time()

    def infinite_batches(ds, seed=0):
        while True:
            for b in iterate_batches(ds, batch_size=1, seed=seed):
                yield b
            seed += 1

    data_iter = infinite_batches(dataset, seed=42)

    pad = mx.zeros((1, 64, 1536), dtype=mx.float16)
    zeros_local = mx.zeros((1, args.crop_len, 257), dtype=mx.float16)
    t_local_pads = [mx.concatenate([pad, layer.to_local_embed(zeros_local)], axis=1) for layer in teacher.transformer.layers]
    mx.eval(*t_local_pads)

    for b_idx in range(args.start_block, total_blocks):
        t_b0 = time.time()
        s_block = student.transformer.layers[b_idx]
        t_block = teacher.transformer.layers[b_idx]

        student.freeze()
        s_block.unfreeze()

        lr_sched = optim.cosine_decay(args.lr_block, args.steps_per_block, end=args.lr_block * 0.05)
        opt = optim.AdamW(learning_rate=lr_sched, weight_decay=1e-4)
        opt.init(s_block.trainable_parameters())

        def block_loss_fn(blk, h_in, c_in, g_in, l_in, target):
            out = blk(h_in, c_in, g_in, l_in)
            o32, t32 = out.astype(mx.float32), target.astype(mx.float32)
            norm_mse = mx.mean((o32 - t32) ** 2) / (mx.mean(t32 ** 2) + 1e-6)
            cos = mx.sum(o32 * t32) / (mx.sqrt(mx.sum(o32 ** 2)) * mx.sqrt(mx.sum(t32 ** 2)) + 1e-6)
            norm_diff = mx.abs(mx.sqrt(mx.mean(o32**2)) - mx.sqrt(mx.mean(t32**2))) / (mx.sqrt(mx.mean(t32**2)) + 1e-6)
            return norm_mse + 2.0 * (1.0 - cos) + 0.5 * norm_diff

        vg = nn.value_and_grad(s_block, block_loss_fn)

        for step in range(1, args.steps_per_block + 1):
            batch = next(data_iter)
            latents = mx.array(batch["latents"][:, :, :args.crop_len]).astype(mx.float16)
            prompt = batch["prompt"][0]
            cross_full = prompt_cache[prompt]

            t_val = active_timesteps[step % len(active_timesteps)]
            t_tensor = mx.array([t_val], dtype=mx.float16)

            noise = mx.random.normal(latents.shape, dtype=latents.dtype)
            noised = latents * (1.0 - t_val) + noise * t_val

            # Embeddings & Conditionings with EXACT temporal conditioning
            c_raw = teacher.to_cond_embed[0](cross_full)
            c_raw = nn.silu(c_raw)
            context = teacher.to_cond_embed[2](c_raw)

            g_raw = teacher.to_global_embed[0](global_cond_val)
            g_raw = nn.silu(g_raw)
            g_pre = teacher.to_global_embed[2](g_raw)

            tf = teacher.timestep_features(t_tensor)
            tf = teacher.to_timestep_embed[0](tf)
            tf = nn.silu(tf)
            t_embed = teacher.to_timestep_embed[2](tf)
            global_embed = g_pre + t_embed

            gc_emb = teacher.transformer.global_cond_embedder[0](global_embed)
            gc_emb = nn.silu(gc_emb)
            g_proj = teacher.transformer.global_cond_embedder[2](gc_emb)

            x_lc = noised.transpose(0, 2, 1)
            x_pp = teacher.preprocess_conv(x_lc) + x_lc
            h_in = teacher.transformer.project_in(x_pp)
            mem = mx.broadcast_to(teacher.transformer.memory_tokens[None], (1, 64, 1536))
            h_in = mx.concatenate([mem, h_in], axis=1)

            # Target: Ideal Teacher representation at block b_idx
            h_teach = h_in
            for prev_idx in range(b_idx):
                h_teach = teacher.transformer.layers[prev_idx](h_teach, context, g_proj, t_local_pads[prev_idx])
            target = t_block(h_teach, context, g_proj, t_local_pads[b_idx])

            # Student input: Actual cumulative output of preceding student blocks
            h_stud = h_in
            for prev_idx in range(b_idx):
                h_stud = student.transformer.layers[prev_idx](h_stud, context, g_proj, t_local_pads[prev_idx])
            mx.eval(target, h_stud)

            loss, grads = vg(s_block, h_stud, context, g_proj, t_local_pads[b_idx], target)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(s_block, grads)
            mx.eval(s_block.parameters(), opt.state)

        # Eval block final cosine on actual student input
        s_eval = s_block(h_stud, context, g_proj, t_local_pads[b_idx])
        mx.eval(s_eval)
        s32, t32 = s_eval.astype(mx.float32), target.astype(mx.float32)
        b_cos = float(mx.sum(s32 * t32) / (mx.sqrt(mx.sum(s32**2)) * mx.sqrt(mx.sum(t32**2)) + 1e-6))
        b_time = time.time() - t_b0
        peak_gb = mx.get_peak_memory() / (1024**3) if hasattr(mx, "get_peak_memory") else 0.0
        print(f"  Bloc {b_idx:02d}/23 | Cos: {b_cos:.4f} | Loss: {float(loss):.4f} | RAM: {peak_gb:.1f}GB | {b_time:.0f}s")

        del opt, vg, target, h_teach, h_stud, h_in
        gc.collect()
        mx.clear_cache()

        if (b_idx + 1) % 4 == 0 or b_idx == total_blocks - 1:
            flat_master = dict(tree_flatten(student.parameters()))
            tmp_m = master_path.with_suffix(".tmp.npz")
            mx.savez(str(tmp_m), **{k: v.astype(mx.float16) for k, v in flat_master.items()})
            os.replace(str(tmp_m), str(master_path))

    t_total = time.time() - t_total_start
    print(f"\n[Done] 24 blocs calibrés en {t_total:.0f}s ({t_total/60:.1f} min)")

    print("[5] Packing hybrid BitNet/Bonsai INT2 model...")
    export_bit_perfect_int2(student, final_int2_path, group_size=args.group_size)

    print("\n[6] Measuring End-to-End Diffusion Alignment (Teacher vs Student)...")
    sigmas_test = [1.0, 0.891, 0.746, 0.512, 0.274]
    key_test = mx.random.key(2026106744)
    x_test = mx.random.normal((1, 256, args.crop_len), dtype=mx.float16, key=key_test)
    c_test = list(prompt_cache.values())[0]
    g_test = global_cond_val

    print(f"  {'Sigma':>6} | {'Cos Sim':>8} | {'Teacher RMS':>11} | {'Student RMS':>11}")
    for s in sigmas_test:
        s_arr = mx.array([s], dtype=mx.float16)
        v_teach = teacher(x_test, s_arr, c_test, g_test)
        v_stud = student(x_test, s_arr, c_test, g_test)
        mx.eval(v_teach, v_stud)
        vt, vs = v_teach.astype(mx.float32), v_stud.astype(mx.float32)
        cos = float(mx.sum(vt * vs) / (mx.sqrt(mx.sum(vt**2)) * mx.sqrt(mx.sum(vs**2)) + 1e-6))
        t_rms = float(mx.sqrt(mx.mean(vt**2)))
        s_rms = float(mx.sqrt(mx.mean(vs**2)))
        print(f"  {s:6.3f} | {cos:8.4f} | {t_rms:11.4f} | {s_rms:11.4f}")

    print("=== Complete ===")


if __name__ == "__main__":
    main()
