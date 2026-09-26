"""Multi-Dataset Student-Forced Ternary Distillation & Global Smoothing.

Combines:
- sftberlin: 153 tracks (minimal, techno, deep-house, microhouse)
- sftminimal: 58 pristine clips (Bodzin, Eulberg, Extrawelt, Rekorder, Kalkbrenner)
- sftvoices: 256 vocal textures and a capella phrases (clean vocal chops)

Stages:
1. Student-Forced block-wise distillation (24 blocks, 160 steps/block).
2. Global End-to-End smoothing pass (100 steps) across upper blocks.
3. Hybrid BitNet/Bonsai INT2 export (192 ternary layers + 10 FP16 outer layers).
"""

from __future__ import annotations

import argparse
import gc
import json
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

from models.defs import dit_mlx_medium, lora_merge
from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz
from models.defs.t5gemma_mlx import T5Gemma
from models.defs.latent_dataset import PreEncodedLatentDataset
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


class MultiDatasetSampler:
    """Balanced sampler across Berlin, Minimal, and Voices."""

    def __init__(self, datasets: list[PreEncodedLatentDataset], weights: list[float], seed: int = 42):
        self.datasets = datasets
        self.weights = [w / sum(weights) for w in weights]
        self.rng = random.Random(seed)

    def next_sample(self):
        ds = self.rng.choices(self.datasets, weights=self.weights, k=1)[0]
        idx = self.rng.randint(0, len(ds) - 1)
        item = ds[idx]
        if item is None:
            return self.next_sample()
        return item


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Ternary Distillation & Smoothing")
    parser.add_argument("--steps-per-block", type=int, default=160)
    parser.add_argument("--smoothing-steps", type=int, default=100)
    parser.add_argument("--lr-block", type=float, default=2e-4)
    parser.add_argument("--crop-len", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--teacher-weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--lora-path", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-runs/sftberlin-medium-lora-r8-spectral-body-safe-audio-native-hd-clean-content010-tmax085-temporal010-1e-4-20260910-retry2/5477321e/checkpoints/sftberlin-medium-lora-r8-spectral-body-safe-audio-native-hd-clean-content010-tmax085-temporal010-1e-4-20260910-retry2-step=50-epoch=0.safetensors"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/quantized-models"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_path = args.output_dir / "dit_medium_bonsai_distilled_master.npz"
    final_int2_path = args.output_dir / "dit_medium_bonsai_ternary_int2_group64.npz"

    print("=== Multi-Dataset SFT Berlin + Minimal + Voices Ternary Distillation ===")
    print(f"Steps/bloc: {args.steps_per_block} | Smoothing: {args.smoothing_steps} | Start block: {args.start_block}")

    # 1. Datasets & Conditioning Cache
    print("\n[1/4] Loading Datasets & Pre-computing Conditionings...")
    berlin_cfg = json.loads(Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json").read_text())
    ds_berlin = PreEncodedLatentDataset("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s", args.crop_len, random_crop=True, prompt_config=berlin_cfg, seed=42)

    minimal_dir = Path("output/sample-expertise-pilot/sftminimal/latents-12s")
    ds_minimal = PreEncodedLatentDataset(str(minimal_dir), args.crop_len, random_crop=True, prompt_config=None, seed=43)

    voices_dir = Path("output/sample-expertise-pilot/sftvoices/sft-training-peak-safe-strict-12s/latents-train")
    voices_cfg = json.loads(Path("output/sample-expertise-pilot/sftvoices/sft-training-peak-safe-strict-12s/prompt-config-vocal.json").read_text())
    ds_voices = PreEncodedLatentDataset(str(voices_dir), args.crop_len, random_crop=True, prompt_config=voices_cfg, seed=44)

    broad_dir = Path("output/sample-expertise-pilot/broad-music/latents-12s")
    ds_broad = PreEncodedLatentDataset(str(broad_dir), args.crop_len, random_crop=True, prompt_config=None, seed=45)

    print(f"  Loaded datasets: Berlin ({len(ds_berlin)}), Minimal ({len(ds_minimal)}), Broad ({len(ds_broad)}), Voices ({len(ds_voices)})")
    sampler = MultiDatasetSampler([ds_berlin, ds_minimal, ds_broad, ds_voices], weights=[0.40, 0.25, 0.20, 0.15], seed=42)

    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(args.teacher_weights), prefix="cond.")
    sec_tok = secs_embedder(12.0).astype(mx.float16)
    global_cond_val = sec_tok[:, 0, :]
    mx.eval(sec_tok, global_cond_val)

    all_prompts = set()
    for ds in [ds_berlin, ds_minimal, ds_broad, ds_voices]:
        for i in range(len(ds)):
            it = ds[i]
            if it and "prompt" in it:
                all_prompts.add(it["prompt"])

    prompt_cache = {}
    for p in all_prompts:
        emb, mask = t5.encode([p], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32)).astype(mx.float16)
        cross_full = mx.concatenate([padded, sec_tok], axis=1)
        mx.eval(cross_full)
        prompt_cache[p] = cross_full

    print(f"  Cached {len(prompt_cache)} unique multi-domain prompts with 257-token conditioning. Freeing T5...")
    del t5, padding_emb, secs_embedder
    gc.collect()
    mx.clear_cache()

    # 2. Frozen Teacher with SFT Berlin Merged
    print("\n[2/4] Loading and Merging SFT Teacher...")
    t_weights = dict(mx.load(str(args.teacher_weights)))
    if args.lora_path.is_file():
        stats = lora_merge.merge_loras_into_weights(t_weights, [str(args.lora_path)], strength=1.0)
        print(f"  Merged SFT Berlin adapter: {stats['merged']} layers merged into Teacher.")

    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(list(t_weights.items()), strict=False)
    teacher.freeze()
    mx.eval(teacher.parameters())

    # 3. Student Initialization
    print("\n[3/4] Initializing Ternary Student from Teacher...")
    student = dit_mlx_medium.DiT(T_lat=args.crop_len)
    student.load_weights(list(t_weights.items()), strict=False)
    del t_weights
    gc.collect()

    n_ternary = apply_ternary_qat(student, group_size=args.group_size)
    print(f"  Converted {n_ternary} layers to TernaryQATLinear.")

    sigmas = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    active_timesteps = [float(s) for s in sigmas[:-1]]

    pad = mx.zeros((1, 64, 1536), dtype=mx.float16)
    zeros_local = mx.zeros((1, args.crop_len, 257), dtype=mx.float16)
    t_local_pads = [mx.concatenate([pad, layer.to_local_embed(zeros_local)], axis=1) for layer in teacher.transformer.layers]
    mx.eval(*t_local_pads)

    # 4. Stage 1: Student-Forced Block-Wise Distillation
    print(f"\n[4/5] Stage 1: Student-Forced Distillation (blocks {args.start_block}..23)...")
    total_blocks = len(student.transformer.layers)
    t_stage1_start = time.time()

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
            sample = sampler.next_sample()
            latents = mx.array(sample["latents"][:, :args.crop_len]).astype(mx.float16)
            prompt = sample.get("prompt", "minimal-techno; 127 BPM; minimal groove")
            cross_full = prompt_cache.get(prompt, list(prompt_cache.values())[0])

            t_val = active_timesteps[step % len(active_timesteps)]
            t_tensor = mx.array([t_val], dtype=mx.float16)

            noise = mx.random.normal(latents.shape, dtype=latents.dtype)
            noised = latents * (1.0 - t_val) + noise * t_val
            if noised.ndim == 2:
                noised = noised[None, ...]

            c_raw = nn.silu(teacher.to_cond_embed[0](cross_full))
            context = teacher.to_cond_embed[2](c_raw)

            g_raw = nn.silu(teacher.to_global_embed[0](global_cond_val))
            g_pre = teacher.to_global_embed[2](g_raw)

            tf = nn.silu(teacher.to_timestep_embed[0](teacher.timestep_features(t_tensor)))
            global_embed = g_pre + teacher.to_timestep_embed[2](tf)

            gc_emb = nn.silu(teacher.transformer.global_cond_embedder[0](global_embed))
            g_proj = teacher.transformer.global_cond_embedder[2](gc_emb)

            x_lc = noised.transpose(0, 2, 1)
            x_pp = teacher.preprocess_conv(x_lc) + x_lc
            h_in = teacher.transformer.project_in(x_pp)
            mem = mx.broadcast_to(teacher.transformer.memory_tokens[None], (1, 64, 1536))
            h_in = mx.concatenate([mem, h_in], axis=1)

            # Target: Teacher block output
            h_teach = h_in
            for prev_idx in range(b_idx):
                h_teach = teacher.transformer.layers[prev_idx](h_teach, context, g_proj, t_local_pads[prev_idx])
            target = t_block(h_teach, context, g_proj, t_local_pads[b_idx])

            # Student input: cumulative student representation
            h_stud = h_in
            for prev_idx in range(b_idx):
                h_stud = student.transformer.layers[prev_idx](h_stud, context, g_proj, t_local_pads[prev_idx])
            mx.eval(target, h_stud)

            loss, grads = vg(s_block, h_stud, context, g_proj, t_local_pads[b_idx], target)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(s_block, grads)
            mx.eval(s_block.parameters(), opt.state)

        # Eval block final cosine
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

    dt_stage1 = time.time() - t_stage1_start
    print(f"[Done Stage 1] 24 blocs en {dt_stage1/60:.1f} min")

    # 5. Stage 2: Global End-to-End Smoothing Pass
    print(f"\n[5/5] Stage 2: Global End-to-End Smoothing ({args.smoothing_steps} steps)...")
    student.freeze()
    # Unfreeze upper blocks (18..23) for smooth terminal velocity alignment
    for i in range(18, 24):
        student.transformer.layers[i].unfreeze()

    smooth_opt = optim.AdamW(learning_rate=5e-6, weight_decay=1e-4)
    smooth_opt.init(student.trainable_parameters())

    def end_to_end_loss(model, x, t, c, g, target_v):
        v_pred = model(x, t, c, g)
        v32, tg32 = v_pred.astype(mx.float32), target_v.astype(mx.float32)
        norm_mse = mx.mean((v32 - tg32) ** 2) / (mx.mean(tg32 ** 2) + 1e-6)
        cos = mx.sum(v32 * tg32) / (mx.sqrt(mx.sum(v32 ** 2)) * mx.sqrt(mx.sum(tg32 ** 2)) + 1e-6)
        return norm_mse + 2.0 * (1.0 - cos)

    vg_smooth = nn.value_and_grad(student, end_to_end_loss)

    for step in range(1, args.smoothing_steps + 1):
        sample = sampler.next_sample()
        latents = mx.array(sample["latents"][:, :args.crop_len]).astype(mx.float16)
        if latents.ndim == 2:
            latents = latents[None, ...]
        prompt = sample.get("prompt", "minimal-techno; 127 BPM; minimal groove")
        cross_full = prompt_cache.get(prompt, list(prompt_cache.values())[0])

        t_val = active_timesteps[step % len(active_timesteps)]
        t_tensor = mx.array([t_val], dtype=mx.float16)

        noise = mx.random.normal(latents.shape, dtype=latents.dtype)
        noised = latents * (1.0 - t_val) + noise * t_val

        v_teach = teacher(noised, t_tensor, cross_full, global_cond_val)
        mx.eval(v_teach)

        loss, grads = vg_smooth(student, noised, t_tensor, cross_full, global_cond_val, v_teach)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        smooth_opt.update(student, grads)
        mx.eval(student.parameters(), smooth_opt.state)

        if step % 25 == 0 or step == args.smoothing_steps:
            print(f"  Smoothing step {step:03d}/{args.smoothing_steps} | Loss: {float(loss):.4f}")

    print("\n[6] Packing Hybrid BitNet/Bonsai INT2 Model...")
    export_bit_perfect_int2(student, final_int2_path, group_size=args.group_size)

    print("\n[7] End-to-End Diffusion Quality Audit (Teacher vs Student)...")
    sigmas_test = [1.0, 0.891, 0.746, 0.512, 0.274]
    key_test = mx.random.key(2026106744)
    x_test = mx.random.normal((1, 256, args.crop_len), dtype=mx.float16, key=key_test)
    c_test = list(prompt_cache.values())[0]

    print(f"  {'Sigma':>6} | {'Cos Sim':>8} | {'Teacher RMS':>11} | {'Student RMS':>11}")
    for s in sigmas_test:
        s_arr = mx.array([s], dtype=mx.float16)
        v_teach = teacher(x_test, s_arr, c_test, global_cond_val)
        v_stud = student(x_test, s_arr, c_test, global_cond_val)
        mx.eval(v_teach, v_stud)
        vt, vs = v_teach.astype(mx.float32), v_stud.astype(mx.float32)
        cos = float(mx.sum(vt * vs) / (mx.sqrt(mx.sum(vt**2)) * mx.sqrt(mx.sum(vs**2)) + 1e-6))
        t_rms = float(mx.sqrt(mx.mean(vt**2)))
        s_rms = float(mx.sqrt(mx.mean(vs**2)))
        print(f"  {s:6.3f} | {cos:8.4f} | {t_rms:11.4f} | {s_rms:11.4f}")

    print("=== Distillation & Smoothing Complete ===")


if __name__ == "__main__":
    main()
