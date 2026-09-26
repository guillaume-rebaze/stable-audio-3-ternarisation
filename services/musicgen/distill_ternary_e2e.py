"""End-to-End Ternary QAT Distillation for Stable Audio 3 DiT Medium.

Key improvement over block-wise: trains the FULL student forward pass against
the teacher, preventing inter-block error accumulation.

- TernaryQATLinear with STE: forward uses quantized weights, backward flows
  through identity (straight-through estimator).
- Loss: velocity field MSE + cosine alignment at full-network output level.
- 350 steps, cosine LR decay, AdamW.
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
# Ternary QAT Linear with STE + optimal least-squares scale
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
        # Mean-abs scale (base threshold)
        scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5
        # Ternary quantization
        q = mx.clip(mx.round(g / scale), -1.0, 1.0)
        # Least-squares optimal magnitude
        s_opt = mx.sum(g * q, axis=-1, keepdims=True) / (mx.sum(q**2, axis=-1, keepdims=True) + 1e-5)
        # Quantized weight via STE
        w_q = (q * s_opt).reshape(out_d, in_d)
        w_eff = w + mx.stop_gradient(w_q - w)  # STE: grad flows through continuous w
        y = x @ w_eff.astype(x.dtype).T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


def apply_ternary_qat(model: nn.Module, group_size: int = 64) -> int:
    """Replace all quantizable Linear layers in the FULL model with TernaryQATLinear."""
    count = 0

    def convert(path: str, layer: nn.Module) -> nn.Module:
        nonlocal count
        if isinstance(layer, nn.Linear) and layer.weight.shape[1] % group_size == 0:
            count += 1
            return TernaryQATLinear.from_linear(layer, group_size=group_size)
        return layer

    leaves = tree_map_with_path(convert, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return count


# ---------------------------------------------------------------------------
# Bit-perfect packing
# ---------------------------------------------------------------------------

def pack_bit_perfect_ternary(student_model: nn.Module, teacher_weights_path: Path,
                              out_path: Path, group_size: int = 64):
    """Pack quantized weights into MLX native 2-bit affine container."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--teacher-weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--crop-len", type=int, default=64)
    parser.add_argument("--output-model", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_ternary_int2_group64.npz"))
    parser.add_argument("--init-from-master", action="store_true")
    args = parser.parse_args()

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    print("=== End-to-End Ternary QAT Distillation ===")

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
    print("[3/4] Initializing student with TernaryQATLinear...")
    master_path = args.output_model.parent / "dit_medium_bonsai_distilled_master.npz"

    if args.init_from_master and master_path.exists():
        print(f"  Init from master: {master_path}")
        student = dit_mlx_medium.load_dit(str(master_path), T_lat=args.crop_len,
                                           dtype=mx.float16, compile_=False)
    else:
        print("  Init from teacher FP16 weights (clean start)")
        student = dit_mlx_medium.load_dit(str(args.teacher_weights), T_lat=args.crop_len,
                                           dtype=mx.float16, compile_=False)

    n_ternary = apply_ternary_qat(student, group_size=args.group_size)
    print(f"  Converted {n_ternary} layers to TernaryQATLinear.")

    sigmas = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    active_timesteps = [float(s) for s in sigmas[:-1]]

    # 4. End-to-end training
    print(f"\n[4/4] End-to-end distillation ({args.steps} steps)...")

    lr_sched = optim.cosine_decay(args.lr, args.steps, end=args.lr * 0.05)
    optimizer = optim.AdamW(learning_rate=lr_sched, weight_decay=1e-4)
    optimizer.init(student.trainable_parameters())

    def loss_fn(student_model, noised, t_tensor, cross_ctx, global_raw, v_teacher):
        v_student = student_model(noised, t_tensor, cross_ctx, global_raw)
        vt = v_teacher.astype(mx.float32)
        vs = v_student.astype(mx.float32)
        norm_mse = mx.mean((vs - vt) ** 2) / (mx.mean(vt ** 2) + 1e-6)
        cos = mx.sum(vs * vt) / (mx.sqrt(mx.sum(vs ** 2)) * mx.sqrt(mx.sum(vt ** 2)) + 1e-6)
        norm_diff = mx.abs(mx.sqrt(mx.mean(vs ** 2)) - mx.sqrt(mx.mean(vt ** 2))) / (mx.sqrt(mx.mean(vt ** 2)) + 1e-6)
        return norm_mse + 2.0 * (1.0 - cos) + 0.5 * norm_diff

    vg_fn = nn.value_and_grad(student, loss_fn)

    def infinite_batches(ds, seed=0):
        ep = 0
        while True:
            for b in iterate_batches(ds, batch_size=1, seed=seed + ep):
                yield b
            ep += 1

    data_iter = infinite_batches(dataset, seed=42)
    t_start = time.time()
    best_cos = -1.0

    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        latents = mx.array(batch["latents"][:, :, :args.crop_len]).astype(mx.float16)
        prompt = batch["prompt"][0]
        cross_ctx = prompt_cache[prompt]
        t_val = active_timesteps[(step - 1) % len(active_timesteps)]
        t_tensor = mx.array([t_val], dtype=mx.float16)

        noise = mx.random.normal(latents.shape, dtype=latents.dtype)
        noised = latents * (1.0 - t_val) + noise * t_val

        # Pre-compute teacher output (no grad stored)
        v_teacher = mx.stop_gradient(teacher(noised, t_tensor, cross_ctx, global_cond_base))
        mx.eval(v_teacher)

        loss, grads = vg_fn(student, noised, t_tensor, cross_ctx, global_cond_base, v_teacher)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(student, grads)
        mx.eval(student.parameters(), optimizer.state, loss)

        if step % 25 == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t_start
            # Eval cos sim
            vs = student(noised, t_tensor, cross_ctx, global_cond_base).astype(mx.float32)
            vt = v_teacher.astype(mx.float32)
            cos = float(mx.sum(vt * vs) / (mx.sqrt(mx.sum(vt**2)) * mx.sqrt(mx.sum(vs**2)) + 1e-6))
            peak_gb = mx.get_peak_memory() / (1024**3) if hasattr(mx, "get_peak_memory") else 0.0
            eta = (args.steps - step) * (elapsed / max(step, 1))
            print(f"  Step {step:4d}/{args.steps} | Loss: {float(loss):.4f} | Cos: {cos:.4f} | "
                  f"GradNorm: {float(grad_norm):.3f} | RAM: {peak_gb:.1f}GB | ETA: {eta:.0f}s")

            if cos > best_cos:
                best_cos = cos
                ckpt = args.output_model.parent / "dit_medium_bonsai_e2e_best_master.npz"
                flat = dict(tree_flatten(student.parameters()))
                tmp = ckpt.with_suffix(".tmp.npz")
                mx.savez(str(tmp), **{k: v.astype(mx.float16) for k, v in flat.items()})
                os.replace(str(tmp), str(ckpt))
                print(f"    -> Best checkpoint saved (cos={cos:.4f})")

        del v_teacher
        gc.collect()
        mx.clear_cache()

    print(f"\n[Done] {args.steps} steps | Best cos: {best_cos:.4f} | Time: {time.time()-t_start:.0f}s")

    # 5. Save master FP16
    print("[5] Saving master FP16...")
    flat_master = dict(tree_flatten(student.parameters()))
    tmp_master = master_path.with_suffix(".tmp.npz")
    mx.savez(str(tmp_master), **{k: v.astype(mx.float16) for k, v in flat_master.items()})
    os.replace(str(tmp_master), str(master_path))
    print(f"  Saved: {master_path}")

    # 6. Pack ternary
    del teacher
    gc.collect()
    mx.clear_cache()
    print("[6] Packing bit-perfect ternary...")
    pack_bit_perfect_ternary(student, args.teacher_weights, args.output_model, group_size=args.group_size)
    print("=== Complete ===")


if __name__ == "__main__":
    main()
