"""Teacher-Guided Distillation & Recalibration for Ternary Stable Audio 3 DiT.

Aligns the Ternary QAT student directly with the FP16 teacher:
L = ||v_student - v_teacher||^2 + (1 - cos_sim)
Runs strictly under 12 GB RAM.
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
sys.path = [str(MLX_RUNTIME_ROOT), str(MLX_RUNTIME_ROOT / "scripts")] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map_with_path, tree_flatten
import numpy as np

from sa3_mlx import T5GEMMA_NPZ_REL, DIT_CHOICES
from weights import ensure_local
from models.defs.sa3_pipeline import apply_prompt_padding, load_conditioner_from_npz
from models.defs.t5gemma_mlx import T5Gemma
from models.defs.training import sample_training_timesteps, shift_training_timesteps
from models.defs.latent_dataset import PreEncodedLatentDataset, iterate_batches
from models.defs import dit_mlx_medium


class TernaryQATLinear(nn.Module):
    """Bonsai-style ternary linear layer with FP32 master weights and STE."""

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
        w_q = (q * scale).reshape(out_d, in_d)
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


def grad_checkpoint(layer):
    fn = type(layer).__call__
    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)
    type(layer).__call__ = checkpointed_fn


def quantize_dit(model: nn.Module, bits: int = 2, group_size: int = 64):
    def predicate(path: str, layer: nn.Module) -> bool:
        if isinstance(layer, nn.Linear):
            shape = tuple(int(v) for v in layer.weight.shape)
            if shape[-1] % group_size == 0:
                return True
        return False
    nn.quantize(model, group_size=group_size, bits=bits, mode="affine", class_predicate=predicate)


def pack_exact_ternary_int2(student_model: nn.Module, out_path: Path, group_size: int = 64):
    """Pack student ternary weights into MLX native 2-bit affine container."""
    
    flat_params = {}
    for k, v in tree_flatten(student_model.parameters()):
        if "weight" in k and v.ndim == 2 and v.shape[1] % group_size == 0:
            out_d, in_d = v.shape
            g = v.reshape(out_d, -1, group_size).astype(mx.float32)
            scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5
            q = mx.clip(mx.round(g / scale), -1.0, 1.0)
            w_q = (q * scale).reshape(out_d, in_d).astype(mx.float16)
            flat_params[k] = w_q
        else:
            flat_params[k] = v.astype(mx.float16)
            
    temp_fp16 = out_path.with_suffix(".tmp_f16.npz")
    mx.savez(str(temp_fp16), **flat_params)
    
    target_dit = dit_mlx_medium.DiT(T_lat=128)
    target_dit.load_weights(str(temp_fp16), strict=False)
    
    quantize_dit(target_dit, bits=2, group_size=group_size)
    target_dit.save_weights(str(out_path))
    temp_fp16.unlink(missing_ok=True)
    print(f"[Export] Packed 2-bit model saved to {out_path} ({out_path.stat().st_size / (1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Teacher-Guided Distillation & Recalibration for Ternary DiT")
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--teacher-weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--init-weights", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/qat-bonsai-runs/dit_medium_bonsai_ternary_final.npz"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--crop-len", type=int, default=54)
    parser.add_argument("--blocks-per-stage", type=int, default=4)
    parser.add_argument("--steps-per-stage", type=int, default=30)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/qat-distill-runs"))
    parser.add_argument("--final-int2-path", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_ternary_int2_group64.npz"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_file = args.output_dir / "distill_status.json"
    log_file = args.output_dir / "distill_log.jsonl"

    print("=== Bonsai Teacher-Guided Distillation & Recalibration ===")

    # 1. Conditioning cache
    print("[Conditioner] Pre-computing full conditioning (prompt + seconds + global)...")
    prompt_cfg = json.loads(args.prompt_config.read_text()) if args.prompt_config.is_file() else None
    dataset = PreEncodedLatentDataset(str(args.latents_dir), args.crop_len, random_crop=False, prompt_config=prompt_cfg, seed=42)
    
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(args.teacher_weights), prefix="cond.")
    
    unique_prompts = set()
    for i in range(len(dataset)):
        it = dataset[i]
        if it and "prompt" in it:
            unique_prompts.add(it["prompt"])

    seconds = 5.0
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

    print(f"[Conditioner] Cached {len(prompt_cache)} conditionings. Freeing T5...")
    del t5
    gc.collect()
    mx.clear_cache()

    # 2. Teacher (Frozen FP16)
    print(f"[Teacher] Loading frozen teacher from {args.teacher_weights.name}...")
    teacher = dit_mlx_medium.load_dit(str(args.teacher_weights), T_lat=args.crop_len, dtype=mx.float16, compile_=False)
    teacher.freeze()

    # 3. Student (Ternary QAT)
    source_weights = args.init_weights if args.init_weights.is_file() else args.teacher_weights
    print(f"[Student] Initializing student from {source_weights.name}...")
    student = dit_mlx_medium.load_dit(str(source_weights), T_lat=args.crop_len, dtype=mx.float16, compile_=False)
    converted = apply_ternary_qat(student, group_size=args.group_size)
    print(f"[Student] Converted {converted} layers to TernaryQATLinear.")
    grad_checkpoint(student.transformer.layers[0])

    total_blocks = len(student.transformer.layers)
    num_stages = math.ceil(total_blocks / args.blocks_per_stage)
    total_steps = args.cycles * num_stages * args.steps_per_stage
    global_step = 0

    rng = np.random.default_rng(42)
    log_handle = log_file.open("w")
    t_start = time.time()

    print(f"[Training] Distilling {total_blocks} blocks ({num_stages} stages/cycle, {args.cycles} cycles, {total_steps} total steps)...")

    for cycle in range(1, args.cycles + 1):
        print(f"\n>>> Distillation Cycle {cycle}/{args.cycles} <<<")
        for stage in range(1, num_stages + 1):
            s_idx = stage - 1
            start_block = s_idx * args.blocks_per_stage
            end_block = min(start_block + args.blocks_per_stage, total_blocks)
            print(f"\n--- Stage {stage}/{num_stages}: Distilling Blocks {start_block} to {end_block-1} ---")

            student.freeze()
            for b_idx in range(start_block, end_block):
                for m in student.transformer.layers[b_idx].modules():
                    if isinstance(m, TernaryQATLinear):
                        m.unfreeze()

            optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)
            optimizer.init(student.trainable_parameters())

            def distill_loss(model, latents, timesteps, cross_cond, gcond, v_target):
                v_stud = model(latents, timesteps, cross_cond, gcond)
                diff = v_stud.astype(mx.float32) - v_target.astype(mx.float32)
                mse = mx.mean(diff ** 2)
                
                dot = mx.sum(v_stud.astype(mx.float32) * v_target.astype(mx.float32))
                norm_prod = mx.sqrt(mx.sum(v_stud.astype(mx.float32)**2)) * mx.sqrt(mx.sum(v_target.astype(mx.float32)**2)) + 1e-6
                cos_loss = 1.0 - (dot / norm_prod)
                
                return mse + 2.0 * cos_loss

            vg = nn.value_and_grad(student, distill_loss)
            data_iter = iterate_batches(dataset, batch_size=1, seed=42 + global_step)

            for step in range(1, args.steps_per_stage + 1):
                global_step += 1
                t0 = time.time()

                batch = next(data_iter)
                latents = mx.array(batch["latents"][:, :, :args.crop_len])
                prompt = batch["prompt"][0]
                cross_cond = prompt_cache[prompt]

                raw_t = sample_training_timesteps("uniform", 1, rng=rng)
                t_shifted = shift_training_timesteps(raw_t, args.crop_len, shift_type="full", options={"min_length": 256, "max_length": 4096})
                timesteps = mx.array(t_shifted)
                
                t_val = timesteps.astype(mx.float32)[:, None, None]
                noise = mx.random.normal(latents.shape, dtype=latents.dtype)
                noised_latents = latents * (1.0 - t_val) + noise * t_val

                # Evaluate teacher forward cleanly outside the autograd tape
                v_teach = teacher(noised_latents, timesteps, cross_cond, global_cond)
                mx.eval(v_teach)

                loss, grads = vg(student, noised_latents, timesteps, cross_cond, global_cond, v_teach)
                grads, _ = optim.clip_grad_norm(grads, 1.0)
                optimizer.update(student, grads)
                mx.eval(student.parameters(), optimizer.state)
                mx.clear_cache()

                step_sec = time.time() - t0
                peak_gb = mx.get_peak_memory() / (1024 ** 3) if hasattr(mx, "get_peak_memory") else 0.0

                entry = {
                    "step": global_step,
                    "total": total_steps,
                    "cycle": cycle,
                    "stage": stage,
                    "blocks": f"{start_block}-{end_block-1}",
                    "loss": round(float(loss.item()), 4),
                    "ram_gb": round(peak_gb, 2),
                    "sec": round(step_sec, 2),
                }
                log_handle.write(json.dumps(entry) + "\n")
                log_handle.flush()

                if step % 10 == 0 or step == 1:
                    print(f"Step {global_step:03d}/{total_steps:03d} | Loss: {entry['loss']:.4f} | RAM: {peak_gb:.2f} GB | {step_sec:.2f}s/step")

    log_handle.close()
    del teacher
    gc.collect()
    mx.clear_cache()

    distill_master = args.output_dir / "dit_medium_bonsai_distilled_master.npz"
    flat_params = dict(tree_flatten(student.parameters()))
    mx.savez(str(distill_master), **{k: v.astype(mx.float16) for k, v in flat_params.items()})
    print(f"[Master] Distilled FP16 master weights saved: {distill_master}")

    print(f"[Packing] Packing calibrated ternary weights into {args.final_int2_path.name}...")
    pack_exact_ternary_int2(student, args.final_int2_path, group_size=args.group_size)
    print("=== Distillation & Calibration Finished Successfully ===")


if __name__ == "__main__":
    main()
