"""Complete Bonsai-style Block-wise Progressive 1-bit (Binary) QAT for Stable Audio 3 DiT.

Runs strictly under 12 GB RAM (~9.0 GB peak):
- Binary sign weights {-s, +s} with group scaling and STE.
- FP32 master weights for AdamW numerical stability.
- Rolling single checkpoint in FP16 to keep disk footprint <= 2.8 GB.
- Progressively adapts all 24 transformer blocks in stages of 4 blocks.
- Gradient checkpointing + explicit MLX memory cache flushing.
- Exports final packed 1-bit model (~240 MB) upon completion.
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
sys.path.insert(0, str(MLX_RUNTIME_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map_with_path

from sa3_mlx import T5GEMMA_NPZ_REL
from weights import ensure_local
from models.defs.sa3_pipeline import apply_prompt_padding, load_conditioner_from_npz
from models.defs.t5gemma_mlx import T5Gemma
from models.defs.training import sample_training_timesteps, shift_training_timesteps
from models.defs.latent_dataset import PreEncodedLatentDataset, iterate_batches
from models.defs import dit_mlx_medium
from run_sftberlin_quantized import _int1_classes
import numpy as np


class BinaryQATLinear(nn.Module):
    """Group-wise binary weights {-s, +s} with FP32 master weights and STE."""

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
    def from_linear(cls, layer: nn.Linear, group_size: int = 64) -> BinaryQATLinear:
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
        
        # Binary 1-bit projection {-s, +s}
        q = mx.where(g >= 0, 1.0, -1.0)
        w_q = (q * scale).reshape(out_d, in_d)
        
        # STE: forward uses binary weights, backward updates continuous FP32 master weights
        w_eff = w + mx.stop_gradient(w_q - w)
        y = x @ w_eff.astype(x.dtype).T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


def apply_binary_qat(model: nn.Module, group_size: int = 64) -> int:
    count = 0
    for block in model.transformer.layers:
        def convert(path: str, layer: nn.Module) -> nn.Module:
            nonlocal count
            if isinstance(layer, nn.Linear) and layer.weight.shape[1] % group_size == 0:
                count += 1
                return BinaryQATLinear.from_linear(layer, group_size=group_size)
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


def save_compact_checkpoint(model: nn.Module, path: Path):
    temp_path = path.with_name(path.name + ".tmp.npz")
    flat = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
    mx.savez(str(temp_path), **flat)
    temp_path.replace(path)


def export_packed_1bit(model: nn.Module, destination: Path, group_size: int = 64):
    """Convert binary QAT layers to packed INT1 (1-bit) representation and save."""
    binary_linear, _ = _int1_classes()
    temp_path = destination.with_name(destination.name + ".tmp.npz")
    
    # Clone and convert to packed format
    for block in model.transformer.layers:
        def convert_to_packed(path: str, layer: nn.Module) -> nn.Module:
            if isinstance(layer, BinaryQATLinear):
                # Create a standard linear view to feed from_linear
                dummy = nn.Linear(layer.input_dims, layer.output_dims, bias=(layer.bias is not None))
                dummy.weight = layer.weight.astype(mx.float16)
                if layer.bias is not None:
                    dummy.bias = layer.bias.astype(mx.float16)
                return binary_linear.from_linear(dummy, group_size)
            return layer

        leaves = tree_map_with_path(convert_to_packed, block.leaf_modules(), is_leaf=nn.Module.is_module)
        block.update_modules(leaves)
        
    flat = {k: v for k, v in tree_flatten(model.parameters())}
    mx.savez(str(temp_path), **flat)
    temp_path.replace(destination)


def main():
    parser = argparse.ArgumentParser(description="Bonsai 1-bit (Binary) QAT for Stable Audio 3 DiT")
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--init-weights", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/qat-bonsai-runs/dit_medium_bonsai_ternary_final.npz"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--blocks-per-stage", type=int, default=4)
    parser.add_argument("--steps-per-stage", type=int, default=40)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/qat-bonsai-1bit-runs"))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_file = args.output_dir / "training_status.json"
    log_file = args.output_dir / "training_log.jsonl"

    print(f"=== Bonsai 1-bit (Binary) QAT (Blocks: 24, Stages of {args.blocks_per_stage}, Cycles: {args.cycles}) ===")

    # 1. Dataset
    prompt_cfg = json.loads(args.prompt_config.read_text()) if args.prompt_config.is_file() else None
    dataset = PreEncodedLatentDataset(str(args.latents_dir), args.crop_len, random_crop=False, prompt_config=prompt_cfg, seed=42)
    print(f"[Data] Loaded {len(dataset)} pre-encoded latent samples.")

    # 2. Pre-encode prompts & evict T5
    print("[Conditioner] Pre-encoding unique prompts and freeing T5 from RAM...")
    base_weights = MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(base_weights), prefix="cond.")

    unique_prompts = set()
    for i in range(len(dataset)):
        it = dataset[i]
        if it and "prompt" in it:
            unique_prompts.add(it["prompt"])

    prompt_cache = {}
    for p in unique_prompts:
        emb, mask = t5.encode([p], max_len=256)
        padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32))
        mx.eval(padded)
        prompt_cache[p] = padded

    print(f"[Conditioner] Cached {len(prompt_cache)} prompt embeddings. Evicting T5...")
    del t5
    gc.collect()
    mx.clear_cache()

    # 3. Load Model (init from ternary model for high-efficiency adaptation)
    init_path = args.init_weights if args.init_weights.is_file() else base_weights
    print(f"[Model] Loading initial weights from {init_path.name}...")
    dit_model = dit_mlx_medium.load_dit(str(init_path), T_lat=args.crop_len, dtype=mx.float16, compile_=False)

    # 4. Convert all Linear layers to BinaryQATLinear with FP32 master weights
    converted = apply_binary_qat(dit_model, group_size=args.group_size)
    print(f"[QAT 1-bit] Converted {converted} Linear layers to BinaryQATLinear (group={args.group_size}).")

    # 5. Gradient Checkpointing
    grad_checkpoint(dit_model.transformer.layers[0])
    print(f"[Memory] Gradient checkpointing active.")

    total_blocks = len(dit_model.transformer.layers)
    num_stages = math.ceil(total_blocks / args.blocks_per_stage)
    total_steps = args.cycles * num_stages * args.steps_per_stage

    global_step = 0
    t_start = time.time()
    rng = np.random.default_rng(42)
    log_handle = log_file.open("w")

    print(f"[Schedule] {total_blocks} blocks -> {num_stages} stages/cycle -> {args.cycles} cycles -> Total: {total_steps} steps.")

    for cycle in range(1, args.cycles + 1):
        print(f"\n>>> Starting 1-bit Cycle {cycle}/{args.cycles} <<<")
        for stage in range(1, num_stages + 1):
            s_idx = stage - 1
            start_block = s_idx * args.blocks_per_stage
            end_block = min(start_block + args.blocks_per_stage, total_blocks)
            print(f"\n--- Stage {stage}/{num_stages}: Training Blocks {start_block} to {end_block-1} in 1-bit ---")

            dit_model.freeze()
            for b_idx in range(start_block, end_block):
                for m in dit_model.transformer.layers[b_idx].modules():
                    if isinstance(m, BinaryQATLinear):
                        m.unfreeze()

            optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
            optimizer.init(dit_model.trainable_parameters())

            def loss_fn(model, latents, timesteps, prompt_emb):
                cross16 = prompt_emb.astype(mx.float16)
                global16 = mx.zeros((latents.shape[0], 768), dtype=mx.float16)
                
                noise = mx.random.normal(latents.shape, dtype=latents.dtype)
                t = timesteps.astype(mx.float32)[:, None, None]
                noised = latents * (1.0 - t) + noise * t
                target = noise - latents
                
                pred = model(noised, timesteps, cross16, global16)
                return mx.mean((pred.astype(mx.float32) - target.astype(mx.float32)) ** 2)

            vg = nn.value_and_grad(dit_model, loss_fn)
            data_iter = iterate_batches(dataset, batch_size=1, seed=42 + global_step)

            for step_in_stage in range(1, args.steps_per_stage + 1):
                global_step += 1
                t_step0 = time.time()

                batch = next(data_iter)
                latents = mx.array(batch["latents"])
                prompt = batch["prompt"][0]
                prompt_emb = prompt_cache[prompt]

                raw_t = sample_training_timesteps("uniform", 1, rng=rng)
                t_shifted = shift_training_timesteps(raw_t, args.crop_len, shift_type="full", options={"min_length": 256, "max_length": 4096})
                timesteps = mx.array(t_shifted)

                loss, grads = vg(dit_model, latents, timesteps, prompt_emb)
                if args.grad_clip > 0.0:
                    grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
                optimizer.update(dit_model, grads)
                mx.eval(dit_model.parameters(), optimizer.state)
                mx.clear_cache()

                step_sec = time.time() - t_step0
                peak_gb = mx.get_peak_memory() / (1024 ** 3) if hasattr(mx, "get_peak_memory") else 0.0

                entry = {
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "cycle": cycle,
                    "stage": stage,
                    "blocks": f"{start_block}-{end_block-1}",
                    "loss": round(float(loss.item()), 4),
                    "peak_ram_gb": round(peak_gb, 2),
                    "step_sec": round(step_sec, 3),
                    "elapsed_min": round((time.time() - t_start) / 60.0, 1),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                log_handle.write(json.dumps(entry) + "\n")
                log_handle.flush()

                with open(status_file, "w") as sf:
                    json.dump(entry, sf, indent=2)

                if step_in_stage % 10 == 0 or step_in_stage == 1:
                    print(f"Step {global_step:04d}/{total_steps:04d} (C{cycle}S{stage}) | Loss: {entry['loss']:.4f} | RAM: {peak_gb:.2f} GB | {step_sec:.2f}s/step")

            # Rolling compact master checkpoint
            save_compact_checkpoint(dit_model, args.output_dir / "dit_medium_bonsai_1bit_latest.npz")

    log_handle.close()

    # Final Master Checkpoint
    final_master_path = args.output_dir / "dit_medium_bonsai_1bit_final.npz"
    save_compact_checkpoint(dit_model, final_master_path)
    print(f"\n[COMPLETE] 1-bit QAT finished! Master weights saved to {final_master_path}")

    # Export Packed 1-bit INT1 model for inference
    packed_path = Path("output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_1bit_packed_group64.npz")
    print(f"[Export] Packing into true 1-bit bit-packed INT1 format...")
    export_packed_1bit(dit_model, packed_path, group_size=args.group_size)
    print(f"[Export] Packed 1-bit model saved to {packed_path} ({packed_path.stat().st_size / (1024**2):.1f} MB)!")


if __name__ == "__main__":
    main()
