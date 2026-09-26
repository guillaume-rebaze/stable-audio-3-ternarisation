"""Bonsai-style Quantization-Aware Training (QAT) for Stable Audio 3 DiT on Apple Silicon.

Implements:
- Ternary {-s, 0, +s} weights with Straight-Through Estimator (STE) and group scaling.
- Gradient checkpointing + compiled training graph.
- Batch size = 1 with pre-encoded latents cache for strict < 12 GB RAM footprint.
- Memory and loss telemetry per step.
"""

from __future__ import annotations

import argparse
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
import numpy as np


class TernaryQATLinear(nn.Module):
    """Linear layer using group-wise ternary weights {-s, 0, +s} with STE."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = False, group_size: int = 64):
        super().__init__()
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.group_size = int(group_size)
        self.weight = mx.zeros((self.output_dims, self.input_dims), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((self.output_dims,), dtype=mx.float16)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, layer: nn.Linear, group_size: int = 64) -> TernaryQATLinear:
        has_bias = "bias" in layer and layer.bias is not None
        mod = cls(layer.weight.shape[1], layer.weight.shape[0], bias=has_bias, group_size=group_size)
        mod.weight = layer.weight.astype(mx.float16)
        if has_bias:
            mod.bias = layer.bias.astype(mx.float16)
        return mod

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight
        out_d, in_d = w.shape
        g = w.reshape(out_d, -1, self.group_size)
        scale = mx.mean(mx.abs(g), axis=-1, keepdims=True) + 1e-5
        scaled = g / scale
        q = mx.clip(mx.round(scaled), -1.0, 1.0)
        w_q = (q * scale).reshape(out_d, in_d)
        
        # Straight-Through Estimator (STE)
        w_eff = w + mx.stop_gradient(w_q - w)
        y = x @ w_eff.astype(x.dtype).T
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


def apply_ternary_qat(model: nn.Module, group_size: int = 64) -> int:
    """Convert only transformer blocks linear layers to TernaryQATLinear."""
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
    """Enable activation recomputation for transformer block class."""
    fn = type(layer).__call__
    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)
    type(layer).__call__ = checkpointed_fn


def main():
    parser = argparse.ArgumentParser(description="Bonsai QAT for Stable Audio 3 DiT")
    parser.add_argument("--latents-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s"))
    parser.add_argument("--prompt-config", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/prompt-config.json"))
    parser.add_argument("--weights", type=Path,
                        default=MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/sample-expertise-pilot/sftberlin/qat-bonsai-runs"))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Bonsai QAT Training (Group size: {args.group_size}, LR: {args.lr}) ===")
    
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    # 1. Load Dataset
    prompt_cfg = json.loads(args.prompt_config.read_text()) if args.prompt_config.is_file() else None
    dataset = PreEncodedLatentDataset(str(args.latents_dir), args.crop_len, random_crop=False, prompt_config=prompt_cfg, seed=42)
    print(f"[Data] Loaded {len(dataset)} pre-encoded latent samples (crop: {args.crop_len}).")

    # 2. Load Base Model
    print(f"[Model] Loading base DiT weights from {args.weights.name}...")
    dit_model = dit_mlx_medium.load_dit(str(args.weights), T_lat=args.crop_len, dtype=mx.float16, compile_=False)
    
    # 3. Apply Ternary QAT Layers with STE
    converted = apply_ternary_qat(dit_model, group_size=args.group_size)
    print(f"[QAT] Converted {converted} Linear layers to TernaryQATLinear (group={args.group_size}).")

    # 4. Enable Gradient Checkpointing
    grad_checkpoint(dit_model.transformer.layers[0])
    print(f"[Memory] Gradient Checkpointing enabled across {len(dit_model.transformer.layers)} transformer blocks.")

    # 5. Conditioner & T5 Cache
    print("[Conditioner] Loading T5 and text projections...")
    t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = load_conditioner_from_npz(str(args.weights), prefix="cond.")
    t5_cache = {}

    def get_prompt_emb(prompt_text: str):
        if prompt_text not in t5_cache:
            emb, mask = t5.encode([prompt_text], max_len=256)
            padded = apply_prompt_padding(emb.astype(mx.float32), mask, padding_emb.astype(mx.float32))
            mx.eval(padded)
            t5_cache[prompt_text] = padded
        return t5_cache[prompt_text]

    # 6. Training Loss and Step
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(model, latents, timesteps, prompt_emb, seconds_total):
        cross16 = prompt_emb.astype(mx.float16)
        global16 = mx.zeros((latents.shape[0], 768), dtype=mx.float16)
        
        noise = mx.random.normal(latents.shape, dtype=latents.dtype)
        t = timesteps.astype(mx.float32)[:, None, None]
        noised = latents * (1.0 - t) + noise * t
        target = noise - latents
        
        pred = model(noised, timesteps, cross16, global16)
        loss = mx.mean((pred.astype(mx.float32) - target.astype(mx.float32)) ** 2)
        return loss

    vg = nn.value_and_grad(dit_model, loss_fn)
    optimizer.init(dit_model.trainable_parameters())
    state = [dit_model.state, optimizer.state, mx.random.state]

    def train_step(latents, timesteps, prompt_emb, seconds_total):
        loss, grads = vg(dit_model, latents, timesteps, prompt_emb, seconds_total)
        if args.grad_clip > 0.0:
            grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(dit_model, grads)
        return loss

    train_step = mx.compile(train_step, inputs=state, outputs=state)

    print("\n[Start] Launching QAT Training loop...")
    step = 0
    rng = np.random.default_rng(42)
    t_start = time.time()
    
    log_file = args.output_dir / "training_log.jsonl"
    log_handle = log_file.open("w")

    for batch in iterate_batches(dataset, batch_size=1, seed=42):
        if step >= args.max_steps:
            break
            
        step += 1
        t_step0 = time.time()

        latents = mx.array(batch["latents"])
        seconds_total = batch["seconds_total"]
        prompt = batch["prompt"][0]
        prompt_emb = get_prompt_emb(prompt)

        raw_t = sample_training_timesteps("uniform", 1, rng=rng)
        t_shifted = shift_training_timesteps(raw_t, args.crop_len, shift_type="full", options={"min_length": 256, "max_length": 4096})
        timesteps = mx.array(t_shifted)

        loss = train_step(latents, timesteps, prompt_emb, seconds_total)
        mx.eval(state)

        step_time = time.time() - t_step0
        peak_gb = mx.get_peak_memory() / (1024 ** 3) if hasattr(mx, "get_peak_memory") else 0.0

        log_entry = {
            "step": step,
            "loss": float(loss.item()),
            "step_sec": round(step_time, 3),
            "peak_ram_gb": round(peak_gb, 2),
            "elapsed_sec": round(time.time() - t_start, 1)
        }
        log_handle.write(json.dumps(log_entry) + "\n")
        log_handle.flush()

        if step % 5 == 0 or step == 1:
            print(f"Step {step:03d}/{args.max_steps:03d} | Loss: {loss.item():.4f} | RAM: {peak_gb:.2f} GB | Time: {step_time:.2f}s/step")

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            ckpt_path = args.output_dir / f"dit_medium_bonsai_qat_step{step:03d}.npz"
            dit_model.save_weights(str(ckpt_path))
            print(f"  --> Saved checkpoint: {ckpt_path.name}")

    log_handle.close()
    print(f"\n[Done] Training completed in {time.time() - t_start:.1f}s. Artifacts saved in {args.output_dir}")


if __name__ == "__main__":
    main()
