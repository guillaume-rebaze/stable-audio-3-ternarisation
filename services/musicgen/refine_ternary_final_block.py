"""Refine the terminal ternary block with a true E2E velocity loss.

The first quality run showed that blocks 0--22 remained close to the teacher,
while block 23 collapsed the final hidden state.  This targeted pass keeps the
already validated prefix from disk and only reopens the last block.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np

import train_ternary_quality as tq
from audit_ternary_quality import load_manifest, load_student
from models.defs import dit_mlx_medium
from ternary_contract import dequantize_packed, quantize_weight, write_json
from mlx.utils import tree_flatten

mx = tq.mx
nn = tq.nn
optim = tq.optim


def replace_quantized_with_qat(block: nn.Module, group_size: int) -> None:
    for name in tq.CORE_NAMES:
        quantized = tq.module_at(block, name)
        packed = np.array(quantized.weight)
        scales = np.array(quantized.scales)
        biases = np.array(quantized.biases)
        dense = dequantize_packed(packed, scales, biases, group_size)
        qat = tq.TernaryQATLinear(
            int(dense.shape[1]),
            int(dense.shape[0]),
            getattr(quantized, "bias", None) is not None,
            group_size,
        )
        qat.weight = mx.array(dense, dtype=mx.float32)
        if qat.bias is not None:
            qat.bias = quantized.bias.astype(mx.float32)
        tq.set_module_at(block, name, qat)


def velocity_loss(
    block: nn.Module,
    h_student: mx.array,
    context: mx.array,
    global_cond: mx.array,
    local_pad: mx.array,
    target_hidden: mx.array,
    project_out: nn.Module,
    postprocess_conv: nn.Module,
    target_velocity: mx.array,
) -> mx.array:
    output = block(h_student, context, global_cond, local_pad)
    output_audio = output[:, dit_mlx_medium.NUM_MEMORY_TOKENS :, :].astype(mx.float32)
    target_audio = target_hidden[:, dit_mlx_medium.NUM_MEMORY_TOKENS :, :].astype(mx.float32)
    hidden_mse = mx.mean((output_audio - target_audio) ** 2) / (
        mx.mean(target_audio * target_audio) + 1e-6
    )
    hidden_cos = mx.sum(output_audio * target_audio) / (
        mx.sqrt(mx.sum(output_audio * output_audio))
        * mx.sqrt(mx.sum(target_audio * target_audio))
        + 1e-6
    )

    predicted = project_out(output_audio)
    predicted = postprocess_conv(predicted) + predicted
    target = target_velocity.transpose(0, 2, 1).astype(mx.float32)
    velocity_mse = mx.mean((predicted - target) ** 2) / (
        mx.mean(target * target) + 1e-6
    )
    velocity_cos = mx.sum(predicted * target) / (
        mx.sqrt(mx.sum(predicted * predicted))
        * mx.sqrt(mx.sum(target * target))
        + 1e-6
    )
    return (
        hidden_mse
        + 1.5 * (1.0 - hidden_cos)
        + 1.25 * velocity_mse
        + 1.0 * (1.0 - velocity_cos)
    )


def export_model(student: nn.Module, output_path: Path) -> None:
    params = dict(tree_flatten(student.parameters()))
    mx.eval(*params.values())
    arrays = {key: np.array(value) for key, value in params.items()}
    tmp_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_path), **arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine terminal block 23 of a ternary artifact")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"))
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--block-index", type=int, default=23)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.source_manifest or args.source_artifact.with_suffix(".json")
    source_manifest = load_manifest(manifest_path)
    group_size = int(source_manifest["model"]["group_size"])
    samples = tq.load_samples(args.dataset_dir, 0)
    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    student = load_student(args.source_artifact, source_manifest, args.crop_len)
    _, cross_cache, global_cond, global_pre = tq.cache_conditioning(
        teacher, args.teacher_weights, [s["prompt"] for s in samples], args.seconds
    )
    teacher_local = tq.local_pads(teacher, args.crop_len)
    block = student.transformer.layers[args.block_index]
    replace_quantized_with_qat(block, group_size)
    student.freeze()
    block.unfreeze()
    optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(5e-5, args.steps, end=2e-6),
        weight_decay=1e-4,
    )
    optimizer.init(block.trainable_parameters())
    rng = random.Random(args.seed)
    sigmas = [0.95, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]

    def loss_fn(model, h_student, context, g, local, target_hidden, target_velocity):
        return velocity_loss(
            model,
            h_student,
            context,
            g,
            local,
            target_hidden,
            student.transformer.project_out,
            student.postprocess_conv,
            target_velocity,
        )

    value_grad = nn.value_and_grad(block, loss_fn)
    started = time.time()
    latest = None
    print(f"[Refine] block={args.block_index} steps={args.steps} group={group_size}", flush=True)
    for step in range(args.steps):
        sample = samples[rng.randrange(len(samples))]
        sigma = sigmas[step % len(sigmas)]
        x = tq.noised_latent(sample, args.crop_len, sigma, args.seed + step)
        context = (  # the block consumes the teacher's projected condition
            teacher.to_cond_embed[2](
                nn.silu(teacher.to_cond_embed[0](cross_cache[sample["prompt"]]))
            )
        )
        g = tq.timestep_projection(teacher, global_pre, sigma)
        h_teacher, h_student = tq.prefix_states(
            teacher, student, x, args.block_index, context, g, teacher_local
        )
        target_hidden = teacher.transformer.layers[args.block_index](
            h_teacher, context, g, teacher_local[args.block_index]
        )
        t = mx.array([sigma], dtype=mx.float16)
        target_velocity = teacher(x, t, cross_cache[sample["prompt"]], global_cond)
        mx.eval(h_student, target_hidden, target_velocity)
        loss, grads = value_grad(
            block,
            h_student,
            context,
            g,
            teacher_local[args.block_index],
            target_hidden,
            target_velocity,
        )
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(block, grads)
        mx.eval(block.parameters(), optimizer.state, loss)
        latest = float(loss)
        if step == 0 or (step + 1) % max(1, args.steps // 8) == 0:
            print(f"[Refine] step {step + 1}/{args.steps} loss={latest:.5f} sigma={sigma:.2f}", flush=True)

    records = {}
    hard_metrics = tq.hard_freeze_block(block, args.block_index, group_size, records)
    artifact = args.output_dir / (
        f"{args.source_artifact.stem}_refined_block{args.block_index}.npz"
    )
    export_model(student, artifact)
    manifest = dict(source_manifest)
    manifest["artifact"] = str(artifact)
    manifest["created_at_unix"] = time.time()
    manifest["size_bytes"] = artifact.stat().st_size
    manifest["refinement"] = {
        "source_artifact": str(args.source_artifact),
        "block_index": args.block_index,
        "steps": args.steps,
        "last_loss": latest,
        "hard_metrics": hard_metrics,
        "loss": "hidden_audio_tokens + final_velocity_mse/cosine",
    }
    manifest_out = artifact.with_suffix(".json")
    write_json(manifest_out, manifest)
    print(f"[Refine] exported {artifact} bytes={artifact.stat().st_size}", flush=True)
    print(json.dumps({"hard_metrics": hard_metrics, "last_loss": latest}, indent=2), flush=True)
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
