"""End-to-end QAT repair for a sequential ternary artifact.

Sequential hard freezing prevents later blocks from changing earlier code
choices.  This phase reopens every ternary matrix from the serialized artifact,
optimizes the complete DiT velocity with STE, and serializes the final codes
again through the same affine contract.
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
from ternary_contract import dequantize_packed, write_json

mx = tq.mx
nn = tq.nn
optim = tq.optim


def reopen_all_core(student: nn.Module, group_size: int) -> None:
    for block in student.transformer.layers:
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


def e2e_loss(model, x, t, cross, global_cond, target):
    prediction = model(x, t, cross, global_cond)
    p = prediction.astype(mx.float32)
    q = target.astype(mx.float32)
    mse = mx.mean((p - q) ** 2) / (mx.mean(q * q) + 1e-6)
    cosine = mx.sum(p * q) / (
        mx.sqrt(mx.sum(p * p)) * mx.sqrt(mx.sum(q * q)) + 1e-6
    )
    # Keep a modest amplitude term so the optimizer cannot improve cosine by
    # shrinking the velocity toward zero.
    rms_p = mx.sqrt(mx.mean(p * p) + 1e-6)
    rms_q = mx.sqrt(mx.mean(q * q) + 1e-6)
    rms_loss = ((rms_p - rms_q) / rms_q) ** 2
    return mse + 2.0 * (1.0 - cosine) + 0.25 * rms_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end QAT repair for a ternary DiT")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"))
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=525252)
    parser.add_argument("--checkpoint-every", type=int, default=0)
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
    _, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher, args.teacher_weights, [s["prompt"] for s in samples], args.seconds
    )
    reopen_all_core(student, group_size)

    # Reopen the block-local FP16 adapters and final projection as allowed by
    # the quality plan; keep conditioning and input projections anchored.
    student.freeze()
    for block in student.transformer.layers:
        block.unfreeze()
    student.transformer.project_out.unfreeze()
    student.postprocess_conv.unfreeze()
    trainable = student.trainable_parameters()
    optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(1.5e-5, args.steps, end=1e-6),
        weight_decay=5e-5,
    )
    optimizer.init(trainable)
    value_grad = nn.value_and_grad(student, e2e_loss)
    rng = random.Random(args.seed)
    sigmas = [0.95, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]
    started = time.time()
    losses: list[float] = []
    print(
        f"[E2E] source={args.source_artifact} steps={args.steps} "
        f"group={group_size} samples={len(samples)} memory={tq.memory_snapshot()}",
        flush=True,
    )
    for step in range(args.steps):
        sample = samples[rng.randrange(len(samples))]
        sigma = sigmas[step % len(sigmas)]
        x = tq.noised_latent(sample, args.crop_len, sigma, args.seed + step)
        t = mx.array([sigma], dtype=mx.float16)
        cross = cross_cache[sample["prompt"]]
        target = teacher(x, t, cross, global_cond)
        mx.eval(target)
        loss, grads = value_grad(student, x, t, cross, global_cond, target)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(student, grads)
        mx.eval(student.parameters(), optimizer.state, loss)
        latest = float(loss)
        losses.append(latest)
        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"[E2E] step {step + 1}/{args.steps} loss={latest:.5f} "
                f"sigma={sigma:.2f} memory={tq.memory_snapshot()}",
                flush=True,
            )

    records: dict[str, tq.TernaryWeights] = {}
    for index, block in enumerate(student.transformer.layers):
        tq.hard_freeze_block(block, index, group_size, records)
    artifact = args.output_dir / (
        f"{args.source_artifact.stem}_e2e_qat.npz"
    )
    manifest_out = artifact.with_suffix(".json")
    config = {
        "source_artifact": str(args.source_artifact),
        "steps": args.steps,
        "group_size": group_size,
        "crop_len": args.crop_len,
        "seed": args.seed,
        "loss": "full_velocity_mse_cosine_rms",
    }
    manifest = tq.export_artifact(
        student,
        records,
        artifact,
        manifest_out,
        group_size,
        args.crop_len,
        config,
    )
    write_json(
        args.output_dir / "e2e_metrics.json",
        {
            "status": "e2e_qat_exported",
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "elapsed_seconds": time.time() - started,
            "memory": tq.memory_snapshot(),
            "manifest": str(manifest_out),
            "artifact_bytes": artifact.stat().st_size,
        },
    )
    print(
        f"[E2E] exported {artifact} bytes={artifact.stat().st_size} "
        f"scope={manifest['scope']['count']} memory={tq.memory_snapshot()}",
        flush=True,
    )
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
