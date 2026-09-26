"""Memory-bounded E2E refinement of a sliding block window.

Full-DiT QAT is useful but exceeds the 12 GB machine budget.  This variant
keeps the same complete velocity objective while reopening only one contiguous
window of ternary blocks; the rest of the artifact stays hard-quantized.
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
from finetune_ternary_e2e import e2e_loss
from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_contract import (
    TernaryWeights,
    dequantize_packed,
    scope_digest,
    unpack_codes,
    write_json,
)
from ternary_runtime_contract import pingpong_trace, pingpong_transition, timestep_tensor
from mlx.utils import tree_flatten

mx = tq.mx
nn = tq.nn
optim = tq.optim


def rollout_case(
    model: nn.Module,
    cross: mx.array,
    global_cond: mx.array,
    latent_len: int,
    steps: int,
    seed: int,
) -> dict:
    """Capture a sampler trace and the fixed noises needed to replay it."""
    sigmas = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
    sigma_values = [float(value) for value in sigmas]
    initial = mx.random.normal(
        (1, 256, latent_len), dtype=mx.float16, key=mx.random.key(seed)
    )
    trace = pingpong_trace(
        lambda x, t: model(x, t, cross, global_cond),
        initial,
        sigmas,
        sampler_seed=seed + 1,
    )
    states = [
        (
            mx.array(record["state"], dtype=mx.float16),
            timestep_tensor(float(record["sigma"])),
        )
        for record in trace[:-1]
    ]
    noises = [
        mx.zeros_like(initial)
        if record["noise"] is None
        else mx.array(record["noise"], dtype=mx.float16)
        for record in trace[:-1]
    ]
    return {
        "initial": initial,
        "states": states,
        "noises": noises,
        "sigmas": sigma_values,
    }


def differentiable_rollout_loss(
    student: nn.Module,
    initial: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    teacher: nn.Module,
    sigmas: list[float],
    noises: list[mx.array],
) -> mx.array:
    """Distill teacher velocities along the student-generated two-step path."""
    x = initial
    losses: list[mx.array] = []
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        t = timestep_tensor(sigma)
        student_velocity = student(x, t, cross, global_cond)
        teacher_velocity = teacher(mx.stop_gradient(x), t, cross, global_cond)
        p = student_velocity.astype(mx.float32)
        q = mx.stop_gradient(teacher_velocity.astype(mx.float32))
        mse = mx.mean((p - q) ** 2) / (mx.mean(q * q) + 1e-6)
        cosine = mx.sum(p * q) / (
            mx.sqrt(mx.sum(p * p)) * mx.sqrt(mx.sum(q * q)) + 1e-6
        )
        rms_p = mx.sqrt(mx.mean(p * p) + 1e-6)
        rms_q = mx.sqrt(mx.mean(q * q) + 1e-6)
        losses.append(mse + 2.0 * (1.0 - cosine) + 0.25 * ((rms_p - rms_q) / rms_q) ** 2)
        t_curr = mx.array(sigma, dtype=mx.float32)
        t_next = mx.array(sigmas[index + 1], dtype=mx.float32)
        x = pingpong_transition(
            x,
            student_velocity,
            t_curr,
            t_next,
            noises[index],
            index,
            len(sigmas) - 1,
        )
    return mx.mean(mx.stack(losses))


def reopen_window(
    student: nn.Module,
    start: int,
    end: int,
    group_size: int,
    quantizer_mode: str,
) -> None:
    for index in range(start, end + 1):
        block = student.transformer.layers[index]
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
                quantizer_mode,
            )
            qat.weight = mx.array(dense, dtype=mx.float32)
            if qat.bias is not None:
                qat.bias = quantized.bias.astype(mx.float32)
            tq.set_module_at(block, name, qat)


def records_from_quantized_model(
    student: nn.Module, scope: list[str], group_size: int, quantizer_mode: str
) -> dict[str, TernaryWeights]:
    records: dict[str, TernaryWeights] = {}
    for prefix in scope:
        module = tq.module_at(student, prefix)
        packed = np.array(module.weight)
        scales = np.array(module.scales)
        biases = np.array(module.biases)
        q = unpack_codes(packed, group_size)
        records[prefix] = TernaryWeights(
            packed_codes=packed,
            scales=scales,
            biases=biases,
            q=q,
            group_means=np.zeros_like(scales, dtype=np.float32),
            group_size=group_size,
            mode=quantizer_mode,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Windowed E2E ternary refinement")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"))
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-block", type=int, default=20)
    parser.add_argument("--end-block", type=int, default=23)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument(
        "--differentiable-rollout",
        action="store_true",
        help="mix a real student-generated rollout loss into the window updates",
    )
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=626262)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.source_manifest or args.source_artifact.with_suffix(".json")
    source_manifest = load_manifest(manifest_path)
    group_size = int(source_manifest["model"]["group_size"])
    quantizer_mode = str(source_manifest["model"].get("quantizer_mode", "symmetric"))
    scope = list(source_manifest["scope"]["paths"])
    teacher = dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    student = load_student(args.source_artifact, source_manifest, args.crop_len)
    samples = tq.load_samples(args.dataset_dir, 0)
    _, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher, args.teacher_weights, [s["prompt"] for s in samples], args.seconds
    )
    rollout_states: list[tuple[str, str, mx.array, mx.array]] = []
    rollout_cases: list[dict] = []
    real_states: list[tuple[str, str, mx.array, mx.array]] = []
    if args.rollout_steps:
        prompts = sorted(cross_cache)
        print(
            f"[Window] caching {args.rollout_steps}-step teacher+student rollouts "
            f"for {len(prompts)} prompts",
            flush=True,
        )
        for index, prompt in enumerate(prompts):
            teacher_case = rollout_case(
                teacher,
                cross_cache[prompt],
                global_cond,
                args.crop_len,
                args.rollout_steps,
                args.seed + 100000 + index,
            )
            student_case = rollout_case(
                student,
                cross_cache[prompt],
                global_cond,
                args.crop_len,
                args.rollout_steps,
                args.seed + 100000 + index,
            )
            rollout_cases.append(
                {
                    "prompt": prompt,
                    "initial": student_case["initial"],
                    "noises": student_case["noises"],
                    "sigmas": student_case["sigmas"],
                }
            )
            for x_state, t_state in teacher_case["states"]:
                rollout_states.append(("teacher", prompt, x_state, t_state))
            for x_state, t_state in student_case["states"]:
                rollout_states.append(("student", prompt, x_state, t_state))
        interpolation_sigmas = (0.02, 0.10, 0.35, 0.60, 0.90, 0.98)
        for index, sample in enumerate(samples):
            sigma = interpolation_sigmas[index % len(interpolation_sigmas)]
            real_states.append(
                (
                    "real_interpolation",
                    sample["prompt"],
                    tq.noised_latent(sample, args.crop_len, sigma, args.seed + 200000 + index),
                    mx.array([sigma], dtype=mx.float16),
                )
            )
        print(
            f"[Window] cached states={len(rollout_states)} "
            f"cases={len(rollout_cases)} real={len(real_states)}",
            flush=True,
        )
    reopen_window(
        student, args.start_block, args.end_block, group_size, quantizer_mode
    )
    student.freeze()
    for index in range(args.start_block, args.end_block + 1):
        block = student.transformer.layers[index]
        for module in tq.core_modules(block).values():
            module.unfreeze()
        block.unfreeze(recurse=False, keys=["to_scale_shift_gate"])
        block.pre_norm.unfreeze()
        block.cross_attend_norm.unfreeze()
        block.ff_norm.unfreeze()
    student.transformer.project_out.unfreeze()
    student.postprocess_conv.unfreeze()
    optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(
            args.learning_rate, args.steps, end=max(args.learning_rate * 0.05, 1e-6)
        ),
        eps=1e-4,
        weight_decay=5e-5,
    )
    optimizer.init(student.trainable_parameters())
    value_grad = nn.value_and_grad(student, e2e_loss)
    rollout_grad = (
        nn.value_and_grad(student, differentiable_rollout_loss)
        if args.differentiable_rollout and rollout_cases
        else None
    )
    rng = random.Random(args.seed)
    sigmas = [0.95, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]
    losses: list[float] = []
    started = time.time()
    print(
        f"[Window] blocks={args.start_block}:{args.end_block} steps={args.steps} "
        f"memory={tq.memory_snapshot()}",
        flush=True,
    )
    for step in range(args.steps):
        use_rollout = bool(rollout_grad and step % 3 == 2)
        if use_rollout:
            case = rollout_cases[rng.randrange(len(rollout_cases))]
            prompt = case["prompt"]
            cross = cross_cache[prompt]
            sigma = case["sigmas"][0]
            loss, grads = rollout_grad(
                student,
                case["initial"],
                cross,
                global_cond,
                teacher,
                case["sigmas"],
                case["noises"],
            )
            label = "student_rollout"
        elif rollout_states:
            source, prompt, x, t = rollout_states[rng.randrange(len(rollout_states))]
            cross = cross_cache[prompt]
            sigma = float(t[0])
            target = teacher(x, t, cross, global_cond)
            mx.eval(target)
            loss, grads = value_grad(student, x, t, cross, global_cond, target)
            label = source
        elif real_states:
            source, prompt, x, t = real_states[rng.randrange(len(real_states))]
            cross = cross_cache[prompt]
            sigma = float(t[0])
            target = teacher(x, t, cross, global_cond)
            mx.eval(target)
            loss, grads = value_grad(student, x, t, cross, global_cond, target)
            label = source
        else:
            sample = samples[rng.randrange(len(samples))]
            sigma = sigmas[step % len(sigmas)]
            x = tq.noised_latent(sample, args.crop_len, sigma, args.seed + step)
            t = mx.array([sigma], dtype=mx.float16)
            cross = cross_cache[sample["prompt"]]
            target = teacher(x, t, cross, global_cond)
            mx.eval(target)
            loss, grads = value_grad(student, x, t, cross, global_cond, target)
            label = "real_latent"
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(student, grads)
        mx.eval(student.parameters(), optimizer.state, loss)
        latest = float(loss)
        losses.append(latest)
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(
                f"[Window] step {step + 1}/{args.steps} loss={latest:.5f} "
                f"sigma={sigma:.2f} source={label} memory={tq.memory_snapshot()}",
                flush=True,
            )

    # Re-serialize every reopened block through the same contract.  The blocks
    # outside the window are already MLX QuantizedLinear and are carried over.
    records: dict[str, TernaryWeights] = {}
    for index in range(args.start_block, args.end_block + 1):
        tq.hard_freeze_block(
            student.transformer.layers[index],
            index,
            group_size,
            records,
            quantizer_mode,
        )
    all_records = records_from_quantized_model(
        student, scope, group_size, quantizer_mode
    )
    artifact = args.output_dir / (
        f"{args.source_artifact.stem}_window{args.start_block}-{args.end_block}.npz"
    )
    manifest = tq.export_artifact(
        student,
        all_records,
        artifact,
        artifact.with_suffix(".json"),
        group_size,
        args.crop_len,
        {
            "source_artifact": str(args.source_artifact),
            "steps": args.steps,
            "group_size": group_size,
            "quantizer_mode": quantizer_mode,
            "rollout_steps": args.rollout_steps,
            "differentiable_rollout": args.differentiable_rollout,
        },
        quantizer_mode,
        "symmetric_compact" if quantizer_mode == "symmetric" else "full_affine",
    )
    manifest = json.loads(json.dumps(manifest))
    manifest["artifact"] = str(artifact)
    manifest["created_at_unix"] = time.time()
    manifest["size_bytes"] = artifact.stat().st_size
    manifest["window_refinement"] = {
        "start_block": args.start_block,
        "end_block": args.end_block,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "rollout_steps": args.rollout_steps,
        "differentiable_rollout": args.differentiable_rollout,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "elapsed_seconds": time.time() - started,
        "memory": tq.memory_snapshot(),
    }
    manifest["scope"]["digest"] = scope_digest(scope)
    write_json(artifact.with_suffix(".json"), manifest)
    write_json(args.output_dir / "window_metrics.json", manifest["window_refinement"])
    print(
        f"[Window] exported {artifact} bytes={artifact.stat().st_size} "
        f"memory={tq.memory_snapshot()}",
        flush=True,
    )
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
