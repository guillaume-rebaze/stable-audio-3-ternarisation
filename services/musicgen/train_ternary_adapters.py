"""Train explicit low-rank residuals on top of a reloadable ternary base."""

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
from finetune_ternary_window import teacher_rollout_states
from ternary_adapter import TernaryAdapterLinear

mx = tq.mx
nn = tq.nn
optim = tq.optim


def wrap_core(student: nn.Module, scope: list[str], rank: int, alpha: float) -> None:
    for index, path in enumerate(sorted(scope)):
        base = tq.module_at(student, path)
        tq.set_module_at(
            student,
            path,
            TernaryAdapterLinear(base, rank=rank, alpha=alpha, seed=9000 + index),
        )


def adapter_arrays(student: nn.Module, scope: list[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in scope:
        module = tq.module_at(student, path)
        if not isinstance(module, TernaryAdapterLinear):
            raise TypeError(f"Expected adapter at {path}, got {type(module)}")
        mx.eval(module.down, module.up)
        arrays[f"{path}.down"] = np.array(module.down)
        arrays[f"{path}.up"] = np.array(module.up)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Train low-rank residuals on a ternary base")
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"))
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=737373)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.base_manifest or args.base_artifact.with_suffix(".json")
    base_manifest = load_manifest(manifest_path)
    scope = list(base_manifest["scope"]["paths"])
    samples = tq.load_samples(args.dataset_dir, 0)

    teacher = tq.dit_mlx_medium.DiT(T_lat=args.crop_len)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    student = load_student(args.base_artifact, base_manifest, args.crop_len)
    _, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher, args.teacher_weights, [s["prompt"] for s in samples], args.seconds
    )
    wrap_core(student, scope, args.rank, args.alpha)
    student.freeze()
    for path in scope:
        tq.module_at(student, path).unfreeze(recurse=False, keys=["down", "up"])

    rollout_states: list[tuple[str, mx.array, mx.array]] = []
    if args.rollout_steps:
        prompts = sorted(cross_cache)
        print(f"[Adapter] caching {args.rollout_steps}-step rollouts for {len(prompts)} prompts", flush=True)
        for index, prompt in enumerate(prompts):
            for state, timestep in teacher_rollout_states(
                teacher,
                cross_cache[prompt],
                global_cond,
                args.crop_len,
                args.rollout_steps,
                args.seed + 100000 + index,
            ):
                rollout_states.append((prompt, state, timestep))

    optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(
            args.learning_rate, args.steps, end=max(args.learning_rate * 0.05, 1e-6)
        ),
        weight_decay=1e-5,
    )
    optimizer.init(student.trainable_parameters())
    value_grad = nn.value_and_grad(student, e2e_loss)
    rng = random.Random(args.seed)
    sigmas = [0.95, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10]
    losses: list[float] = []
    started = time.time()
    print(
        f"[Adapter] rank={args.rank} scope={len(scope)} steps={args.steps} "
        f"memory={tq.memory_snapshot()}",
        flush=True,
    )
    for step in range(args.steps):
        if rollout_states:
            prompt, x, t = rollout_states[rng.randrange(len(rollout_states))]
            cross = cross_cache[prompt]
        else:
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
        losses.append(float(loss))
        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            print(
                f"[Adapter] step {step + 1}/{args.steps} loss={float(loss):.5f} "
                f"memory={tq.memory_snapshot()}",
                flush=True,
            )

    arrays = adapter_arrays(student, scope)
    adapter_path = args.output_dir / f"ternary_lora_r{args.rank}.npz"
    tmp = adapter_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), **arrays)
    tmp.replace(adapter_path)
    adapter_manifest = {
        "schema": "onus.ternary-adapter/v1",
        "base_artifact": str(args.base_artifact),
        "base_manifest": str(manifest_path),
        "adapter": {
            "rank": args.rank,
            "alpha": args.alpha,
            "scope": scope,
            "scope_count": len(scope),
        },
        "training": {
            "steps": args.steps,
            "rollout_steps": args.rollout_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "loss": "full_velocity_mse_cosine_rms",
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        },
        "size_bytes": adapter_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
        "memory": tq.memory_snapshot(),
    }
    tq.write_json(args.output_dir / "adapter_manifest.json", adapter_manifest)
    print(
        f"[Adapter] exported {adapter_path} bytes={adapter_path.stat().st_size} "
        f"memory={tq.memory_snapshot()}",
        flush=True,
    )
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
