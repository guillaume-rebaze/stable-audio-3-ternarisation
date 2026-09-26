"""Activation-aware local distillation for one ternary DiT block.

Global velocity QAT plateaued because seven ternary projections compound their
errors before the audit sees them.  This trainer isolates block 0, keeps its
real ternary forward active, and matches the dense teacher block output on the
independent V7 state cache.  It starts from an existing hard TTQ checkpoint,
then serializes and reloads the block before audit.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

import train_ternary_quality as tq
import train_ternary_window_v6 as tw
from models.defs import dit_mlx_medium
from ternary_contract import scope_digest, write_json

mx = tq.mx
nn = tq.nn
optim = tq.optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Activation-aware local TTQ distillation for one DiT block"
    )
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--quantizer-mode", choices=("ttq", "ttq_hadamard"), default="ttq"
    )
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--fixed-state-count", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--learning-rate-end", type=float, default=5e-7)
    parser.add_argument("--scale-learning-rate", type=float, default=1e-5)
    parser.add_argument("--scale-learning-rate-end", type=float, default=1e-5)
    parser.add_argument("--optimizer-eps", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--quantizer-surrogate", choices=("smooth", "identity"), default="smooth"
    )
    parser.add_argument("--soft-end-updates", type=int, default=0)
    parser.add_argument("--soft-sharpness-start", type=float, default=1.5)
    parser.add_argument("--soft-sharpness-end", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--max-metal-bytes", type=int, default=12_000_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    return parser.parse_args()


def _tree_has_leaves(tree: object) -> bool:
    if tree is None:
        return False
    if isinstance(tree, dict):
        return any(_tree_has_leaves(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return any(_tree_has_leaves(value) for value in tree)
    return True


def load_teacher(path: Path, crop_len: int) -> nn.Module:
    model = dit_mlx_medium.DiT(T_lat=crop_len)
    model.load_weights(str(path), strict=False)
    model.freeze()
    mx.eval(model.parameters())
    return model


def prepare_conditioning(
    teacher: nn.Module,
    cross_cache: list[mx.array],
    global_cond: mx.array,
    sigmas: np.ndarray,
) -> tuple[list[mx.array], dict[float, mx.array], mx.array]:
    contexts: list[mx.array] = []
    for cross in cross_cache:
        context = teacher.to_cond_embed[2](
            nn.silu(teacher.to_cond_embed[0](cross))
        )
        mx.eval(context)
        contexts.append(context)
    global_pre = teacher.to_global_embed[2](
        nn.silu(teacher.to_global_embed[0](global_cond))
    )
    mx.eval(global_pre)
    projections: dict[float, mx.array] = {}
    for sigma in sorted({float(value) for value in sigmas}):
        projections[sigma] = tq.timestep_projection(teacher, global_pre, sigma)
    return contexts, projections, global_pre


def build_student(
    teacher_weights: Path,
    crop_len: int,
    source_records_path: Path,
    group_size: int,
    quantizer_mode: str,
    surrogate: str,
) -> tuple[nn.Module, dict, dict]:
    student = load_teacher(teacher_weights, crop_len)
    source_records, source_metadata = tq.load_records_checkpoint(source_records_path)
    expected = {
        f"transformer.layers.0.{name}" for name in tq.CORE_NAMES
    }
    if set(source_records) != expected:
        raise ValueError(
            "local block trainer requires exactly block-0 core records: "
            f"missing={sorted(expected - set(source_records))} "
            f"extra={sorted(set(source_records) - expected)}"
        )
    if int(source_metadata["group_size"]) != group_size:
        raise ValueError("source records group_size does not match")
    for prefix, record in source_records.items():
        if record.mode != quantizer_mode:
            raise ValueError(
                f"source record {prefix} mode={record.mode!r} does not match "
                f"quantizer_mode={quantizer_mode!r}"
            )

    student = tq.apply_records_to_model(student, source_records, group_size)
    block = student.transformer.layers[0]
    for name in tq.CORE_NAMES:
        prefix = f"transformer.layers.0.{name}"
        current = tq.module_at(block, name)
        qat = tw.quantized_linear_to_qat(
            current,
            source_records[prefix],
            group_size,
            quantizer_mode,
        )
        tq.set_module_at(block, name, qat)
    student.freeze()
    for module in tq.core_modules(block).values():
        module.unfreeze()
        module.set_surrogate_mode(surrogate)
    mx.eval(student.parameters())
    return student, source_records, source_metadata


def code_snapshot(records: dict) -> dict[str, np.ndarray]:
    return {
        prefix: np.asarray(record.q, dtype=np.int8).copy()
        for prefix, record in sorted(records.items())
    }


def code_change_count(before: dict[str, np.ndarray], after: dict) -> int:
    count = 0
    for prefix, record in after.items():
        count += int(np.count_nonzero(before[prefix] != np.asarray(record.q)))
    return count


def roundtrip_check(
    student: nn.Module,
    teacher_weights: Path,
    records: dict,
    states: np.ndarray,
    sigmas: np.ndarray,
    prompt_indices: np.ndarray,
    contexts: list[mx.array],
    projections: dict[float, mx.array],
    local_pad: mx.array,
    crop_len: int,
    group_size: int,
    indices: list[int],
) -> dict:
    reloaded = load_teacher(teacher_weights, crop_len)
    reloaded = tq.apply_records_to_model(reloaded, records, group_size)
    expected_cosines: list[float] = []
    relative_errors: list[float] = []
    student_block = student.transformer.layers[0]
    reload_block = reloaded.transformer.layers[0]
    for index in indices[: min(8, len(indices))]:
        x = mx.array(states[index][None], dtype=mx.float16)
        h_in = tq.model_input(student, x)
        context = contexts[int(prompt_indices[index])]
        global_cond = projections[float(sigmas[index])]
        expected = student_block(h_in, context, global_cond, local_pad)
        actual = reload_block(h_in, context, global_cond, local_pad)
        mx.eval(expected, actual)
        left = np.asarray(expected, dtype=np.float32).ravel()
        right = np.asarray(actual, dtype=np.float32).ravel()
        relative_errors.append(tq.relative_error(left, right))
        expected_cosines.append(
            float(
                np.dot(left, right)
                / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-8)
            )
        )
    del reloaded
    gc.collect()
    mx.clear_cache()
    result = {
        "checked": len(relative_errors),
        "max_relative_error": float(max(relative_errors or [0.0])),
        "min_cosine": float(min(expected_cosines or [1.0])),
    }
    result["passed"] = (
        result["max_relative_error"] <= 1e-3
        and result["min_cosine"] >= 0.99999
    )
    return result


def main() -> None:
    args = parse_args()
    if args.group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    if args.steps <= 0 or args.fixed_state_count <= 0:
        raise ValueError("steps and fixed-state-count must be positive")
    if args.gradient_accumulation <= 0:
        raise ValueError("gradient-accumulation must be positive")
    if args.teacher_weights is None:
        args.teacher_weights = (
            tq.MLX_RUNTIME_ROOT / "models" / "mlx" / "dit_medium_f16.npz"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    states, sigmas, prompt_indices, sources, cross_cache, global_cond, cache_manifest = (
        tw.load_state_cache(args.state_cache)
    )
    indices = tw.balanced_fixed_state_indices(
        prompt_indices, sigmas, sources, min(args.fixed_state_count, len(states))
    )
    order = list(indices)
    random.Random(args.seed).shuffle(order)

    teacher = load_teacher(args.teacher_weights, args.crop_len)
    contexts, projections, _ = prepare_conditioning(
        teacher, cross_cache, global_cond, sigmas
    )
    local_pad = tq.local_pads(teacher, args.crop_len)[0]
    student, source_records, source_metadata = build_student(
        args.teacher_weights,
        args.crop_len,
        args.source_records,
        args.group_size,
        args.quantizer_mode,
        args.quantizer_surrogate,
    )
    block = student.transformer.layers[0]
    before_codes = code_snapshot(source_records)

    initial_weights, initial_scales = tw.split_scale_parameter_tree(
        block.trainable_parameters()
    )
    if not _tree_has_leaves(initial_weights) or not _tree_has_leaves(initial_scales):
        raise RuntimeError("TTQ local trainer found no trainable master or scale parameters")
    weight_optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(
            args.learning_rate, args.steps, end=args.learning_rate_end
        ),
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
    )
    scale_optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(
            args.scale_learning_rate,
            args.steps,
            end=args.scale_learning_rate_end,
        ),
        eps=args.optimizer_eps,
        weight_decay=0.0,
    )
    weight_optimizer.init(initial_weights)
    scale_optimizer.init(initial_scales)

    def loss_fn(
        current_block: nn.Module,
        h_in: mx.array,
        context: mx.array,
        projected_timestep: mx.array,
        target: mx.array,
    ) -> mx.array:
        return tq.block_loss(
            current_block,
            h_in,
            context,
            projected_timestep,
            local_pad,
            target,
        )

    value_grad = nn.value_and_grad(block, loss_fn)
    losses: list[float] = []
    started = time.time()
    peak_bytes = 0
    for step in range(args.steps):
        tw.set_quantizer_schedule(
            student,
            0,
            0,
            *tw.quantizer_schedule(
                step,
                args.soft_end_updates,
                args.soft_sharpness_start,
                args.soft_sharpness_end,
            ),
        )
        accumulated = None
        micro_losses: list[float] = []
        for micro in range(args.gradient_accumulation):
            index = order[(step * args.gradient_accumulation + micro) % len(order)]
            x = mx.array(states[index][None], dtype=mx.float16)
            context = contexts[int(prompt_indices[index])]
            projected_timestep = projections[float(sigmas[index])]
            h_in = tq.model_input(teacher, x)
            target = teacher.transformer.layers[0](
                h_in, context, projected_timestep, local_pad
            )
            mx.eval(h_in, target)
            loss, grads = value_grad(
                block, h_in, context, projected_timestep, target
            )
            mx.eval(loss, grads)
            loss_value = float(loss)
            if not np.isfinite(loss_value) or not tw.tree_is_finite(grads):
                raise RuntimeError(
                    f"non-finite local loss/gradient at step={step + 1} micro={micro + 1}"
                )
            accumulated = (
                grads
                if accumulated is None
                else tq.tree_add(accumulated, grads)
            )
            micro_losses.append(loss_value)
        grads = tq.tree_scale(
            accumulated, 1.0 / float(args.gradient_accumulation)
        )
        grads, raw_norm = optim.clip_grad_norm(grads, args.gradient_clip)
        if not np.isfinite(float(raw_norm)):
            raise RuntimeError(f"non-finite gradient norm at step={step + 1}")
        weight_grads, scale_grads = tw.split_scale_parameter_tree(grads)
        current_weights, current_scales = tw.split_scale_parameter_tree(
            block.trainable_parameters()
        )
        block.update(weight_optimizer.apply_gradients(weight_grads, current_weights))
        block.update(scale_optimizer.apply_gradients(scale_grads, current_scales))
        mx.eval(
            block.parameters(), weight_optimizer.state, scale_optimizer.state
        )
        if not tw.tree_is_finite(block.trainable_parameters()):
            raise RuntimeError(f"non-finite parameter at step={step + 1}")
        memory = tq.memory_snapshot()
        peak_bytes = max(peak_bytes, int(memory["metal_peak_gb"] * (1024**3)))
        if peak_bytes > args.max_metal_bytes:
            raise RuntimeError(
                f"Metal memory guard exceeded: {peak_bytes} > {args.max_metal_bytes}"
            )
        latest = float(np.mean(micro_losses))
        losses.append(latest)
        if step == 0 or (step + 1) % max(1, args.checkpoint_every) == 0:
            print(
                f"[LocalV7] step={step + 1}/{args.steps} loss={latest:.6f} "
                f"grad_norm={float(raw_norm):.5g} memory={memory}",
                flush=True,
            )

    mx.eval(block.parameters())
    records: dict = {}
    hard_metrics = tq.hard_freeze_block(
        block, 0, args.group_size, records, args.quantizer_mode
    )
    records_path = args.output_dir / "records_checkpoint.npz"
    tq.save_records_checkpoint(
        records_path,
        records,
        1,
        args.group_size,
        args.crop_len,
        args.quantizer_mode,
    )
    roundtrip = roundtrip_check(
        student,
        args.teacher_weights,
        records,
        states,
        sigmas,
        prompt_indices,
        contexts,
        projections,
        local_pad,
        args.crop_len,
        args.group_size,
        order,
    )
    if not roundtrip["passed"]:
        raise RuntimeError(f"local hard roundtrip failed: {roundtrip}")
    summary = {
        "schema": "onus.ternary-quality/v7-local-block-summary",
        "status": "completed",
        "block": 0,
        "group_size": args.group_size,
        "quantizer_mode": args.quantizer_mode,
        "quantizer_surrogate": args.quantizer_surrogate,
        "steps": args.steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": [args.learning_rate, args.learning_rate_end],
        "scale_learning_rate": [
            args.scale_learning_rate,
            args.scale_learning_rate_end,
        ],
        "fixed_state_count": len(indices),
        "state_cache": str(args.state_cache.resolve()),
        "state_cache_manifest": cache_manifest,
        "source_records": str(args.source_records.resolve()),
        "source_records_metadata": source_metadata,
        "teacher": tq.file_fingerprint(args.teacher_weights),
        "scope": sorted(records),
        "scope_digest": scope_digest(records),
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
        },
        "hard_metrics": hard_metrics,
        "codes_changed_vs_source": code_change_count(before_codes, records),
        "roundtrip": roundtrip,
        "memory": {
            "peak_gb": peak_bytes / (1024**3),
            "final": tq.memory_snapshot(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "trainer": str(Path(__file__).resolve()),
        },
        "seconds": time.time() - started,
    }
    write_json(args.output_dir / "local_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
