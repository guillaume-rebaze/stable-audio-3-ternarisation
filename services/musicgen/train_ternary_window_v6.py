"""Paired V7 pilot for one ternary window with a full-DiT objective.

Only the selected window is trainable, but the loss is measured after the
complete student DiT.  This prevents the local hidden-state score from hiding
errors introduced by the remaining suffix.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

import train_ternary_quality as tq
from ternary_contract import TernaryWeights, scope_digest, unpack_codes, write_json
from ternary_teacher_targets import file_identity, load_teacher_targets, sha256_file
from ternary_runtime_contract import (
    build_state_cache_contract,
    build_teacher_target_contract,
    pingpong_transition,
    timestep_tensor,
)
import training_checkpoint as tc

mx = tq.mx
nn = tq.nn
optim = tq.optim
TTQ_MODES = {"ttq", "ttq_hadamard"}
SYMMETRIC_HADAMARD_MODE = "learned_symmetric_hadamard"
LEARNED_MODES = {
    "learned_symmetric",
    SYMMETRIC_HADAMARD_MODE,
    "learned_affine",
    *TTQ_MODES,
}


def checkpointed_module_call(module, *args):
    """Rematerialize a module while keeping its trainable parameters explicit."""
    def apply_module(parameters, *inputs):
        module.update(parameters)
        return module(*inputs)

    return mx.checkpoint(apply_module)(module.trainable_parameters(), *args)


def checkpointed_dit(model, x, t, cross_attn_cond_raw, global_cond_raw):
    """Forward equivalent of DiT.__call__ with per-block rematerialization."""
    batch = x.shape[0]
    context = model.to_cond_embed[2](nn.silu(model.to_cond_embed[0](cross_attn_cond_raw)))
    global_pre = model.to_global_embed[2](nn.silu(model.to_global_embed[0](global_cond_raw)))
    timestep = model.timestep_features(t)
    timestep = model.to_timestep_embed[2](nn.silu(model.to_timestep_embed[0](timestep)))
    global_embed = global_pre + timestep

    x_lc = x.transpose(0, 2, 1)
    x_pp = model.preprocess_conv(x_lc) + x_lc
    local = mx.zeros((batch, x.shape[-1], tq.dit_mlx_medium.LOCAL_ADD_COND_DIM))
    h = model.transformer.project_in(x_pp)
    memory = mx.broadcast_to(
        model.transformer.memory_tokens[None],
        (batch, tq.dit_mlx_medium.NUM_MEMORY_TOKENS, tq.dit_mlx_medium.EMBED_DIM),
    )
    h = mx.concatenate([memory, h], axis=1)
    global_cond = model.transformer.global_cond_embedder[2](
        nn.silu(model.transformer.global_cond_embedder[0](global_embed))
    )
    for layer in model.transformer.layers:
        local_emb = layer.to_local_embed(local)
        pad = mx.zeros(
            (batch, tq.dit_mlx_medium.NUM_MEMORY_TOKENS, tq.dit_mlx_medium.EMBED_DIM),
            dtype=local_emb.dtype,
        )
        local_padded = mx.concatenate([pad, local_emb], axis=1)

        h = checkpointed_module_call(
            layer, h, context, global_cond, local_padded
        )
    h = h[:, tq.dit_mlx_medium.NUM_MEMORY_TOKENS :, :]
    h = model.transformer.project_out(h)
    h = model.postprocess_conv(h) + h
    return h.transpose(0, 2, 1)


def quantized_linear_to_qat(
    module: nn.Module,
    record: TernaryWeights,
    group_size: int,
    quantizer_mode: str,
    master_weight: np.ndarray | None = None,
    master_bias: np.ndarray | None = None,
) -> tq.TernaryQATLinear:
    """Reopen a serialized ternary layer with an explicit master initializer."""
    if not isinstance(
        module,
        (nn.QuantizedLinear, tq.TernaryHadamardLinear, tq.TernaryTTQLinear),
    ):
        raise TypeError(f"expected quantized ternary module, got {type(module)}")
    if record.group_size != group_size:
        raise ValueError(
            f"record group_size={record.group_size} does not match {group_size}"
        )
    if isinstance(module, nn.QuantizedLinear):
        if tuple(module.scales.shape) != tuple(record.scales.shape):
            raise ValueError(
                f"record scale shape {record.scales.shape} does not match "
                f"module {module.scales.shape}"
            )

    module_bias = getattr(module, "bias", None)
    qat = tq.TernaryQATLinear(
        record.in_dim,
        record.out_dim,
        bias=module_bias is not None,
        group_size=group_size,
        quantizer_mode=quantizer_mode,
    )
    if master_weight is None:
        master = tq.dequantize_record(record)
        # ``dequantize_record`` resolves MLX's affine code convention to the
        # physical q*s levels.  A serialized symmetric_hadamard record is
        # already in the rotated (W H) basis; older/direct records remain in
        # the original basis and need an explicit Hadamard rotation.  Mixing
        # these two cases silently destroys a continuation warm start.
        if record.mode != "symmetric_hadamard" and quantizer_mode in {
            "ttq_hadamard",
            SYMMETRIC_HADAMARD_MODE,
        }:
            master = tq.rotate_weight_hadamard(master, group_size)
    else:
        # dense_teacher capture is already rotated by the caller for the
        # Hadamard mode; never rotate an explicitly supplied master twice.
        master = np.asarray(master_weight, dtype=np.float32)
    if master.shape != (record.out_dim, record.in_dim):
        raise ValueError(
            f"master weight shape {master.shape} does not match "
            f"record {(record.out_dim, record.in_dim)}"
        )
    qat.weight = mx.array(master, dtype=mx.float32)
    if module_bias is not None:
        bias = (
            np.asarray(master_bias, dtype=np.float32)
            if master_bias is not None
            else np.asarray(module_bias, dtype=np.float32)
        )
        qat.bias = mx.array(bias, dtype=mx.float32)
    qat.initialize_learned_scales()
    if quantizer_mode in TTQ_MODES and record.mode in TTQ_MODES:
        assert record.positive_scales is not None
        assert record.negative_scales is not None
        qat.log_scales = mx.log(
            mx.maximum(mx.array(record.positive_scales, dtype=mx.float32), 1e-6)
        )
        qat.log_negative_scales = mx.log(
            mx.maximum(mx.array(record.negative_scales, dtype=mx.float32), 1e-6)
        )
        qat.group_biases = mx.array(record.group_means, dtype=mx.float32)
    return qat


def load_state_cache(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[mx.array], mx.array, dict]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "onus.ternary-quality/v7-state-cache":
        raise ValueError(f"unsupported state cache schema: {manifest.get('schema')!r}")
    runtime_contract = manifest.get("runtime_contract", {})
    expected_contract = build_state_cache_contract(
        tq.MLX_RUNTIME_ROOT,
        Path(__file__).resolve().parent,
        int(manifest["cache"]["crop_len"]),
        float(manifest["cache"]["seconds"]),
        int(manifest["cache"]["trajectory_steps"]),
    )
    if runtime_contract != expected_contract:
        raise ValueError("state cache runtime contract does not match current code and precision")
    with np.load(cache_dir / "states.npz", allow_pickle=False) as arrays:
        states = np.array(arrays["states"], dtype=np.float16)
        sigmas = np.array(arrays["sigmas"], dtype=np.float32)
        prompt_indices = np.array(arrays["prompt_indices"], dtype=np.int32)
        sources = np.array(arrays["sources"], dtype=str)
    with np.load(cache_dir / "conditions.npz", allow_pickle=False) as arrays:
        cross = [mx.array(arrays[f"cross_{index:04d}"]) for index in range(len(manifest["cache"]["prompt_index"]))]
        global_cond = mx.array(arrays["global_cond"])
    if not (
        len(states) == len(sigmas) == len(prompt_indices) == len(sources)
    ):
        raise ValueError("state cache arrays have inconsistent lengths")
    if np.any(prompt_indices < 0) or np.any(
        prompt_indices >= len(manifest["cache"]["prompt_index"])
    ):
        raise ValueError("state cache has an invalid prompt index")
    return states, sigmas, prompt_indices, sources, cross, global_cond, manifest


def balanced_fixed_state_indices(
    prompt_indices: np.ndarray,
    sigmas: np.ndarray,
    sources: np.ndarray,
    count: int,
) -> list[int]:
    """Choose a deterministic pilot subset balanced over source, prompt and t."""
    if not 0 < count <= len(prompt_indices):
        raise ValueError(f"fixed-state-count must be in [1,{len(prompt_indices)}]")
    prompt_values = sorted(int(value) for value in np.unique(prompt_indices))
    source_values = sorted(str(value) for value in np.unique(sources))
    sigma_values = sorted((float(value) for value in np.unique(sigmas)), reverse=True)
    prompt_count = min(
        len(prompt_values), max(1, count // (len(source_values) * 2))
    )
    sigma_count = max(1, count // (len(source_values) * prompt_count))
    if len(source_values) * prompt_count * sigma_count > count:
        sigma_count = max(1, count // (len(source_values) * prompt_count))
    prompt_positions = np.linspace(
        0, len(prompt_values) - 1, prompt_count, dtype=np.int32
    )
    sigma_positions = np.linspace(
        0, len(sigma_values) - 1, sigma_count, dtype=np.int32
    )
    selected_prompts = [prompt_values[int(value)] for value in prompt_positions]
    selected_sigmas = [sigma_values[int(value)] for value in sigma_positions]
    chosen: list[int] = []
    for prompt in selected_prompts:
        for sigma in selected_sigmas:
            for source in source_values:
                candidates = np.flatnonzero(
                    (prompt_indices == prompt)
                    & np.isclose(sigmas, sigma, rtol=0.0, atol=1e-7)
                    & (sources == source)
                )
                if len(candidates):
                    chosen.append(int(candidates[0]))
    if len(chosen) < count:
        for index in range(len(prompt_indices)):
            if index not in chosen:
                chosen.append(index)
                if len(chosen) == count:
                    break
    if len(chosen) != count:
        raise ValueError(f"could select only {len(chosen)} of {count} fixed states")
    return chosen


def _tree_has_leaves(tree) -> bool:
    if tree is None:
        return False
    if isinstance(tree, dict):
        return any(_tree_has_leaves(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return any(_tree_has_leaves(value) for value in tree)
    return True


def checkpoint_run_signature(
    args,
    cache_dir: Path,
    target_metadata: dict | None,
    trajectory_metadata: dict | None,
    full_rollout_metadata: dict | None,
    student: nn.Module,
    warm_start_parent: dict[str, object] | None = None,
) -> dict[str, object]:
    trainable = dict(tq.tree_flatten(student.trainable_parameters()))
    return {
        "schema": "onus.ternary-quality/v7-window-run",
        "output_dir": str(args.output_dir.resolve()),
        "window": [int(args.start_block), int(args.end_block)],
        "group_size": int(args.group_size),
        "quantizer_mode": str(args.quantizer_mode),
        "quantizer_surrogate": str(args.quantizer_surrogate),
        "ttq_train_parameters": str(args.ttq_train_parameters),
        "master_init": str(args.master_init),
        "pair_sampling_mode": str(args.pair_sampling_mode),
        "trajectory_pointwise_source": str(args.trajectory_pointwise_source),
        "steps": int(args.steps),
        "fixed_state_count": int(args.fixed_state_count),
        "focus_prompt_index": int(args.focus_prompt_index),
        "state_sampling": str(args.state_sampling),
        "max_metal_bytes": int(args.max_metal_bytes),
        "gradient_accumulation": int(args.gradient_accumulation),
        "learning_rate": [float(args.learning_rate), float(args.learning_rate_end)],
        "scale_learning_rate": [
            float(args.scale_learning_rate),
            float(args.scale_learning_rate_end),
        ],
        "soft_end_updates": int(args.soft_end_updates),
        "soft_sharpness": [
            float(args.soft_sharpness_start),
            float(args.soft_sharpness_end),
        ],
        "weight_decay": float(args.weight_decay),
        "optimizer_eps": float(args.optimizer_eps),
        "gradient_clip": float(args.gradient_clip),
        "seed": int(args.seed),
        "teacher": tq.file_fingerprint(args.teacher_weights),
        "state_cache": {
            name: tq.file_fingerprint(cache_dir / name)
            for name in ("manifest.json", "states.npz", "conditions.npz")
        },
        "teacher_targets": (
            {
                "path": target_metadata.get("cache_file"),
                "sha256": target_metadata.get("target_sha256"),
                "contract": target_metadata.get("inputs", {}).get("target_contract"),
            }
            if target_metadata
            else None
        ),
        "trajectory_pairs": (
            {
                "path": trajectory_metadata.get("cache_file"),
                "sha256": trajectory_metadata.get("cache_sha256"),
                "manifest_sha256": trajectory_metadata.get("manifest_sha256"),
                "contract_digest": trajectory_metadata.get("contract_digest"),
                "loss_weight": float(args.trajectory_loss_weight),
                "trajectory_microbatch_fraction": (
                    0.5 if args.trajectory_loss_weight > 0 else 0.0
                ),
            }
            if trajectory_metadata
            else None
        ),
        "full_rollout_targets": (
            {
                "path": full_rollout_metadata.get("cache_file"),
                "sha256": full_rollout_metadata.get("cache_sha256"),
                "manifest_sha256": full_rollout_metadata.get("manifest_sha256"),
                "contract_digest": full_rollout_metadata.get("contract_digest"),
                "loss_weight": float(args.full_rollout_loss_weight),
                "window_steps": int(args.full_rollout_window_steps),
                "on_policy_stitch": bool(args.full_rollout_on_policy_stitch),
            }
            if full_rollout_metadata
            else None
        ),
        "source_records": (
            tq.file_fingerprint(args.source_records)
            if args.source_records is not None
            else None
        ),
        "warm_start_parent": warm_start_parent,
        "trainer": tq.file_fingerprint(Path(__file__)),
        "trainable_inventory": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(trainable.items())
        },
    }


def validate_warm_start_payload(
    checkpoint_path: Path,
    checkpoint_state: dict,
    expected_model_parameters: object,
    source_records_path: Path,
    teacher_weights: Path,
    window: tuple[int, int],
    group_size: int,
    quantizer_mode: str,
    weight_decay: float,
    optimizer_eps: float,
    gradient_accumulation: int,
    seed: int,
) -> dict[str, object]:
    """Validate a deliberate cross-run transfer without weakening exact resume."""
    metadata = checkpoint_state.get("metadata", {})
    if metadata.get("schema") != "onus.ternary-quality/v4-step-checkpoint":
        raise ValueError("warm-start requires a complete v4 step checkpoint")
    if metadata.get("complete") is not True:
        raise ValueError("warm-start checkpoint is not marked complete")
    if metadata.get("window") != [int(window[0]), int(window[1])]:
        raise ValueError("warm-start checkpoint window does not match")
    if int(metadata.get("group_size", -1)) != int(group_size):
        raise ValueError("warm-start checkpoint group size does not match")
    if metadata.get("quantizer_mode") != quantizer_mode:
        raise ValueError("warm-start checkpoint quantizer mode does not match")
    step_next = int(metadata.get("step_next", -1))
    if step_next < 1:
        raise ValueError("warm-start checkpoint has an invalid step counter")

    parent_dir = checkpoint_path.resolve().parent.parent
    parent_records = parent_dir / "records_checkpoint.npz"
    if not parent_records.is_file() or not source_records_path.is_file():
        raise ValueError("warm-start parent and source records must both exist")
    parent_signature = metadata.get("run_signature")
    if not isinstance(parent_signature, dict):
        raise ValueError("warm-start checkpoint omits its source run signature")
    if Path(str(parent_signature.get("output_dir", ""))).resolve() != parent_dir:
        raise ValueError("warm-start checkpoint path disagrees with its source run")
    if parent_signature.get("window") != [int(window[0]), int(window[1])]:
        raise ValueError("warm-start source signature window does not match")
    if int(parent_signature.get("group_size", -1)) != int(group_size):
        raise ValueError("warm-start source signature group size does not match")
    if parent_signature.get("quantizer_mode") != quantizer_mode:
        raise ValueError("warm-start source signature quantizer mode does not match")
    if float(parent_signature.get("weight_decay", float("nan"))) != float(weight_decay):
        raise ValueError("warm-start weight decay differs from source optimizer")
    if float(parent_signature.get("optimizer_eps", float("nan"))) != float(optimizer_eps):
        raise ValueError("warm-start optimizer epsilon differs from source optimizer")
    if int(parent_signature.get("gradient_accumulation", -1)) != int(
        gradient_accumulation
    ):
        raise ValueError("warm-start gradient accumulation differs from source run")
    if int(parent_signature.get("seed", -1)) != int(seed):
        raise ValueError("warm-start seed differs from source run")
    parent_teacher = parent_signature.get("teacher", {})
    current_teacher = tq.file_fingerprint(teacher_weights)
    if not parent_teacher.get("sha256") or (
        parent_teacher.get("sha256") != current_teacher.get("sha256")
    ):
        raise ValueError("warm-start teacher weights do not match source checkpoint")
    parent_records_sha = sha256_file(parent_records)
    source_records_sha = sha256_file(source_records_path)
    if parent_records_sha != source_records_sha:
        raise ValueError("warm-start source records differ from parent run records")

    expected_parameters = dict(tq.tree_flatten(expected_model_parameters))
    checkpoint_parameters = dict(tq.tree_flatten(checkpoint_state["model_state"]))
    if set(expected_parameters) != set(checkpoint_parameters):
        raise ValueError("warm-start trainable parameter keys do not match")
    for key, expected in expected_parameters.items():
        expected_array = np.asarray(expected)
        saved_array = np.asarray(checkpoint_parameters[key])
        if expected_array.shape != saved_array.shape:
            raise ValueError(f"warm-start shape mismatch for {key}")
        if expected_array.dtype != np.float32 or saved_array.dtype != np.float32:
            raise ValueError(f"warm-start master must remain FP32 for {key}")

    optimizer_parameters = dict(
        tq.tree_flatten(checkpoint_state["optimizer_state"])
    )
    expected_optimizer_keys = {"main.step", "main.learning_rate"}
    for key in expected_parameters:
        expected_optimizer_keys.add(f"main.{key}.m")
        expected_optimizer_keys.add(f"main.{key}.v")
    if set(optimizer_parameters) != expected_optimizer_keys:
        raise ValueError("warm-start Adam state keys do not match trainable masters")
    for key in expected_parameters:
        for moment in ("m", "v"):
            saved = np.asarray(optimizer_parameters[f"main.{key}.{moment}"])
            expected = np.asarray(expected_parameters[key])
            if saved.shape != expected.shape or saved.dtype != np.float32:
                raise ValueError(f"warm-start Adam {moment} state mismatch for {key}")
    optimizer_step = np.asarray(optimizer_parameters["main.step"])
    learning_rate = np.asarray(optimizer_parameters["main.learning_rate"])
    if optimizer_step.shape != () or int(optimizer_step) != step_next:
        raise ValueError("warm-start optimizer step disagrees with checkpoint metadata")
    if learning_rate.shape != () or learning_rate.dtype != np.float32:
        raise ValueError("warm-start optimizer learning rate must be scalar FP32")
    effective_learning_rate = float(learning_rate)
    if not np.isfinite(effective_learning_rate) or effective_learning_rate <= 0:
        raise ValueError("warm-start optimizer learning rate is invalid")

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "payload_sha256": metadata.get("payload_sha256"),
        "parent_run_dir": str(parent_dir),
        "parent_records_sha256": parent_records_sha,
        "step_next": step_next,
        "optimizer_step": int(optimizer_step),
        "effective_learning_rate": effective_learning_rate,
        "master_parameter_count": int(
            sum(np.asarray(value).size for value in checkpoint_parameters.values())
        ),
        "master_parameter_keys": len(checkpoint_parameters),
    }


def paired_epoch_pair_index(pair_order: list[int], update: int, microbatch: int) -> int:
    """Use two unique pair anchors per update, repeated once for paired loss."""
    if not pair_order:
        raise ValueError("paired-epoch sampling requires a non-empty pair order")
    if update < 0 or microbatch not in range(4):
        raise ValueError("paired-epoch sampling requires update >= 0 and four microbatches")
    pair_slot = update * 2 + microbatch // 2
    return int(pair_order[pair_slot % len(pair_order)])


def paired_epoch_trajectory_coverage(
    pair_order: list[int], updates: int
) -> set[int]:
    return {
        paired_epoch_pair_index(pair_order, update, microbatch)
        for update in range(updates)
        for microbatch in (1, 3)
    }


def compare_restored_records(
    student,
    source_records: dict[str, TernaryWeights],
    start_block: int,
    end_block: int,
    group_size: int,
) -> dict[str, int]:
    """Require checkpoint masters to re-quantize to source codes and scales."""
    code_mismatches = 0
    scale_mismatches = 0
    bias_mismatches = 0
    compared = 0
    for block_index in range(start_block, end_block + 1):
        for name in tq.CORE_NAMES:
            path = f"transformer.layers.{block_index}.{name}"
            record = source_records.get(path)
            if record is None:
                raise ValueError(f"warm-start records omit {path}")
            module = tq.module_at(student.transformer.layers[block_index], name)
            if not isinstance(module, tq.TernaryQATLinear):
                raise ValueError(f"warm-start module was not reopened for QAT: {path}")
            mx.eval(module.weight)
            current = tq.quantize_symmetric_weight(
                np.asarray(module.weight, dtype=np.float32), group_size=group_size
            )
            code_mismatches += int(np.count_nonzero(current.q != record.q))
            scale_mismatches += int(
                np.count_nonzero(current.scales != record.scales)
            )
            bias_mismatches += int(
                np.count_nonzero(current.biases != record.biases)
            )
            compared += int(record.q.size)
    report = {
        "matrices": (end_block - start_block + 1) * len(tq.CORE_NAMES),
        "codes_compared": compared,
        "code_mismatches": code_mismatches,
        "scale_mismatches": scale_mismatches,
        "derived_bias_mismatches": bias_mismatches,
    }
    if code_mismatches or scale_mismatches or bias_mismatches:
        raise ValueError(f"warm-start hard records differ from source: {report}")
    return report


def compare_hard_forward_fixture(student, fixture_path: Path) -> dict[str, object]:
    """Compare restored QAT forward with the established hard-reload tolerance."""
    if not fixture_path.is_file():
        raise ValueError(f"warm-start source forward fixture is missing: {fixture_path}")
    errors: list[dict[str, float]] = []
    with np.load(fixture_path, allow_pickle=False) as fixture:
        count = int(np.asarray(fixture["sample_count"]))
        global_cond = mx.array(fixture["global_cond"])
        for index in range(count):
            x = mx.array(fixture[f"x_{index}"])
            t = mx.array(fixture[f"t_{index}"], dtype=mx.float32)
            cross = mx.array(fixture[f"cross_{index}"])
            expected = np.asarray(fixture[f"expected_{index}"])
            actual = student(x, t, cross, global_cond)
            mx.eval(actual)
            actual = np.asarray(actual)
            actual32 = actual.astype(np.float32).reshape(-1)
            expected32 = expected.astype(np.float32).reshape(-1)
            difference = actual32 - expected32
            relative_l2 = float(
                np.linalg.norm(difference.reshape(-1))
                / max(np.linalg.norm(expected32), 1e-12)
            )
            cosine = float(
                np.dot(actual32, expected32)
                / max(
                    float(np.linalg.norm(actual32) * np.linalg.norm(expected32)),
                    1e-12,
                )
            )
            errors.append(
                {
                    "max_abs": float(np.max(np.abs(difference))),
                    "relative_l2": relative_l2,
                    "cosine": cosine,
                }
            )
    maximum_absolute = max((item["max_abs"] for item in errors), default=0.0)
    maximum_relative = max((item["relative_l2"] for item in errors), default=0.0)
    minimum_cosine = min((item["cosine"] for item in errors), default=1.0)
    if maximum_relative > 1e-3 or minimum_cosine < 0.99999:
        raise ValueError(
            "warm-start hard forward differs from source fixture: "
            f"max_abs={maximum_absolute:.8g} relative_l2={maximum_relative:.8g} "
            f"cosine={minimum_cosine:.9g}"
        )
    return {
        "fixture": str(fixture_path.resolve()),
        "samples": len(errors),
        "max_abs": maximum_absolute,
        "max_relative_l2": maximum_relative,
        "min_cosine": minimum_cosine,
        "tolerance_relative_l2_max": 1e-3,
        "tolerance_cosine_min": 0.99999,
        "passed": True,
    }


def split_scale_parameter_tree(
    tree,
    key: str | None = None,
    parameter_policy: str = "all",
):
    """Separate master weights and selected learned quantizer parameters."""
    quantizer_parameter_keys = {
        "log_scales",
        "log_negative_scales",
        "log_threshold_multiplier",
        "group_biases",
    }
    frozen_quantizer_keys = {
        "log_scales",
        "log_negative_scales",
        "group_biases",
    }
    if parameter_policy not in {"all", "thresholds"}:
        raise ValueError(f"unknown quantizer parameter policy: {parameter_policy}")
    if parameter_policy == "thresholds" and key in frozen_quantizer_keys:
        return None, None
    if isinstance(tree, dict):
        weights = {}
        scales = {}
        for key, value in tree.items():
            if key in quantizer_parameter_keys:
                if parameter_policy == "thresholds" and key in frozen_quantizer_keys:
                    continue
                scales[key] = value
                continue
            child_weights, child_scales = split_scale_parameter_tree(
                value, key, parameter_policy
            )
            if _tree_has_leaves(child_weights):
                weights[key] = child_weights
            if _tree_has_leaves(child_scales):
                scales[key] = child_scales
        return weights, scales
    if isinstance(tree, list):
        split = [
            split_scale_parameter_tree(value, parameter_policy=parameter_policy)
            for value in tree
        ]
        return (
            [weights if _tree_has_leaves(weights) else {} for weights, _ in split],
            [scales if _tree_has_leaves(scales) else {} for _, scales in split],
        )
    if isinstance(tree, tuple):
        split = [
            split_scale_parameter_tree(value, parameter_policy=parameter_policy)
            for value in tree
        ]
        return (
            tuple(weights if _tree_has_leaves(weights) else {} for weights, _ in split),
            tuple(scales if _tree_has_leaves(scales) else {} for _, scales in split),
        )
    return (None, tree) if key in quantizer_parameter_keys else (tree, None)


def learned_quantizer_diagnostics(
    student, start_block: int, end_block: int, group_size: int
) -> dict[str, object] | None:
    """Summarize threshold movement and hard-code changes before export."""
    threshold_logs: list[np.ndarray] = []
    changed_codes = 0
    total_codes = 0
    for block_index in range(start_block, end_block + 1):
        for module in tq.core_modules(student.transformer.layers[block_index]).values():
            if not isinstance(module, tq.TernaryQATLinear):
                continue
            mx.eval(module.weight, module.log_threshold_multiplier)
            weight = np.asarray(module.weight, dtype=np.float32)
            groups = weight.reshape(
                module.out_dim, module.in_dim // group_size, group_size
            )
            logs = np.clip(
                np.asarray(module.log_threshold_multiplier, dtype=np.float32),
                tq.LEARNED_THRESHOLD_LOG_BOUNDS[0],
                tq.LEARNED_THRESHOLD_LOG_BOUNDS[1],
            )
            base = np.maximum(np.mean(np.abs(groups), axis=-1), 1e-6)
            fixed_q = np.clip(
                np.rint(groups / base[..., None]), -1, 1
            ).astype(np.int8)
            learned_q = np.clip(
                np.rint(groups / (base * np.exp(logs))[..., None]), -1, 1
            ).astype(np.int8)
            threshold_logs.append(logs.reshape(-1))
            changed_codes += int(np.sum(fixed_q != learned_q))
            total_codes += int(fixed_q.size)
    if not threshold_logs:
        return None
    logs = np.concatenate(threshold_logs)
    multipliers = np.exp(logs)
    bound = max(abs(value) for value in tq.LEARNED_THRESHOLD_LOG_BOUNDS)
    return {
        "group_count": int(logs.size),
        "threshold_log_abs_mean": float(np.mean(np.abs(logs))),
        "threshold_multiplier_min": float(np.min(multipliers)),
        "threshold_multiplier_median": float(np.median(multipliers)),
        "threshold_multiplier_mean": float(np.mean(multipliers)),
        "threshold_multiplier_max": float(np.max(multipliers)),
        "groups_with_nonzero_threshold_update": int(np.sum(np.abs(logs) > 1e-7)),
        "fraction_at_threshold_bound": float(
            np.mean(np.abs(logs) >= bound - 1e-6)
        ),
        "hard_code_values_changed_vs_absmean": changed_codes,
        "hard_code_value_count": total_codes,
        "hard_code_change_fraction_vs_absmean": float(
            changed_codes / max(total_codes, 1)
        ),
    }


def learned_threshold_snapshot(student, start_block: int, end_block: int) -> np.ndarray:
    leaves = []
    for block_index in range(start_block, end_block + 1):
        for module in tq.core_modules(student.transformer.layers[block_index]).values():
            if isinstance(module, tq.TernaryQATLinear):
                mx.eval(module.log_threshold_multiplier)
                leaves.append(
                    np.asarray(module.log_threshold_multiplier, dtype=np.float32).reshape(-1)
                )
    return np.concatenate(leaves) if leaves else np.zeros((0,), dtype=np.float32)


def symmetric_code_snapshot(
    student, start_block: int, end_block: int, group_size: int
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Measure the exact hard codes, threshold proximity and fitted scales."""
    codes: dict[str, np.ndarray] = {}
    per_module: dict[str, dict[str, object]] = {}
    total_histogram = {-1: 0, 0: 0, 1: 0}
    scale_values: list[np.ndarray] = []
    distance_values: list[np.ndarray] = []
    zero_groups = 0
    group_count = 0
    for block_index in range(start_block, end_block + 1):
        for name, module in tq.core_modules(
            student.transformer.layers[block_index]
        ).items():
            if not isinstance(module, tq.TernaryQATLinear):
                continue
            mx.eval(module.weight)
            weight = np.asarray(module.weight, dtype=np.float32)
            quantized = tq.quantize_symmetric_weight(weight, group_size=group_size)
            q = np.asarray(quantized.q, dtype=np.int8)
            path = f"transformer.layers.{block_index}.{name}"
            codes[path] = q.copy()
            histogram = {value: int(np.sum(q == value)) for value in (-1, 0, 1)}
            for value, count in histogram.items():
                total_histogram[value] += count
            groups = weight.reshape(
                module.out_dim, module.in_dim // group_size, group_size
            )
            base = np.maximum(np.mean(np.abs(groups), axis=-1), 1e-6)
            normalized = np.abs(groups) / base[..., None]
            distances = np.abs(normalized - 0.5).reshape(-1)
            scales = np.abs(np.asarray(quantized.scales, dtype=np.float32)).reshape(-1)
            module_zero_groups = int(np.sum(np.sum(q != 0, axis=-1) == 0))
            module_group_count = int(q.shape[0] * q.shape[1])
            zero_groups += module_zero_groups
            group_count += module_group_count
            scale_values.append(scales)
            distance_values.append(distances)
            per_module[path] = {
                "code_count": int(q.size),
                "code_histogram": {str(key): value for key, value in histogram.items()},
                "zero_groups": module_zero_groups,
                "group_count": module_group_count,
                "scale_abs_quantiles": [
                    float(value)
                    for value in np.quantile(scales, [0.0, 0.5, 0.9, 1.0])
                ],
                "threshold_distance_quantiles": [
                    float(value)
                    for value in np.quantile(distances, [0.01, 0.1, 0.5])
                ],
                "fraction_within_1pct_threshold": float(np.mean(distances <= 0.01)),
            }
    if not codes:
        raise RuntimeError("symmetric code audit found no trainable QAT modules")
    scales_all = np.concatenate(scale_values)
    distances_all = np.concatenate(distance_values)
    return codes, {
        "code_count": int(sum(total_histogram.values())),
        "code_histogram": {str(key): value for key, value in total_histogram.items()},
        "zero_groups": zero_groups,
        "group_count": group_count,
        "scale_abs_quantiles": [
            float(value)
            for value in np.quantile(scales_all, [0.0, 0.5, 0.9, 1.0])
        ],
        "threshold_distance_quantiles": [
            float(value)
            for value in np.quantile(distances_all, [0.01, 0.1, 0.5])
        ],
        "fraction_within_1pct_threshold": float(np.mean(distances_all <= 0.01)),
        "per_module": per_module,
    }


def attach_code_transitions(
    metrics: dict[str, object],
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    initial: dict[str, np.ndarray],
) -> dict[str, object]:
    per_module = metrics["per_module"]
    changed_previous = 0
    changed_initial = 0
    total = 0
    for path, values in current.items():
        flips_previous = int(np.count_nonzero(values != previous[path]))
        flips_initial = int(np.count_nonzero(values != initial[path]))
        per_module[path]["flips_since_previous_audit"] = flips_previous
        per_module[path]["flips_since_start"] = flips_initial
        changed_previous += flips_previous
        changed_initial += flips_initial
        total += int(values.size)
    metrics["flips_since_previous_audit"] = changed_previous
    metrics["flips_since_start"] = changed_initial
    metrics["code_count"] = total
    metrics["flip_fraction_since_previous_audit"] = changed_previous / max(total, 1)
    metrics["flip_fraction_since_start"] = changed_initial / max(total, 1)
    return metrics


def master_weight_snapshot(student, start_block: int, end_block: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for block_index in range(start_block, end_block + 1):
        for name, module in tq.core_modules(
            student.transformer.layers[block_index]
        ).items():
            if isinstance(module, tq.TernaryQATLinear):
                mx.eval(module.weight)
                result[f"transformer.layers.{block_index}.{name}"] = np.asarray(
                    module.weight, dtype=np.float32
                ).copy()
    return result


def master_update_diagnostics(
    current: dict[str, np.ndarray], previous: dict[str, np.ndarray]
) -> dict[str, object]:
    per_module: dict[str, dict[str, float]] = {}
    update_sq = 0.0
    weight_sq = 0.0
    for path, values in current.items():
        delta = values - previous[path]
        module_update = float(np.sum(delta * delta, dtype=np.float64))
        module_weight = float(np.sum(values * values, dtype=np.float64))
        update_sq += module_update
        weight_sq += module_weight
        per_module[path] = {
            "update_l2": float(np.sqrt(module_update)),
            "weight_l2": float(np.sqrt(module_weight)),
            "update_to_weight_ratio": float(
                np.sqrt(module_update) / max(np.sqrt(module_weight), 1e-12)
            ),
        }
    return {
        "update_l2": float(np.sqrt(update_sq)),
        "weight_l2": float(np.sqrt(weight_sq)),
        "update_to_weight_ratio": float(
            np.sqrt(update_sq) / max(np.sqrt(weight_sq), 1e-12)
        ),
        "per_module": per_module,
    }


def tree_is_finite(tree) -> bool:
    checks = [mx.all(mx.isfinite(value)) for _, value in tq.tree_flatten(tree)]
    if not checks:
        return True
    mx.eval(*checks)
    return all(bool(np.asarray(value)) for value in checks)


def optimizer_learning_rate(optimizer) -> float:
    values = dict(tq.tree_flatten(optimizer.state))
    matches = [
        value for key, value in values.items()
        if str(key).split(".")[-1] == "learning_rate"
    ]
    if len(matches) != 1:
        raise ValueError("optimizer state must expose exactly one learning_rate")
    rate = np.asarray(matches[0])
    if rate.shape != () or not np.isfinite(float(rate)) or float(rate) <= 0:
        raise ValueError("optimizer effective learning rate is invalid")
    return float(rate)


def cosine_schedule_value(start: float, end: float, steps: int, step: int) -> float:
    progress = min(max(step, 0) / max(steps, 1), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * progress)))


def quantizer_schedule(
    step: int,
    soft_end_updates: int,
    sharpness_start: float,
    sharpness_end: float,
) -> tuple[bool, float]:
    """Return the deterministic smooth-to-hard QAT schedule for one update."""
    if soft_end_updates <= 0:
        return False, float(sharpness_end)
    if step < soft_end_updates:
        progress = step / max(soft_end_updates - 1, 1)
        sharpness = sharpness_start + (sharpness_end - sharpness_start) * progress
        return True, float(sharpness)
    return False, float(sharpness_end)


def set_quantizer_schedule(
    student: nn.Module,
    start_block: int,
    end_block: int,
    soft_forward: bool,
    sharpness: float,
) -> None:
    for block_index in range(start_block, end_block + 1):
        for module in tq.core_modules(student.transformer.layers[block_index]).values():
            if isinstance(module, tq.TernaryQATLinear):
                module.set_quantization_schedule(soft_forward, sharpness)


def set_quantizer_surrogate(
    student: nn.Module,
    start_block: int,
    end_block: int,
    mode: str,
) -> None:
    for block_index in range(start_block, end_block + 1):
        for module in tq.core_modules(student.transformer.layers[block_index]).values():
            if isinstance(module, tq.TernaryQATLinear):
                module.set_surrogate_mode(mode)


def save_hard_forward_fixture(
    path: Path,
    student,
    states: np.ndarray,
    sigmas: np.ndarray,
    prompt_indices: np.ndarray,
    cross_cache: list[mx.array],
    global_cond: mx.array,
    sample_indices: list[int],
) -> None:
    selected = sorted({sample_indices[0], sample_indices[len(sample_indices) // 2], sample_indices[-1]})
    arrays: dict[str, np.ndarray] = {
        "sample_count": np.asarray(len(selected), dtype=np.int32),
        "global_cond": np.asarray(global_cond),
    }
    for fixture_index, state_index in enumerate(selected):
        x = mx.array(states[state_index][None], dtype=mx.float16)
        t = timestep_tensor(float(sigmas[state_index]))
        cross = cross_cache[int(prompt_indices[state_index])]
        expected = student(x, t, cross, global_cond)
        mx.eval(expected)
        arrays[f"x_{fixture_index}"] = np.asarray(x)
        arrays[f"t_{fixture_index}"] = np.asarray(t, dtype=np.float32)
        arrays[f"cross_{fixture_index}"] = np.asarray(cross)
        arrays[f"expected_{fixture_index}"] = np.asarray(expected)
    np.savez_compressed(path, **arrays)


def full_velocity_loss(student, x, t, cross, global_cond, target):
    prediction = checkpointed_dit(student, x, t, cross, global_cond)
    return velocity_loss_from_prediction(prediction, target)


def velocity_loss_from_prediction(prediction, target):
    p = prediction.astype(mx.float32)
    q = mx.stop_gradient(target.astype(mx.float32))
    mse = mx.mean((p - q) ** 2) / (mx.mean(q * q) + 1e-6)
    cosine = mx.sum(p * q) / (
        mx.sqrt(mx.sum(p * p)) * mx.sqrt(mx.sum(q * q)) + 1e-6
    )
    rms_p = mx.sqrt(mx.mean(p * p) + 1e-6)
    rms_q = mx.sqrt(mx.mean(q * q) + 1e-6)
    rms = ((rms_p - rms_q) / rms_q) ** 2
    return mse + 0.1 * (1.0 - cosine) + 0.1 * rms


def two_step_student_trace(
    model_fn,
    anchor: mx.array,
    sigmas: tuple[float, float, float],
    noise_first: mx.array,
    noise_second: mx.array,
    pair_start: int,
    total_steps: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Differentiate exactly two production sampler steps from one anchor."""
    if total_steps < 2 or not 0 <= pair_start < total_steps - 1:
        raise ValueError("pair_start must identify two consecutive sampler steps")
    if len(sigmas) != 3:
        raise ValueError("two-step trace requires three consecutive sigmas")
    sigma0, sigma1, sigma2 = (float(value) for value in sigmas)
    velocity0 = model_fn(anchor, timestep_tensor(sigma0, anchor.shape[0]))
    state1 = pingpong_transition(
        anchor, velocity0, sigma0, sigma1, noise_first, pair_start, total_steps
    )
    velocity1 = model_fn(state1, timestep_tensor(sigma1, anchor.shape[0]))
    second_noise = (
        noise_second
        if pair_start + 1 < total_steps - 1 and sigma2 > 0.0
        else None
    )
    state2 = pingpong_transition(
        state1,
        velocity1,
        sigma1,
        sigma2,
        second_noise,
        pair_start + 1,
        total_steps,
    )
    return velocity0, state1, velocity1, state2


def trajectory_pair_loss(
    student,
    anchor: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    sigmas: tuple[float, float, float],
    noise_first: mx.array,
    noise_second: mx.array,
    target_velocity: mx.array,
    target_endpoint: mx.array,
    pair_start: int,
    total_steps: int,
    trajectory_weight: float,
):
    model_fn = lambda x, t: checkpointed_dit(student, x, t, cross, global_cond)
    velocity0, _state1, _velocity1, state2 = two_step_student_trace(
        model_fn,
        anchor,
        sigmas,
        noise_first,
        noise_second,
        pair_start,
        total_steps,
    )
    velocity_loss = velocity_loss_from_prediction(velocity0, target_velocity)
    prediction = state2.astype(mx.float32)
    target = mx.stop_gradient(target_endpoint.astype(mx.float32))
    trajectory_loss = mx.mean((prediction - target) ** 2) / (
        mx.mean(target * target) + 1e-6
    )
    return velocity_loss + trajectory_weight * trajectory_loss


def terminal_state_loss(prediction: mx.array, target: mx.array) -> mx.array:
    """Stable terminal objective: direction, energy and raw endpoint error."""
    p = prediction.astype(mx.float32)
    q = mx.stop_gradient(target.astype(mx.float32))
    mse = mx.mean((p - q) ** 2) / (mx.mean(q * q) + 1e-6)
    cosine = mx.sum(p * q) / (
        mx.sqrt(mx.sum(p * p)) * mx.sqrt(mx.sum(q * q)) + 1e-6
    )
    rms_p = mx.sqrt(mx.mean(p * p) + 1e-6)
    rms_q = mx.sqrt(mx.mean(q * q) + 1e-6)
    rms = ((rms_p - rms_q) / rms_q) ** 2
    return mse + 1.5 * (1.0 - cosine) + 0.25 * rms


def checkpointed_sampler_step(
    student,
    state: mx.array,
    noise: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    sigma: float,
    next_sigma: float,
    step: int,
    total_steps: int,
) -> mx.array:
    """Rematerialize one whole DiT sampler step to cap multi-step memory."""
    def apply_step(parameters, current_state, current_noise):
        student.update(parameters)
        velocity = checkpointed_dit(
            student,
            current_state,
            timestep_tensor(sigma, current_state.shape[0]),
            cross,
            global_cond,
        )
        return pingpong_transition(
            current_state,
            velocity,
            sigma,
            next_sigma,
            current_noise,
            step,
            total_steps,
        )

    return mx.checkpoint(apply_step)(
        student.trainable_parameters(), state, noise
    )


def rollout_state(
    student,
    initial: mx.array,
    noises: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    sigma_grid: tuple[float, ...],
    sampler_start_step: int = 0,
    sampler_total_steps: int = 8,
) -> mx.array:
    """Differentiate a consecutive production sampler segment."""
    total_steps = len(sigma_grid) - 1
    if total_steps not in {2, 4, 8}:
        raise ValueError("rollout requires a 2-, 4- or 8-step grid")
    if noises.shape[1] < total_steps:
        raise ValueError("rollout noise window is shorter than the sigma grid")
    if not 0 <= sampler_start_step < sampler_total_steps:
        raise ValueError("invalid sampler start step for rollout window")
    if sampler_start_step + total_steps > sampler_total_steps:
        raise ValueError("rollout window exceeds the production sampler")
    state = initial
    for step in range(total_steps):
        state = checkpointed_sampler_step(
            student,
            state,
            noises[:, step],
            cross,
            global_cond,
            sigma_grid[step],
            sigma_grid[step + 1],
            sampler_start_step + step,
            sampler_total_steps,
        )
    return state


def rollout_state_without_grad(
    student,
    initial: mx.array,
    noises: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    sigma_grid: tuple[float, ...],
    sampler_start_step: int,
    sampler_total_steps: int = 8,
) -> mx.array:
    """Capture an on-policy segment boundary without retaining its graph."""
    state = mx.stop_gradient(initial)
    total_steps = len(sigma_grid) - 1
    for step in range(total_steps):
        state = mx.stop_gradient(
            checkpointed_sampler_step(
                student,
                state,
                noises[:, step],
                cross,
                global_cond,
                sigma_grid[step],
                sigma_grid[step + 1],
                sampler_start_step + step,
                sampler_total_steps,
            )
        )
        mx.eval(state)
    return state


def full_rollout_loss(
    student,
    initial: mx.array,
    noises: mx.array,
    cross: mx.array,
    global_cond: mx.array,
    sigma_grid: tuple[float, ...],
    target_terminal: mx.array,
    rollout_weight: float,
    sampler_start_step: int = 0,
    sampler_total_steps: int = 8,
) -> mx.array:
    """Differentiate a production sampler window with a terminal endpoint."""
    state = rollout_state(
        student,
        initial,
        noises,
        cross,
        global_cond,
        sigma_grid,
        sampler_start_step,
        sampler_total_steps,
    )
    return rollout_weight * terminal_state_loss(state, target_terminal)


def load_trajectory_pair_cache(
    path: Path,
    state_cache: Path,
    source_records: Path,
    teacher_weights: Path,
    crop_len: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load pair anchors/targets only when their exact producer inputs match."""
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "onus.ternary-quality/v7-trajectory-pairs":
        raise ValueError("unsupported trajectory-pair cache schema")
    expected_inputs = {
        "state_cache_manifest": file_identity(state_cache / "manifest.json"),
        "states": file_identity(state_cache / "states.npz"),
        "conditions": file_identity(state_cache / "conditions.npz"),
        "source_records": file_identity(source_records),
        "teacher_weights": file_identity(teacher_weights),
    }
    if metadata.get("inputs") != expected_inputs:
        raise ValueError("trajectory-pair cache provenance does not match current inputs")
    cache_identity = file_identity(path)
    if metadata.get("cache_file") != str(path.resolve()):
        raise ValueError("trajectory-pair cache path disagrees with its manifest")
    if metadata.get("cache_sha256") != cache_identity["sha256"]:
        raise ValueError("trajectory-pair cache checksum mismatch")
    if metadata.get("arrays_sha256") != sha256_file(path):
        raise ValueError("trajectory-pair cache archive checksum mismatch")

    with np.load(path, allow_pickle=False) as archive:
        expected_names = {
            "anchors",
            "noise_first",
            "noise_second",
            "target_velocity",
            "target_state_one",
            "target_endpoint",
            "sigmas",
            "prompt_indices",
            "pair_steps",
            "generation_seeds",
            "second_noise_present",
        }
        if set(archive.files) != expected_names:
            raise ValueError("trajectory-pair cache arrays do not match the schema")
        arrays = {name: np.array(archive[name]) for name in expected_names}

    count = int(metadata.get("pair_count", -1))
    latent_shape = (count, 256, crop_len)
    for name in (
        "anchors",
        "noise_first",
        "noise_second",
        "target_velocity",
        "target_state_one",
        "target_endpoint",
    ):
        if arrays[name].shape != latent_shape or arrays[name].dtype != np.float16:
            raise ValueError(f"invalid {name} shape/dtype in trajectory-pair cache")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite values in trajectory-pair array {name}")
    if arrays["sigmas"].shape != (count, 3) or arrays["sigmas"].dtype != np.float32:
        raise ValueError("invalid sigma triplets in trajectory-pair cache")
    for name, dtype in (
        ("prompt_indices", np.int32),
        ("pair_steps", np.int8),
        ("generation_seeds", np.int64),
        ("second_noise_present", np.bool_),
    ):
        if arrays[name].shape != (count,) or arrays[name].dtype != dtype:
            raise ValueError(f"invalid {name} shape/dtype in trajectory-pair cache")
    if count <= 0 or np.any(arrays["pair_steps"] < 0) or np.any(arrays["pair_steps"] > 6):
        raise ValueError("trajectory-pair cache has invalid pair indices")
    if np.any(arrays["prompt_indices"] < 0) or np.any(
        arrays["prompt_indices"] >= len(metadata.get("prompts", []))
    ):
        raise ValueError("trajectory-pair cache has invalid prompt indices")
    step_counts = np.bincount(arrays["pair_steps"].astype(np.int32), minlength=7)
    if np.any(step_counts == 0) or int(step_counts.max() - step_counts.min()) > 1:
        raise ValueError("trajectory-pair cache is not balanced over its seven step pairs")
    if count != len(metadata.get("prompts", [])) * 2 * 7:
        raise ValueError("trajectory-pair cache is not 16-prompts x 2-seeds x 7-pairs")
    prompt_counts = np.bincount(
        arrays["prompt_indices"].astype(np.int32),
        minlength=len(metadata.get("prompts", [])),
    )
    if np.any(prompt_counts != 14):
        raise ValueError("trajectory-pair cache is not balanced over train prompts")
    sigma_grid = np.asarray(metadata.get("sampler", {}).get("sigma_grid", []), dtype=np.float32)
    if sigma_grid.shape != (9,):
        raise ValueError("trajectory-pair cache omits the production nine-point sigma grid")
    expected_triplets = np.stack(
        [sigma_grid[int(pair_step) : int(pair_step) + 3] for pair_step in arrays["pair_steps"]]
    )
    if not np.array_equal(arrays["sigmas"], expected_triplets):
        raise ValueError("trajectory-pair sigmas do not match the production schedule")
    if not np.array_equal(
        arrays["second_noise_present"], arrays["pair_steps"] < 6
    ):
        raise ValueError("trajectory-pair cache has invalid second-noise flags")
    loaded_metadata = {
        "cache_file": str(path.resolve()),
        "cache_sha256": cache_identity["sha256"],
        "manifest_sha256": file_identity(metadata_path)["sha256"],
        "contract_digest": metadata.get("contract_digest"),
        "metadata": metadata,
    }
    return arrays, loaded_metadata


def load_full_rollout_cache(
    path: Path,
    state_cache: Path,
    source_records: Path,
    teacher_weights: Path,
    crop_len: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load full-rollout targets only when all producer inputs match exactly."""
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "onus.ternary-quality/v7-full-rollout-targets":
        raise ValueError("unsupported full-rollout target cache schema")
    expected_inputs = {
        "state_cache_manifest": file_identity(state_cache / "manifest.json"),
        "states": file_identity(state_cache / "states.npz"),
        "conditions": file_identity(state_cache / "conditions.npz"),
        "source_records": file_identity(source_records),
        "teacher_weights": file_identity(teacher_weights),
    }
    audit_contract = metadata.get("inputs", {}).get("audit_contract")
    if isinstance(audit_contract, dict):
        audit_path = audit_contract.get("path")
        if not isinstance(audit_path, str) or not audit_path:
            raise ValueError("full-rollout cache has an invalid audit contract identity")
        expected_inputs["audit_contract"] = file_identity(Path(audit_path))
    if metadata.get("inputs") != expected_inputs:
        raise ValueError("full-rollout cache provenance does not match current inputs")
    cache_identity = file_identity(path)
    if metadata.get("cache_file") != str(path.resolve()):
        raise ValueError("full-rollout cache path disagrees with its manifest")
    if metadata.get("cache_sha256") != cache_identity["sha256"]:
        raise ValueError("full-rollout cache checksum mismatch")
    if metadata.get("arrays_sha256") != sha256_file(path):
        raise ValueError("full-rollout cache archive checksum mismatch")

    with np.load(path, allow_pickle=False) as archive:
        expected_names = {
            "initial_states",
            "source_states",
            "noises",
            "target_velocity",
            "target_terminal",
            "target_states",
            "prompt_indices",
            "generation_seeds",
        }
        if set(archive.files) != expected_names:
            raise ValueError("full-rollout cache arrays do not match the schema")
        arrays = {name: np.array(archive[name]) for name in expected_names}

    count = int(metadata.get("pair_count", -1))
    latent_shape = (count, 256, crop_len)
    for name in ("initial_states", "target_velocity", "target_terminal"):
        if arrays[name].shape != latent_shape or arrays[name].dtype != np.float16:
            raise ValueError(f"invalid {name} in full-rollout target cache")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite values in full-rollout array {name}")
    if arrays["source_states"].shape != (count, 9, 256, crop_len):
        raise ValueError("invalid source_states shape in full-rollout target cache")
    if arrays["target_states"].shape != (count, 9, 256, crop_len):
        raise ValueError("invalid target_states shape in full-rollout target cache")
    if arrays["source_states"].dtype != np.float16 or arrays["target_states"].dtype != np.float16:
        raise ValueError("full-rollout state arrays must be float16")
    if not np.isfinite(arrays["source_states"]).all() or not np.isfinite(
        arrays["target_states"]
    ).all():
        raise ValueError("non-finite values in full-rollout state arrays")
    if arrays["noises"].shape != (count, 7, 256, crop_len):
        raise ValueError("invalid noises shape in full-rollout target cache")
    if arrays["noises"].dtype != np.float16 or not np.isfinite(arrays["noises"]).all():
        raise ValueError("invalid noises values in full-rollout target cache")
    for name, dtype in (("prompt_indices", np.int32), ("generation_seeds", np.int64)):
        if arrays[name].shape != (count,) or arrays[name].dtype != dtype:
            raise ValueError(f"invalid {name} in full-rollout target cache")
    prompts = metadata.get("prompts", [])
    if count <= 0 or len(prompts) <= 0:
        raise ValueError("full-rollout target cache is empty")
    sampling_policy = str(metadata.get("sampling_policy", "balanced"))
    if sampling_policy == "balanced":
        if count != len(prompts) * int(metadata.get("repeats", 0)):
            raise ValueError("full-rollout target cache count is not prompt-balanced")
        prompt_counts = np.bincount(
            np.asarray(arrays["prompt_indices"], dtype=np.int32),
            minlength=len(prompts),
        )
        if not np.all(prompt_counts == int(metadata.get("repeats", 0))):
            raise ValueError("balanced full-rollout cache has unequal prompt counts")
    elif sampling_policy == "weighted":
        declared_counts = metadata.get("prompt_counts")
        if not isinstance(declared_counts, list) or len(declared_counts) != len(prompts):
            raise ValueError("weighted full-rollout cache omits prompt_counts")
        prompt_counts = np.bincount(
            np.asarray(arrays["prompt_indices"], dtype=np.int32),
            minlength=len(prompts),
        )
        if sum(int(value) for value in declared_counts) != count:
            raise ValueError("weighted full-rollout prompt_counts do not sum to count")
        if not np.array_equal(
            prompt_counts,
            np.asarray(declared_counts, dtype=np.int64),
        ):
            raise ValueError("weighted full-rollout prompt_counts disagree with rows")
    else:
        raise ValueError(f"unsupported full-rollout sampling policy: {sampling_policy}")
    if np.any(arrays["prompt_indices"] < 0) or np.any(
        arrays["prompt_indices"] >= len(prompts)
    ):
        raise ValueError("full-rollout target cache has invalid prompt indices")
    sigma_grid = np.asarray(
        metadata.get("sampler", {}).get("sigma_grid", []), dtype=np.float32
    )
    if sigma_grid.shape != (9,):
        raise ValueError("full-rollout target cache omits the production sigma grid")
    arrays["sigma_grid"] = sigma_grid
    loaded_metadata = {
        "cache_file": str(path.resolve()),
        "cache_sha256": cache_identity["sha256"],
        "manifest_sha256": file_identity(metadata_path)["sha256"],
        "contract_digest": metadata.get("contract_digest"),
        "metadata": metadata,
    }
    return arrays, loaded_metadata


def records_from_window(
    student,
    start_block: int,
    end_block: int,
    group_size: int,
    training_quantizer_mode: str,
) -> dict[str, TernaryWeights]:
    records: dict[str, TernaryWeights] = {}
    for block_index in range(start_block, end_block + 1):
        tq.hard_freeze_block(
            student.transformer.layers[block_index],
            block_index,
            group_size,
            records,
            training_quantizer_mode,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 full-DiT ternary-window pilot")
    parser.add_argument("--dataset-dir", type=Path, default=Path("output/sample-expertise-pilot/ternary-quality-v6-20260923/authorized-independent-sftvoices-v1/train"))
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument(
        "--source-records",
        type=Path,
        default=None,
        help="completed-block records checkpoint; dense teacher is rebuilt then these blocks are applied",
    )
    parser.add_argument(
        "--master-init",
        choices=("record_dequantized", "dense_teacher"),
        default="record_dequantized",
        help=(
            "initialize selected-block QAT masters from the source record's "
            "dequantized weights or from the original dense teacher weights"
        ),
    )
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument(
        "--teacher-target-cache",
        type=Path,
        default=None,
        help="provenance-checked fp16 teacher outputs; avoids loading the teacher during QAT",
    )
    parser.add_argument(
        "--trajectory-pair-cache",
        type=Path,
        default=None,
        help="matched student-rollout anchors and teacher branch targets for P3",
    )
    parser.add_argument(
        "--full-rollout-target-cache",
        type=Path,
        default=None,
        help="on-policy initial states/noises with exact teacher terminal targets",
    )
    parser.add_argument(
        "--trajectory-loss-weight",
        type=float,
        default=0.0,
        help="weight on the two-step endpoint NMSE; pair-cache batches use half trajectory and half pointwise microbatches",
    )
    parser.add_argument(
        "--full-rollout-loss-weight",
        type=float,
        default=0.0,
        help="weight on the differentiable production 8-step terminal objective",
    )
    parser.add_argument(
        "--full-rollout-window-steps",
        type=int,
        choices=(2, 4, 8),
        default=4,
        help="number of consecutive production steps differentiated per rollout sample",
    )
    parser.add_argument(
        "--full-rollout-on-policy-stitch",
        action="store_true",
        help=(
            "for a 4-step budget, train two sequential segments: the second "
            "starts from the detached student midpoint, keeping the 8-step "
            "terminal signal under the 12 GB memory guard"
        ),
    )
    parser.add_argument(
        "--pair-sampling-mode",
        choices=("with_replacement", "paired_epoch"),
        default="with_replacement",
        help="paired_epoch cycles every pair once and shares anchors across pointwise/trajectory microbatches",
    )
    parser.add_argument(
        "--trajectory-pointwise-source",
        choices=("pair", "state_cache"),
        default="pair",
        help=(
            "when trajectory pairs are enabled, use their matched anchors for "
            "the pointwise half (pair) or use independent cached states (state_cache)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--quantizer-mode",
        choices=(
            "symmetric",
        "learned_symmetric",
        "learned_symmetric_hadamard",
            "learned_affine",
            "ttq",
            "ttq_hadamard",
            "affine_centered",
        ),
        default="symmetric",
        help="symmetric=s*q; learned_symmetric learns symmetric levels; "
        "learned_affine learns affine levels/thresholds; ttq learns independent "
            "positive/negative levels; ttq_hadamard adds a block Hadamard "
            "rotation; affine_centered is fixed",
    )
    parser.add_argument(
        "--quantizer-surrogate",
        choices=("smooth", "identity"),
        default="smooth",
        help=(
            "gradient surrogate for learned ternary bins; identity is an "
            "LSQ-style straight-through derivative that can change source codes"
        ),
    )
    parser.add_argument(
        "--ttq-train-parameters",
        choices=("all", "thresholds"),
        default="all",
        help=(
            "TTQ calibration policy: train all levels/means/thresholds, or "
            "freeze levels and means and train thresholds only"
        ),
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--fixed-state-count",
        type=int,
        default=0,
        help="sample only this deterministic, balanced subset for a micro-overfit check",
    )
    parser.add_argument(
        "--focus-prompt-index",
        type=int,
        default=-1,
        help=(
            "pointwise-only diagnostic: restrict state-cache samples to one "
            "prompt index; -1 keeps the normal balanced/all-state pool"
        ),
    )
    parser.add_argument(
        "--state-sampling",
        choices=("with_replacement", "without_replacement"),
        default="with_replacement",
        help=(
            "pointwise state sampler; without_replacement completes each "
            "cached-state epoch before repeating"
        ),
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="stop after this many updates while preserving the --steps LR schedule for later resume",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--learning-rate-end", type=float, default=1e-6)
    parser.add_argument("--scale-learning-rate", type=float, default=3e-4)
    parser.add_argument("--scale-learning-rate-end", type=float, default=3e-5)
    parser.add_argument(
        "--soft-end-updates",
        type=int,
        default=0,
        help="keep learned quantizers on a smooth forward until this update, then hard STE",
    )
    parser.add_argument(
        "--soft-sharpness-start",
        type=float,
        default=1.5,
        help="sigmoid sharpness at the beginning of the smooth quantizer schedule",
    )
    parser.add_argument(
        "--soft-sharpness-end",
        type=float,
        default=8.0,
        help="sigmoid sharpness at the end of the smooth quantizer schedule",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-eps", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--resume-step-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--warm-start-step-checkpoint",
        type=Path,
        default=None,
        help="explicit cross-run transfer of FP32 masters and Adam state; exact resume remains signature-locked",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate warm-start provenance, hard records, forward fixture, LR and pair coverage without updating weights",
    )
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="write the cumulative records checkpoint without exporting the large dense-rest artifact",
    )
    args = parser.parse_args()
    if args.start_block < 0 or args.end_block < args.start_block or args.end_block >= 24:
        raise ValueError("invalid block window")
    if args.group_size not in (16, 32, 64, 128):
        raise ValueError("group_size must be 16, 32, 64, or 128")
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and gradient_accumulation must be positive")
    run_until_step = args.steps if args.max_updates is None else args.max_updates
    if not 1 <= run_until_step <= args.steps:
        raise ValueError("max_updates must be between 1 and steps")
    if args.learning_rate <= 0 or args.learning_rate_end <= 0:
        raise ValueError("master learning rates must be positive")
    if args.scale_learning_rate <= 0 or args.scale_learning_rate_end <= 0:
        raise ValueError("scale learning rates must be positive")
    if args.soft_end_updates < 0 or args.soft_end_updates > args.steps:
        raise ValueError("soft-end-updates must be between 0 and steps")
    if (
        args.soft_sharpness_start <= 0
        or args.soft_sharpness_end <= 0
        or not np.isfinite(args.soft_sharpness_start)
        or not np.isfinite(args.soft_sharpness_end)
    ):
        raise ValueError("soft quantizer sharpness values must be finite and positive")
    if args.max_metal_bytes <= 0:
        raise ValueError("max-metal-bytes must be positive")
    if args.trajectory_loss_weight < 0:
        raise ValueError("trajectory-loss-weight must be zero or positive")
    if args.full_rollout_loss_weight < 0:
        raise ValueError("full-rollout-loss-weight must be zero or positive")
    if args.trajectory_loss_weight > 0 and args.trajectory_pair_cache is None:
        raise ValueError("trajectory-loss-weight requires --trajectory-pair-cache")
    if args.full_rollout_loss_weight > 0 and args.full_rollout_target_cache is None:
        raise ValueError(
            "full-rollout-loss-weight requires --full-rollout-target-cache"
        )
    if args.full_rollout_target_cache is not None and args.full_rollout_window_steps > 8:
        raise ValueError("full-rollout window exceeds the cached production grid")
    if args.full_rollout_on_policy_stitch:
        if args.full_rollout_target_cache is None:
            raise ValueError(
                "full-rollout on-policy stitching requires --full-rollout-target-cache"
            )
        if args.full_rollout_window_steps != 4:
            raise ValueError(
                "full-rollout on-policy stitching requires a 4-step segment"
            )
    if args.trajectory_pair_cache is not None and args.full_rollout_target_cache is not None:
        raise ValueError("two-step and full-rollout caches are mutually exclusive")
    if args.pair_sampling_mode == "paired_epoch":
        if args.trajectory_pair_cache is None:
            raise ValueError("paired-epoch sampling requires --trajectory-pair-cache")
        if args.gradient_accumulation != 4:
            raise ValueError("paired-epoch sampling requires gradient accumulation 4")
    if args.trajectory_pair_cache is not None:
        if args.source_records is None:
            raise ValueError("trajectory-pair training requires --source-records")
        if (
            args.trajectory_pointwise_source == "pair"
            and args.fixed_state_count != 0
        ):
            raise ValueError(
                "trajectory-pair mode samples its own balanced anchors; set --fixed-state-count 0"
            )
        if args.gradient_accumulation % 2:
            raise ValueError(
                "trajectory-pair mode requires even gradient accumulation for a 50/50 mix"
            )
    if args.full_rollout_target_cache is not None:
        if args.source_records is None:
            raise ValueError("full-rollout training requires --source-records")
        if args.gradient_accumulation < 2:
            raise ValueError("full-rollout training requires gradient accumulation >= 2")
    if args.resume_step_checkpoint is not None and args.warm_start_step_checkpoint is not None:
        raise ValueError("exact resume and cross-run warm-start are mutually exclusive")
    if args.preflight_only and args.warm_start_step_checkpoint is None:
        raise ValueError("preflight-only requires --warm-start-step-checkpoint")
    if args.warm_start_step_checkpoint is not None:
        if args.source_records is None:
            raise ValueError("warm-start requires the matching --source-records")
        if args.quantizer_mode != "symmetric":
            raise ValueError("warm-start currently supports symmetric QAT only")
        if args.pair_sampling_mode != "paired_epoch":
            raise ValueError("warm-start requires paired-epoch sampling")

    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be zero or positive")
    if run_until_step < args.steps and args.checkpoint_every_steps == 0:
        raise ValueError("partial training requires step checkpoints for safe resume")
    if args.fixed_state_count < 0:
        raise ValueError("fixed-state-count must be zero or positive")
    if args.focus_prompt_index < -1:
        raise ValueError("focus-prompt-index must be -1 or non-negative")
    if args.resume_step_checkpoint is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {args.output_dir}")
    if args.resume_step_checkpoint is not None and (args.output_dir / "window_summary.json").exists():
        raise FileExistsError(f"run already has a final summary: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    (
        states,
        sigmas,
        prompt_indices,
        sources,
        cross_cache,
        global_cond,
        cache_manifest,
    ) = load_state_cache(args.state_cache)
    trajectory_pairs = None
    trajectory_pair_metadata = None
    full_rollout_targets = None
    full_rollout_metadata = None
    pair_indices_by_step: list[np.ndarray] = []
    pair_order: list[int] | None = None
    if args.trajectory_pair_cache is not None:
        trajectory_pairs, trajectory_pair_metadata = load_trajectory_pair_cache(
            args.trajectory_pair_cache,
            args.state_cache,
            args.source_records,
            args.teacher_weights,
            int(cache_manifest["cache"]["crop_len"]),
        )
        pair_indices_by_step = [
            np.flatnonzero(trajectory_pairs["pair_steps"] == pair_step)
            for pair_step in range(7)
        ]
        if args.pair_sampling_mode == "paired_epoch":
            pair_count = len(trajectory_pairs["pair_steps"])
            pair_order = random.Random(args.seed).sample(
                range(pair_count), pair_count
            )
            if args.warm_start_step_checkpoint is not None:
                prompt_count = len(trajectory_pair_metadata["metadata"].get("prompts", []))
                if pair_count != 224 or prompt_count != 16:
                    raise ValueError(
                        "P3.1 requires exactly 224 pairs across 16 train prompts"
                    )
    if args.full_rollout_target_cache is not None:
        full_rollout_targets, full_rollout_metadata = load_full_rollout_cache(
            args.full_rollout_target_cache,
            args.state_cache,
            args.source_records,
            args.teacher_weights,
            int(cache_manifest["cache"]["crop_len"]),
        )
    if args.state_sampling == "without_replacement" and (
        trajectory_pairs is not None or full_rollout_targets is not None
    ):
        raise ValueError(
            "without_replacement state sampling is only valid for pointwise state-cache training"
        )
    if args.focus_prompt_index >= 0:
        focused = np.flatnonzero(prompt_indices == args.focus_prompt_index)
        if not len(focused):
            raise ValueError(
                f"focus-prompt-index={args.focus_prompt_index} has no states in cache"
            )
        fixed_indices = [int(index) for index in focused]
    else:
        fixed_indices = (
            balanced_fixed_state_indices(
                prompt_indices, sigmas, sources, args.fixed_state_count
            )
            if args.fixed_state_count
            else list(range(len(states)))
        )
    prompts = cache_manifest["cache"]["prompt_index"]

    teacher_targets = None
    if args.teacher_target_cache is not None:
        target_contract = build_teacher_target_contract(
            tq.MLX_RUNTIME_ROOT,
            Path(__file__).resolve().parent,
            int(cache_manifest["cache"]["crop_len"]),
            float(cache_manifest["cache"]["seconds"]),
            int(cache_manifest["cache"]["trajectory_steps"]),
        )
        teacher_targets, teacher_target_metadata = load_teacher_targets(
            args.teacher_target_cache,
            args.state_cache,
            args.teacher_weights,
            len(states),
            target_contract,
        )
        teacher = None
    elif trajectory_pairs is not None and args.trajectory_pointwise_source == "pair":
        teacher_target_metadata = None
        teacher = None
    else:
        teacher_target_metadata = None
        teacher = tq.dit_mlx_medium.DiT(T_lat=128)
        teacher.load_weights(str(args.teacher_weights), strict=False)
        teacher.freeze()
    student = tq.dit_mlx_medium.DiT(T_lat=128)
    student.load_weights(str(args.teacher_weights), strict=False)
    dense_master_weights: dict[str, np.ndarray] = {}
    dense_master_biases: dict[str, np.ndarray] = {}
    if args.master_init == "dense_teacher":
        for block_index in range(args.start_block, args.end_block + 1):
            for name in tq.CORE_NAMES:
                prefix = f"transformer.layers.{block_index}.{name}"
                dense_module = tq.module_at(
                    student.transformer.layers[block_index], name
                )
                if not isinstance(dense_module, nn.Linear):
                    raise TypeError(
                        f"dense master init expected Linear at {prefix}, "
                        f"got {type(dense_module)}"
                    )
                mx.eval(dense_module.weight)
                dense_weight = np.asarray(
                    dense_module.weight, dtype=np.float32
                )
                if args.quantizer_mode in {"ttq_hadamard", SYMMETRIC_HADAMARD_MODE}:
                    dense_weight = tq.rotate_weight_hadamard(
                        dense_weight, args.group_size
                    )
                dense_master_weights[prefix] = dense_weight.copy()
                dense_bias = getattr(dense_module, "bias", None)
                if dense_bias is not None:
                    mx.eval(dense_bias)
                    dense_master_biases[prefix] = np.asarray(
                        dense_bias, dtype=np.float32
                    ).copy()
    source_records: dict[str, TernaryWeights] = {}
    reopened_source_modules = 0
    if args.source_records is not None:
        source_records, source_meta = tq.load_records_checkpoint(args.source_records)
        source_group_sizes = sorted({int(record.group_size) for record in source_records.values()})
        if not source_group_sizes:
            raise ValueError("source records are empty")
        # A cumulative Bonsai cascade may use a finer group size for one
        # difficult block while later blocks return to the baseline size.
        # Per-record metadata, validated by load_records_checkpoint, is the
        # source of truth; top-level group_size remains legacy compatibility.
        print(
            f"[WindowV6] source record group sizes={source_group_sizes} "
            f"active group size={args.group_size}",
            flush=True,
        )
        for prefix, record in source_records.items():
            parts = prefix.split(".")
            source_module = tq.module_at(
                student.transformer.layers[int(parts[2])], ".".join(parts[3:])
            )
            if getattr(source_module, "bias", None) is not None and record.linear_bias is None:
                raise ValueError(
                    f"source records omit trained linear bias at {prefix}; "
                    "regenerate them from a V7 hard checkpoint"
                )
        student = tq.apply_records_to_model(student, source_records, args.group_size)
        for block_index in range(args.start_block, args.end_block + 1):
            for name in tq.CORE_NAMES:
                prefix = f"transformer.layers.{block_index}.{name}"
                record = source_records.get(prefix)
                if record is None:
                    continue
                current = tq.module_at(student.transformer.layers[block_index], name)
                qat = quantized_linear_to_qat(
                    current,
                    record,
                    args.group_size,
                    args.quantizer_mode,
                    dense_master_weights.get(prefix),
                    dense_master_biases.get(prefix),
                )
                tq.set_module_at(student.transformer.layers[block_index], name, qat)
                reopened_source_modules += 1
    tq.replace_core_block(
        student.transformer.layers[args.start_block],
        args.group_size,
        args.quantizer_mode,
    )
    for block_index in range(args.start_block + 1, args.end_block + 1):
        tq.replace_core_block(
            student.transformer.layers[block_index],
            args.group_size,
            args.quantizer_mode,
        )
    student.freeze()
    for block_index in range(args.start_block, args.end_block + 1):
        for module in tq.core_modules(student.transformer.layers[block_index]).values():
            module.unfreeze()
    set_quantizer_surrogate(
        student,
        args.start_block,
        args.end_block,
        args.quantizer_surrogate,
    )
    mx.eval(student.parameters())
    if teacher is not None:
        mx.eval(teacher.parameters())

    initial_code_snapshot = None
    if args.quantizer_mode == "symmetric":
        initial_code_snapshot, initial_code_metrics = symmetric_code_snapshot(
            student, args.start_block, args.end_block, args.group_size
        )
    else:
        initial_code_metrics = None

    step_resume_state = None
    warm_start_state = None
    warm_start_parent = None
    warm_start_report = None
    if args.resume_step_checkpoint is not None:
        step_resume_state = tc.load_step_checkpoint(args.resume_step_checkpoint)
        step_metadata = step_resume_state["metadata"]
        warm_start_parent = step_metadata.get("run_signature", {}).get(
            "warm_start_parent"
        )
        student.update(step_resume_state["model_state"])
    elif args.warm_start_step_checkpoint is not None:
        warm_start_state = tc.load_step_checkpoint(args.warm_start_step_checkpoint)
        warm_start_report = validate_warm_start_payload(
            args.warm_start_step_checkpoint,
            warm_start_state,
            student.trainable_parameters(),
            args.source_records,
            args.teacher_weights,
            (args.start_block, args.end_block),
            args.group_size,
            args.quantizer_mode,
            args.weight_decay,
            args.optimizer_eps,
            args.gradient_accumulation,
            args.seed,
        )
        checkpoint_rate = float(warm_start_report["effective_learning_rate"])
        if not (
            np.isclose(args.learning_rate, checkpoint_rate, rtol=1e-7, atol=1e-12)
            and np.isclose(
                args.learning_rate_end, checkpoint_rate, rtol=1e-7, atol=1e-12
            )
        ):
            raise ValueError(
                "warm-start requires fixed --learning-rate and --learning-rate-end "
                f"equal to checkpoint rate {checkpoint_rate:.10g}"
            )
        student.update(warm_start_state["model_state"])
        warm_start_parent = {
            "checkpoint": warm_start_report["checkpoint"],
            "payload_sha256": warm_start_report["payload_sha256"],
            "parent_run_dir": warm_start_report["parent_run_dir"],
            "step_next": warm_start_report["step_next"],
        }
        warm_start_report["records_parity"] = compare_restored_records(
            student,
            source_records,
            args.start_block,
            args.end_block,
            args.group_size,
        )
        warm_start_report["forward_parity"] = compare_hard_forward_fixture(
            student,
            Path(str(warm_start_report["parent_run_dir"]))
            / "hard_forward_fixture.npz",
        )

    run_signature = checkpoint_run_signature(
        args,
        args.state_cache,
        teacher_target_metadata,
        trajectory_pair_metadata,
        full_rollout_metadata,
        student,
        warm_start_parent,
    )
    if step_resume_state is not None:
        step_metadata = step_resume_state["metadata"]
        if step_metadata.get("run_signature") != run_signature:
            raise ValueError("step checkpoint run signature does not match current inputs/config")
        step_start = int(step_metadata.get("step_next", -1))
        if step_start < 1 or step_start > run_until_step:
            raise ValueError(
                f"step checkpoint next update {step_start} is outside this invocation's remaining range"
            )

    optimizer = None
    weight_optimizer = None
    scale_optimizer = None
    if args.quantizer_mode in LEARNED_MODES:
        initial_weights, initial_scales = split_scale_parameter_tree(
            student.trainable_parameters(),
            parameter_policy=args.ttq_train_parameters,
        )
        if not _tree_has_leaves(initial_weights) or not _tree_has_leaves(initial_scales):
            raise RuntimeError(
                f"{args.quantizer_mode} requires trainable masters and quantizer parameters"
            )
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
    else:
        optimizer = optim.AdamW(
            learning_rate=optim.cosine_decay(
                args.learning_rate, args.steps, end=args.learning_rate_end
            ),
            eps=args.optimizer_eps,
            weight_decay=args.weight_decay,
        )
        optimizer.init(student.trainable_parameters())
    checkpoint_state = step_resume_state or warm_start_state
    if checkpoint_state is not None:
        saved_optimizers = checkpoint_state["optimizer_state"]
        if args.quantizer_mode in LEARNED_MODES:
            weight_optimizer.state = saved_optimizers["weight"]
            scale_optimizer.state = saved_optimizers["scale"]
        else:
            optimizer.state = saved_optimizers["main"]
    value_grad = nn.value_and_grad(student, full_velocity_loss)
    trajectory_value_grad = (
        nn.value_and_grad(student, trajectory_pair_loss)
        if args.trajectory_loss_weight > 0
        else None
    )
    full_rollout_value_grad = (
        nn.value_and_grad(student, full_rollout_loss)
        if args.full_rollout_loss_weight > 0
        else None
    )
    rng = random.Random(args.seed)
    step_start = 0
    if step_resume_state is not None:
        rng = tc.restore_rngs(step_resume_state, rng)
        step_start = int(step_resume_state["metadata"]["step_next"])
    elif warm_start_state is not None:
        rng = tc.restore_rngs(warm_start_state, rng)
    losses: list[float] = (
        list(step_resume_state["metadata"].get("loss_history", []))
        if step_resume_state is not None
        else []
    )
    trajectory_objective_history: list[float] = (
        list(step_resume_state["metadata"].get("trajectory_objective_history", []))
        if step_resume_state is not None
        else []
    )
    rollout_objective_history: list[float] = (
        list(step_resume_state["metadata"].get("rollout_objective_history", []))
        if step_resume_state is not None
        else []
    )
    diagnostic_history: list[dict[str, object]] = (
        list(step_resume_state["metadata"].get("diagnostic_history", []))
        if step_resume_state is not None
        else []
    )
    effective_learning_rate_history: list[dict[str, float]] = (
        list(
            step_resume_state["metadata"].get(
                "effective_learning_rate_history", []
            )
        )
        if step_resume_state is not None
        else []
    )
    seen_state_indices: set[int] = (
        set(step_resume_state["metadata"].get("seen_state_indices", []))
        if step_resume_state is not None
        else set()
    )
    state_sampling_orders: dict[int, list[int]] = {}
    gradient_clip_count = (
        int(step_resume_state["metadata"].get("gradient_clip_count", 0))
        if step_resume_state is not None
        else 0
    )
    seen_pair_indices: set[int] = (
        set(step_resume_state["metadata"].get("seen_pair_indices", []))
        if step_resume_state is not None
        else set()
    )
    seen_trajectory_pair_indices: set[int] = (
        set(step_resume_state["metadata"].get("seen_trajectory_pair_indices", []))
        if step_resume_state is not None
        else set()
    )
    seen_rollout_indices: set[int] = (
        set(step_resume_state["metadata"].get("seen_rollout_indices", []))
        if step_resume_state is not None
        else set()
    )
    pair_step_counts: dict[str, int] = (
        {
            str(key): int(value)
            for key, value in step_resume_state["metadata"].get(
                "pair_step_counts", {}
            ).items()
        }
        if step_resume_state is not None
        else {str(index): 0 for index in range(7)}
    )
    previous_code_snapshot = None
    if initial_code_snapshot is not None:
        previous_code_snapshot, _ = symmetric_code_snapshot(
            student, args.start_block, args.end_block, args.group_size
        )
    previous_master_weights = master_weight_snapshot(
        student, args.start_block, args.end_block
    )
    startup_memory = tq.memory_snapshot()
    startup_peak = int(startup_memory["metal_peak_gb"] * (1024**3))
    if startup_peak > args.max_metal_bytes:
        raise RuntimeError(
            f"startup exceeded Metal memory guard: {startup_peak} > "
            f"{args.max_metal_bytes} bytes"
        )
    if warm_start_state is not None:
        assert pair_order is not None
        covered_trajectory_pairs = paired_epoch_trajectory_coverage(
            pair_order, run_until_step
        )
        warm_start_report["planned_pair_coverage"] = {
            "pair_count": len(pair_order),
            "trajectory_unique_at_end": len(covered_trajectory_pairs),
            "updates": run_until_step,
            "complete": len(covered_trajectory_pairs) == len(pair_order),
        }
        if not warm_start_report["planned_pair_coverage"]["complete"]:
            raise ValueError(
                "P3.1 run budget does not cover every trajectory pair exactly once"
            )
    if args.preflight_only:
        preflight_report = {
            "schema": "onus.ternary-quality/v7-warm-start-preflight",
            "status": "passed_no_updates",
            "run_signature": run_signature,
            "warm_start": warm_start_report,
            "memory": startup_memory,
        }
        report_path = args.output_dir / "warm_start_preflight.json"
        write_json(report_path, preflight_report)
        print(json.dumps(preflight_report, indent=2), flush=True)
        return
    print(
        f"[WindowV6] blocks={args.start_block}:{args.end_block} "
        f"mode={args.quantizer_mode} group={args.group_size} "
        f"steps={args.steps} accumulation={args.gradient_accumulation} "
        f"reopened_source_modules={reopened_source_modules} "
        f"scale_lr={args.scale_learning_rate if weight_optimizer else 'n/a'} "
        f"states={len(fixed_indices)}/{len(states)} "
        f"pairs={len(trajectory_pairs['pair_steps']) if trajectory_pairs is not None else 0} "
        f"trajectory_pointwise={args.trajectory_pointwise_source} "
        f"trajectory_weight={args.trajectory_loss_weight:g} "
        f"memory={tq.memory_snapshot()}",
        flush=True,
    )
    for step in range(step_start, run_until_step):
        update_started = time.perf_counter()
        soft_forward, quantizer_sharpness = quantizer_schedule(
            step,
            args.soft_end_updates,
            args.soft_sharpness_start,
            args.soft_sharpness_end,
        )
        set_quantizer_schedule(
            student,
            args.start_block,
            args.end_block,
            soft_forward,
            quantizer_sharpness,
        )
        accumulated = None
        micro_losses: list[float] = []
        update_trajectory_objectives: list[float] = []
        update_rollout_objectives: list[float] = []
        update_pair_steps: list[int] = []
        for micro in range(args.gradient_accumulation):
            use_pair_cache = trajectory_pairs is not None
            use_trajectory = bool(
                use_pair_cache
                and trajectory_value_grad is not None
                and micro % 2 == 1
            )
            use_full_rollout = bool(
                full_rollout_targets is not None
                and full_rollout_value_grad is not None
                and micro % 2 == 1
            )
            use_pair_sample = bool(
                use_trajectory
                or (
                    use_pair_cache
                    and args.trajectory_pointwise_source == "pair"
                )
            )
            if use_full_rollout:
                rollout_count = len(full_rollout_targets["prompt_indices"])
                rollout_index = (
                    step * max(1, args.gradient_accumulation // 2)
                    + micro // 2
                ) % rollout_count
                prompt_index = int(
                    full_rollout_targets["prompt_indices"][rollout_index]
                )
                window_steps = args.full_rollout_window_steps
                if args.full_rollout_on_policy_stitch:
                    # A direct 8-step graph exceeds the 12 GB budget on the
                    # target machine.  Two independently differentiated
                    # four-step segments preserve the terminal target while
                    # feeding the detached student midpoint into segment two.
                    window_start = 0
                    segment_steps = 4
                    initial = mx.array(
                        full_rollout_targets["source_states"][
                            rollout_index, 0
                        ][None],
                        dtype=mx.float16,
                    )

                    def segment_noises(start: int) -> mx.array:
                        noise_window = np.zeros(
                            (
                                segment_steps,
                                256,
                                int(cache_manifest["cache"]["crop_len"]),
                            ),
                            dtype=np.float16,
                        )
                        available_noises = min(segment_steps, 8 - start - 1)
                        if available_noises > 0:
                            noise_window[:available_noises] = full_rollout_targets[
                                "noises"
                            ][
                                rollout_index,
                                start : start + available_noises,
                            ]
                        return mx.array(noise_window[None], dtype=mx.float16)

                    first_noises = segment_noises(0)
                    second_noises = segment_noises(4)
                    first_sigma_grid = tuple(
                        float(value)
                        for value in full_rollout_targets["sigma_grid"][0:5]
                    )
                    second_sigma_grid = tuple(
                        float(value)
                        for value in full_rollout_targets["sigma_grid"][4:9]
                    )
                    target_midpoint = mx.array(
                        full_rollout_targets["target_states"][
                            rollout_index, 4
                        ][None],
                        dtype=mx.float16,
                    )
                    target_terminal = mx.array(
                        full_rollout_targets["target_states"][
                            rollout_index, 8
                        ][None],
                        dtype=mx.float16,
                    )
                    midpoint = rollout_state_without_grad(
                        student,
                        initial,
                        first_noises,
                        cross_cache[prompt_index],
                        global_cond,
                        first_sigma_grid,
                        0,
                        8,
                    )
                    first_loss, first_grads = full_rollout_value_grad(
                        student,
                        initial,
                        first_noises,
                        cross_cache[prompt_index],
                        global_cond,
                        first_sigma_grid,
                        target_midpoint,
                        args.full_rollout_loss_weight,
                        0,
                        8,
                    )
                    mx.eval(first_loss, first_grads)
                    second_loss, second_grads = full_rollout_value_grad(
                        student,
                        mx.stop_gradient(midpoint),
                        second_noises,
                        cross_cache[prompt_index],
                        global_cond,
                        second_sigma_grid,
                        target_terminal,
                        args.full_rollout_loss_weight,
                        4,
                        8,
                    )
                    loss = (first_loss + second_loss) * 0.5
                    grads = tq.tree_scale(
                        tq.tree_add(first_grads, second_grads), 0.5
                    )
                    mx.eval(loss, grads)
                else:
                    window_starts = list(range(0, 9 - window_steps, window_steps))
                    window_start = window_starts[rollout_index % len(window_starts)]
                    initial = mx.array(
                        full_rollout_targets["source_states"][
                            rollout_index, window_start
                        ][None],
                        dtype=mx.float16,
                    )
                    noise_window = np.zeros(
                        (
                            window_steps,
                            256,
                            int(cache_manifest["cache"]["crop_len"]),
                        ),
                        dtype=np.float16,
                    )
                    available_noises = min(window_steps, 8 - window_start - 1)
                    if available_noises > 0:
                        noise_window[:available_noises] = full_rollout_targets[
                            "noises"
                        ][
                            rollout_index,
                            window_start : window_start + available_noises,
                        ]
                    noises = mx.array(
                        noise_window[None],
                        dtype=mx.float16,
                    )
                    target_terminal = mx.array(
                        full_rollout_targets["target_states"][
                            rollout_index, window_start + window_steps
                        ][None],
                        dtype=mx.float16,
                    )
                    sigma_grid = tuple(
                        float(value)
                        for value in full_rollout_targets["sigma_grid"][
                            window_start : window_start + window_steps + 1
                        ]
                    )
                    loss, grads = full_rollout_value_grad(
                        student,
                        initial,
                        noises,
                        cross_cache[prompt_index],
                        global_cond,
                        sigma_grid,
                        target_terminal,
                        args.full_rollout_loss_weight,
                        window_start,
                        8,
                    )
                seen_rollout_indices.add(rollout_index)
                update_rollout_objectives.append(float(loss))
            elif use_pair_sample:
                if args.pair_sampling_mode == "paired_epoch":
                    if pair_order is None:
                        raise RuntimeError("paired-epoch order was not initialized")
                    pair_index = paired_epoch_pair_index(pair_order, step, micro)
                else:
                    pair_step = (step * args.gradient_accumulation + micro) % 7
                    step_candidates = pair_indices_by_step[pair_step]
                    pair_index = int(step_candidates[rng.randrange(len(step_candidates))])
                prompt_index = int(trajectory_pairs["prompt_indices"][pair_index])
                cross = cross_cache[prompt_index]
                anchor = mx.array(
                    trajectory_pairs["anchors"][pair_index][None], dtype=mx.float16
                )
                target = mx.array(
                    trajectory_pairs["target_velocity"][pair_index][None],
                    dtype=mx.float16,
                )
                t = timestep_tensor(
                    float(trajectory_pairs["sigmas"][pair_index, 0]), anchor.shape[0]
                )
                seen_pair_indices.add(pair_index)
                pair_start = int(trajectory_pairs["pair_steps"][pair_index])
                update_pair_steps.append(pair_start)
                pair_step_counts[str(pair_start)] = (
                    pair_step_counts.get(str(pair_start), 0) + 1
                )
                if use_trajectory:
                    seen_trajectory_pair_indices.add(pair_index)
                    noise_first = mx.array(
                        trajectory_pairs["noise_first"][pair_index][None],
                        dtype=mx.float16,
                    )
                    noise_second = mx.array(
                        trajectory_pairs["noise_second"][pair_index][None],
                        dtype=mx.float16,
                    )
                    target_endpoint = mx.array(
                        trajectory_pairs["target_endpoint"][pair_index][None],
                        dtype=mx.float16,
                    )
                    sigma_triplet = tuple(
                        float(value)
                        for value in trajectory_pairs["sigmas"][pair_index]
                    )
                    loss, grads = trajectory_value_grad(
                        student,
                        anchor,
                        cross,
                        global_cond,
                        sigma_triplet,
                        noise_first,
                        noise_second,
                        target,
                        target_endpoint,
                        pair_start,
                        8,
                        args.trajectory_loss_weight,
                    )
                    update_trajectory_objectives.append(float(loss))
                else:
                    loss, grads = value_grad(
                        student, anchor, t, cross, global_cond, target
                    )
            else:
                if args.state_sampling == "without_replacement":
                    global_micro = step * args.gradient_accumulation + micro
                    cycle, offset = divmod(global_micro, len(fixed_indices))
                    if cycle not in state_sampling_orders:
                        state_sampling_orders[cycle] = random.Random(
                            args.seed + cycle
                        ).sample(fixed_indices, len(fixed_indices))
                    index = state_sampling_orders[cycle][offset]
                else:
                    index = fixed_indices[rng.randrange(len(fixed_indices))]
                seen_state_indices.add(index)
                x = mx.array(states[index][None], dtype=mx.float16)
                t = timestep_tensor(float(sigmas[index]))
                cross = cross_cache[int(prompt_indices[index])]
                if teacher_targets is None:
                    target = teacher(x, t, cross, global_cond)
                    mx.eval(target)
                else:
                    target = mx.array(teacher_targets[index][None], dtype=mx.float16)
                loss, grads = value_grad(student, x, t, cross, global_cond, target)
            mx.eval(loss, grads)
            value = float(loss)
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite loss at update={step + 1} micro={micro + 1}")
            if not tree_is_finite(grads):
                raise RuntimeError(
                    f"non-finite gradient at update={step + 1} micro={micro + 1}"
                )
            accumulated = grads if accumulated is None else tq.tree_add(accumulated, grads)
            micro_losses.append(value)
        grads = tq.tree_scale(accumulated, 1.0 / args.gradient_accumulation)
        grads, raw_grad_norm = optim.clip_grad_norm(grads, args.gradient_clip)
        grad_norm_value = float(raw_grad_norm)
        if not np.isfinite(grad_norm_value):
            raise RuntimeError(f"non-finite global gradient norm at update={step + 1}")
        if grad_norm_value > args.gradient_clip:
            gradient_clip_count += 1
        if args.quantizer_mode in LEARNED_MODES:
            weight_grads, scale_grads = split_scale_parameter_tree(
                grads, parameter_policy=args.ttq_train_parameters
            )
            current_weights, current_scales = split_scale_parameter_tree(
                student.trainable_parameters(),
                parameter_policy=args.ttq_train_parameters,
            )
            if step == 0:
                threshold_before = learned_threshold_snapshot(
                    student, args.start_block, args.end_block
                )
                threshold_grad_sq = 0.0
                threshold_grad_leaves = 0
                for path, value in tq.tree_flatten(scale_grads):
                    if "log_threshold_multiplier" in str(path):
                        grad_value = np.asarray(value, dtype=np.float32)
                        threshold_grad_sq += float(np.sum(grad_value * grad_value))
                        threshold_grad_leaves += 1
            weight_updates = weight_optimizer.apply_gradients(weight_grads, current_weights)
            student.update(weight_updates)
            _, current_scales = split_scale_parameter_tree(
                student.trainable_parameters(),
                parameter_policy=args.ttq_train_parameters,
            )
            scale_updates = scale_optimizer.apply_gradients(scale_grads, current_scales)
            student.update(scale_updates)
            mx.eval(
                student.parameters(),
                weight_optimizer.state,
                scale_optimizer.state,
            )
            if step == 0:
                threshold_after = learned_threshold_snapshot(
                    student, args.start_block, args.end_block
                )
                print(
                    "[WindowV6] threshold_update="
                    f"leaves:{threshold_grad_leaves} "
                    f"grad_norm:{np.sqrt(threshold_grad_sq):.8g} "
                    f"param_delta_abs_mean:{np.mean(np.abs(threshold_after - threshold_before)):.8g} "
                    f"param_abs_mean:{np.mean(np.abs(threshold_after)):.8g}",
                    flush=True,
                )
        else:
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state)
        effective_learning_rates = (
            {
                "weight": optimizer_learning_rate(weight_optimizer),
                "scale": optimizer_learning_rate(scale_optimizer),
            }
            if weight_optimizer is not None
            else {"main": optimizer_learning_rate(optimizer)}
        )
        if warm_start_report is not None:
            expected_rate = float(warm_start_report["effective_learning_rate"])
            if not np.isclose(
                effective_learning_rates["main"], expected_rate,
                rtol=1e-7, atol=1e-12,
            ):
                raise RuntimeError(
                    "warm-start effective learning rate changed: "
                    f"{effective_learning_rates['main']:.10g} != {expected_rate:.10g}"
                )
        effective_learning_rate_history.append(effective_learning_rates)
        if not tree_is_finite(student.trainable_parameters()):
            raise RuntimeError(f"non-finite trainable parameter at update={step + 1}")
        update_memory = tq.memory_snapshot()
        update_peak = int(update_memory["metal_peak_gb"] * (1024**3))
        if update_peak > args.max_metal_bytes:
            raise RuntimeError(
                f"update={step + 1} exceeded Metal memory guard: {update_peak} > "
                f"{args.max_metal_bytes} bytes"
            )
        latest = float(np.mean(micro_losses))
        losses.append(latest)
        if update_trajectory_objectives:
            trajectory_objective_history.append(
                float(np.mean(update_trajectory_objectives))
            )
        if update_rollout_objectives:
            rollout_objective_history.append(float(np.mean(update_rollout_objectives)))
        update_seconds = time.perf_counter() - update_started
        if args.checkpoint_every_steps and (
            (step + 1) % args.checkpoint_every_steps == 0
            or step + 1 == run_until_step
        ):
            checkpoint_path = args.output_dir / "checkpoints" / "window_latest.npz"
            optimizer_state = (
                {"weight": weight_optimizer.state, "scale": scale_optimizer.state}
                if args.quantizer_mode in LEARNED_MODES
                else {"main": optimizer.state}
            )
            code_metrics = None
            if initial_code_snapshot is not None:
                current_code_snapshot, code_metrics = symmetric_code_snapshot(
                    student, args.start_block, args.end_block, args.group_size
                )
                code_metrics = attach_code_transitions(
                    code_metrics,
                    current_code_snapshot,
                    previous_code_snapshot,
                    initial_code_snapshot,
                )
                previous_code_snapshot = current_code_snapshot
            current_master_weights = master_weight_snapshot(
                student, args.start_block, args.end_block
            )
            update_metrics = master_update_diagnostics(
                current_master_weights, previous_master_weights
            )
            previous_master_weights = current_master_weights
            learning_rate = cosine_schedule_value(
                args.learning_rate,
                args.learning_rate_end,
                args.steps,
                step,
            )
            scale_learning_rate = (
                cosine_schedule_value(
                    args.scale_learning_rate,
                    args.scale_learning_rate_end,
                    args.steps,
                    step,
                )
                if weight_optimizer is not None
                else None
            )
            checkpoint_metrics = {
                "update": step + 1,
                "loss": latest,
                "learning_rate": learning_rate,
                "effective_learning_rates": effective_learning_rates,
                "scale_learning_rate": scale_learning_rate,
                "gradient_norm_before_clip": grad_norm_value,
                "gradient_clip_applied": grad_norm_value > args.gradient_clip,
                "gradient_clip_fraction": gradient_clip_count / max(step + 1, 1),
                "master_update": update_metrics,
                "code_diagnostics": code_metrics,
                "unique_states_seen": len(seen_state_indices),
                "state_count": len(fixed_indices),
                "trajectory_weight": args.trajectory_loss_weight,
                "trajectory_objective": (
                    float(np.mean(update_trajectory_objectives))
                    if update_trajectory_objectives
                    else None
                ),
                "full_rollout_objective": (
                    float(np.mean(update_rollout_objectives))
                    if update_rollout_objectives
                    else None
                ),
                "trajectory_pair_steps": update_pair_steps,
                "trajectory_pair_count": len(seen_pair_indices),
                "update_seconds": update_seconds,
                "memory": tq.memory_snapshot(),
            }
            diagnostic_history.append(checkpoint_metrics)
            tc.save_step_checkpoint(
                checkpoint_path,
                student.trainable_parameters(),
                optimizer_state,
                {
                    "run_signature": run_signature,
                    "window": [args.start_block, args.end_block],
                    "step_next": step + 1,
                    "last_loss": latest,
                    "group_size": args.group_size,
                    "crop_len": 128,
                    "quantizer_mode": args.quantizer_mode,
                    "soft_end_updates": args.soft_end_updates,
                    "soft_sharpness_start": args.soft_sharpness_start,
                    "soft_sharpness_end": args.soft_sharpness_end,
                    "seed": args.seed,
                    "gradient_accumulation": args.gradient_accumulation,
                    "fixed_state_count": args.fixed_state_count,
                    "seen_state_indices": sorted(seen_state_indices),
                    "loss_history": losses,
                    "effective_learning_rate_history": effective_learning_rate_history,
                    "trajectory_objective_history": trajectory_objective_history,
                    "rollout_objective_history": rollout_objective_history,
                    "diagnostic_history": diagnostic_history,
                    "gradient_clip_count": gradient_clip_count,
                    "seen_pair_indices": sorted(seen_pair_indices),
                    "seen_trajectory_pair_indices": sorted(
                        seen_trajectory_pair_indices
                    ),
                    "pair_step_counts": pair_step_counts,
                    "seen_rollout_indices": sorted(seen_rollout_indices),
                },
                rng.getstate(),
                np.random.get_state(),
                list(getattr(mx.random, "state", [])),
            )
            print(
                f"[StepCheckpoint] update={step + 1} loss={latest:.6f} "
                f"trajectory={float(np.mean(update_trajectory_objectives)):.6f} "
                if update_trajectory_objectives
                else f"[StepCheckpoint] update={step + 1} loss={latest:.6f} trajectory=n/a ",
                flush=True,
            )
            if update_rollout_objectives:
                print(
                    f"[StepCheckpoint] full_rollout="
                    f"{float(np.mean(update_rollout_objectives)):.6f}",
                    flush=True,
                )
            print(
                f"grad_norm={grad_norm_value:.6g} "
                f"update/weight={update_metrics['update_to_weight_ratio']:.6g} "
                f"code_flips={code_metrics['flips_since_previous_audit'] if code_metrics else 'n/a'} "
                f"unique_states={len(seen_state_indices)} "
                f"unique_pairs={len(seen_pair_indices)} "
                f"path={checkpoint_path}",
                flush=True,
            )
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            trajectory_latest = (
                float(np.mean(update_trajectory_objectives))
                if update_trajectory_objectives
                else None
            )
            rollout_latest = (
                float(np.mean(update_rollout_objectives))
                if update_rollout_objectives
                else None
            )
            print(
                f"[WindowV6] update={step + 1}/{args.steps} loss={latest:.6f} "
                f"trajectory={trajectory_latest if trajectory_latest is not None else 'n/a'} "
                f"full_rollout={rollout_latest if rollout_latest is not None else 'n/a'} "
                f"quantizer={'soft' if soft_forward else 'hard'} "
                f"sharpness={quantizer_sharpness:.4g} "
                f"memory={tq.memory_snapshot()}",
                flush=True,
            )

    if run_until_step < args.steps:
        checkpoint_path = args.output_dir / "checkpoints" / "window_latest.npz"
        if not checkpoint_path.is_file():
            raise RuntimeError("partial run ended without a resumable step checkpoint")
        print(
            json.dumps(
                {
                    "status": "partial_checkpoint_saved",
                    "completed_updates": run_until_step,
                    "total_updates": args.steps,
                    "checkpoint": str(checkpoint_path),
                    "loss_last": losses[-1] if losses else None,
                    "trajectory_objective_last": (
                        trajectory_objective_history[-1]
                        if trajectory_objective_history
                        else None
                    ),
                    "memory": tq.memory_snapshot(),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    quantizer_diagnostics = (
        learned_quantizer_diagnostics(
            student, args.start_block, args.end_block, args.group_size
        )
        if args.quantizer_mode in LEARNED_MODES
        else None
    )
    final_code_snapshot = None
    final_code_metrics = None
    if initial_code_snapshot is not None:
        final_code_snapshot, final_code_metrics = symmetric_code_snapshot(
            student, args.start_block, args.end_block, args.group_size
        )
        final_code_metrics = attach_code_transitions(
            final_code_metrics,
            final_code_snapshot,
            previous_code_snapshot,
            initial_code_snapshot,
        )
    final_master_weights = master_weight_snapshot(
        student, args.start_block, args.end_block
    )
    final_master_update = master_update_diagnostics(
        final_master_weights, previous_master_weights
    )
    records = records_from_window(
        student,
        args.start_block,
        args.end_block,
        args.group_size,
        args.quantizer_mode,
    )
    records = {**source_records, **records}
    records_checkpoint = args.output_dir / "records_checkpoint.npz"
    tq.save_records_checkpoint(
        records_checkpoint,
        records,
        args.end_block + 1,
        args.group_size,
        128,
        args.quantizer_mode,
    )
    fixture_path = args.output_dir / "hard_forward_fixture.npz"
    save_hard_forward_fixture(
        fixture_path,
        student,
        states,
        sigmas,
        prompt_indices,
        cross_cache,
        global_cond,
        fixed_indices,
    )
    artifact = args.output_dir / (
        f"dit_medium_v7_window{args.start_block}-{args.end_block}_"
        f"{args.quantizer_mode}_g{args.group_size}.npz"
    )
    config = {
        "schema": "onus.ternary-quality/v7-window-run",
        "dataset_dir": str(args.dataset_dir),
        "state_cache": str(args.state_cache),
        "teacher_weights": str(args.teacher_weights),
        "teacher_target_cache": (
            str(args.teacher_target_cache) if args.teacher_target_cache else None
        ),
        "trajectory_pair_cache": (
            str(args.trajectory_pair_cache) if args.trajectory_pair_cache else None
        ),
        "full_rollout_target_cache": (
            str(args.full_rollout_target_cache)
            if args.full_rollout_target_cache
            else None
        ),
        "full_rollout_loss_weight": args.full_rollout_loss_weight,
        "full_rollout_window_steps": args.full_rollout_window_steps,
        "full_rollout_on_policy_stitch": args.full_rollout_on_policy_stitch,
        "trajectory_pointwise_source": args.trajectory_pointwise_source,
        "trajectory_loss_weight": args.trajectory_loss_weight,
        "pair_sampling_mode": args.pair_sampling_mode,
        "trajectory_microbatch_fraction": (
            0.5 if trajectory_pairs is not None and args.trajectory_loss_weight > 0 else 0.0
        ),
        "start_block": args.start_block,
        "end_block": args.end_block,
        "group_size": args.group_size,
        "training_quantizer_mode": args.quantizer_mode,
        "quantizer_surrogate": args.quantizer_surrogate,
        "master_init": args.master_init,
        "reopened_source_modules": reopened_source_modules,
        "steps": args.steps,
        "fixed_state_count": args.fixed_state_count,
        "state_sampling": args.state_sampling,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "learning_rate_end": args.learning_rate_end,
        "warm_start_parent": warm_start_parent,
        "scale_learning_rate": args.scale_learning_rate,
        "scale_learning_rate_end": args.scale_learning_rate_end,
        "soft_end_updates": args.soft_end_updates,
        "soft_sharpness_start": args.soft_sharpness_start,
        "soft_sharpness_end": args.soft_sharpness_end,
        "threshold_log_bounds": list(tq.LEARNED_THRESHOLD_LOG_BOUNDS),
        "weight_decay": args.weight_decay,
        "optimizer_eps": args.optimizer_eps,
        "gradient_clip": args.gradient_clip,
        "max_metal_bytes": args.max_metal_bytes,
        "seed": args.seed,
    }
    parity = None
    if not args.records_only:
        tq.export_artifact(
            student,
            records,
            artifact,
            artifact.with_suffix(".json"),
            args.group_size,
            128,
            config,
            args.quantizer_mode,
            "symmetric_compact",
        )
    records_scope_digest = scope_digest(records)
    records_scope_count = len(records)
    roundtrip_report_path = args.output_dir / "cross_process_roundtrip.json"
    verifier_command = [
        sys.executable,
        str(Path(__file__).with_name("verify_ternary_records_roundtrip.py")),
        "--teacher-weights",
        str(args.teacher_weights),
        "--records",
        str(records_checkpoint),
        "--fixture",
        str(fixture_path),
        "--output",
        str(roundtrip_report_path),
        "--group-size",
        str(args.group_size),
        "--crop-len",
        "128",
        "--max-metal-bytes",
        str(args.max_metal_bytes),
    ]
    if not args.records_only:
        verifier_command.extend(
            [
                "--artifact",
                str(artifact),
                "--manifest",
                str(artifact.with_suffix(".json")),
            ]
        )
    del teacher, student, records, source_records, states, teacher_targets
    del full_rollout_targets
    del sigmas, prompt_indices, sources, cross_cache, global_cond, final_code_snapshot
    gc.collect()
    mx.clear_cache()
    verifier = subprocess.run(
        verifier_command,
        check=False,
        capture_output=True,
        text=True,
    )
    if verifier.stdout:
        print(verifier.stdout, flush=True)
    if verifier.returncode != 0:
        raise RuntimeError(
            "cross-process hard-forward roundtrip failed: " + verifier.stderr[-4000:]
        )
    cross_process_roundtrip = json.loads(
        roundtrip_report_path.read_text(encoding="utf-8")
    )
    summary = {
        "schema": "onus.ternary-quality/v7-window-summary",
        "status": "records_roundtrip_verified" if args.records_only else "exported_roundtrip_verified",
        "artifact": None if args.records_only else str(artifact),
        "manifest": None if args.records_only else str(artifact.with_suffix(".json")),
        "hard_forward_fixture": str(fixture_path),
        "cross_process_roundtrip": cross_process_roundtrip,
        "run_signature": run_signature,
        "warm_start": warm_start_report,
        "records_checkpoint": str(records_checkpoint),
        "records_only": args.records_only,
        "scope_digest": records_scope_digest,
        "scope_count": records_scope_count,
        "reopened_source_modules": reopened_source_modules,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_history": losses,
        "effective_learning_rate_history": effective_learning_rate_history,
        "trajectory_objective_history": trajectory_objective_history,
        "trajectory_pair_cache": (
            trajectory_pair_metadata["metadata"] if trajectory_pair_metadata else None
        ),
        "full_rollout_target_cache": (
            full_rollout_metadata["metadata"] if full_rollout_metadata else None
        ),
        "full_rollout_loss_weight": args.full_rollout_loss_weight,
        "full_rollout_window_steps": args.full_rollout_window_steps,
        "rollout_objective_history": rollout_objective_history,
        "full_rollout_indices_seen": len(seen_rollout_indices),
        "trajectory_pointwise_source": args.trajectory_pointwise_source,
        "trajectory_pair_count": (
            int(len(trajectory_pairs["pair_steps"])) if trajectory_pairs is not None else 0
        ),
        "trajectory_pair_indices_seen": len(seen_pair_indices),
        "trajectory_unroll_pair_indices_seen": len(seen_trajectory_pair_indices),
        "trajectory_pair_step_counts": pair_step_counts,
        "pair_sampling_mode": args.pair_sampling_mode,
        "soft_end_updates": args.soft_end_updates,
        "soft_sharpness_start": args.soft_sharpness_start,
        "soft_sharpness_end": args.soft_sharpness_end,
        "parameter_reload": parity,
        "learned_quantizer_diagnostics": quantizer_diagnostics,
        "initial_code_diagnostics": initial_code_metrics,
        "final_code_diagnostics": final_code_metrics,
        "final_master_update_since_last_audit": final_master_update,
        "diagnostic_history": diagnostic_history,
        "unique_states_seen": len(seen_state_indices),
        "fixed_state_indices": fixed_indices if args.fixed_state_count else None,
        "teacher_target_cache": teacher_target_metadata,
        "elapsed_seconds": time.time() - started,
        "memory": tq.memory_snapshot(),
        "cache": cache_manifest["cache"],
    }
    write_json(args.output_dir / "window_summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "artifact": None if args.records_only else str(artifact),
        "bytes": None if args.records_only else artifact.stat().st_size,
        "records_checkpoint": str(records_checkpoint),
        "roundtrip": cross_process_roundtrip["status"],
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "scope_count": records_scope_count,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
