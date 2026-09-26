"""Cache teacher inputs/outputs for block-output scoring in V8."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import train_ternary_quality as tq
from models.defs import dit_mlx_medium
from ternary_provenance_v8 import canonical_digest, sha256_file, validate_contract
from ternary_runtime_contract import timestep_tensor
from profile_ternary_activations_v8 import _select_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-states", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    contract_report = validate_contract(args.dataset_contract, project_root)
    state_cache = args.state_cache.resolve()
    states_path = state_cache / "states.npz"
    conditions_path = state_cache / "conditions.npz"
    manifest_path = state_cache / "manifest.json"
    states_file = np.load(states_path, allow_pickle=False)
    conditions_file = np.load(conditions_path, allow_pickle=False)
    states = states_file["states"]
    sigmas = states_file["sigmas"].astype(np.float32)
    prompt_indices = states_file["prompt_indices"].astype(np.int32)
    selected_indices = _select_indices(prompt_indices, sigmas, args.max_states, args.replicas)
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    teacher = dit_mlx_medium.DiT(T_lat=int(states.shape[-1]))
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    block_inputs: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    global_conditions: list[np.ndarray] = []
    local_padded_values: list[np.ndarray] = []
    block_targets: list[np.ndarray] = []
    peak = 0
    for position, state_index in enumerate(selected_indices, start=1):
        sigma = float(sigmas[state_index])
        prompt_index = int(prompt_indices[state_index])
        x = mx.array(states[state_index][None], dtype=mx.float16)
        t = timestep_tensor(sigma)
        cross_raw = mx.array(conditions_file[f"cross_{prompt_index:04d}"], dtype=mx.float16)
        global_raw = mx.array(conditions_file["global_cond"], dtype=mx.float16)

        context = teacher.to_cond_embed[2](tq.nn.silu(teacher.to_cond_embed[0](cross_raw)))
        global_pre = teacher.to_global_embed[2](tq.nn.silu(teacher.to_global_embed[0](global_raw)))
        timestep = teacher.timestep_features(t)
        timestep = teacher.to_timestep_embed[2](tq.nn.silu(teacher.to_timestep_embed[0](timestep)))
        global_embed = global_pre + timestep
        x_lc = x.transpose(0, 2, 1)
        x_pp = teacher.preprocess_conv(x_lc) + x_lc
        h = teacher.transformer.project_in(x_pp)
        memory = mx.broadcast_to(
            teacher.transformer.memory_tokens[None],
            (1, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM),
        )
        h = mx.concatenate([memory, h], axis=1)
        global_cond = teacher.transformer.global_cond_embedder[2](
            tq.nn.silu(teacher.transformer.global_cond_embedder[0](global_embed))
        )
        local = mx.zeros((1, x.shape[-1], dit_mlx_medium.LOCAL_ADD_COND_DIM))
        local_emb = teacher.transformer.layers[0].to_local_embed(local)
        pad = mx.zeros(
            (1, dit_mlx_medium.NUM_MEMORY_TOKENS, dit_mlx_medium.EMBED_DIM),
            dtype=local_emb.dtype,
        )
        local_padded = mx.concatenate([pad, local_emb], axis=1)
        target = teacher.transformer.layers[0](h, context, global_cond, local_padded)
        mx.eval(h, context, global_cond, local_padded, target)
        block_inputs.append(np.asarray(h, dtype=np.float16)[0])
        contexts.append(np.asarray(context, dtype=np.float16)[0])
        global_conditions.append(np.asarray(global_cond, dtype=np.float16)[0])
        local_padded_values.append(np.asarray(local_padded, dtype=np.float16)[0])
        block_targets.append(np.asarray(target, dtype=np.float16)[0])
        snapshot = tq.memory_snapshot()
        peak = max(peak, int(snapshot["metal_peak_gb"] * (1024**3)))
        if peak > args.max_metal_bytes:
            raise RuntimeError("block calibration exceeded Metal guard")
        if position % 16 == 0 or position == len(selected_indices):
            print(json.dumps({"processed": position, "total": len(selected_indices), "peak_gb": peak / (1024**3)}), flush=True)

    arrays = {
        "h_in": np.stack(block_inputs).astype(np.float16),
        "context": np.stack(contexts).astype(np.float16),
        "global_cond": np.stack(global_conditions).astype(np.float16),
        "local_padded": np.stack(local_padded_values).astype(np.float16),
        "target": np.stack(block_targets).astype(np.float16),
        "sigmas": sigmas[selected_indices].astype(np.float32),
        "prompt_indices": prompt_indices[selected_indices].astype(np.int32),
        "state_indices": np.asarray(selected_indices, dtype=np.int32),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "block0_calibration.npz"
    np.savez_compressed(calibration_path, **arrays)
    summary: dict[str, Any] = {
        "schema": "onus.ternary-quality/v8-block-calibration",
        "status": "prepared",
        "state_cache": {
            "path": str(state_cache),
            "states_sha256": sha256_file(states_path),
            "conditions_sha256": sha256_file(conditions_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "dataset_contract": contract_report,
        "teacher": {"path": str(args.teacher_weights), "sha256": sha256_file(args.teacher_weights)},
        "selection": {
            "state_count": len(selected_indices),
            "replicas_per_prompt_sigma": args.replicas,
            "prompts": cache_manifest["dataset"]["selected_prompts"],
        },
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "calibration": {"path": str(calibration_path), "sha256": sha256_file(calibration_path)},
        "calibration_digest": canonical_digest({name: list(value.shape) for name, value in arrays.items()}),
        "memory": {"peak_metal_bytes": peak, "peak_metal_gb": peak / (1024**3)},
    }
    (args.output_dir / "block0_calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir), "calibration_digest": summary["calibration_digest"], "peak_metal_gb": summary["memory"]["peak_metal_gb"]}, indent=2), flush=True)
    del teacher, states_file, conditions_file
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
