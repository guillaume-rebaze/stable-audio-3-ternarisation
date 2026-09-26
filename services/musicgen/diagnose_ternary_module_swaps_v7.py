"""Measure whether one block-0 module swap fixes the audit bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_ternary_quality as aq
import train_ternary_quality as tq
from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_runtime_contract import timestep_tensor

mx = tq.mx
nn = tq.nn


def load_model(
    teacher_weights: Path,
    records: dict,
    group_size: int,
    crop_len: int,
) -> nn.Module:
    model = dit_mlx_medium.DiT(T_lat=crop_len)
    model.load_weights(str(teacher_weights), strict=False)
    model = tq.apply_records_to_model(model, records, group_size)
    model.freeze()
    mx.eval(model.parameters())
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-metal-bytes", type=int, default=12_000_000_000)
    args = parser.parse_args()

    candidate, candidate_meta = tq.load_records_checkpoint(args.candidate)
    alternative, alternative_meta = tq.load_records_checkpoint(args.alternative)
    if int(candidate_meta["group_size"]) != int(alternative_meta["group_size"]):
        raise ValueError("candidate and alternative group sizes differ")
    prefixes = sorted(candidate)
    if prefixes != sorted(alternative):
        raise ValueError("candidate and alternative scopes differ")
    group_size = int(candidate_meta["group_size"])

    samples = tq.load_samples(args.dataset_dir, 0)
    samples, split_report = aq.load_split_manifest(
        args.split_manifest, samples, "debug_train_seen"
    )
    selected = aq.one_sample_per_prompt(samples, args.max_prompts)
    schedule = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    mx.eval(schedule)
    sigmas = [float(value) for value in schedule[:-1]]

    teacher = dit_mlx_medium.DiT(T_lat=128)
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    context_cache, cross_cache, global_cond, _ = tq.cache_conditioning(
        teacher,
        args.teacher_weights,
        [sample["prompt"] for sample in selected],
        seconds=12.0,
    )
    del context_cache
    variants: dict[str, dict] = {"candidate": candidate}
    for prefix in prefixes:
        swapped = dict(candidate)
        swapped[prefix] = alternative[prefix]
        variants[f"swap:{prefix}"] = swapped

    reports: dict[str, dict] = {}
    for name, records in variants.items():
        model = load_model(args.teacher_weights, records, group_size, 128)
        metrics = aq.audit_velocity(
            teacher,
            model,
            selected,
            cross_cache,
            global_cond,
            128,
            sigmas,
            args.seed,
        )
        reports[name] = {
            "mean_cosine": metrics["mean_cosine"],
            "min_cosine": metrics["min_cosine"],
            "candidate_velocity_pass": metrics["candidate_velocity_pass"],
            "release_velocity_pass": metrics["release_velocity_pass"],
            "worst": sorted(
                [
                    {
                        "prompt": row["prompt"],
                        "sigma": row["sigma"],
                        "cosine": row["cosine"],
                    }
                    for row in metrics["records"]
                ],
                key=lambda row: row["cosine"],
            )[:5],
        }
        del model
        tq.gc.collect()
        mx.clear_cache()
        print(
            f"[SwapAudit] {name} mean={reports[name]['mean_cosine']:.8f} "
            f"min={reports[name]['min_cosine']:.8f}",
            flush=True,
        )

    result = {
        "schema": "onus.ternary-quality/v7-module-swap-audit",
        "candidate": str(args.candidate.resolve()),
        "alternative": str(args.alternative.resolve()),
        "teacher": str(args.teacher_weights.resolve()),
        "split": split_report,
        "prompts": len(selected),
        "sigmas": sigmas,
        "reports": reports,
        "memory": tq.memory_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tq.write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
