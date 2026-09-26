"""Benchmark several records-only candidates on the exact V7 velocity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_ternary_quality as aq
import train_ternary_quality as tq
from models.defs import dit_mlx_medium
from models.defs.sa3_pipeline import build_pingpong_schedule
from ternary_provenance_v8 import validate_contract

mx = tq.mx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--dataset-contract",
        type=Path,
        default=None,
        help="V8 contract; preferred over the legacy split manifest",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--split-role",
        choices=("debug_train_seen", "validation", "test"),
        default="debug_train_seen",
    )
    args = parser.parse_args()

    loaded_records = []
    group_size = None
    for path in args.records:
        records, metadata = tq.load_records_checkpoint(path)
        current_group_size = int(metadata["group_size"])
        if group_size is None:
            group_size = current_group_size
        if current_group_size != group_size:
            raise ValueError("all candidates must use the same group size")
        loaded_records.append((path, records))

    samples = tq.load_samples(args.dataset_dir, 0)
    if args.dataset_contract is not None:
        contract_report = aq.validate_dataset_contract(
            args.dataset_contract,
            args.project_root,
            args.dataset_dir,
            args.split_role,
        )
        contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
        contract_role = "train" if args.split_role == "debug_train_seen" else args.split_role
        allowed = {
            str((args.project_root / sample["latent"]["path"]).resolve())
            for sample in contract["splits"][contract_role]["samples"]
        }
        samples = [sample for sample in samples if str(Path(sample["path"]).resolve()) in allowed]
        split_report = {
            "path": str(args.dataset_contract),
            "role": args.split_role,
            "verified": True,
            "sample_count": len(samples),
            "parent_count": None,
            "dataset_contract": contract_report,
        }
    elif args.split_manifest is not None:
        samples, split_report = aq.load_split_manifest(
            args.split_manifest, samples, args.split_role
        )
    else:
        raise ValueError("provide --dataset-contract or --split-manifest")
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

    reports = []
    for path, records in loaded_records:
        model = dit_mlx_medium.DiT(T_lat=128)
        model.load_weights(str(args.teacher_weights), strict=False)
        model = tq.apply_records_to_model(model, records, int(group_size))
        model.freeze()
        mx.eval(model.parameters())
        metrics = aq.audit_velocity(
            teacher, model, selected, cross_cache, global_cond, 128, sigmas, args.seed
        )
        worst = sorted(
            (
                {
                    "prompt": row["prompt"],
                    "sigma": row["sigma"],
                    "cosine": row["cosine"],
                }
                for row in metrics["records"]
            ),
            key=lambda row: row["cosine"],
        )[:5]
        report = {
            "records": str(path),
            "mean_cosine": metrics["mean_cosine"],
            "min_cosine": metrics["min_cosine"],
            "candidate_velocity_pass": metrics["candidate_velocity_pass"],
            "release_velocity_pass": metrics["release_velocity_pass"],
            "worst": worst,
        }
        reports.append(report)
        print(
            f"[RecordBenchmark] {path} mean={report['mean_cosine']:.8f} "
            f"min={report['min_cosine']:.8f}",
            flush=True,
        )
        del model
        tq.gc.collect()
        mx.clear_cache()

    payload = {
        "schema": "onus.ternary-quality/v7-record-benchmark",
        "teacher": str(args.teacher_weights.resolve()),
        "split": split_report,
        "prompts": len(selected),
        "sigmas": sigmas,
        "reports": reports,
        "memory": tq.memory_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tq.write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
