"""Materialize a cumulative ternary-record checkpoint as a reloadable DiT."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_ternary_quality as tq


def expected_cumulative_scope(next_block: int) -> set[str]:
    """Return the exact seven-projection scope for completed blocks [0, N)."""
    if not 1 <= next_block <= 24:
        raise ValueError(f"next_block must be in [1, 24], got {next_block}")
    return {
        f"transformer.layers.{block_index}.{name}"
        for block_index in range(next_block)
        for name in tq.CORE_NAMES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a cumulative records checkpoint with dense teacher remainder"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new .npz artifact path")
    args = parser.parse_args()

    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    manifest_path = args.output.with_suffix(".json")
    reload_path = args.output.with_suffix(".reload.json")
    temporary_path = args.output.with_suffix(".tmp.npz")
    if any(path.exists() for path in (args.output, manifest_path, reload_path, temporary_path)):
        raise FileExistsError(f"refusing to overwrite export target: {args.output}")
    if not args.teacher_weights.is_file():
        raise FileNotFoundError(args.teacher_weights)

    records, checkpoint = tq.load_records_checkpoint(args.checkpoint)
    group_size = int(checkpoint["group_size"])
    crop_len = int(checkpoint["crop_len"])
    quantizer_mode = str(checkpoint["quantizer_mode"])
    next_block = int(checkpoint["next_block"])
    if quantizer_mode not in {"symmetric", "learned_symmetric", "affine_centered"}:
        raise ValueError(f"unsupported checkpoint quantizer: {quantizer_mode}")
    if group_size not in {32, 64, 128}:
        raise ValueError(f"unsupported checkpoint group size: {group_size}")

    expected = expected_cumulative_scope(next_block)
    actual = set(records)
    if actual != expected:
        raise ValueError(
            "checkpoint is not a complete cumulative core: "
            f"missing={sorted(expected - actual)[:5]}, "
            f"unexpected={sorted(actual - expected)[:5]}"
        )

    student = tq.dit_mlx_medium.DiT(T_lat=crop_len)
    student.load_weights(str(args.teacher_weights), strict=False)
    student = tq.apply_records_to_model(student, records, group_size)

    symmetric = quantizer_mode in {"symmetric", "learned_symmetric"}
    storage_mode = "symmetric_compact" if symmetric else "full_affine"
    config = {
        "kind": "cumulative-records-materialization",
        "source_checkpoint": str(args.checkpoint),
        "source_scope_digest": checkpoint["scope_digest"],
        "next_block": next_block,
        "teacher_weights": tq.file_fingerprint(args.teacher_weights),
    }
    tq.export_artifact(
        student,
        records,
        args.output,
        manifest_path,
        group_size,
        crop_len,
        config,
        quantizer_mode,
        storage_mode,
    )
    reloaded = tq.reload_model(
        args.output,
        records,
        group_size,
        crop_len,
        quantizer_mode,
        storage_mode,
    )
    parity = tq.parameter_reload_report(student, reloaded)
    tq.write_json(
        reload_path,
        {
            "artifact": str(args.output),
            "source_checkpoint": str(args.checkpoint),
            "scope_count": len(records),
            "scope_digest": checkpoint["scope_digest"],
            "parameter_reload": parity,
            "bytes": args.output.stat().st_size,
            "memory": tq.memory_snapshot(),
        },
    )
    if not parity["exact"]:
        raise RuntimeError(f"reload parity failed; see {reload_path}")

    print(
        tq.json.dumps(
            {
                "status": "exported_reloadable",
                "artifact": str(args.output),
                "manifest": str(manifest_path),
                "reload_report": str(reload_path),
                "bytes": args.output.stat().st_size,
                "next_block": next_block,
                "scope_count": len(records),
                "reload_exact": parity["exact"],
                "memory": tq.memory_snapshot(),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
