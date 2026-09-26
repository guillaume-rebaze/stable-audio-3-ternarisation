"""Build a symmetric hard-quantized teacher baseline for paired V7 audits."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import train_ternary_quality as tq
import train_ternary_window_v6 as window_v7

mx = tq.mx


def check_memory(max_metal_bytes: int, label: str) -> dict[str, float]:
    snapshot = tq.memory_snapshot()
    peak = int(snapshot["metal_peak_gb"] * (1024**3))
    if peak > max_metal_bytes:
        raise RuntimeError(
            f"{label} exceeded Metal memory guard: {peak} > {max_metal_bytes} bytes"
        )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    if not (0 <= args.start_block <= args.end_block < 24):
        raise ValueError("invalid baseline block window")
    if args.group_size not in {32, 64, 128}:
        raise ValueError("group-size must be 32, 64, or 128")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite baseline: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    states, sigmas, prompt_indices, sources, cross, global_cond, cache_manifest = (
        window_v7.load_state_cache(args.state_cache)
    )
    fixed_indices = window_v7.balanced_fixed_state_indices(
        prompt_indices, sigmas, sources, min(16, len(states))
    )
    model = tq.dit_mlx_medium.DiT(T_lat=args.crop_len)
    model.load_weights(str(args.teacher_weights), strict=False)
    model.freeze()
    mx.eval(model.parameters())
    check_memory(args.max_metal_bytes, "dense teacher load")

    records = {}
    for block_index in range(args.start_block, args.end_block + 1):
        tq.replace_core_block(
            model.transformer.layers[block_index], args.group_size, "symmetric"
        )
        tq.hard_freeze_block(
            model.transformer.layers[block_index],
            block_index,
            args.group_size,
            records,
            "symmetric",
        )

    records_path = args.output_dir / "records_checkpoint.npz"
    tq.save_records_checkpoint(
        records_path,
        records,
        args.end_block + 1,
        args.group_size,
        args.crop_len,
        "symmetric",
    )
    fixture_path = args.output_dir / "hard_forward_fixture.npz"
    window_v7.save_hard_forward_fixture(
        fixture_path,
        model,
        states,
        sigmas,
        prompt_indices,
        cross,
        global_cond,
        fixed_indices,
    )
    check_memory(args.max_metal_bytes, "hard baseline forward")

    roundtrip_path = args.output_dir / "cross_process_roundtrip.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_ternary_records_roundtrip.py")),
        "--teacher-weights",
        str(args.teacher_weights),
        "--records",
        str(records_path),
        "--fixture",
        str(fixture_path),
        "--output",
        str(roundtrip_path),
        "--group-size",
        str(args.group_size),
        "--crop-len",
        str(args.crop_len),
        "--max-metal-bytes",
        str(args.max_metal_bytes),
    ]
    records_digest = tq.scope_digest(records)
    record_count = len(records)
    del model, records, states, sigmas, prompt_indices, sources, cross, global_cond
    gc.collect()
    mx.clear_cache()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeError("hard baseline cross-process roundtrip failed: " + result.stderr[-4000:])

    report = {
        "schema": "onus.ternary-quality/v7-hard-baseline",
        "status": "roundtrip_verified",
        "teacher": tq.file_fingerprint(args.teacher_weights),
        "state_cache_manifest": tq.file_fingerprint(args.state_cache / "manifest.json"),
        "records_checkpoint": str(records_path),
        "fixture": str(fixture_path),
        "scope_count": record_count,
        "scope_digest": records_digest,
        "cross_process_roundtrip": json.loads(
            roundtrip_path.read_text(encoding="utf-8")
        ),
        "memory": tq.memory_snapshot(),
    }
    tq.write_json(args.output_dir / "baseline_summary.json", report)
    print(json.dumps({
        "status": report["status"],
        "records_checkpoint": str(records_path),
        "scope_count": record_count,
        "roundtrip": report["cross_process_roundtrip"]["status"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
