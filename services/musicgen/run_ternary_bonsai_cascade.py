#!/usr/bin/env python3
"""Run a guarded cumulative G32 Bonsai cascade one block at a time.

The driver deliberately keeps pointwise and on-policy attempts separate.  A
block is promoted only after the reloaded full-DiT audit passes; the selected
record checkpoint is written to a manifest after every block so an interrupted
campaign can resume without guessing which prefix was accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEACHER = Path(
    "/Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/"
    "mlx/models/mlx/dit_medium_f16.npz"
)
DEFAULT_DATASET = ROOT / "output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-independent/train"
DEFAULT_STATE_CACHE = ROOT / "output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/state-cache-512-v9-hadamard-audit-exact-v2"
DEFAULT_TARGET_CACHE = ROOT / "output/sample-expertise-pilot/ternary-quality-v9-bonsai-pilot/teacher-targets-512-v9-hadamard-audit-exact-v3/targets.npz"
TRAINER = ROOT / "services/musicgen/train_ternary_window_v6.py"
AUDITOR = ROOT / "services/musicgen/audit_ternary_quality.py"
ROLLOUT_PREPARER = ROOT / "services/musicgen/prepare_ternary_rollout_targets.py"
RENDERER = ROOT / "services/musicgen/render_ternary_bonsai_canary.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-records", required=True, type=Path)
    parser.add_argument("--start-block", required=True, type=int)
    parser.add_argument("--end-block", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_STATE_CACHE)
    parser.add_argument("--target-cache", type=Path, default=DEFAULT_TARGET_CACHE)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--quantizer-mode",
        choices=("learned_symmetric", "learned_symmetric_hadamard"),
        default="learned_symmetric_hadamard",
    )
    parser.add_argument("--quantizer-surrogate", choices=("smooth", "identity"), default="identity")
    parser.add_argument("--master-init", choices=("record_dequantized", "dense_teacher"), default="dense_teacher")
    parser.add_argument(
        "--state-sampling",
        choices=("with_replacement", "without_replacement"),
        default="without_replacement",
    )
    parser.add_argument(
        "--promote-on-quality-fail",
        action="store_true",
        help="continue the structural cascade while recording failed quality gates",
    )
    parser.add_argument(
        "--skip-onpolicy-fallback",
        action="store_true",
        help="do not launch the expensive rollout fallback after a pointwise failure",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, capture: bool = False) -> str:
    print("[Cascade] $ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def audit(records: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    run(
        [
            sys.executable,
            str(AUDITOR),
            "--records-checkpoint",
            str(records),
            "--dataset-dir",
            str(args.dataset_dir),
            "--teacher-weights",
            str(TEACHER),
            "--output-dir",
            str(output_dir),
            "--max-prompts",
            str(args.max_prompts),
            "--seed",
            str(args.seed),
            "--max-metal-bytes",
            str(args.max_metal_bytes),
            "--trajectory-steps",
            "4",
            "--sampler-steps",
            "4",
            "--allow-fail",
        ]
    )
    return read_audit(output_dir / "audit_summary.json")


def read_audit(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    velocity = summary["velocity"]
    result = {
        "mean_cosine": velocity["mean_cosine"],
        "min_cosine": velocity["min_cosine"],
        "candidate_velocity_pass": velocity["candidate_velocity_pass"],
        "release_velocity_pass": velocity["release_velocity_pass"],
        "trajectory": summary.get("trajectory", {}),
        "summary": rel(summary_path),
    }
    print(
        "[Cascade] audit mean={mean_cosine:.6f} min={min_cosine:.6f} "
        "release={release_velocity_pass}".format(**result),
        flush=True,
    )
    return result


def render(records: Path, output_dir: Path) -> dict[str, Any]:
    canary_dir = output_dir / "audio-canary-4steps-128"
    stdout = run(
        [
            sys.executable,
            str(RENDERER),
            "--records-checkpoint",
            str(records),
            "--teacher-weights",
            str(TEACHER),
            "--dataset-dir",
            str(ROOT / "output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-independent/train"),
            "--out-dir",
            str(canary_dir),
            "--crop-len",
            "128",
            "--seconds",
            "12",
            "--steps",
            "4",
            "--seed",
            "20260925",
            "--sample-index",
            "0",
        ],
        capture=True,
    )
    result = json.loads(stdout)
    return read_audio(canary_dir / "audio_canary.json", result)


def read_audio(manifest_path: Path, result: dict[str, Any] | None = None) -> dict[str, Any]:
    if result is None:
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        "[Cascade] audio cosine={:.6f} latent_l2={:.6f} technical={}".format(
            result["audio_cosine"], result["latent_relative_l2"], result["technical_pass"]
        ),
        flush=True,
    )
    return {
        "audio_cosine": result["audio_cosine"],
        "audio_relative_l2": result["audio_relative_l2"],
        "latent_relative_l2": result["latent_relative_l2"],
        "technical_pass": result["technical_pass"],
        "manifest": rel(manifest_path),
    }


def train_pointwise(source: Path, output_dir: Path, block: int, args: argparse.Namespace) -> Path:
    run(
        [
            sys.executable,
            str(TRAINER),
            "--state-cache",
            str(args.state_cache),
            "--teacher-target-cache",
            str(args.target_cache),
            "--source-records",
            str(source),
            "--master-init",
            args.master_init,
            "--teacher-weights",
            str(TEACHER),
            "--output-dir",
            str(output_dir),
            "--start-block",
            str(block),
            "--end-block",
            str(block),
            "--group-size",
            str(args.group_size),
            "--quantizer-mode",
            args.quantizer_mode,
            "--quantizer-surrogate",
            args.quantizer_surrogate,
            "--state-sampling",
            args.state_sampling,
            "--steps",
            str(args.steps),
            "--max-updates",
            str(args.steps),
            "--gradient-accumulation",
            "1",
            "--learning-rate",
            "0.00001",
            "--learning-rate-end",
            "0.000001",
            "--scale-learning-rate",
            "0.0003",
            "--scale-learning-rate-end",
            "0.00003",
            "--soft-end-updates",
            "0",
            "--weight-decay",
            "0",
            "--optimizer-eps",
            "0.000001",
            "--gradient-clip",
            "1",
            "--checkpoint-every-steps",
            "0",
            "--max-metal-bytes",
            str(args.max_metal_bytes),
            "--seed",
            str(args.seed),
            "--records-only",
        ]
    )
    return output_dir / "records_checkpoint.npz"


def prepare_rollout(source: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    target = output_dir / "targets.npz"
    run(
        [
            sys.executable,
            str(ROLLOUT_PREPARER),
            "--state-cache",
            str(args.state_cache),
            "--source-records",
            str(source),
            "--teacher-weights",
            str(TEACHER),
            "--output",
            str(target),
            "--repeats",
            "2",
            "--seed",
            str(args.seed),
            "--rollout-source-mode",
            "student",
            "--max-metal-bytes",
            str(args.max_metal_bytes),
        ]
    )
    return target


def train_on_policy(
    source: Path, rollout: Path, output_dir: Path, block: int, args: argparse.Namespace
) -> Path:
    run(
        [
            sys.executable,
            str(TRAINER),
            "--state-cache",
            str(args.state_cache),
            "--teacher-target-cache",
            str(args.target_cache),
            "--full-rollout-target-cache",
            str(rollout),
            "--source-records",
            str(source),
            "--master-init",
            args.master_init,
            "--teacher-weights",
            str(TEACHER),
            "--output-dir",
            str(output_dir),
            "--start-block",
            str(block),
            "--end-block",
            str(block),
            "--group-size",
            str(args.group_size),
            "--quantizer-mode",
            args.quantizer_mode,
            "--quantizer-surrogate",
            args.quantizer_surrogate,
            "--full-rollout-loss-weight",
            "0.25",
            "--full-rollout-window-steps",
            "4",
            "--full-rollout-on-policy-stitch",
            "--steps",
            str(args.steps),
            "--max-updates",
            str(args.steps),
            "--gradient-accumulation",
            "2",
            "--learning-rate",
            "0.00001",
            "--learning-rate-end",
            "0.000001",
            "--scale-learning-rate",
            "0.0003",
            "--scale-learning-rate-end",
            "0.00003",
            "--soft-end-updates",
            "0",
            "--weight-decay",
            "0",
            "--optimizer-eps",
            "0.000001",
            "--gradient-clip",
            "1",
            "--checkpoint-every-steps",
            "0",
            "--max-metal-bytes",
            str(args.max_metal_bytes),
            "--seed",
            str(args.seed),
            "--records-only",
        ]
    )
    return output_dir / "records_checkpoint.npz"


def initial_manifest(args: argparse.Namespace, source: Path) -> dict[str, Any]:
    return {
        "schema": "onus.ternary-quality/v9-bonsai-cascade",
        "group_size": args.group_size,
        "quantizer_mode": args.quantizer_mode,
        "quantizer_surrogate": args.quantizer_surrogate,
        "master_init": args.master_init,
        "state_sampling": args.state_sampling,
        "dataset_dir": rel(args.dataset_dir),
        "state_cache": rel(args.state_cache),
        "target_cache": rel(args.target_cache),
        "seed": args.seed,
        "max_metal_bytes": args.max_metal_bytes,
        "steps": args.steps,
        "initial_records": {"path": rel(source), "sha256": sha256(source)},
        "blocks": {},
    }


def main() -> int:
    args = parse_args()
    source = args.initial_records.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.start_block < 0 or args.end_block < args.start_block or args.end_block > 23:
        raise ValueError("expected 0 <= start-block <= end-block <= 23")

    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        blocks = manifest.get("blocks", {})
        if blocks:
            last = max(int(key) for key in blocks)
            source = (ROOT / blocks[str(last)]["selected_records"]).resolve()
            args.start_block = last + 1
            print(f"[Cascade] resume after block {last}; source={rel(source)}", flush=True)
    else:
        manifest = initial_manifest(args, source)
        write_json(manifest_path, manifest)

    for block in range(args.start_block, args.end_block + 1):
        if block in {0, 1, 2, 3, 4} and block < args.start_block:
            continue
        block_root = output_root / f"cascade-block{block:02d}-g{args.group_size}"
        pointwise_root = block_root / "pointwise"
        pointwise_records = pointwise_root / "records_checkpoint.npz"
        if pointwise_records.exists():
            print(f"[Cascade] reuse pointwise records for block {block}", flush=True)
        else:
            incomplete = pointwise_root / "records_checkpoint.tmp.npz"
            if incomplete.exists():
                incomplete.unlink()
            pointwise_records = train_pointwise(source, pointwise_root, block, args)
        pointwise_audit_path = pointwise_root / f"audit-{args.max_prompts}p-4s/audit_summary.json"
        pointwise_audit = (
            read_audit(pointwise_audit_path)
            if pointwise_audit_path.exists()
            else audit(pointwise_records, pointwise_root / f"audit-{args.max_prompts}p-4s", args)
        )
        pointwise_audio_path = pointwise_root / "audio-canary-4steps-128/audio_canary.json"
        pointwise_audio = (
            read_audio(pointwise_audio_path)
            if pointwise_audio_path.exists()
            else render(pointwise_records, pointwise_root)
        )

        attempts: dict[str, Any] = {
            "pointwise": {
                "records": rel(pointwise_records),
                "records_sha256": sha256(pointwise_records),
                "audit": pointwise_audit,
                "audio": pointwise_audio,
            }
        }
        selected_records = pointwise_records
        selected_mode = "pointwise"

        if (
            not args.skip_onpolicy_fallback
            and (not pointwise_audit["release_velocity_pass"] or not pointwise_audio["technical_pass"])
        ):
            rollout_root = block_root / "rollout-targets"
            rollout = rollout_root / "targets.npz"
            if rollout.exists():
                print(f"[Cascade] reuse rollout targets for block {block}", flush=True)
            else:
                rollout = prepare_rollout(source, rollout_root, args)
            on_policy_root = block_root / "onpolicy4"
            on_policy_records = on_policy_root / "records_checkpoint.npz"
            if on_policy_records.exists():
                print(f"[Cascade] reuse on-policy records for block {block}", flush=True)
            else:
                incomplete = on_policy_root / "records_checkpoint.tmp.npz"
                if incomplete.exists():
                    incomplete.unlink()
                on_policy_records = train_on_policy(source, rollout, on_policy_root, block, args)
            on_policy_audit_path = on_policy_root / f"audit-{args.max_prompts}p-4s/audit_summary.json"
            on_policy_audit = (
                read_audit(on_policy_audit_path)
                if on_policy_audit_path.exists()
                else audit(on_policy_records, on_policy_root / f"audit-{args.max_prompts}p-4s", args)
            )
            on_policy_audio_path = on_policy_root / "audio-canary-4steps-128/audio_canary.json"
            on_policy_audio = (
                read_audio(on_policy_audio_path)
                if on_policy_audio_path.exists()
                else render(on_policy_records, on_policy_root)
            )
            attempts["onpolicy4"] = {
                "rollout_targets": rel(rollout),
                "records": rel(on_policy_records),
                "records_sha256": sha256(on_policy_records),
                "audit": on_policy_audit,
                "audio": on_policy_audio,
            }
            if (
                not args.promote_on_quality_fail
                and (not on_policy_audit["release_velocity_pass"] or not on_policy_audio["technical_pass"])
            ):
                write_json(manifest_path, manifest)
                raise RuntimeError(f"block {block} failed pointwise and on-policy gates")
            selected_records = on_policy_records
            selected_mode = "onpolicy4"

        if (
            not args.promote_on_quality_fail
            and (not pointwise_audit["release_velocity_pass"] or not pointwise_audio["technical_pass"])
            and selected_mode == "pointwise"
        ):
            write_json(manifest_path, manifest)
            raise RuntimeError(f"block {block} failed pointwise quality gates")

        manifest["blocks"][str(block)] = {
            "selected_mode": selected_mode,
            "selected_records": rel(selected_records),
            "selected_records_sha256": sha256(selected_records),
            "attempts": attempts,
        }
        write_json(manifest_path, manifest)
        summary = json.loads((selected_records.parent / "window_summary.json").read_text(encoding="utf-8"))
        print(
            f"[Cascade] promoted block {block} mode={selected_mode} "
            f"scope_count={summary.get('scope_count')}",
            flush=True,
        )
        source = selected_records

    manifest["final_records"] = {"path": rel(source), "sha256": sha256(source)}
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "final_records": rel(source), "manifest": rel(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
