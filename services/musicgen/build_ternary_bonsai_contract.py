#!/usr/bin/env python3
"""Build a header-only Bonsai P0 contract for a dense NPZ checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ternary_bonsai_contract import (
    build_experiment_contract,
    build_weight_scope,
    inspect_npz_headers,
    storage_report,
    validate_experiment_contract,
    write_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--code-path",
        action="append",
        type=Path,
        default=[],
        help="Source file to hash into the contract; repeatable.",
    )
    parser.add_argument("--teacher-json", type=Path)
    parser.add_argument("--data-json", type=Path)
    parser.add_argument("--sampler-json", type=Path)
    parser.add_argument("--resources-json", type=Path)
    return parser.parse_args()


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    args = parse_args()
    teacher_path = args.teacher_weights.expanduser().resolve()
    headers = inspect_npz_headers(teacher_path)
    scope = build_weight_scope(headers)
    storage = storage_report(scope, args.group_size)
    code_paths = args.code_path or [Path(__file__).with_name("ternary_bonsai_contract.py")]
    contract = build_experiment_contract(
        scope=scope,
        storage=storage,
        files=[(teacher_path, "teacher_weights")] + [(path, "source") for path in code_paths],
        teacher=load_json(args.teacher_json),
        data=load_json(args.data_json),
        sampler=load_json(args.sampler_json),
        resources=load_json(args.resources_json),
    )
    validate_experiment_contract(contract)
    output = write_contract(contract, args.output)
    print(
        json.dumps(
            {
                "contract": str(output),
                "core_tensors": scope["core_tensor_count"],
                "support_tensors": scope["support_tensor_count"],
                "core_fraction": scope["core_fraction"],
                "group_size": storage["group_size"],
                "packed_payload_bytes": storage["packed_payload_bytes"],
                "within_package_envelope": storage["within_package_envelope"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

