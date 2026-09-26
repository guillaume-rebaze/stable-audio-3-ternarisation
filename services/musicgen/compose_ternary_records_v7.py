"""Compose a records-only ablation from a base and selected module swaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_ternary_quality as tq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--swap",
        nargs=2,
        action="append",
        metavar=("PREFIX", "RECORDS"),
        default=[],
        help="replace one module prefix with the record from another checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records, metadata = tq.load_records_checkpoint(args.base)
    group_size = int(metadata["group_size"])
    for prefix, source_path in args.swap:
        alternative, alternative_metadata = tq.load_records_checkpoint(Path(source_path))
        if int(alternative_metadata["group_size"]) != group_size:
            raise ValueError("swap group size does not match base")
        if prefix not in records or prefix not in alternative:
            raise ValueError(f"swap prefix is absent from both records: {prefix}")
        records[prefix] = alternative[prefix]

    tq.save_records_checkpoint(
        args.output,
        records,
        next_block=int(metadata.get("next_block", 1)),
        group_size=group_size,
        crop_len=int(metadata.get("crop_len", 128)),
        quantizer_mode=str(metadata["quantizer_mode"]),
    )
    print(
        {
            "status": "composed",
            "base": str(args.base),
            "swaps": {prefix: source for prefix, source in args.swap},
            "output": str(args.output),
            "scope_count": len(records),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
