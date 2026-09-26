"""Create a block-boundary checkpoint that reopens the last N blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_ternary_quality as tq


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a targeted ternary retrain")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reopen-last", type=int, default=1)
    args = parser.parse_args()

    records, metadata = tq.load_records_checkpoint(args.source)
    completed_blocks = int(metadata["next_block"])
    if args.reopen_last <= 0 or args.reopen_last > completed_blocks:
        raise ValueError("reopen-last must be within completed checkpoint blocks")
    first_reopened = completed_blocks - args.reopen_last
    kept = {
        prefix: record
        for prefix, record in records.items()
        if int(prefix.split(".")[2]) < first_reopened
    }
    tq.save_records_checkpoint(
        args.destination,
        kept,
        first_reopened,
        int(metadata["group_size"]),
        int(metadata["crop_len"]),
        str(metadata["quantizer_mode"]),
    )
    print({
        "status": "prepared",
        "source": str(args.source),
        "destination": str(args.destination),
        "reopened_blocks": args.reopen_last,
        "next_block": first_reopened,
        "kept_records": len(kept),
    })


if __name__ == "__main__":
    main()
