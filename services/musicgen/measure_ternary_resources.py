"""Measure reload RSS/Metal/swap for a ternary artifact without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import train_ternary_quality as tq
from audit_ternary_quality import load_manifest, load_student


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ternary reload resources")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    manifest = load_manifest(args.manifest)
    before = {"rss_bytes": rss_bytes(), "metal": tq.memory_snapshot()}
    student = load_student(args.artifact, manifest, args.crop_len)
    after = {"rss_bytes": rss_bytes(), "metal": tq.memory_snapshot()}
    payload = {
        "schema": "onus.ternary-quality-v3/resources",
        "artifact": str(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "platform": platform.platform(),
        "before_reload": before,
        "after_reload": after,
        "rss_max_bytes": rss_bytes(),
        "vm_stat": command_output(["/usr/bin/vm_stat"]),
        "swapusage": command_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"]),
        "elapsed_seconds": time.time() - started,
        "note": "RSS and Metal are separate counters; unified-memory pages may overlap.",
    }
    del student
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "measured",
        "artifact_bytes": payload["artifact_bytes"],
        "rss_max_bytes": payload["rss_max_bytes"],
        "after_reload": payload["after_reload"],
        "swapusage": payload["swapusage"],
    }, indent=2))


if __name__ == "__main__":
    main()
