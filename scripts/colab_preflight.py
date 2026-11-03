"""Read-only Colab capacity report for Strata IQ3_S (stdlib only).

The thresholds are conservative experiment gates, not a Strata compatibility
certificate. Strata's own launch plan is the next source of fit information.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

GB = 10**9
MIN_RAM_TOTAL_GB = 64
MIN_RAM_AVAILABLE_GB = 60
MIN_GPU_TOTAL_GB = 40
MIN_GPU_FREE_GB = 35
MIN_DISK_FREE_GB = 100


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.strip().split()[0]) * 1024
    return values


def gpu_info() -> list[dict[str, object]]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    cards = []
    for line in result.stdout.splitlines():
        name, total, free, driver, capability = [x.strip() for x in line.split(",", 4)]
        cards.append({"name": name, "vram_total_bytes": int(total) * 1024**2,
                      "vram_free_bytes": int(free) * 1024**2,
                      "driver": driver, "compute_capability": capability})
    return cards


def inspect(path: str = "/content") -> dict[str, object]:
    memory = meminfo()
    disk = shutil.disk_usage(path)
    try:
        cards = gpu_info()
        gpu_error = None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        cards, gpu_error = [], str(exc)
    best = max(cards, key=lambda x: x["vram_free_bytes"], default=None)
    checks = {
        "linux_x86_64": platform.system() == "Linux" and platform.machine() == "x86_64",
        "ram_total_at_least_64_gb": memory.get("MemTotal", 0) >= MIN_RAM_TOTAL_GB * GB,
        "ram_available_at_least_60_gb": memory.get("MemAvailable", 0) >= MIN_RAM_AVAILABLE_GB * GB,
        "gpu_total_at_least_40_gb": bool(best and best["vram_total_bytes"] >= MIN_GPU_TOTAL_GB * GB),
        "gpu_free_at_least_35_gb": bool(best and best["vram_free_bytes"] >= MIN_GPU_FREE_GB * GB),
        "disk_free_at_least_100_gb": disk.free >= MIN_DISK_FREE_GB * GB,
    }
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "memory_bytes": memory,
        "disk": {"path": path, "total_bytes": disk.total, "free_bytes": disk.free},
        "gpus": cards, "gpu_probe_error": gpu_error,
        "cuda_runtime": os.environ.get("CUDA_VERSION"),
        "checks": checks, "gate_passed": all(checks.values()),
        "hardware_compatibility": "A100 sm_80 requires Strata Linux CUDA 12 engine; verify selected package in launch plan/runtime",
        "basis": "IQ3_S: 50 GB host RAM experts plus ~10 GB headroom, 54.8 GB RAM+VRAM weights; ~83.6 GB main download plus ~5.2 GB MTP; 100 GB free disk is a conservative experiment threshold",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/content/strata-data")
    parser.add_argument("--output", default="/content/strata-artifacts/preflight.json")
    args = parser.parse_args()
    report = inspect(str(Path(args.data_dir).parent))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Preflight {'PASS' if report['gate_passed'] else 'FAIL'}; report: {output}")
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
