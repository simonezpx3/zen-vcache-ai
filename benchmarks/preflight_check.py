#!/usr/bin/env python3
"""Pre-flight environment noise check for Zen 4 3D V-Cache benchmarks."""

import argparse
import json
import os
from pathlib import Path


def _read_sysfs(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else "unknown"


def check_environment() -> dict:
    governor = _read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    thp = _read_sysfs("/sys/kernel/mm/transparent_hugepage/enabled")
    paranoid = _read_sysfs("/proc/sys/kernel/perf_event_paranoid")
    load1, load5, load15 = os.getloadavg()

    warnings = []
    if governor not in ("performance", "unknown"):
        warnings.append(f"CPU governor is '{governor}' (recommend 'performance' to reduce variance)")
    if not thp.startswith(chr(91) + "always") and thp != "unknown":
        warnings.append(f"THP is '{thp}' (recommend 'always' for 2MB TLB residency)")
    if load1 > 1.5:
        warnings.append(f"High 1-min system load ({load1:.2f} > 1.50); desktop noise may inflate p99/std")

    return {
        "cpu_governor": governor,
        "thp_enabled": thp,
        "perf_event_paranoid": paranoid,
        "load_avg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "warnings": warnings,
        "clean": len(warnings) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight noise check for zen-vcache-ai benchmarks")
    parser.add_argument("--json", type=str, help="Optional path to write JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if warnings are present")
    args = parser.parse_args()

    report = check_environment()
    l1 = report["load_avg"][0]
    print(
        f"Pre-flight: governor={report['cpu_governor']} | THP={report['thp_enabled']} | "
        f"load1={l1:.2f} | paranoid={report['perf_event_paranoid']}"
    )
    for w in report["warnings"]:
        print(f"  WARN // {w}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 1 if (args.strict and report["warnings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
