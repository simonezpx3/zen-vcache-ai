"""
CPU Topology and L3 3D V-Cache Hardware Inspector
Analyzes Zen 4 / Zen 5 cache structure, NUMA / CCD bounds, and affinity masks.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class L3CacheDomain:
    index: int
    size_kib: int
    shared_cpus: list[int]
    is_vcache: bool


@dataclass
class CpuTopology:
    model_name: str
    total_logical_cpus: int
    physical_cores: int
    smt_enabled: bool
    l3_domains: list[L3CacheDomain]
    has_3d_vcache: bool
    is_monolithic_ccd: bool
    recommended_infer_cores: str
    avx512_vbmi_supported: bool
    avx512_vnni_supported: bool


def _read_sysfs(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _parse_cpu_list(cpu_list_str: str) -> list[int]:
    """Parse comma/hyphen separated CPU lists, e.g., '0-3,5,7' -> [0, 1, 2, 3, 5, 7]."""
    cpus = set()
    if not cpu_list_str:
        return []
    for part in cpu_list_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def get_cpu_topology() -> CpuTopology:
    """Inspect system hardware and return CpuTopology configuration."""
    model_name = "Unknown Processor"
    flags = set()
    core_ids = set()
    logical_count = os.cpu_count() or 1

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name") and model_name == "Unknown Processor":
                model_name = line.split(":", 1)[1].strip()
            elif line.startswith("flags"):
                flags.update(line.split(":", 1)[1].strip().split())
            elif line.startswith("core id"):
                core_ids.add(line.split(":", 1)[1].strip())

    physical_cores = len(core_ids) if core_ids else logical_count
    smt_enabled = logical_count > physical_cores

    # Inspect L3 caches in /sys/devices/system/cpu/
    l3_domains: list[L3CacheDomain] = []
    seen_shared_lists = set()
    sysfs_cpu_dir = Path("/sys/devices/system/cpu")

    cpu_dirs = [p for p in sysfs_cpu_dir.iterdir() if p.name.startswith("cpu") and p.name[3:].isdigit()]
    for cpu_path in sorted(cpu_dirs, key=lambda p: int(p.name[3:])):
        index3 = cpu_path / "cache/index3"
        if not index3.exists():
            continue

        level = _read_sysfs(index3 / "level")
        if level != "3":
            continue

        shared_cpus_str = _read_sysfs(index3 / "shared_cpu_list")
        if shared_cpus_str in seen_shared_lists:
            continue
        seen_shared_lists.add(shared_cpus_str)

        size_str = _read_sysfs(index3 / "size")  # e.g. "98304K"
        size_kib = 0
        if size_str.endswith("K"):
            size_kib = int(size_str[:-1])
        elif size_str.endswith("M"):
            size_kib = int(size_str[:-1]) * 1024

        shared_cpus = _parse_cpu_list(shared_cpus_str)
        # 3D V-Cache indicator: L3 cache > 32MB (32768 KiB) per CCD, typically 96MB (98304 KiB)
        is_vcache = size_kib >= 65536

        l3_domains.append(
            L3CacheDomain(
                index=len(l3_domains),
                size_kib=size_kib,
                shared_cpus=shared_cpus,
                is_vcache=is_vcache,
            )
        )

    has_3d_vcache = any(d.is_vcache for d in l3_domains)
    is_monolithic_ccd = len(l3_domains) == 1

    # Recommended infer cores:
    # On 7800X3D (single 8C CCD): pin to cores 4-7 (isolating cores 0-3 for system/IO)
    # On multi-CCD parts: pin strictly to the V-Cache CCD
    if has_3d_vcache and is_monolithic_ccd:
        recommended_cores = "4-7"
    elif has_3d_vcache:
        vcache_cpus = next((d.shared_cpus for d in l3_domains if d.is_vcache), [])
        if len(vcache_cpus) >= 8:
            c_start = vcache_cpus[4]
            c_end = vcache_cpus[7]
            recommended_cores = f"{c_start}-{c_end}"
        elif vcache_cpus:
            c_start = vcache_cpus[0]
            c_end = vcache_cpus[-1]
            recommended_cores = f"{c_start}-{c_end}"
        else:
            recommended_cores = "4-7"
    else:
        recommended_cores = "4-7"

    return CpuTopology(
        model_name=model_name,
        total_logical_cpus=logical_count,
        physical_cores=physical_cores,
        smt_enabled=smt_enabled,
        l3_domains=l3_domains,
        has_3d_vcache=has_3d_vcache,
        is_monolithic_ccd=is_monolithic_ccd,
        recommended_infer_cores=recommended_cores,
        avx512_vbmi_supported=bool(flags & {"avx512vbmi", "avx512_vbmi"}),
        avx512_vnni_supported="avx512_vnni" in flags,
    )


def main():
    topo = get_cpu_topology()
    print("=" * 70)
    print(" CPU TOPOLOGY & L3 3D V-CACHE DETECTION")
    print("=" * 70)
    print(f"Model Name        : {topo.model_name}")
    print(f"Logical / Physical: {topo.total_logical_cpus} / {topo.physical_cores} (SMT: {topo.smt_enabled})")
    print(f"3D V-Cache Present: {topo.has_3d_vcache}")
    print(f"Monolithic CCD    : {topo.is_monolithic_ccd}")
    print(f"AVX-512 VBMI      : {topo.avx512_vbmi_supported}")
    print(f"AVX-512 VNNI      : {topo.avx512_vnni_supported}")
    print(f"Recommended Cores : {topo.recommended_infer_cores}")
    print("-" * 70)
    print("L3 Cache Domains:")
    for d in topo.l3_domains:
        vcache_flag = " · 3D V-CACHE" if d.is_vcache else ""
        print(f"  • Domain #{d.index}: {d.size_kib // 1024} MiB ({d.size_kib} KiB) -> CPUs {d.shared_cpus}{vcache_flag}")
    print("=" * 70)


if __name__ == "__main__":
    main()
