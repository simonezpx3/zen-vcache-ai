# Cache-Resident Working Sets: Exploiting Zen 4 96MB V-Cache for Sub-Millisecond SQLite & Edge Micro-AI

Hardware PMU telemetry and benchmarks demonstrating low-latency memory residency on the AMD Ryzen 7 7800X3D (Zen 4, 8C/16T, 96MB unified L3 SRAM).

---

## Executive Summary

AMD's 3D V-Cache is widely recognized for accelerating cache-sensitive gaming workloads. In systems programming and edge database engineering, however, the **monolithic 96MB L3 SRAM pool** on the Ryzen 7 7800X3D offers a distinctive advantage: an entire active working set—including database pages, index structures, and INT8 model weights—can reside entirely within L3 cache with ~10–11 ns access latency and ~2.5 TB/s aggregate bandwidth, completely bypassing the ~70 ns DDR5 bus.

This repository provides reproducible benchmarks and raw Linux PMU hardware performance counter logs (`perf stat`) characterizing the memory boundary, SQLite zero-copy performance, and TLB mitigation.

---

## Architectural Context: Single-CCD Advantage

Unlike dual-CCD models (e.g., Ryzen 9 7900X3D and 7950X3D), where only one CCD carries the stacked 3D V-Cache tile—introducing asymmetric cross-CCD NUMA-like latency penalties and requiring OS thread parking—the **Ryzen 7 7800X3D consists of a single monolithic 8C/16T CCD**:

```
+---------------------------------------------------------------+
|                      Ryzen 7 7800X3D CCD                      |
|  [Core 0] [Core 1] [Core 2] [Core 3] [Core 4] ... [Core 7]    |
+---------------------------------------------------------------+
|         Unified 96 MB L3 Cache (~10-11 ns latency)            |
|         (32 MB CCD base + 64 MB stacked 3D V-Cache)           |
+---------------------------------------------------------------+
                               |  Memory Bus (~70 ns latency)
                               v
+---------------------------------------------------------------+
|                 System Memory (DDR5 Dual-Channel)             |
+---------------------------------------------------------------+
```

All 8 physical cores and 16 execution threads share direct, uniform access to the 96MB L3 slice without inter-core die-crossing latency.

---

## Benchmark 1: The Hardware Cache Cliff (Pointer Chasing)

To isolate true memory subsystem latency and eliminate hardware stream/stride prefetcher speculation, `benchmarks/cache_cliff.c` constructs a cyclic pseudo-random permutation across 64-byte cache lines. Each cache line contains a single pointer to the next pseudo-random line in the cycle (`curr = *(void**)curr`).

### Empirical Latency by Working Set Size

Measured on Linux 6.14 (Zen 4, AMD Ryzen 7 7800X3D @ 4.2–5.0 GHz):

| Buffer Size | Mean Latency (ns) | Residency Domain | Notes |
|:---|:---:|:---|:---|
| **8 MB** | **10.66 ns** | L3 CCD Base | Baseline Zen 4 L3 |
| **16 MB** | **10.95 ns** | L3 CCD Base | Baseline Zen 4 L3 |
| **24 MB** | **10.83 ns** | L3 CCD Base | Baseline Zen 4 L3 |
| **32 MB** | **11.04 ns** | L3 CCD Base Boundary | Upper limit of standard Zen 4 L3 |
| **48 MB** | **11.19 ns** | **3D V-Cache** | *Non-X3D CPUs spill to DRAM here* |
| **64 MB** | **11.35 ns** | **3D V-Cache** | Flat ~11 ns latency sustained |
| **80 MB** | **14.75 ns** | **3D V-Cache** | Approaching L3 capacity boundary |
| **96 MB** | **29.61 ns** | **3D V-Cache Ceiling**| Mixed eviction / OS competition |
| **112 MB** | **38.32 ns** | DDR5 Spillover | Hardware cache boundary crossed |
| **128 MB** | **45.96 ns** | DDR5 Memory Bus | Full DDR5 latency penalty |
| **160 MB** | **52.16 ns** | DDR5 Memory Bus | DRAM bandwidth bound |
| **256 MB** | **60.79 ns** | DDR5 Memory Bus | Saturated DDR5 bus access |

### Key Observation
Between **8 MB and 64 MB**, access latency remains completely flat at **10.6–11.3 ns**. On a standard non-V-Cache processor (such as the Ryzen 7 7700X with 32MB L3), buffers of 48MB and 64MB immediately degrade to >40 ns DRAM access. The 3D V-Cache maintains sub-12 ns memory access for working sets twice the size of conventional L3 caches.

---

## Benchmark 2: SQLite L3 Memory-Mapped Residency

For embedded databases and search indices whose dataset fits under 64MB, configuring SQLite for zero-copy memory mapping keeps the active B-tree and FTS5 inverted index pages hot in L3 SRAM:

```sql
PRAGMA mmap_size = 67108864;  -- 64 MB zero-copy memory map
PRAGMA cache_size = -32768;    -- 32 MB page cache allocation
PRAGMA temp_store = MEMORY;    -- Ephemeral structures held in RAM
PRAGMA synchronous = NORMAL;
```

### Linux PMU Telemetry (`perf stat`, 500 FTS5 search queries)

- **L1 Data Cache Load Misses**: Dropped from 6.99M to 5.05M (**-27.7%**)
- **L3 / DRAM Cache Misses**: Dropped from 893k to 794k (**-11.1%**)
- **Kernel System Time**: Reduced from 7.8 ms to 4.9 ms (**-37.2%** with `jemalloc`)
- **Query Latency**: Up to **33% reduction** in total query execution time

---

## Benchmark 3: TLB Mitigation via 2MB HugePages

When traversing large in-memory buffers (64MB–96MB) using standard 4KB pages, page table traversal requires up to 24,576 TLB entries, exceeding the hardware L2 dTLB (2,048–3,072 entries on Zen 4) and causing repetitive page walks.

By advising the kernel via `madvise(addr, size, MADV_HUGEPAGE)`:
- Working sets of 64MB require only **32 entries** in the TLB.
- **dTLB-load-misses**: Dropped by **98.1%** (from 899,498 to 17,170).
- **Kernel sys time**: Reduced by **65.5%** (from 22.9 ms to 7.9 ms).

---

## Production Gotchas & Engineering Trade-offs

When designing software around L3 cache residency, several operational trade-offs must be evaluated:

1. **SQLite `mmap` SIGBUS Hazard**:
   If an external process truncates or overwrites a memory-mapped database file, or if underlying storage encounters an I/O fault, the kernel raises a `SIGBUS` signal. In multi-process write-heavy environments, `PRAGMA mmap_size` should be used with read-only replicas or tightly controlled single-writer processes.
2. **Pseudo-LRU Eviction (No True Hardware Lock)**:
   Modern x86 hardware does not provide explicit userspace pinning of cache lines without platform-specific QoS features (like AMD Platform QoS / Intel CAT). If background system daemons or heavy interrupts thrash memory, L3 lines will be evicted to DRAM. Working sets should be sized conservatively below the 96MB ceiling (~64MB is the empirical sweet spot).
3. **Transparent Huge Pages (THP) Compaction Spikes**:
   While 2MB HugePages drastically reduce TLB misses, runtime page allocation under fragmented memory can trigger kernel memory compaction (`khugepaged`), introducing p99 latency spikes. For predictable latency, pre-allocate buffers using `MADV_HUGEPAGE` or static `hugetlbfs`.

---

## Reproducing the Benchmarks

### Prerequisites
- Linux running on AMD Zen 4/5 X3D processor (e.g., Ryzen 7 7800X3D, 9800X3D).
- `gcc` with AVX-512 and `-march=native` support.
- Python 3.8+ with `sqlite3`.
- `perf` (Linux tools) for hardware PMU telemetry.

### Running the Suite
```bash
git clone <repo-url>
cd ryzen-7800x3d-vcache-benchmarks
chmod +x reproduce.sh
./reproduce.sh
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
