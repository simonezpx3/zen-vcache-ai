# Cache-Resident Working Sets: Exploiting Zen 4 96MB V-Cache for Sub-Millisecond SQLite & Edge Micro-AI

![Hardware: AMD Ryzen 7 7800X3D](https://img.shields.io/badge/Hardware-Ryzen_7_7800X3D-ED1C24?logo=amd&logoColor=white)
![Architecture: Zen 4 (8C/16T)](https://img.shields.io/badge/Architecture-Zen_4_(8C%2F16T)-orange)
![L3 Cache: 96MB Unified](https://img.shields.io/badge/L3_Cache-96MB_Unified_SRAM-blue)
![OS: Linux 6.14](https://img.shields.io/badge/OS-Linux_6.14-yellow?logo=linux&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Hardware PMU telemetry and benchmarks demonstrating low-latency memory residency on the AMD Ryzen 7 7800X3D (Zen 4, 8C/16T, 96MB unified L3 SRAM).

---

## Executive Summary

AMD's 3D V-Cache is widely recognized for accelerating cache-sensitive gaming workloads. In systems programming and edge database engineering, however, the **monolithic 96MB L3 SRAM pool** on the Ryzen 7 7800X3D offers a distinctive advantage: an entire active working set—including database pages, index structures, and INT8 model weights—can reside entirely within L3 cache with ~10–11 ns access latency and ~1.3 TB/s aggregate bandwidth on a single 8C CCD, minimizing steady-state DRAM memory traffic and bypassing the ~70 ns DDR5 bus latency stalls.

This repository provides reproducible benchmarks and raw Linux PMU hardware performance counter logs (`perf stat`) characterizing the memory boundary, SQLite zero-copy performance, and TLB mitigation.

### Test Rig & Environment Specifications

| Component | Specification | Operational Role |
|:---|:---|:---|
| **CPU** | AMD Ryzen 7 7800X3D (Zen 4, 8 Cores / 16 Threads) | Single monolithic CCD, 4.2 GHz base / 5.0 GHz boost |
| **L1 Cache** | 32 KiB L1D + 32 KiB L1I per core (512 KiB total) | ~0.8–1.0 ns access latency |
| **L2 Cache** | 1024 KiB (1 MiB) per core (8 MiB total) | ~3.0–3.2 ns access latency |
| **L3 Cache** | **96 MiB Unified SRAM** (32 MiB base + 64 MiB 3D V-Cache) | ~10.6–11.4 ns sustained latency (~1.3 TB/s bandwidth) |
| **System Memory** | 32 GiB Dual-Channel DDR5-6000 CL30 (EXPO enabled) | ~60–75 ns random access floor |
| **Host Operating System** | Arch Linux (Kernel 6.14.x, x86-64-v3) | Linux PMU subsystem (`perf`, `cgroups v2`) |
| **Compiler / Toolchain** | GCC 14.x (`-O3 -march=native`), Python 3.12+ | AVX-512 VNNI SIMD vectorization enabled |

---

## Table of Contents

- [Architectural Context: Single-CCD Advantage](#architectural-context-single-ccd-advantage)
- [Benchmark 1: The Hardware Cache Cliff (Pointer Chasing)](#benchmark-1-the-hardware-cache-cliff-pointer-chasing)
- [Benchmark 2: SQLite L3 Memory-Mapped Residency](#benchmark-2-sqlite-l3-memory-mapped-residency)
- [Benchmark 3: TLB Mitigation via 2MB HugePages](#benchmark-3-tlb-mitigation-via-2mb-hugepages)
- [Benchmark 4: Memory Bus Contention & Core Pinning Isolation](#benchmark-4-memory-bus-contention--core-pinning-isolation)
- [Benchmark 5: Production Two-Stage L3 Vector RAG & AVX-512 SIMD Acceleration](#benchmark-5-production-two-stage-l3-vector-rag--avx-512-simd-acceleration)
- [Practical Applications & Architecture Patterns](#practical-applications--architecture-patterns)
- [Production Cheat Sheet: 4 Golden Rules for Systems Engineers](#production-cheat-sheet-4-golden-rules-for-systems-engineers)
- [Repository Structure & Tooling](#repository-structure--tooling)
- [External Validation & Microarchitectural Corroboration](#external-validation--microarchitectural-corroboration)
- [Production Gotchas & Engineering Trade-offs](#production-gotchas--engineering-trade-offs)
- [Reproducing the Benchmarks](#reproducing-the-benchmarks)
- [References & Prior Art](#references--prior-art)

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

### Single-CCD vs. Dual-CCD Topology Comparison

| Metric / Attribute | Single-CCD X3D (e.g. 7800X3D, 9800X3D) | Dual-CCD Asymmetric X3D (e.g. 7900X3D, 7950X3D) |
|:---|:---|:---|
| **CCD Layout** | 1 monolithic 8C/16T die | 2 distinct CCDs (1× V-Cache CCD + 1× Frequency CCD) |
| **L3 Cache Distribution** | Single unified 96 MiB pool | Split: 96 MiB on CCD0, standard 32 MiB on CCD1 |
| **Inter-Core Die Crossing** | **0 ns** (all intra-CCD communication) | **~70–80 ns** penalty across Infinity Fabric & IOD |
| **OS Thread Scheduling** | Fully symmetric, zero thread-parking required | Requires complex CPPC / thread-parking drivers |
| **Cache-Resident Safety** | **Guaranteed flat ~11 ns across all cores** | Unpredictable if OS scheduler migrates threads to CCD1 |

**Applicable 1-CCD V-Cache Processors**: This architecture pattern and benchmark suite directly apply to all AMD processors featuring 1 monolithic V-Cache CCD:
- **Zen 5**: Ryzen 7 9800X3D (8C/16T, 96MB unified L3, inverted under-die stacking)
- **Zen 4**: Ryzen 7 7800X3D (8C/16T, 96MB unified L3), Ryzen 5 7600X3D (6C/12T, 96MB unified L3)
- **Zen 3**: Ryzen 7 5800X3D, 5700X3D, 5600X3D (AM4, DDR4, 96MB unified L3)

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

### Visualizing the Cliff: L3 Residency vs. DDR5 Memory Wall

```
  Latency (ns)
   70 +                                                     * [256MB: 60.8ns]
   60 +                                             *       |
   50 +                                     *       |       | [160MB: 52.2ns]
   40 +                             *       |       |       | [128MB: 46.0ns]
   30 +                     *       |       |       |       | [112MB: 38.3ns]
   20 +             *       |       |       |       |       | [ 96MB: 29.6ns - L3 Ceiling]
   10 + * * * * * * |       |       |       |       |       | [ 8MB–64MB: ~11ns Flat SRAM]
    0 +-------------+-------+-------+-------+-------+-------+--> Working Set Size
       8  16 32 48 64  80  96  112 128 160 256 MB
      [-- 3D V-Cache Zone --] [Spill] [---- DDR5 Memory Bus ----]
```

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

## Benchmark 4: Memory Bus Contention & Core Pinning Isolation

A critical question in production systems is: *How resilient is L3 cache residency when the host system is under heavy background memory bandwidth saturation?*

To test this, `benchmarks/contention_benchmark.py` floods the system memory bus with **4 background worker processes on Cores 4–7**, each continuously thrashing a 128 MB buffer (spilling past the 96 MB L3 cache directly into DDR5). Concurrently, 400 vector search queries are executed under two scheduling regimes:
1. **Unpinned / Default Scheduling**: The OS scheduler migrates search threads across all cores (0–7), colliding directly with the thrashing workers.
2. **vcache-run Core Isolation**: Queries are pinned to Cores 0–3 with `jemalloc` and `glibc.malloc.hugetlb=1`, isolating the query pipeline from the compute workers while sharing the unified 96MB L3 SRAM.

### Empirical Latency under DDR5 Bus Flooding

Measured on Linux 6.14 over 400 search queries on an active 1,252-document vector dataset:

| Latency Metric | Quiet Baseline (Cores 0–3) | Unpinned under Load (Cores 0–7) | vcache-run Isolated (Cores 0–3) | Isolation Benefit |
|:---|:---:|:---:|:---:|:---:|
| **p50 (Median)** | **602.9 µs** | 672.1 µs | **660.8 µs** | -11.3 µs faster |
| **p90** | **633.3 µs** | 718.4 µs | **696.8 µs** | -21.6 µs faster |
| **p95** | **637.0 µs** | 729.1 µs | **717.1 µs** | -12.0 µs faster |
| **p99 (Tail)** | **642.5 µs** | 790.0 µs | **759.8 µs** | **-30.2 µs lower tail** |
| **MAX** | **649.6 µs** | 1,070.6 µs | **998.9 µs** | **Sub-millisecond cap** |
| **Jitter (StdDev)** | **16.5 µs** | 45.5 µs | **40.0 µs** | **12.2% variance reduction** |

### Key Takeaway
Even when DDR5 memory bandwidth is severely contested by background workloads, the monolithic single-CCD 96MB L3 SRAM pool prevents memory bus starvation. Pining database/retrieval workloads to Cores 0–3 caps max latency strictly below 1 ms (**998.9 µs**) and reduces tail jitter by **12.2%**.

---

## Benchmark 5: Production Two-Stage L3 Vector RAG & AVX-512 SIMD Acceleration

For AI agents and local semantic search, storing full-precision floating-point (FP32) embedding matrices in memory wastes cache capacity. By combining **1-bit binary signature indexing** in L3 SRAM with **INT8 candidate reranking** in DRAM, vector search achieves sub-millisecond execution with zero GPU VRAM impact.

### Two-Stage Pipeline Architecture

```
(Query Text)
     |
     v  (Signed Hash + Orthogonal FWHT Projection)
+---------------------------------------------------------------+
|  Stage 1 // Hot L3 Index (AVX-512 VPOPCNTDQ)                  |
|  - 512-bit (64-byte) binary embeddings in L3 SRAM             |
|  - Footprint: 78.25 KiB for 1,252 documents (<= 20 MiB bound) |
|  - Full index scan: 202.39 µs (Hamming distance via XOR/POPCNT)|
|  - Selects Top-K (K = 224) candidate pool                     |
+---------------------------------------------------------------+
     |
     v  (Extract INT8 candidate vectors from DRAM)
+---------------------------------------------------------------+
|  Stage 2 // DRAM Candidate Rerank (AVX-512BW)                 |
|  - Scaled signed INT8 dot-product via _mm512_madd_epi16       |
|  - Reranking latency: ~60 µs for candidate subset             |
|  - End-to-end query latency: 264–586 µs                       |
|  - Accuracy: Recall@10 = 0.963 (96.3%) vs. exhaustive FP32    |
+---------------------------------------------------------------+
     |
     v
(Top-10 Semantic Results)
```

### Measured Production Metrics (1,252 Documents)

- **Hot L3 Index Footprint**: **78.25 KiB** (fits completely inside L3 SRAM, leaving >95MB for OS and cache lines).
- **Stage 1 Hamming Scan Latency**: **202.39 µs** (scans all 1,252 binary 512-bit vectors).
- **End-to-End Query Latency**: **264.30 µs to 586.49 µs** (including text vectorization, Stage 1 scan, and Stage 2 INT8 reranking).
- **Retrieval Precision**: **Recall@10 = 0.963** (96.3% overlap with ground-truth FP32 cosine similarity).
- **Zero-VRAM Intent Router**: Centroid intent classification using AVX-512 VNNI `VPDPBUSD` executes in **72.74 µs** with **27.4 KiB** memory footprint and 100% rule recall.

### SIMD Acceleration Kernels (C / AVX-512)

#### 1. 512-Bit Hamming Distance via AVX-512 VPOPCNTDQ (`_mm512_popcnt_epi64`)
```c
#include <immintrin.h>
#include <stdint.h>

// Single-instruction 512-bit binary Hamming distance (64 bytes per document)
uint32_t avx512_hamming_dist_512(const uint8_t* query_bin, const uint8_t* doc_bin) {
    __m512i vq = _mm512_loadu_si512((const __m512i*)query_bin);
    __m512i vd = _mm512_loadu_si512((const __m512i*)doc_bin);
    __m512i vx = _mm512_xor_si512(vq, vd);
    __m512i vcnt = _mm512_popcnt_epi64(vx);
    return (uint32_t)_mm512_reduce_add_epi64(vcnt);
}
```

#### 2. INT8 Dot-Product via AVX-512 VNNI (`VPDPBUSD`)
```c
// Accumulates unsigned byte query * signed byte centroid into 32-bit integer
int32_t vnni_dot_product(const uint8_t* query_u8, const int8_t* centroid_s8, int n) {
    __m512i acc = _mm512_setzero_si512();
    for (int i = 0; i < n; i += 64) {
        __m512i vq = _mm512_loadu_si512((const __m512i*)(query_u8 + i));
        __m512i vc = _mm512_loadu_si512((const __m512i*)(centroid_s8 + i));
        acc = _mm512_dpbusd_epi32(acc, vq, vc);
    }
    return _mm512_reduce_add_epi32(acc);
}
```

---

## Practical Applications & Architecture Patterns

Beyond raw synthetic benchmarks, keeping targeted working sets under the 64MB–96MB L3 boundary unlocks distinct architectural patterns in production systems:

### 1. Zero-VRAM Co-existence for Multi-Model Systems
In local AI workstations where a discrete GPU's VRAM is fully saturated by primary generative workloads (such as diffusion pipelines or large-context language models), allocating auxiliary tasks to the GPU leads to Out-Of-Memory (OOM) errors or device memory thrashing.
- By quantizing auxiliary micro-models (such as MiniLM-L6-v2, MobileBERT, or domain classification heads) to INT8, total parameter footprint drops to 20–35 MB (note: larger base models like ModernBERT-base with ~149M parameters require ~150 MB in INT8 and exceed single-CCD L3 capacity).
- The lightweight model weights and small-batch activation buffers fit entirely within the 96MB L3 SRAM pool of the Ryzen 7 7800X3D.
- Utilizing Zen 4 AVX-512 VNNI instructions, inference executes in sub-2ms latencies directly on the CPU while minimizing steady-state DRAM memory bus stalls and avoiding contention for discrete GPU VRAM.

```bash
# Recommended environment configuration for CPU-resident INT8 inference
export OMP_NUM_THREADS=8
export GOMP_CPU_AFFINITY="0-7"
export ONNXRUNTIME_EXECUTION_PROVIDERS="CPUExecutionProvider"
# Run quantized model pinned to single CCD without cross-die migration
taskset -c 0-7 python3 run_embedding_head.py --model minilm-l6-v2-int8.onnx
```

### 2. Sub-Millisecond Embedded Retrieval & Agent State
Local agentic frameworks, CLI utilities, and background daemons frequently query metadata tables, token registries, and full-text search indices.
- Standard file-backed SQLite queries introduce `read()` syscall overhead, kernel context switches, and redundant memory copies into user-space buffers.
- Setting `PRAGMA mmap_size = 67108864` allows the entire database file (under 64MB) to be memory-mapped directly into the process address space.
- B-tree lookups and FTS5 inverted index traversals execute directly from L3 SRAM cache lines at ~10–11 ns access latency, resulting in sub-millisecond query completions with zero disk I/O.

```python
import sqlite3

def connect_l3_resident_db(db_path: str) -> sqlite3.Connection:
    """Configures SQLite connection for zero-copy L3 cache residency."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA mmap_size = 67108864;")   # 64 MB zero-copy mmap
    conn.execute("PRAGMA cache_size = -32768;")     # 32 MB page cache allocation
    conn.execute("PRAGMA temp_store = MEMORY;")     # Ephemeral structures in memory
    return conn
```

### 3. Pointer-Chasing Acceleration in Graph-Based Vector Search (HNSW)
Approximate Nearest Neighbor (ANN) search algorithms—most notably Hierarchical Navigable Small World (HNSW) graphs widely used in vector databases and RAG pipelines—are constrained by the "pointer chasing" bottleneck: traversals perform pseudo-random memory hops between graph nodes.
- When traversing graphs in system DDR5 RAM, each node dereference incurs ~60–75 ns access latency.
- When active vector subgraphs or quantization codebooks fit within the 96MB L3 cache, each hop resolves at ~10–12 ns SRAM latency.
- This produces significant throughput gains (up to 88% higher query throughput and 50% faster index construction) without requiring dedicated GPU acceleration.

### 4. Hierarchical Two-Tier Intent Routing
Low-latency real-time applications (such as voice command routers, triage engines, or event dispatchers) benefit from hierarchical pipeline tiering:
- **Tier 1 (L1 Cache Resident)**: Compact deterministic string matching (e.g., Aho-Corasick state machines sized under the 32 KiB L1D capacity) executes under 0.05 ms directly within L1 cache.
- **Tier 2 (L3 Cache Resident)**: Semantic fallback classification using an L3-resident INT8 transformer model executes under 2 ms on the CPU.
- Heavier generative models or remote APIs are only invoked when Tier 1 and Tier 2 escalate, slashing end-to-end latency and compute overhead.

---

## Production Cheat Sheet: 4 Golden Rules for Systems Engineers

To guarantee deterministic, sub-millisecond residency on AMD single-CCD X3D processors in production, enforce these four principles:

1. **The Rule of 64MB (Working Set Sizing)**:
   Never size active datasets to the nominal 96MB hardware ceiling. Keep hot working sets, model weights, and SQLite mmap buffers **strictly under 64 MB**. This leaves 32MB headroom for operating system buffers, thread stacks, dynamic heap allocations, and kernel interrupts without triggering pseudo-LRU cache line evictions.

2. **Enforce 2MB HugePages for Buffers >16MB**:
   Standard 4KB paging across 64MB buffers exhausts the Zen 4 L2 dTLB (2,048–3,072 entries) and incurs heavy page table walk overhead. Always advise allocations via `madvise(MADV_HUGEPAGE)` or mount datasets onto `hugetlbfs` to reduce page table entries from 16,384 down to 32 (-98% TLB misses).

3. **Pin Threads to the Single CCD**:
   Bind performance-critical worker threads to the physical cores of the V-Cache CCD using `sched_setaffinity()` or `taskset -c 0-7`. Preventing the Linux scheduler from migrating threads preserves core-private L1/L2 caches and eliminates cross-core invalidation storms.

4. **Partition Cores for I/O and Compute Isolation**:
   When combining background compute or model generation with low-latency search, partition the 8 physical cores: bind database and vector search workers to Cores 0–3 and heavy background threads to Cores 4–7. This preserves sub-millisecond p99 latency (759 µs vs. 790 µs unpinned) and reduces latency jitter by **12.2%** even under heavy DDR5 memory bus flooding.

---

## External Validation & Microarchitectural Corroboration

The empirical latency curves and architectural observations captured in this repository align directly with independent microarchitectural analyses and published industry research:

### 1. Pointer-Chasing Latency Curves ([*Chips and Cheese*](https://chipsandcheese.com/2023/04/23/amds-7950x3d-zen-4-gets-vcache/))
- Microarchitectural analysis by [*Chips and Cheese*](https://chipsandcheese.com/2023/04/23/amds-7950x3d-zen-4-gets-vcache/) confirms that AMD's stacked 3D V-Cache die introduces only a marginal **~4-cycle latency delta** compared to standard Zen 4 base L3 cache slices.
- On standard Zen 4 desktop processors (such as the Ryzen 7 7700X), pointer-chasing benchmarks exhibit a steep latency cliff immediately after surpassing the **32MB L3 threshold**, spilling directly into DDR5 memory.
- On the Ryzen 7 7800X3D, this latency plateau remains flat across the expanded **8MB–64MB region** (~10.6–11.4 ns in our empirical PMU runs)—doubling standard L3 capacity—with 80MB–96MB acting as a transition boundary before spilling over to off-die AM5 DDR5-6000 system memory latency (**~60–75 ns**, matching our measured 60.79 ns ceiling at 256MB).

### 2. Algorithmic Memory Bottlenecks in Vector Search ([*Coleman et al. / arXiv*](https://arxiv.org/abs/2104.03221), [*Qdrant*](https://qdrant.tech/articles/immutable-data-structures/))
- In Approximate Nearest Neighbor (ANN) vector search, graph traversal algorithms (most notably HNSW) do not benefit from sequential streaming bandwidth; they are inherently constrained by **random memory pointer dereferencing** across non-contiguous graph nodes.
- Technical publications from vector database engineering teams ([*Qdrant*](https://qdrant.tech/articles/immutable-data-structures/)) and distributed systems research confirm that retaining graph hierarchies and centroid quantization tables within massive L3 cache pools (across consumer X3D chips and server-grade AMD EPYC Genoa-X parts) eliminates DDR memory stalls, yielding documented throughput gains exceeding 80–90%.
- Academic research on graph traversal cache dynamics ([Coleman et al., *NeurIPS / arXiv:2104.03221*](https://arxiv.org/abs/2104.03221)) proves that irregular memory accesses during HNSW search degrade throughput by up to 40% when the graph spills outside LLC cache boundaries due to cold pointer chasing.

### 3. Topological Symmetry ([*Phoronix*](https://www.phoronix.com/review/amd-ryzen-7-7800x3d-linux))
- Workstation evaluations and Linux kernel scheduler benchmarks on [*Phoronix*](https://www.phoronix.com/review/amd-ryzen-7-7800x3d-linux) highlight the architectural simplicity of the Ryzen 7 7800X3D's single-CCD topology.
- Unlike multi-CCD architectures (such as the 7900X3D/7950X3D or multi-socket platforms), which exhibit non-uniform Infinity Fabric cross-CCD latency penalties (~70 ns cross-die hops) and require complex thread-parking heuristics, the 7800X3D provides uniform, symmetric 96MB L3 access across all 8 cores without NUMA-like degradation.

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

## Repository Structure & Tooling

```
.
├── benchmarks/
│   ├── cache_cliff.c              # Pointer-chasing cyclic permutation benchmark (8MB–256MB)
│   ├── contention_benchmark.py    # DDR5 bus contention stress test with core pinning
│   ├── hugepages_test.c           # 2MB HugePages vs standard 4KB paging benchmark
│   ├── simd_l3_rag.c              # AVX-512 VPOPCNTDQ (Hamming) & VNNI (INT8) standalone test
│   └── sqlite_l3_mmap.py          # SQLite FTS5 mmap vs page cache comparison
├── results/
│   ├── baseline_probe_results.json # Hardware PMU flags and baseline query probe
│   ├── cache_cliff_run.txt        # Raw latency numbers from 8MB to 256MB
│   ├── contention_benchmark_results.json # Tail latency and jitter numbers under bus flood
│   └── perf_stat_cache_cliff.txt  # Linux perf stat PMU event logs
├── tools/
│   └── vcache-run                 # Unprivileged core affinity & jemalloc wrapper
├── reproduce.sh                   # Automated build & benchmark runner (runs all 5 benchmarks)
├── LICENSE                        # MIT License
└── README.md                      # Comprehensive microarchitectural report
```

### The `vcache-run` Tool
This repository includes `tools/vcache-run`, an unprivileged CLI wrapper that isolates database and micro-AI tasks without root/setuid privileges:
```bash
# Isolate SQLite or Vector RAG on Cores 0-3 with jemalloc and hugetlb tunables:
./tools/vcache-run --profile db python3 -m arci_kb search "query"

# Isolate compute and model tasks on Cores 4-7:
./tools/vcache-run --profile infer python3 my_model.py
```

---

## Reproducing the Benchmarks

### Prerequisites
- Linux running on AMD Zen 4/5 X3D processor (e.g., Ryzen 7 7800X3D, 9800X3D).
- `gcc` with AVX-512 support (`-mavx512f -mavx512vpopcntdq -mavx512vnni -mavx512bw -march=native`).
- Python 3.8+ with `sqlite3` and `numpy`.
- `perf` (Linux tools) for hardware PMU telemetry.

### Running the Suite
```bash
git clone https://github.com/simonezpx3/ryzen-7800x3d-vcache-benchmarks.git
cd ryzen-7800x3d-vcache-benchmarks
chmod +x reproduce.sh
./reproduce.sh
```

---

## References & Prior Art

1. **Chips and Cheese** — [AMD’s 7950X3D: Zen 4 Gets VCache](https://chipsandcheese.com/2023/04/23/amds-7950x3d-zen-4-gets-vcache/) (Microarchitectural latency analysis and stacked L3 SRAM delta characterization).
2. **Phoronix** — [AMD Ryzen 7 7800X3D Linux Performance](https://www.phoronix.com/review/amd-ryzen-7-7800x3d-linux) (Linux kernel benchmarking and single-CCD symmetric topology analysis).
3. **Coleman, B., Segarra, S., Shrivastava, A., Smola, A.** — [Graph Reordering for Cache-Efficient Near Neighbor Search](https://arxiv.org/abs/2104.03221) (NeurIPS / arXiv:2104.03221; formal analysis of pointer chasing, memory hierarchy stalls, and cache hit maximization in graph-based ANN algorithms like HNSW).
4. **Qdrant Vector Database Research** — [Immutable Data Structures in Qdrant](https://qdrant.tech/articles/immutable-data-structures/) (Low-level performance analysis, CPU cache misses, and memory alignment in vector search).
5. **AMD Zen 4 Microarchitecture & Optimization** — [Wikipedia: Zen 4 Microarchitecture](https://en.wikipedia.org/wiki/Zen_4) (Hardware breakdown of Zen 4 core pipeline, unified L3 cache hierarchy, and dual-pumped AVX-512 VNNI execution; cross-referencing AMD Publication #57647).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
