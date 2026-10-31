# Cache-Resident Working Sets: Exploiting Zen 4 96MB V-Cache for Sub-Millisecond SQLite & Edge Micro-AI

Hardware PMU telemetry and benchmarks demonstrating low-latency memory residency on the AMD Ryzen 7 7800X3D (Zen 4, 8C/16T, 96MB unified L3 SRAM).

---

## Executive Summary

AMD's 3D V-Cache is widely recognized for accelerating cache-sensitive gaming workloads. In systems programming and edge database engineering, however, the **monolithic 96MB L3 SRAM pool** on the Ryzen 7 7800X3D offers a distinctive advantage: an entire active working set—including database pages, index structures, and INT8 model weights—can reside entirely within L3 cache with ~10–11 ns access latency and ~1.3 TB/s aggregate bandwidth on a single 8C CCD, minimizing steady-state DRAM memory traffic and bypassing the ~70 ns DDR5 bus latency stalls.

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

## Practical Applications & Architecture Patterns

Beyond raw synthetic benchmarks, keeping targeted working sets under the 64MB–96MB L3 boundary unlocks distinct architectural patterns in production systems:

### 1. Zero-VRAM Co-existence for Multi-Model Systems
In local AI workstations where a discrete GPU's VRAM is fully saturated by primary generative workloads (such as diffusion pipelines or large-context language models), allocating auxiliary tasks to the GPU leads to Out-Of-Memory (OOM) errors or device memory thrashing.
- By quantizing auxiliary micro-models (such as MiniLM-L6-v2, MobileBERT, or domain classification heads) to INT8, total parameter footprint drops to 20–35 MB (note: larger base models like ModernBERT-base with ~149M parameters require ~150 MB in INT8 and exceed single-CCD L3 capacity).
- The lightweight model weights and small-batch activation buffers fit entirely within the 96MB L3 SRAM pool of the Ryzen 7 7800X3D.
- Utilizing Zen 4 AVX-512 VNNI instructions, inference executes in sub-2ms latencies directly on the CPU while minimizing steady-state DRAM memory bus stalls and avoiding contention for discrete GPU VRAM.

### 2. Sub-Millisecond Embedded Retrieval & Agent State
Local agentic frameworks, CLI utilities, and background daemons frequently query metadata tables, token registries, and full-text search indices.
- Standard file-backed SQLite queries introduce `read()` syscall overhead, kernel context switches, and redundant memory copies into user-space buffers.
- Setting `PRAGMA mmap_size = 67108864` allows the entire database file (under 64MB) to be memory-mapped directly into the process address space.
- B-tree lookups and FTS5 inverted index traversals execute directly from L3 SRAM cache lines at ~10–11 ns access latency, resulting in sub-millisecond query completions with zero disk I/O.

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

## Reproducing the Benchmarks

### Prerequisites
- Linux running on AMD Zen 4/5 X3D processor (e.g., Ryzen 7 7800X3D, 9800X3D).
- `gcc` with AVX-512 and `-march=native` support.
- Python 3.8+ with `sqlite3`.
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
5. **AMD Zen 4 Microarchitecture & Optimization** — [WikiChip Zen 4 Architecture Reference](https://en.wikichip.org/wiki/amd/microarchitectures/zen_4) (Comprehensive hardware analysis of Zen 4 core pipeline, L1/L2/L3 cache latencies, dTLB hierarchies, and dual-pumped AVX-512 VNNI execution; cross-referencing AMD Publication #57647).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
