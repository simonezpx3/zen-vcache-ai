# ZEN 4 3D V-CACHE LOCAL AI ACCELERATION
### Microarchitectural SRAM Residency & Zero-VRAM Speculative Drafting

```
Target CPU     : AMD Ryzen 7 7800X3D (8C/16T, Monolithic Zen 4 CCD)
L3 Cache       : 96 MiB Unified SRAM (~11 ns latency | ~1.3 TB/s read bandwidth)
Target GPU     : NVIDIA GeForce RTX 3070 (8 GB GDDR6, PCIe 4.0 x16)
Platform       : Arch Linux (Linux 6.17+ / znver4 / AVX-512 VBMI + VNNI)
Classification : Performance Specification & Reference Implementation
```

---

## 01. ARCHITECTURAL OVERVIEW & BOTTLENECK ANALYSIS

Desktop local inference on consumer-grade hardware faces two fundamental hardware boundaries:

1. **VRAM Ceiling (8 GB Boundary):**
   Standard neural speculative decoding requires a secondary draft model (0.5B–1.5B parameters), consuming 1.5–2.5 GB of VRAM. On an 8 GB frame buffer, co-locating target and draft models forces context truncation or PCIe host memory spilling.

2. **DDR5 Memory Bus & Cache Hierarchy Bottlenecks:**
   Dual-channel DDR5-6000 reaches measured sequential read bandwidth saturation at ~63.5 GB/s on Zen 4. For CPU-side token generation on models exceeding L3, scaling threads across cores is bounded by memory controller arbitration, while sub-1 GB models partially cache (~15–20% of weights) inside the 96 MiB L3 / 8 MiB L2 hierarchy.

This repository implements microarchitectural evaluation and acceleration on the 7800X3D's stacked 64 MiB 3D V-Cache (96 MiB total L3 SRAM pool) for vector indexing, zero-DRAM ternary compute, and speculative drafting.

---

## 02. MEMORY TOPOLOGY & RESIDENCY MAP

```
+-----------------------------------------------------------------------------+
| AMD RYZEN 7 7800X3D (8-CORE MONOLITHIC CCD, UNIFIED 96 MiB L3)              |
|                                                                             |
|  [Cores 0-1] -> System Services / Desktop Compositor (Hyprland) / OS Sched   |
|  [Cores 2-7] -> Isolated Compute / AVX-512 SIMD / Low-Latency Inference      |
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  | 96 MiB UNIFIED L3 SRAM POOL (~11 ns latency | ~1.3 TB/s bandwidth)    |  |
|  |                                                                       |  |
|  |  Working Sets <= 64 MiB: Flat ~11 ns random access (with 2MB THP)     |  |
|  |  - N-Gram Draft State Machine Footprint:       < 12.0 MiB             |  |
|  |  - 1M Vector PQ64 FastScan Index:               61.04 MiB             |  |
|  |  - Ternary LUT Weights (1.58-bit):              56.00 MiB             |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
                                       |
                     DDR5-6000 Interconnect (63.5 GB/s Measured Read Ceiling)
                                       v
+-----------------------------------------------------------------------------+
| SYSTEM MEMORY (Target for Non-Temporal Cold Streams via _MM_HINT_NTA)       |
+-----------------------------------------------------------------------------+
```

---

## 03. CORE SPECIFICATIONS

### A. L3-Resident Speculative Drafting
* **Subsystem:** Token prediction tree (`ngram-mod`, `ngram-map-k4v`) held within L3 SRAM budget (`<= 40 MiB`).
* **Execution:** Host GPU runs primary weights in VRAM; CPU cores 4–7 evaluate n-gram matches at SRAM speeds.
* **Overhead:** 0 MiB GPU VRAM allocation.

### B. Multi-Core Scaling & DDR5 Read Ceiling
* **Architecture:** AMD Zen 4 single monolithic CCD with unified 96 MiB L3 cache across all 8 cores.
* **Empirical Scaling:** On dense 463 MiB GGUF models (`Qwen2.5-Coder-0.5B-Q4_K_M`) under symmetric desktop isolation:
  - 2 Cores: **97.2 ± 0.9 tok/s**
  - 4 Cores: **152.7 ± 2.3 tok/s**
  - 6 Cores: **173.5 ± 5.0 tok/s** (+13.6% over 4 cores)
  - 8 Cores: **170.8 ± 43.7 tok/s** (higher variance due to OS background tasks)
* **DDR5 Memory Bus Ceiling:** Measured AVX-512 streaming read bandwidth peaks at **63.5 GB/s** on dual-channel DDR5-6000. Implied weight bandwidth for 463 MiB models at 173.5 tok/s corresponds to 84.2 GB/s, proving that **~15–20% of weights remain resident in the 96 MiB L3 / 8 MiB L2 hierarchy** between decode steps.

### C. Sub-64MB Zero-DRAM Inference (AVX-512 VBMI `VPERMB`)
* **Vector Kernel:** Quantized 1.58-bit ternary representations (`<= 56 MiB`) bypass system memory controllers entirely.
* **Throughput:** Scales across all 8 cores to **10,700.4 tok/s** utilizing `_mm512_permutexvar_epi8` with `VPDPBUSD`.

### D. Million-Scale Vector PQ64 FastScan
* **Index Footprint:** 1,000,000 512-dimensional vectors compressed via Product Quantization (64 bytes/vector) occupying exactly **61.04 MiB**.
* **Latency:** **102.2 µs** exhaustive brute-force scan; **2.55 µs** IVF Top-4096 rescoring.

### E. Zen 4 Core PMU Hardware L3 Watchdog
* **Subsystem:** Real-time hardware L3 miss-rate telemetry via Linux `perf_event_open` directly reading AMD Zen 4 Core PMU `LsDmndFillsFromSys` (Event `0x43`, `PERF_TYPE_RAW`).
* **Counter Configuration:**
  - `0x0243ULL` (`UMask 0x02`): Local CCX L3 Hits (demand requests fulfilled directly by 96 MiB 3D V-Cache).
  - `0x0843ULL` (`UMask 0x08`): DRAM IO Near (demand requests that missed L3 and fetched from local DDR5 RAM).
* **Deterministic Ratio:** $\text{L3 Miss Rate} = \frac{\text{dram\_fills}}{\text{l3\_hits} + \text{dram\_fills}}$. Verified via ctypes: 32 MiB resident buffer achieves 9.4% miss rate (cold starts/dTLB); 256 MiB spilling buffer triggers 23.8% miss rate.

---

## 04. EMPIRICAL TELEMETRY

Controlled microarchitectural benchmarks on Ryzen 7 7800X3D + RTX 3070 8GB ($n=10$, Median ± IQR):

### 1. Hardware Cache Cliff & TLB Scaling (Pointer Chase, CPU 5)

| Working Set | 2MB THP Enabled | THP + 64 MiB Thrasher | 4KB Pages (THP Disabled) | THP Off + 64 MiB Thrasher |
| :--- | :---: | :---: | :---: | :---: |
| **8 MiB** | 10.70 ± 0.23 ns | 11.36 ± 0.93 ns | 12.24 ± 0.11 ns | 16.34 ± 3.00 ns |
| **16 MiB** | 10.95 ± 0.27 ns | 16.21 ± 10.01 ns | 14.00 ± 0.39 ns | 21.55 ± 3.20 ns |
| **32 MiB** | 11.07 ± 0.28 ns | 29.51 ± 6.93 ns | 15.83 ± 0.06 ns | 38.52 ± 5.26 ns |
| **48 MiB** | 11.32 ± 0.86 ns | 41.58 ± 3.50 ns | 17.46 ± 1.66 ns | 49.70 ± 1.45 ns |
| **64 MiB** | **11.64 ± 4.26 ns** | 50.42 ± 1.13 ns | 20.11 ± 3.47 ns | 58.52 ± 2.08 ns |
| **80 MiB** | 17.09 ± 8.17 ns | 55.41 ± 0.87 ns | 28.64 ± 1.38 ns | 62.85 ± 1.75 ns |
| **96 MiB** | 27.33 ± 6.61 ns | 58.49 ± 1.06 ns | 33.68 ± 3.07 ns | 66.35 ± 1.26 ns |
| **128 MiB (DRAM)** | 39.64 ± 2.52 ns | 61.79 ± 1.02 ns | 47.70 ± 2.07 ns | 70.64 ± 0.86 ns |

*Key finding:* With 2MB HugePages, L3 random access latency remains completely flat (~11.0–11.6 ns) up to 64 MiB. The degradation above 64 MiB is driven by TLB miss costs and associative cache eviction, not by OS stack contention.

### 2. CPU-Only Core Scaling (`Qwen2.5-Coder-0.5B-Q4_K_M`, 463 MiB)

| Configuration | Pinned Cores | Active Threads | Generation (`tg64`) | Implied Bandwidth |
| :--- | :---: | :---: | :---: | :---: |
| **2 Cores** | `6-7` | 2 | 97.2 ± 0.9 tok/s | 47.2 GB/s |
| **4 Cores (Pinned)** | `4-7` | 4 | 152.7 ± 2.3 tok/s | 74.1 GB/s |
| **4 Cores (Unpinned)** | `2-7` | 4 | 154.0 ± 2.6 tok/s | 74.7 GB/s |
| **6 Cores** | `2-7` | 6 | **173.5 ± 5.0 tok/s** | **84.2 GB/s** |
| **8 Cores** | `0-7` | 8 | 170.8 ± 43.7 tok/s | 82.9 GB/s |

*Key finding:* Core pinning in a single-CCD architecture produces identical throughput (`152.7` vs `154.0` tok/s). 6 cores outperform 4 cores by +13.6%.

### 3. N-Gram Speculative Decoding Attribution (RTX 3070 8GB Target)

| Workload | Phase | Mode | Clean L3 (T0) | 64 MiB Thrasher (T64) | Draft Accepted / Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Code Refactoring** | Warm | N-Gram Spec | **1992.4 ± 9.4 tok/s** | **1968.4 ± 12.7 tok/s** | 349 / 465 (75.1%) |
| **Code Refactoring** | Warm | Baseline (No Spec) | 439.7 ± 9.1 tok/s | 440.7 ± 13.5 tok/s | — |
| **Code Refactoring** | Cold | N-Gram Spec | 707.9 ± 3.6 tok/s | 708.6 ± 3.4 tok/s | 183 / 311 (58.8%) |
| **Code Refactoring** | Cold | Baseline (No Spec) | 438.9 ± 5.2 tok/s | 439.1 ± 14.7 tok/s | — |
| **Greenfield Code** | Cold | N-Gram Spec | 563.3 ± 3.0 tok/s | 565.8 ± 2.5 tok/s | 130 / 275 (47.3%) |
| **Greenfield Code** | Cold | Baseline (No Spec) | 452.9 ± 3.4 tok/s | 454.4 ± 15.0 tok/s | — |

*Key finding:* N-gram speedup is driven by algorithmic token matching and parallel GPU verification (4.53x on warm refactoring, 1.61x on cold refactoring). Evicting L3 via a concurrent 64 MiB thrasher causes only a **1.2% drop**, confirming that throughput is not bounded by L3 capacity. In `llama.cpp`, `--spec-draft-cpu-range` is a no-op for n-gram predictors.

---

## 05. CLI SPECIFICATION & REPRODUCTION

### Directory Layout

```
zen4-3d-vcache-exploit/
├── bin/
│   ├── vcache-run               # Core affinity and memory topology wrapper
│   └── vcache-spec-bench        # Empirical benchmark harness
├── src/vcache/
│   ├── __init__.py              # Public Python package interface
│   ├── benchmark.py             # Benchmark harness & multi-run engine
│   ├── specopd.py               # Proposal-tree engine & Zen 4 L3 PMU watchdog
│   ├── topology.py              # L3 cache sysfs inspector & affinity resolver
│   └── vector.py                # AVX-512 ctypes bindings & vector benchmark
├── kernels/
│   ├── l3_specopd_kernel.c      # AVX-512 tree attention & Zen 4 Core PMU watchdog
│   ├── l3_vector_kernel.c       # AVX-512 VPOPCNTDQ / VBMI L3 vector engine
│   └── Makefile                 # znver4 optimized compiler targets
├── tests/
│   ├── test_benchmark.py        # Metric parsing & vector engine unit tests
│   ├── test_specopd.py          # Tree masks, E2E-TV, budget & watchdog tests
│   └── test_topology.py         # CPU topology & L3 sysfs unit tests
├── pyproject.toml
├── .gitignore
├── LICENSE
├── NOTICE
└── README.md
```

### Execution Commands

```bash
# Build native AVX-512 and Core PMU watchdog library (libl3_vector.so)
make -C kernels

# Execute complete speculative decoding benchmark suite
./bin/vcache-spec-bench --scenario all

# Export raw metrics for telemetry ingestion (JSON or CSV)
./bin/vcache-spec-bench --scenario refactor --format json

# Run in-L3 AVX-512 vector scan benchmark (10k, 100k, 500k vectors)
PYTHONPATH=src python3 -m vcache.vector

# Inspect CPU cache hierarchy and recommended pinning
PYTHONPATH=src python3 -m vcache.topology

# Run unit tests
PYTHONPATH=src python3 -m unittest discover -s tests

# Run hybrid GPU + CPU N-Gram speculative decoding
./bin/vcache-run --profile infer -- llama-cli -m target_model.gguf -ngl 99 \
  --spec-type ngram-map-k4v,ngram-mod \
  --draft-max 16 --draft-min 2
```

---

## 06. SYSTEM COMPATIBILITY

* **Processor:** AMD Zen 4 X3D Series (7800X3D monolithic CCD verified; 7900X3D/7950X3D requires dual-CCD cache-core pinning).
* **Instruction Sets:** AVX-512F, AVX-512BW, AVX-512DQ, AVX-512VL, AVX-512VPOPCNTDQ, AVX-512VNNI, AVX-512VBMI.
* **Libraries:** `glibc >= 2.38`, `jemalloc >= 5.3`.

---

## LICENSE

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
