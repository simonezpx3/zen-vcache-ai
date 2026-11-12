# zen-vcache-ai

**AMD Zen 4/5 3D V-Cache Local AI Acceleration, Hardware PMU Telemetry & Hybrid MoE Benchmarks**

```text
Target CPU     : AMD Ryzen 7 7800X3D (8C/8T BIOS SMT-Off, Monolithic Zen 4 CCD)
L3 Cache       : 96 MiB Unified SRAM (32 MiB Base + 64 MiB 3D V-Cache | ~11 ns latency)
Target GPU     : NVIDIA GeForce RTX 3070 (8 GB GDDR6, PCIe 4.0 x16)
Memory         : Dual-Channel DDR5-6000 (63.5 GB/s AVX-512 streaming read ceiling)
Platform       : Arch Linux (Linux 6.17+ / znver4 / AVX-512 VBMI + VNNI + VPOPCNTDQ)
License        : Apache-2.0 (see LICENSE and NOTICE)
```

---

## 01. Key Empirical Findings

This repository consolidates our microarchitectural C/Python benchmark suite (`benchmarks/`, `results/`) and installable tooling (`bin/`, `src/vcache/`, `kernels/`) for running local AI workloads on consumer 8 GB VRAM + 96 MiB 3D V-Cache desktops.

1. **The 64 MiB L3 Residency Ceiling & 2MB THP Requirement:**
   With 2MB Transparent HugePages (`THP=always`), L3 random pointer-chase latency stays flat at **~11.0–11.6 ns** up to **64 MiB**. Above 64 MiB, dTLB misses and 16-way associative eviction push latency toward DRAM (~39.6 ns at 128 MiB).
2. **8-Thread OpenMP AVX-512 Vector Scan Inside L3 (`<= 61 MiB`):**
   Exhaustive 512-bit Hamming (`VPOPCNTDQ`) scans fit inside L3 up to 1,000,000 vectors (**61.04 MiB**) and scale cleanly across all 8 cores:
   - **100k vectors (6.10 MiB):** `112.60 µs` (1T) → **`15.86 µs`** (8T, **63,051.7 QPS**, **375.82 GB/s**)
   - **500k vectors (30.52 MiB):** `576.98 µs` (1T) → **`77.36 µs`** (8T, **12,926.2 QPS**, **385.23 GB/s**)
   - **1M vectors (61.04 MiB):** `1,165.87 µs` (1T) → **`162.76 µs`** (8T, **6,143.9 QPS**, **366.21 GB/s**)
3. **Hybrid MoE Offloading (8 GB VRAM + DDR5) — `6C (2-7)` + `--moe-cache-mib`:**
   On a monolithic single-CCD 7800X3D, cores `4-7` have **no privileged L3 access** over `0-3`. Under desktop isolation (leaving cores `0-1` for OS/compositor), **6 cores (`2-7`)** is the sharp optimum for DDR5-spilling MoE experts, and caching hot experts in leftover VRAM via `--moe-cache-mib` pushes **35B MoE to 76.3 tok/s** on an 8 GB RTX 3070.
4. **N-Gram Speculative Decoding vs. Untrained Micro-Draft (`SpecOPD` POC):**
   Algorithmic N-Gram matching (`ngram-mod` / `ngram-simple`, 0 MiB extra VRAM) accelerates GPU-resident inference (**+19.6 %** to **4.53×** on warm code refactoring) independently of L3 capacity. Conversely, running an un-trained submatrix-distilled 12M `Q8_0` draft model (`draft-simple` on CPU) degrades throughput (`43.4–45.2 tok/s` vs `121.8 tok/s` baseline) due to low draft acceptance and 151k-vocab LM-head overhead.

---

## 02. Empirical Benchmarks

### A. Hybrid GPU+CPU MoE Scaling: Core Count (`4C` vs `6C` vs `7C` vs `8C`)

Measured via `llama-bench -ngl 99 -ot "\.ffn_.*_exps\.=CPU" -p 128 -n 64 -r 3` under `vcache-run`:

| Model | `4C (4-7)` `tg64` | `6C (2-7)` `tg64` | `7C (1-7)` `tg64` | `8C (0-7)` `tg64` | Optimal (`6C` vs `4C`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 26B.A4B MXFP4 MoE** (`15.40 GiB`) | `38.18 ± 0.23 t/s` | **`40.54 ± 0.21 t/s`** | `38.69 ± 1.26 t/s` | `37.11 ± 1.33 t/s` | **+6.2 %** (`pp128`: **+9.2 %**) |
| **Qwen 3.8 35B.A3B Q3_K_M** (`16.44 GiB`) | `59.45 ± 0.73 t/s` | **`64.97 ± 0.76 t/s`** | `63.14 ± 1.80 t/s` | `62.77 ± 2.92 t/s` | **+9.3 %** (`pp128`: **+4.5 %**) |

### B. VRAM Hot-Expert Cache (`--moe-cache-mib`) on 8 GB RTX 3070 (`6C 2-7`)

Measured via `llama-cli -ngl 99 -t 6 -n 48..64`:

| Model | `-ot` (All Experts CPU) | `--moe-cache-mib 2048` | `--moe-cache-mib 3072` | `--moe-cache-mib 4096` | `--moe-cache-mib 4608` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 26B.A4B** (`c=8192, q8_0`) | `35.5 t/s` | `31.4 t/s` | **`40.1 t/s`** (+13.0 %) | OOM | OOM |
| **Qwen 3.8 35B.A3B** (`c=4096`) | `60.8 t/s` | `55.5 t/s` | `63.1 t/s` | **`72.4 t/s`** (+19.1 %) | **`76.3 t/s`** (+25.5 %) |

### C. Long-Context (`pp4096`) KV Cache Quantization (`Gemma 4 26B.A4B`, `6C`)

| KV Cache (`K / V`) | `pp4096` | `tg64` | Notes |
| :--- | :---: | :---: | :--- |
| `f16 / f16` | `825.89 t/s` | `40.51 t/s` | Baseline precision |
| **`q8_0 / q8_0`** | **`822.81 t/s`** | **`40.43 t/s`** | **~0 % speed penalty, 50 % KV VRAM** |
| `q4_0 / q4_0` | `798.59 t/s` | `40.21 t/s` | -3.3 % prefill dequant overhead |

### D. Unified L3 Sensitivity Test: Clean L3 vs. Concurrent 64 MiB Thrasher (Cores `0-1`)

Testing all three workload classes against an active C memory thrasher evicting the shared 96 MiB L3:

| Workload | Clean L3 | + 64 MiB L3 Thrasher | Impact |
| :--- | :---: | :---: | :--- |
| **16 MiB Buffer (`LsDmndFillsFromSys` PMU)** | `0.05 %` miss rate | `57.05 %` miss rate | **1,141× jump in L3 misses** |
| **500k Vector Scan (`30.52 MiB`, 8T AVX-512)** | `80.86 µs` (`368.56 GB/s`) | `4,010.78 µs` (`7.43 GB/s`) | **49.6× slower** (L3-bound) |
| **GPU N-Gram Speculative (`MiMo 9B` RTX 3070)** | `80.7 tok/s` | `78.4 tok/s` | **-2.8 %** (L3-insensitive) |

### E. Speculative Drafting Attribution (`Qwen3-4B Q4_K_M`, `--reasoning-budget 0`)

| Scenario | Baseline (`none`) | `ngram-mod` (0 MiB VRAM) | `SpecOPD 12M Q8_0` (`draft-simple`, CPU) |
| :--- | :---: | :---: | :---: |
| **Code Refactoring** | `121.8 tok/s` | **`127.3 tok/s`** | `43.4 tok/s` |
| **Greenfield Rust** | `119.7 tok/s` | **`120.6 tok/s`** | `45.2 tok/s` |

> **Note on `SpecOPD` (`src/vcache/specopd.py`, `kernels/l3_specopd_kernel.c`, `bin/vcache-distill-specopd`):**
> The SpecOPD proposal-tree AVX-512 kernel, `LsDmndFillsFromSys` (`0x0243` / `0x0843`) hardware PMU watchdog, and GGUF distillation script are **standalone C/Python prototypes and unit-tested modules** — not an upstream `llama.cpp` training/integration patch. Without on-policy distillation training, submatrix truncation yields near-zero draft acceptance in `llama.cpp`.

> **Note on BIOS SMT:** All measurements on the reference workstation were recorded with `SMT=Off` in UEFI/BIOS (`/sys/devices/system/cpu/smt/control` = `notsupported`).

---

## 03. Repository Layout

```text
zen-vcache-ai/
├── bin/
│   ├── vcache-run               # Unprivileged CPU affinity + jemalloc + hugetlb wrapper
│   ├── vcache-spec-bench        # Speculative decoding benchmark CLI
│   └── vcache-distill-specopd   # Prototype Qwen3 12M Q8_0 micro-draft GGUF extractor
├── src/vcache/
│   ├── topology.py              # Sysfs L3 3D V-Cache & AVX-512 flag inspector
│   ├── vector.py                # ctypes wrapper for OpenMP AVX-512 vector kernels
│   ├── specopd.py               # ProposalTree, E2E-TV, and Zen 4 PMU L3CacheWatchdog
│   └── benchmark.py             # llama-cli speculative benchmark harness
├── kernels/
│   ├── l3_vector_kernel.c       # OpenMP AVX-512 VPOPCNTDQ / INT8 vector search
│   ├── l3_specopd_kernel.c      # AVX-512 tree attention, CLFLUSHOPT sanitizer & PMU watchdog
│   └── Makefile                 # -O3 -march=znver4 -fopenmp shared library build
├── benchmarks/                  # Standalone C/Python L3 & DDR5 microbenchmarks (1–13)
├── examples/                    # Lockless /dev/shm IPC & Zstd L2/L3 stream examples
├── results/                     # Saved raw outputs from Benchmarks 1–13
├── tests/                       # Unit tests (topology, vector, specopd, benchmark)
├── reproduce.sh                 # Automated C/Python microbenchmark reproduction script
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── NOTICE
```

---

## 04. Quick Start & Verification

```bash
# 1. Build OpenMP AVX-512 + Zen 4 PMU shared library
make -C kernels

# 2. Run unit tests
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 3. Inspect CPU topology, L3 cache domains, and AVX-512 flags
PYTHONPATH=src python3 -m vcache.topology

# 4. Run 1T vs 8T AVX-512 L3 vector scan benchmark (100k, 500k, 1M vectors)
PYTHONPATH=src python3 -m vcache.vector

# 5. Launch hybrid MoE inference pinned to optimal 6 cores (2-7) with jemalloc + hugetlb
./bin/vcache-run -c 2-7 -- llama-server -m model.gguf -ngl 99 -cmoe --moe-cache-mib 4096 -t 6

# 6. Reproduce standalone C/Python microbenchmarks (1–13)
./reproduce.sh
```
