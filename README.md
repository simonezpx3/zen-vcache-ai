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
4. **N-Gram Speculative Decoding vs. On-Policy Micro-Draft (`SpecOPD` `d64`/`d128`):**
   Algorithmic N-Gram matching (`ngram-mod` / `ngram-simple`, 0 MiB extra VRAM) accelerates repetitive GPU-resident workflows (`+19.6 %` to `4.53×` on warm code refactoring) independently of L3 capacity. Running standalone Qwen3 micro-drafts (`d64`: `10.02 M`, `15.81 MiB`; `d128`: `20.63 M`, `26.56 MiB` Q8_0) inside the 96 MiB 3D V-Cache executes faster on 6 CPU cores (`5,042` / `4,140 tok/s`) than on the RTX 3070 (`3,235` / `3,466 tok/s`). After PyTorch On-Policy SpecOPD training (`122` rollouts, `17,129` tokens), out-of-sample `llama-speculative` acceptance rises from `0.00 %` to `5.00 %` (`d64`) and `13.10 %` (`d128`, `101.2 tok/s` on refactoring), yet remains below the ~45–60 % breakeven required to beat a fast `122 tok/s` `Qwen3-4B` GPU baseline without multi-million-token distillation or EAGLE-style hidden-state reuse.

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

| Scenario | Baseline (`none`) | `ngram-mod` (0 MiB VRAM) | Untrained `12M` (`draft-simple`, CPU) |
| :--- | :---: | :---: | :---: |
| **Code Refactoring** | `121.8 tok/s` | **`127.3 tok/s`** | `43.4 tok/s` |
| **Greenfield Rust** | `119.7 tok/s` | **`120.6 tok/s`** | `45.2 tok/s` |

### F. On-Policy SpecOPD PyTorch Training (`d64` `10.02 M` vs. `d128` `20.63 M` vs. `d192` `31.53 M` Q8_0)

Trained via [`vcache-train-specopd`](bin/vcache-train-specopd) on `122` on-policy rollouts (`17,129` tokens from `Qwen3-4B Q4_K_M`, 50 training prompts) for `d64`/`d128`, and on the expanded `150`-prompt corpus (`--expanded --max-predict 192`: `310` rollouts / `58,749` tokens for `Qwen3-4B`; `412` rollouts / `78,957` tokens for `DeepSeek-R1-0528-Qwen3-8B`, strictly disjoint from evaluation prompts) for `d192`, using GPU-PCA tied embeddings, Stage 1 Top-K Forward KD (`45` epochs), and gradient-accumulated Stage 2 Reverse-KL On-Policy SpecOPD (`accum=16`, damped LR):

1. **Standalone Micro-Draft Roofline (`llama-bench`, `vcache-run -c 2-7 -n 128 -r 3`):**

| Micro-Draft | Params / File (`L3` Payload) | CPU 6C (`-ngl 0`) `tg128` | GPU RTX 3070 (`-ngl 99`) `tg128` | Winner |
| :--- | :---: | :---: | :---: | :--- |
| **`d64` (`L=4, d=64`)** | `10.02 M` / `15.81 MiB` | **`5,042.37 ± 29.03 tok/s`** (`0.20 ms/tok`) | `3,234.52 ± 94.03 tok/s` | **CPU 3D V-Cache (+55.9 %)** |
| **`d128` (`L=4, d=128`)** | `20.63 M` / `26.56 MiB` | **`4,140.33 ± 85.54 tok/s`** (`0.24 ms/tok`) | `3,466.42 ± 44.81 tok/s` | **CPU 3D V-Cache (+19.4 %)** |
| **`d192` (`L=4, d=192`)** | `31.53 M` / `37.61 MiB` (`31.96 MiB` tensors) | **`3,217.65 ± 44.41 tok/s`** (`0.31 ms/tok`) | `3,102.41 ± 67.06 tok/s` | **CPU 3D V-Cache (+3.7 %)** |

2. **Training Convergence & Out-of-Sample `llama-speculative` (`n_max=4`, `n=96`):**

| Model / Mode | Stage 1 `top1_acc` | Stage 2 `prefix_accept` (train) | Refactor (OOS `accept` / `tok/s`) | Chat Rust (OOS `accept` / `tok/s`) | Greenfield (OOS `accept` / `tok/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline `Qwen3-4B Q4_K_M` (`none`, GPU)** | — | — | `—` / **`120.2–122.0 tok/s`** | `—` / **`120.2–121.8 tok/s`** | `—` / **`120.2–121.4 tok/s`** |
| **Trained `d64` -> `Qwen3-4B` (`ngld=0` CPU)** | `95.84 %` | `85.94 %` (`5.16/6`) | `5.00 %` / `76.8 tok/s` | `1.97 %` / `68.7 tok/s` | `0.26 %` / `62.5 tok/s` |
| **Trained `d128` -> `Qwen3-4B` (`ngld=0` CPU)** | `96.53 %` | `89.32 %` (`5.36/6`) | `12.50 %` / `87.9 tok/s` (`94.2` @ `n_max=8`) | `5.77 %` / `81.6 tok/s` | `0.81 %` / `65.5 tok/s` |
| **Trained `d128` -> `Qwen3-4B` (`ngld=99` GPU)** | `96.53 %` | `89.32 %` (`5.36/6`) | `13.10 %` / `101.2 tok/s` | `5.77 %` / `80.3 tok/s` | `3.57 %` / `76.4 tok/s` |
| **Trained `d192` -> `Qwen3-4B` (`ngld=0` CPU)** | `96.34 %` | `94.01 %` (`5.64/6`) | **`18.64 %`** / **`110.8 tok/s`** | **`13.10 %`** / **`96.6 tok/s`** | **`3.92 %`** / `73.7 tok/s` |
| **Trained `d192` -> `Qwen3-4B` (`ngld=99` GPU)** | `96.34 %` | `94.01 %` (`5.64/6`) | **`18.64 %`** / `110.2 tok/s` | `12.50 %` / `95.4 tok/s` | **`3.92 %`** / `73.9 tok/s` |
| **Baseline `DeepSeek-R1-0528-Qwen3-8B` (`none`, GPU)** | — | — | `—` / **`76.0 tok/s`** | `—` / **`76.0 tok/s`** | `—` / **`76.0 tok/s`** |
| **Trained `d192` -> `DeepSeek-R1-8B` (`ngld=0` CPU)** | `92.32 %` | `85.42 %` (`5.12/6`) | `12.50 %` / `61.8 tok/s` | `9.03 %` / `56.1 tok/s` | `4.22 %` / `48.3 tok/s` |
| **Trained `d192` -> `DeepSeek-R1-8B` (`ngld=99` GPU)** | `92.32 %` | `85.42 %` (`5.12/6`) | **`13.10 %`** / `62.8 tok/s` | `9.03 %` / `55.9 tok/s` | **`4.57 %`** / `48.9 tok/s` |

> **Key Takeaways:**
> - **V-Cache Speed Ceiling Verified:** Disproving the 151k LM-head CPU bottleneck hypothesis, `d64`, `d128`, and `d192` (`31.96 MiB` tensor payload) all stay within the 96 MiB L3 and run faster on 6 Ryzen 7800X3D cores (`5,042` / `4,140` / `3,218 tok/s`) than on RTX 3070 CUDA (`3,235` / `3,466` / `3,102 tok/s`).
> - **Generalization vs. Scale:** Expanding capacity to `d192` and rollouts to `58.7k` tokens (`Qwen3-4B`) lifted out-of-sample acceptance from `13.10 %` to `18.64 %` (`110.8 tok/s`) on Refactor and more than doubled Chat Rust (`5.77 %` → `13.10 %`, `96.6 tok/s`), yet `59k–79k` tokens still remain below the ~45–60 % acceptance breakeven needed to beat GPU baselines (`120.2` / `76.0 tok/s`) without multi-million-token distillation or EAGLE-style hidden-state reuse.
> - **Byte-Exact Tokenizer Cloning:** Cloning raw token/merge bytes and teacher special-token metadata (`tokenizer.ggml.model`, `pre`, `bos`/`eos`/`padding`, `add_bos`/`add_eos`) in `_copy_teacher_tokenizer` resolves the `DeepSeek-R1-0528-Qwen3-8B` vs. `Qwen3` 4-token mismatch (`151643`/`151645`/`151669`/`151670`) so `llama-speculative` validates and runs `d192` drafts cleanly on both teachers.

> **Note on `SpecOPD` C/Python Runtime Modules (`src/vcache/specopd.py`, `kernels/l3_specopd_kernel.c`):**
> The SpecOPD proposal-tree AVX-512 kernel and `LsDmndFillsFromSys` (`0x0243` / `0x0843`) hardware PMU watchdog are standalone C/Python prototypes and unit-tested modules alongside the PyTorch On-Policy SpecOPD trainer (`src/vcache/qwen3_draft.py`, `src/vcache/train_specopd.py`).

> **Note on BIOS SMT:** All measurements on the reference workstation were recorded with `SMT=Off` in UEFI/BIOS (`/sys/devices/system/cpu/smt/control` = `notsupported`).

---

## 03. Repository Layout

```text
zen-vcache-ai/
├── bin/
│   ├── vcache-run               # Unprivileged CPU affinity + jemalloc + hugetlb wrapper
│   ├── vcache-spec-bench        # Speculative decoding benchmark CLI
│   ├── vcache-distill-specopd   # Prototype Qwen3 12M Q8_0 micro-draft GGUF extractor
│   └── vcache-train-specopd     # On-Policy SpecOPD PyTorch trainer & GGUF Q8_0 exporter
├── src/vcache/
│   ├── topology.py              # Sysfs L3 3D V-Cache & AVX-512 flag inspector
│   ├── vector.py                # ctypes wrapper for OpenMP AVX-512 vector kernels
│   ├── specopd.py               # ProposalTree, E2E-TV, and Zen 4 PMU L3CacheWatchdog
│   ├── qwen3_draft.py           # Pure PyTorch Qwen3 micro-draft (d64/d128/d192) + GGUF exporter
│   ├── prompts_specopd.py       # 50/150 training prompts disjoint from benchmark evals
│   ├── train_specopd.py         # LlamaTeacherClient, Top-K KD Stage 1 & Reverse-KL Stage 2
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

# 6. Reproduce standalone C/Python microbenchmarks (with pre-flight, Zen 4 PMU, and E2E)
./reproduce.sh --perf --full
```

---

## 05. Zen 4 PMU Verification, NTA Guard & Practical Takeaways

- **Hardware PMU Verification (`perf stat`, 0% multiplexing on `cache_cliff`):**
  `ls_dmnd_fills_from_sys.local_ccx = 2,312,927,028` (L3 CCX hits) vs `ls_dmnd_fills_from_sys.dram_io_near = 697,828,326` (DDR5 fills). With 2MB THP, latency stays flat at **10.45–11.25 ns** up to 64 MiB (`20.42 ns` at 96 MiB) before spilling to DDR5 (**31.62–56.52 ns** at 112–256 MiB).
- **Shared L3 Contention & Non-Temporal Guard (`_mm_stream_si64`):**
  Because the 7800X3D has a single shared 96 MiB L3 without PQoS/CAT partitioning, temporal memory thrashers on cores `4-7` still evict L3 lines used by cores `0-3` (`P99 = 2772.7 µs`). Switching background streaming writes to non-temporal stores (`_mm_stream_si64`) restores P99 to **`2596.2 µs`** (`-176.5 µs`) and cuts jitter from `116.9 µs` to `83.5 µs`.
- **Asymmetric Hybrid MoE (`GGML_MOE_NTA=1`):**
  Adding `_MM_HINT_NTA` prefetching for cold DDR5 MoE experts in `ggml-cpu.c` (`Qwen3.8-35B-A3B`, `-t 6`) improves prompt processing by **`+22.2 %`** (`239.77 → 292.88 t/s`) and token generation by **`+6.7 %`** (`60.61 → 64.66 t/s`).
- **When 3D V-Cache Helps vs. When It Does Not:**
  - **High Impact (2×–5×):** 32–96 MiB working sets (vector RAG / HNSW / PQ64, `SQLite` mmap), L3-resident intent classifiers, and keeping `< 12 MiB` speculative n-gram tables hot under desktop load.
  - **Low / Zero Impact:** Large dense or MoE weights (`> 96 MiB`) streaming from DDR5 (bounded by dual-channel DDR5 bandwidth, not L3 size) and fully VRAM-resident GPU models.
