#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${SCRIPT_DIR}/benchmarks"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "===================================================================="
echo " AMD Ryzen 7 7800X3D Hardware Cache Cliff & L3 Benchmark Suite"
echo "===================================================================="

echo "Step 1/12 // Compiling Cache Cliff Pointer-Chasing Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/cache_cliff.c" -o "${BENCHMARK_DIR}/cache_cliff" -lm

echo "Step 2/12 // Compiling HugePages vs Standard Paging Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/hugepages_test.c" -o "${BENCHMARK_DIR}/hugepages_test"

echo "Step 3/12 // Compiling AVX-512 Micro-AI SIMD Benchmark..."
gcc -O3 -march=native -mavx512f -mavx512vpopcntdq -mavx512vnni -mavx512bw "${BENCHMARK_DIR}/simd_l3_rag.c" -o "${BENCHMARK_DIR}/simd_l3_rag"

echo "Step 4/12 // Compiling L3 Lockless Ring-Buffer IPC Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/l3_lockless_ipc.c" -o "${BENCHMARK_DIR}/l3_lockless_ipc" -lrt

echo "Step 5/12 // Compiling Two-Tier L3 HNSW Vector Graph Benchmark..."
gcc -O3 -march=native -mavx512f -mavx512vnni -mavx512bw "${BENCHMARK_DIR}/l3_hnsw_two_tier.c" -o "${BENCHMARK_DIR}/l3_hnsw_two_tier" -lm

echo "Step 6/12 // Compiling In-L3 Zstandard Dictionary & LZ4 Stream Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/l3_zstd_dict_stream.c" -o "${BENCHMARK_DIR}/l3_zstd_dict_stream" -lzstd -llz4

echo "Step 7/12 // Compiling AVX-512 VBMI (VPERMB) 1.58-Bit Ternary LUT Benchmark..."
gcc -O3 -march=native -fopenmp "${BENCHMARK_DIR}/l3_vpermb_ternary_lut.c" -o "${BENCHMARK_DIR}/l3_vpermb_ternary_lut"

echo "Step 8/12 // Compiling Non-Temporal L3 Pollution Guard Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/l3_nta_pollution_guard.c" -o "${BENCHMARK_DIR}/l3_nta_pollution_guard"

echo "Step 9/12 // Compiling 1,000,000-Vector PQ64 FastScan Benchmark..."
gcc -O3 -march=native -fopenmp "${BENCHMARK_DIR}/l3_pq64_fastscan_1m.c" -o "${BENCHMARK_DIR}/l3_pq64_fastscan_1m"

echo "Step 10/12 // Compiling Hybrid MoE L3 Shared-Expert Pinning Benchmark..."
gcc -O3 -march=native -fopenmp "${BENCHMARK_DIR}/l3_moe_shared_expert_pinning.c" -o "${BENCHMARK_DIR}/l3_moe_shared_expert_pinning"

echo "Step 11/12 // Running Core C & SIMD Benchmarks..."
"${BENCHMARK_DIR}/cache_cliff" | tee "${RESULTS_DIR}/cache_cliff_run.txt"
echo ""
"${BENCHMARK_DIR}/hugepages_test" | tee "${RESULTS_DIR}/hugepages_run.txt"
echo ""
"${BENCHMARK_DIR}/simd_l3_rag" | tee "${RESULTS_DIR}/simd_l3_rag_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_lockless_ipc" | tee "${RESULTS_DIR}/l3_lockless_ipc_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_hnsw_two_tier" | tee "${RESULTS_DIR}/l3_hnsw_two_tier_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_zstd_dict_stream" | tee "${RESULTS_DIR}/l3_zstd_dict_stream_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_vpermb_ternary_lut" | tee "${RESULTS_DIR}/l3_vpermb_ternary_lut_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_nta_pollution_guard" | tee "${RESULTS_DIR}/l3_nta_pollution_guard_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_pq64_fastscan_1m" | tee "${RESULTS_DIR}/l3_pq64_fastscan_1m_run.txt"
echo ""
"${BENCHMARK_DIR}/l3_moe_shared_expert_pinning" | tee "${RESULTS_DIR}/l3_moe_shared_expert_run.txt"

echo ""
echo "Step 12/12 // Running SQLite MMAP & DDR5 Bus Contention Benchmarks..."
python3 "${BENCHMARK_DIR}/sqlite_l3_mmap.py" --mode both | tee "${RESULTS_DIR}/sqlite_mmap_run.txt"
echo ""
python3 "${BENCHMARK_DIR}/contention_benchmark.py" | tee "${RESULTS_DIR}/contention_run.txt"

echo "===================================================================="
echo " Suite completed successfully. Raw results saved in results/."
echo "===================================================================="
