#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${SCRIPT_DIR}/benchmarks"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "===================================================================="
echo " AMD Ryzen 7 7800X3D Hardware Cache Cliff & L3 Benchmark Suite"
echo "===================================================================="

echo "Step 1/8 // Compiling Cache Cliff Pointer-Chasing Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/cache_cliff.c" -o "${BENCHMARK_DIR}/cache_cliff" -lm

echo "Step 2/8 // Compiling HugePages vs Standard Paging Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/hugepages_test.c" -o "${BENCHMARK_DIR}/hugepages_test"

echo "Step 3/8 // Compiling AVX-512 Micro-AI SIMD Benchmark..."
gcc -O3 -march=native -mavx512f -mavx512vpopcntdq -mavx512vnni -mavx512bw "${BENCHMARK_DIR}/simd_l3_rag.c" -o "${BENCHMARK_DIR}/simd_l3_rag"

echo "Step 4/8 // Compiling L3 Lockless Ring-Buffer IPC Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/l3_lockless_ipc.c" -o "${BENCHMARK_DIR}/l3_lockless_ipc" -lrt

echo "Step 5/8 // Compiling Two-Tier L3 HNSW Vector Graph Benchmark..."
gcc -O3 -march=native -mavx512f -mavx512vnni -mavx512bw "${BENCHMARK_DIR}/l3_hnsw_two_tier.c" -o "${BENCHMARK_DIR}/l3_hnsw_two_tier" -lm

echo "Step 6/8 // Compiling In-L3 Zstandard Dictionary & LZ4 Stream Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/l3_zstd_dict_stream.c" -o "${BENCHMARK_DIR}/l3_zstd_dict_stream" -lzstd -llz4

echo "Step 7/8 // Running Core C & SIMD Benchmarks..."
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
echo "Step 8/8 // Running SQLite MMAP & DDR5 Bus Contention Benchmarks..."
python3 "${BENCHMARK_DIR}/sqlite_l3_mmap.py" --mode both | tee "${RESULTS_DIR}/sqlite_mmap_run.txt"
echo ""
python3 "${BENCHMARK_DIR}/contention_benchmark.py" | tee "${RESULTS_DIR}/contention_run.txt"

echo "===================================================================="
echo " Suite completed successfully. Raw results saved in results/."
echo "===================================================================="
