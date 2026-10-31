#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${SCRIPT_DIR}/benchmarks"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "===================================================================="
echo " AMD Ryzen 7 7800X3D Hardware Cache Cliff & L3 Benchmark Suite"
echo "===================================================================="

echo "Step 1/5 // Compiling Cache Cliff Pointer-Chasing Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/cache_cliff.c" -o "${BENCHMARK_DIR}/cache_cliff" -lm

echo "Step 2/5 // Compiling HugePages vs Standard Paging Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/hugepages_test.c" -o "${BENCHMARK_DIR}/hugepages_test"

echo "Step 3/5 // Compiling AVX-512 Micro-AI SIMD Benchmark..."
gcc -O3 -march=native -mavx512f -mavx512vpopcntdq -mavx512vnni -mavx512bw "${BENCHMARK_DIR}/simd_l3_rag.c" -o "${BENCHMARK_DIR}/simd_l3_rag"

echo "Step 4/5 // Running Hardware Cache Cliff Benchmark..."
"${BENCHMARK_DIR}/cache_cliff" | tee "${RESULTS_DIR}/cache_cliff_run.txt"

echo ""
echo "Running HugePages L3 Working Buffer Test..."
"${BENCHMARK_DIR}/hugepages_test" | tee "${RESULTS_DIR}/hugepages_run.txt"

echo ""
echo "Running AVX-512 Micro-AI SIMD Kernels Benchmark..."
"${BENCHMARK_DIR}/simd_l3_rag" | tee "${RESULTS_DIR}/simd_l3_rag_run.txt"

echo ""
echo "Running SQLite L3 MMAP Benchmark..."
python3 "${BENCHMARK_DIR}/sqlite_l3_mmap.py" --mode both | tee "${RESULTS_DIR}/sqlite_mmap_run.txt"

echo ""
echo "Step 5/5 // Running Memory Bus Contention & Core Isolation Benchmark..."
python3 "${BENCHMARK_DIR}/contention_benchmark.py" | tee "${RESULTS_DIR}/contention_run.txt"

echo "===================================================================="
echo " Suite completed successfully. Raw results saved in results/."
echo "===================================================================="
