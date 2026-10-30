#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="${SCRIPT_DIR}/benchmarks"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "===================================================================="
echo " AMD Ryzen 7 7800X3D Hardware Cache Cliff & L3 Benchmark Suite"
echo "===================================================================="

echo "[1/3] Compiling Cache Cliff Pointer-Chasing Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/cache_cliff.c" -o "${BENCHMARK_DIR}/cache_cliff"

echo "[2/3] Compiling HugePages vs Standard Paging Benchmark..."
gcc -O3 -march=native "${BENCHMARK_DIR}/hugepages_test.c" -o "${BENCHMARK_DIR}/hugepages_test"

echo "[3/3] Running Hardware Cache Cliff Benchmark..."
"${BENCHMARK_DIR}/cache_cliff" | tee "${RESULTS_DIR}/cache_cliff_run.txt"

echo ""
echo "Running HugePages L3 Working Buffer Test..."
"${BENCHMARK_DIR}/hugepages_test" | tee "${RESULTS_DIR}/hugepages_run.txt"

echo ""
echo "Running SQLite L3 MMAP Benchmark..."
python3 "${BENCHMARK_DIR}/sqlite_l3_mmap.py" --mode both | tee "${RESULTS_DIR}/sqlite_mmap_run.txt"

echo "===================================================================="
echo " Suite completed successfully. Raw results saved in results/."
echo "===================================================================="
