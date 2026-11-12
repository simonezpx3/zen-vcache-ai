"""
AVX-512 L3 Vector Search Engine & Benchmark
Hardware-accelerated binary Hamming scans and signed INT8 rescoring.
"""

import ctypes
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
LIB_DIR = PROJECT_ROOT / "kernels"
LIB_PATH = LIB_DIR / "libl3_vector.so"


class L3VectorEngine:
    """Wrapper around libl3_vector.so for AVX-512 VBMI / VPOPCNTDQ acceleration."""

    def __init__(self, lib_path: Path = LIB_PATH):
        self.lib_path = lib_path
        self._ensure_library()
        self._load_library()

    def _ensure_library(self):
        if not self.lib_path.exists():
            makefile = LIB_DIR / "Makefile"
            if makefile.exists():
                subprocess.run(["make", "-C", str(LIB_DIR)], check=True, capture_output=True)
            if not self.lib_path.exists():
                raise FileNotFoundError(f"Shared library not found at {self.lib_path}")

    def _load_library(self):
        self.lib = ctypes.CDLL(str(self.lib_path))

        # avx512_hamming_dist_512
        self.lib.avx512_hamming_dist_512.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.avx512_hamming_dist_512.restype = ctypes.c_uint32

        # avx512_batch_hamming_512
        self.lib.avx512_batch_hamming_512.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.lib.avx512_batch_hamming_512.restype = None

        # avx512_batch_hamming_512_mt
        self.lib.avx512_batch_hamming_512_mt.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
        ]
        self.lib.avx512_batch_hamming_512_mt.restype = None

        # avx512_int8_dot_512
        self.lib.avx512_int8_dot_512.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.avx512_int8_dot_512.restype = ctypes.c_int32

        # avx512_batch_rerank_candidates
        self.lib.avx512_batch_rerank_candidates.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.avx512_batch_rerank_candidates.restype = None

    def batch_hamming(
        self, query: bytes, doc_matrix: bytes, num_docs: int, threads: int = 1
    ) -> list[int]:
        assert len(query) == 64, "Query must be 64 bytes (512 bits)"
        assert len(doc_matrix) >= num_docs * 64, "Doc matrix buffer too small"
        distances = (ctypes.c_uint16 * num_docs)()
        if threads > 1:
            self.lib.avx512_batch_hamming_512_mt(
                query, doc_matrix, num_docs, distances, threads
            )
        else:
            self.lib.avx512_batch_hamming_512(query, doc_matrix, num_docs, distances)
        return list(distances)

    def benchmark_scan(
        self, num_vectors: int = 100_000, iterations: int = 50, threads: int = 1
    ) -> dict:
        """Measure exhaustive L3 scan performance."""
        # Create synthetic 64-byte vectors
        raw_matrix = os.urandom(num_vectors * 64)
        query = os.urandom(64)
        out_buf = (ctypes.c_uint16 * num_vectors)()

        def _scan():
            if threads > 1:
                self.lib.avx512_batch_hamming_512_mt(
                    query, raw_matrix, num_vectors, out_buf, threads
                )
            else:
                self.lib.avx512_batch_hamming_512(
                    query, raw_matrix, num_vectors, out_buf
                )

        # Warmup
        _scan()

        durations = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _scan()
            durations.append(time.perf_counter_ns() - t0)

        durations.sort()
        median_ns = durations[len(durations) // 2]
        median_us = median_ns / 1_000.0

        qps = 1_000_000_000.0 / median_ns
        bytes_scanned = num_vectors * 64
        bandwidth_gb_s = (bytes_scanned / (median_ns / 1e9)) / (1024**3)
        vectors_per_sec = (num_vectors / (median_ns / 1e9))

        return {
            "num_vectors": num_vectors,
            "index_size_mib": round(bytes_scanned / (1024**2), 2),
            "median_latency_us": round(median_us, 2),
            "qps": round(qps, 1),
            "bandwidth_gb_s": round(bandwidth_gb_s, 2),
            "vectors_per_sec": round(vectors_per_sec, 0),
        }


def main():
    print("=" * 70)
    print(" AVX-512 L3 RESIDENT VECTOR SCAN BENCHMARK (1T vs 8T OpenMP)")
    print("=" * 70)
    engine = L3VectorEngine()
    max_threads = min(8, os.cpu_count() or 1)

    for count in [100_000, 500_000, 1_000_000]:
        for t in sorted({1, max_threads}):
            res = engine.benchmark_scan(num_vectors=count, iterations=30, threads=t)
            print(
                f"Index: {count:>9,} vec ({res['index_size_mib']:>5.2f} MiB) | T={t} | "
                f"{res['median_latency_us']:>7.2f} µs | {res['qps']:>9,.1f} QPS | "
                f"{res['bandwidth_gb_s']:>6.2f} GB/s"
            )
        print("-" * 70)


if __name__ == "__main__":
    main()
