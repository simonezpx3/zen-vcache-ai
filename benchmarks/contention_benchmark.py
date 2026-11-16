#!/usr/bin/env python3
"""
3D V-Cache Contention & Isolation Benchmark (Ryzen 7 7800X3D)
Measures p50/p90/p99 tail latency of L3 Vector Search under heavy memory bus pressure.
Compares:
  1. Unpinned / Unisolated (threads migrate across all cores 0-7, colliding with heavy workers).
  2. Isolated via vcache-run policy (pinned to cores 0-3, isolating database from compute).
"""

import ctypes
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Optional external L3VectorIndex; fall back to self-contained L3-resident index
_kb_dir = Path.home() / ".agents/knowledge"
if _kb_dir.exists():
    sys.path.insert(0, str(_kb_dir))
try:
    from l3_vector_index import L3VectorIndex  # type: ignore
except ImportError:
    class L3VectorIndex:  # type: ignore[no-redef]
        """Self-contained ~200 KiB L3-resident quantized vector index fallback."""

        def __init__(self, num_docs: int = 1614, dim: int = 128):
            rng = np.random.default_rng(42)
            self.num_docs = num_docs
            self.vectors = rng.integers(-127, 128, size=(num_docs, dim), dtype=np.int8)
            self.hot_l3_footprint_bytes = self.vectors.nbytes
            self._q = rng.integers(-127, 128, size=(dim,), dtype=np.int8)

        def query(self, _text: str, top_k: int = 10, candidate_k: int = 128):
            scores = self.vectors @ self._q.astype(np.int32)
            idx = np.argpartition(scores, -candidate_k)[-candidate_k:]
            top = idx[np.argsort(scores[idx])[::-1][:top_k]]
            return top.tolist()

STRESS_BUFFER_SIZE_MB = 128  # Exceeds 96MB L3 to force DDR5 memory bus flooding
NUM_STRESS_WORKERS = 4       # Saturate cores 4-7
NUM_QUERIES = 400

SAMPLE_QUERIES = [
    "vcache l3 latence 3d cache amd ryzen 7800x3d",
    "beads tracker issues sync",
    "sqlite wal mmap performance zero-copy",
    "hyprland quickshell bar widget qml",
    "avx512 vnni dot product int8 micro model",
    "zero russian policy security hardening",
    "android adb poco f8 ultra crdt",
    "git secret guard scan commit release"
]


def stress_worker(core_id: int, stop_flag: mp.Event):
    """Generates continuous DDR5 memory bus traffic spilling past L3 cache."""
    try:
        os.sched_setaffinity(0, {core_id})
    except OSError as exc:
        sys.stderr.write(f"affinity warning: {exc}\n")

    # Allocate 128MB buffer of 64-bit integers
    size = (STRESS_BUFFER_SIZE_MB * 1024 * 1024) // 8
    buf = np.ones(size, dtype=np.int64)

    stride = 64 // 8  # 64-byte cache line stride
    while not stop_flag.is_set():
        # Sequential strided access thrashing L3 cache lines
        buf[::stride] += 1
        # Random sample read
        _ = buf[::stride * 4].sum()


def _build_nta_lib(tmp_dir: str) -> str | None:
    """Compile a minimal x86_64 _mm_stream_si64 non-temporal thrasher."""
    c_src = Path(tmp_dir) / "nta_thrash.c"
    so_out = Path(tmp_dir) / "libnta_thrash.so"
    c_src.write_text(
        "#include <immintrin.h>\n"
        "#include <stdint.h>\n"
        "void nta_sweep(int64_t *buf, size_t n) {\n"
        "    for (size_t i = 0; i < n; i += 8) {\n"
        "        _mm_stream_si64((long long *)(buf + i), (long long)(i + 1));\n"
        "    }\n"
        "    _mm_sfence();\n"
        "}\n",
        encoding="utf-8",
    )
    res = subprocess.run(
        ["gcc", "-O3", "-march=native", "-shared", "-fPIC", str(c_src), "-o", str(so_out)],
        check=False,
        capture_output=True,
    )
    return str(so_out) if res.returncode == 0 else None


def stress_worker_nta(core_id: int, stop_flag: mp.Event, so_path: str | None):
    """Generates DDR5 traffic using non-temporal stores to avoid L3 eviction."""
    try:
        os.sched_setaffinity(0, {core_id})
    except OSError as exc:
        sys.stderr.write(f"affinity warning: {exc}\n")

    size = (STRESS_BUFFER_SIZE_MB * 1024 * 1024) // 8
    buf = np.ones(size, dtype=np.int64)

    if so_path:
        lib = ctypes.CDLL(so_path)
        lib.nta_sweep.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        ptr = buf.ctypes.data
        while not stop_flag.is_set():
            lib.nta_sweep(ptr, size)
    else:
        stride = 64 // 8
        while not stop_flag.is_set():
            buf[::stride] += 1


def run_benchmark(idx: L3VectorIndex, affinity_cores: set[int] | None, desc: str) -> list[float]:
    if affinity_cores is not None:
        os.sched_setaffinity(0, affinity_cores)
    else:
        # Allow all cores (0-7)
        os.sched_setaffinity(0, set(range(8)))

    # Warmup
    for q in SAMPLE_QUERIES[:3]:
        _ = idx.query(q, top_k=10, candidate_k=128)

    latencies_us = []
    for i in range(NUM_QUERIES):
        q = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        t0 = time.perf_counter_ns()
        _ = idx.query(q, top_k=10, candidate_k=128)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    return latencies_us


def main():
    print("=" * 70)
    print("AMD Ryzen 7 7800X3D: 3D V-Cache Contention & Isolation Benchmark")
    print("=" * 70)

    idx = L3VectorIndex()
    if idx.num_docs == 0:
        print("Error: L3 Vector Index is empty.")
        sys.exit(1)

    print(f"Index loaded: {idx.num_docs} documents, {idx.hot_l3_footprint_bytes / 1024:.2f} KiB L3 Hot Index.")

    # 1. Baseline: Quiet system (no background thrashing)
    print("\nPhase 1 // Measuring Quiet Baseline (No Contention)...")
    quiet_latencies = run_benchmark(idx, {0, 1, 2, 3}, "Quiet System (Cores 0-3)")

    def stop_workers(workers_list, ev):
        ev.set()
        for p in workers_list:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
            if p.is_alive():
                p.kill()
                p.join()

    # 2. Start Temporal Stress Workers on Cores 4-7
    print(f"\nPhase 2 // Spawning {NUM_STRESS_WORKERS} Temporal Thrashers on Cores 4-7 ({STRESS_BUFFER_SIZE_MB}MB each -> DDR5)...")
    stop_event = mp.Event()
    workers = []
    for cid in range(4, 4 + NUM_STRESS_WORKERS):
        p = mp.Process(target=stress_worker, args=(cid, stop_event))
        p.start()
        workers.append(p)

    time.sleep(1.0)  # Let thrashers saturate cache lines and DRAM bus

    try:
        # Scenario A: Unpinned / Unisolated (Cores 0-7)
        print("Phase 3 // Running Scenario A: Unpinned / Default Scheduling (Cores 0-7)...")
        unpinned_latencies = run_benchmark(idx, None, "Unpinned under load")

        # Scenario B: Isolated via vcache-run policy (Pinned to Cores 0-3)
        print("Phase 4 // Running Scenario B: vcache-run Core Isolation (Cores 0-3)...")
        pinned_latencies = run_benchmark(idx, {0, 1, 2, 3}, "Isolated (Cores 0-3)")

    finally:
        print("\nStopping temporal stress workers...")
        stop_workers(workers, stop_event)

    # Scenario C: Non-Temporal (NTA) Thrashers on Cores 4-7
    with tempfile.TemporaryDirectory() as tmp_dir:
        so_path = _build_nta_lib(tmp_dir)
        print(f"\nPhase 5 // Spawning {NUM_STRESS_WORKERS} NTA Streaming Thrashers on Cores 4-7 (lib={'OK' if so_path else 'fallback'})...")
        stop_nta = mp.Event()
        nta_workers = []
        for cid in range(4, 4 + NUM_STRESS_WORKERS):
            p = mp.Process(target=stress_worker_nta, args=(cid, stop_nta, so_path))
            p.start()
            nta_workers.append(p)
        time.sleep(1.0)
        try:
            nta_latencies = run_benchmark(idx, {0, 1, 2, 3}, "Isolated vs NTA (Cores 0-3)")
        finally:
            print("Stopping NTA stress workers...")
            stop_workers(nta_workers, stop_nta)

    # Calculate statistics
    def calc_stats(lats):
        arr = np.array(lats)
        return {
            "p50_us": float(np.percentile(arr, 50)),
            "p90_us": float(np.percentile(arr, 90)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(np.max(arr)),
            "mean_us": float(np.mean(arr)),
            "std_us": float(np.std(arr))
        }

    s_quiet = calc_stats(quiet_latencies)
    s_unpinned = calc_stats(unpinned_latencies)
    s_pinned = calc_stats(pinned_latencies)
    s_nta = calc_stats(nta_latencies)

    # Format table
    print("\n" + "=" * 98)
    print(f"{'Metric':<16} | {'Quiet (0-3)':<14} | {'Unpinned':<14} | {'Pinned (Temp)':<16} | {'Pinned vs NTA':<16}")
    print("-" * 98)
    for m in ["p50_us", "p90_us", "p95_us", "p99_us", "max_us", "std_us"]:
        name = m.replace("_us", "").upper()
        if name == "STD": name = "JITTER (STD)"
        vq, vu, vp, vn = s_quiet[m], s_unpinned[m], s_pinned[m], s_nta[m]
        print(
            f"{name:<16} | {vq:>9.1f} µs   | {vu:>9.1f} µs   | "
            f"{vp:>11.1f} µs   | {vn:>11.1f} µs"
        )
    print("=" * 98)

    p99_diff = s_unpinned["p99_us"] - s_pinned["p99_us"]
    jitter_reduction = (s_unpinned["std_us"] - s_pinned["std_us"]) / s_unpinned["std_us"] * 100.0
    nta_p99_diff = s_pinned["p99_us"] - s_nta["p99_us"]

    if p99_diff >= 0:
        print(f"\nResult (Temporal): vcache-run isolation reduces p99 tail latency by {p99_diff:.1f} µs ({jitter_reduction:.1f}% lower std).")
    else:
        print(
            f"\nResult (Temporal): vcache-run isolation increases p99 by {-p99_diff:.1f} µs "
            f"(7800X3D shares 96MB L3 across all 8 cores without PQoS/CAT; pinning alone does not block L3 eviction)."
        )
    print(f"Result (NTA Guard): Non-temporal thrashers change pinned p99 by {-nta_p99_diff:+.1f} µs ({s_nta['p99_us']:.1f} µs vs {s_pinned['p99_us']:.1f} µs).")

    # Save results to JSON
    out_file = Path(__file__).resolve().parent.parent / "results/contention_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "hardware": "AMD Ryzen 7 7800X3D (8C/8T, 96MB 3D V-Cache)",
        "workload": "L3 Vector RAG Search (1252 docs)",
        "stress_condition": "4x 128MB DDR5 memory thrashing threads on cores 4-7",
        "num_queries": NUM_QUERIES,
        "quiet": s_quiet,
        "unpinned": s_unpinned,
        "vcache_run_isolated": s_pinned,
        "vcache_run_vs_nta": s_nta,
        "p99_improvement_us": p99_diff,
        "nta_p99_improvement_us": nta_p99_diff,
        "jitter_reduction_pct": jitter_reduction
    }
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Detailed metrics saved to: results/{out_file.name}")


if __name__ == "__main__":
    main()
